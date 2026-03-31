import argparse
import json
import os
import sys
from pathlib import Path

HF_CACHE_DIR = Path(
    os.getenv("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")
)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    str(HF_CACHE_DIR / "sentence_transformers"),
)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", os.getenv("HF_HUB_ETAG_TIMEOUT", "60"))
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "120"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.deps.container import get_llm
from app.services.rag_service import PROMPT_TEMPLATE
from app.services.rerank_service import retrieve_documents


TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "ragas_results.json"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current RAG pipeline with Ragas."
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=TEST_SET_PATH,
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-n documents to keep after reranking for each question.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI chat model used by Ragas for LLM-based evaluation.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model used by Ragas metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the evaluation summary and per-sample rows.",
    )
    return parser.parse_args()


def ensure_openai_api_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to the project root .env file before running evaluation."
    )


def load_test_dataset(path: Path) -> Dataset:
    dataset_dict = load_dataset("json", data_files=str(path))
    return dataset_dict["train"]


def build_reference_answer(row: dict) -> str:
    reference_points = row.get("reference_points", [])
    if not reference_points:
        return ""
    return " ".join(reference_points)


def generate_response(question: str, retrieved_contexts: list[str]) -> str:
    llm = get_llm()
    prompt = PROMPT_TEMPLATE.format(
        context="\n\n".join(retrieved_contexts),
        question=question,
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def prepare_ragas_dataset(test_dataset: Dataset, k: int) -> Dataset:
    rows = {
        "question_id": [],
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": [],
        "chapter": [],
        "section": [],
    }

    for sample in tqdm(
        test_dataset,
        total=len(test_dataset),
        desc="Preparing evaluation samples",
        unit="sample",
    ):
        question = sample["question"]
        docs = retrieve_documents(question, top_n=k)
        retrieved_contexts = [doc.page_content for doc in docs]
        response = generate_response(question, retrieved_contexts)

        rows["question_id"].append(sample["id"])
        rows["user_input"].append(question)
        rows["retrieved_contexts"].append(retrieved_contexts)
        rows["response"].append(response)
        rows["reference"].append(build_reference_answer(sample))
        rows["chapter"].append(sample.get("chapter", ""))
        rows["section"].append(sample.get("section", ""))

    return Dataset.from_dict(rows)


def run_ragas_eval(
    ragas_dataset: Dataset, judge_model: str, embedding_model: str
):
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    evaluator_llm = llm_factory(
        judge_model,
        client=openai_client,
        temperature=0,
    )
    answer_relevancy_embeddings = OpenAIEmbeddings(model=embedding_model)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=answer_relevancy_embeddings,
            strictness=1,
        ),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    return evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=False,
        show_progress=True,
    )


def save_results(result, ragas_dataset: Dataset, output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rows = ragas_dataset.to_list()
    scored_rows = []
    for sample_row, score_row in zip(sample_rows, result.scores):
        scored_rows.append({**sample_row, **score_row})

    payload = {
        "config": {
            "test_set": str(args.test_set),
            "k": args.k,
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model,
        },
        "metrics": result._repr_dict,
        "samples": scored_rows,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(result) -> None:
    print("Ragas evaluation complete")
    for metric_name, metric_value in result._repr_dict.items():
        print(f"{metric_name}: {metric_value:.4f}")


def main() -> None:
    args = parse_args()
    ensure_openai_api_key()
    test_dataset = load_test_dataset(args.test_set)
    print("Preparing RAG responses...")
    ragas_dataset = prepare_ragas_dataset(test_dataset, k=args.k)
    print("Running Ragas evaluation...")
    result = run_ragas_eval(
        ragas_dataset=ragas_dataset,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )
    save_results(result, ragas_dataset, args.output, args)
    print_summary(result)
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
