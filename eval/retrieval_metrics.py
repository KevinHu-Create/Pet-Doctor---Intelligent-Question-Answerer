import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.deps.container import get_vectorstore


TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.json"


@dataclass
class QueryMetrics:
    question_id: str
    question: str
    gold_chapter: str
    retrieved_chapters: list[str]
    hits: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float


def load_test_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload["qas"]


def compute_precision_at_k(hits: int, k: int) -> float:
    return hits / k if k else 0.0


def compute_recall_at_k(hits: int) -> float:
    # Relevance is defined only at the chapter level.
    # Each question has exactly one target chapter, so recall@k is binary:
    # 1.0 if at least one retrieved chunk comes from that chapter, else 0.0.
    return 1.0 if hits > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_query(vectorstore, question_row: dict, k: int) -> QueryMetrics:
    question = question_row["question"]
    gold_chapter = question_row["chapter"]

    docs = vectorstore.similarity_search(question, k=k)
    retrieved_chapters = [doc.metadata.get("chapter", "") for doc in docs]
    hits = sum(1 for chapter in retrieved_chapters if chapter == gold_chapter)

    precision_at_k = compute_precision_at_k(hits, k)
    recall_at_k = compute_recall_at_k(hits)
    f1_at_k = compute_f1(precision_at_k, recall_at_k)

    return QueryMetrics(
        question_id=question_row["id"],
        question=question,
        gold_chapter=gold_chapter,
        retrieved_chapters=retrieved_chapters,
        hits=hits,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        f1_at_k=f1_at_k,
    )


def evaluate_retrieval(k: int, test_set_path: Path) -> tuple[dict, list[QueryMetrics]]:
    vectorstore = get_vectorstore()
    qas = load_test_set(test_set_path)
    results = [evaluate_query(vectorstore, question_row, k) for question_row in qas]

    summary = {
        "num_questions": len(results),
        "k": k,
        "macro_precision_at_k": mean(result.precision_at_k for result in results),
        "macro_recall_at_k": mean(result.recall_at_k for result in results),
        "macro_f1_at_k": mean(result.f1_at_k for result in results),
    }
    return summary, results


def print_report(summary: dict, results: list[QueryMetrics]) -> None:
    print(f"Evaluated {summary['num_questions']} questions")
    print(f"k = {summary['k']}")
    print(f"Macro Precision@{summary['k']}: {summary['macro_precision_at_k']:.4f}")
    print(f"Macro Recall@{summary['k']}: {summary['macro_recall_at_k']:.4f}")
    print(f"Macro F1@{summary['k']}: {summary['macro_f1_at_k']:.4f}")
    print()
    print("Per-question results")

    for result in results:
        print(
            f"{result.question_id} | "
            f"P@{summary['k']}={result.precision_at_k:.4f} | "
            f"R@{summary['k']}={result.recall_at_k:.4f} | "
            f"F1@{summary['k']}={result.f1_at_k:.4f} | "
            f"hits={result.hits} | "
            f"gold={result.gold_chapter}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance using chapter-level relevance."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k documents to retrieve for each question.",
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=TEST_SET_PATH,
        help="Path to the evaluation test set JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, results = evaluate_retrieval(k=args.k, test_set_path=args.test_set)
    print_report(summary, results)


if __name__ == "__main__":
    main()
