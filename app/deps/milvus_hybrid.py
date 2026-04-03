from functools import lru_cache
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
# `pymilvus.model` is a runtime loader that forwards to `milvus_model`.
# Import the real package directly so static analyzers can resolve it.
from milvus_model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_array

from app.core.device import get_torch_device
from app.core.settings import settings


class _BGEM3Encoder:
    def __init__(self) -> None:
        self._model = BGEM3EmbeddingFunction(
            model_name=settings.HF_EMBED_MODEL,
            batch_size=settings.BGE_M3_BATCH_SIZE,
            device=get_torch_device(),
            normalize_embeddings=True,
            use_fp16=settings.BGE_M3_USE_FP16,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self._last_mode: str | None = None
        self._last_inputs: tuple[str, ...] | None = None
        self._last_output: dict[str, Any] | None = None

    def encode_documents(self, texts: Sequence[str]) -> dict[str, Any]:
        return self._encode(texts, mode="documents")

    def encode_queries(self, queries: Sequence[str]) -> dict[str, Any]:
        return self._encode(queries, mode="queries")

    def _encode(self, texts: Sequence[str], mode: str) -> dict[str, Any]:
        normalized_texts = tuple(texts)
        if (
            self._last_mode == mode
            and self._last_inputs == normalized_texts
            and self._last_output is not None
        ):
            return self._last_output

        if mode == "queries":
            output = self._model.encode_queries(list(normalized_texts))
        else:
            output = self._model.encode_documents(list(normalized_texts))

        self._last_mode = mode
        self._last_inputs = normalized_texts
        self._last_output = output
        return output


class BGEM3DenseEmbeddings(Embeddings):
    def __init__(self, encoder: _BGEM3Encoder) -> None:
        self._encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        dense_vectors = self._encoder.encode_documents(texts)["dense"]
        return [_dense_vector_to_list(vector) for vector in dense_vectors]

    def embed_query(self, text: str) -> list[float]:
        dense_vectors = self._encoder.encode_queries([text])["dense"]
        return _dense_vector_to_list(dense_vectors[0])


class BGEM3SparseEmbeddings(BaseSparseEmbedding):
    def __init__(self, encoder: _BGEM3Encoder) -> None:
        self._encoder = encoder

    def embed_query(self, query: str) -> dict[int, float]:
        sparse_matrix = self._encoder.encode_queries([query])["sparse"]
        return _sparse_matrix_to_dicts(sparse_matrix)[0]

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        sparse_matrix = self._encoder.encode_documents(texts)["sparse"]
        return _sparse_matrix_to_dicts(sparse_matrix)


def _dense_vector_to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]


def _sparse_matrix_to_dicts(sparse_matrix: csr_array) -> list[dict[int, float]]:
    sparse_rows: list[dict[int, float]] = []
    for row_index in range(sparse_matrix.shape[0]):
        start = sparse_matrix.indptr[row_index]
        end = sparse_matrix.indptr[row_index + 1]
        row = {
            int(index): float(value)
            for index, value in zip(
                sparse_matrix.indices[start:end],
                sparse_matrix.data[start:end],
            )
        }
        sparse_rows.append(row)
    return sparse_rows


def hybrid_retrieval_enabled() -> bool:
    return settings.RAG_HYBRID_ENABLED


@lru_cache
def get_bge_m3_encoder() -> _BGEM3Encoder:
    return _BGEM3Encoder()


@lru_cache
def get_dense_embeddings() -> Embeddings:
    if hybrid_retrieval_enabled():
        return BGEM3DenseEmbeddings(get_bge_m3_encoder())

    return HuggingFaceEmbeddings(
        model_name=settings.HF_EMBED_MODEL,
        cache_folder=str(settings.HF_CACHE_DIR / "sentence_transformers"),
        model_kwargs={"device": get_torch_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_sparse_embeddings() -> BaseSparseEmbedding:
    if not hybrid_retrieval_enabled():
        raise RuntimeError("Sparse embeddings are only available when hybrid retrieval is enabled.")
    return BGEM3SparseEmbeddings(get_bge_m3_encoder())


def get_vectorstore_embedding_function() -> Embeddings | list[Embeddings | BaseSparseEmbedding]:
    if hybrid_retrieval_enabled():
        return [get_dense_embeddings(), get_sparse_embeddings()]
    return get_dense_embeddings()


def get_vectorstore_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "consistency_level": settings.MILVUS_CONSISTENCY_LEVEL,
        "index_params": _build_index_params(),
        "search_params": _build_search_params(),
    }

    if hybrid_retrieval_enabled():
        kwargs["vector_field"] = [
            settings.MILVUS_DENSE_VECTOR_FIELD,
            settings.MILVUS_SPARSE_VECTOR_FIELD,
        ]

    return kwargs


def _build_index_params() -> dict[str, Any] | list[dict[str, Any]]:
    dense_index = _build_dense_index_params()
    if not hybrid_retrieval_enabled():
        return dense_index

    return [
        dense_index,
        {
            "index_type": settings.MILVUS_SPARSE_INDEX_TYPE,
            "metric_type": settings.MILVUS_SPARSE_METRIC_TYPE,
            "params": {
                "drop_ratio_build": settings.MILVUS_SPARSE_DROP_RATIO_BUILD,
            },
        },
    ]


def _build_search_params() -> dict[str, Any] | list[dict[str, Any]]:
    dense_search = _build_dense_search_params()
    if not hybrid_retrieval_enabled():
        return dense_search

    return [
        dense_search,
        {
            "metric_type": settings.MILVUS_SPARSE_METRIC_TYPE,
            "params": {
                "drop_ratio_search": settings.MILVUS_SPARSE_DROP_RATIO_SEARCH,
            },
        },
    ]


def _build_dense_index_params() -> dict[str, Any]:
    index_type = settings.MILVUS_DENSE_INDEX_TYPE
    metric_type = settings.MILVUS_DENSE_METRIC_TYPE

    if index_type == "AUTOINDEX":
        return {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {},
        }

    if index_type in {
        "IVF_FLAT",
        "IVF_SQ8",
        "IVF_PQ",
        "GPU_IVF_FLAT",
        "GPU_IVF_PQ",
    }:
        return {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": settings.MILVUS_DENSE_INDEX_NLIST},
        }

    return {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": {
            "M": settings.MILVUS_DENSE_INDEX_M,
            "efConstruction": settings.MILVUS_DENSE_EF_CONSTRUCTION,
        },
    }


def _build_dense_search_params() -> dict[str, Any]:
    index_type = settings.MILVUS_DENSE_INDEX_TYPE
    metric_type = settings.MILVUS_DENSE_METRIC_TYPE

    if index_type == "AUTOINDEX":
        return {"metric_type": metric_type, "params": {}}

    if index_type in {
        "IVF_FLAT",
        "IVF_SQ8",
        "IVF_PQ",
        "GPU_IVF_FLAT",
        "GPU_IVF_PQ",
    }:
        return {
            "metric_type": metric_type,
            "params": {"nprobe": settings.MILVUS_DENSE_SEARCH_NPROBE},
        }

    return {
        "metric_type": metric_type,
        "params": {"ef": settings.MILVUS_DENSE_SEARCH_EF},
    }
