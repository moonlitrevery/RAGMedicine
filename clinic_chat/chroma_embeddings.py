"""Adaptador Chroma para embeddings TF-IDF+SVD (offline, sem download ONNX)."""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from clinic_chat.config import EMBEDDER_PATH
from clinic_chat.local_embeddings import embed_texts, load_pipeline


class TfidfSvdEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embeddings locais — evita timeout ao descarregar o modelo ONNX do Chroma."""

    def __init__(self, pipeline_path=None):
        self._path = pipeline_path or EMBEDDER_PATH
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is None:
            self._pipe = load_pipeline(self._path)

    def __call__(self, input: Documents) -> Embeddings:
        self._ensure()
        return embed_texts(self._pipe, list(input))

    def name(self) -> str:
        return "tfidf_svd_local"
