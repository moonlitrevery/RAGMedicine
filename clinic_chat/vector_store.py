from __future__ import annotations

import json
from pathlib import Path

import chromadb

from clinic_chat.chroma_embeddings import TfidfSvdEmbeddingFunction
from clinic_chat.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDER_PATH, KNOWLEDGE_JSON, PROJECT_ROOT
from clinic_chat.local_embeddings import fit_and_save_pipeline


def get_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Coleção com embeddings TF-IDF+SVD (ficheiro em data/tfidf_svd.joblib)."""
    if not EMBEDDER_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo de embeddings em falta: {EMBEDDER_PATH}. Corra a ingestão (main.py ou scripts/ingest.py)."
        )
    client = get_client()
    ef = TfidfSvdEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Base de conhecimento da clínica"},
    )


def ingest_from_json(json_path: Path | None = None) -> int:
    """Carrega documentos no ChromaDB. Retorna número de documentos indexados."""
    path = json_path or KNOWLEDGE_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de conhecimento não encontrado: {path}")

    with path.open(encoding="utf-8") as f:
        docs = json.load(f)

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [{"title": d.get("title", ""), "source": str(path.relative_to(PROJECT_ROOT))} for d in docs]

    fit_and_save_pipeline(texts)
    collection = get_collection()
    ef = TfidfSvdEmbeddingFunction()
    embeddings = ef(texts)

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(docs)


def retrieve(query: str, n_results: int = 4) -> list[str]:
    """Devolve trechos de contexto mais relevantes para a pergunta."""
    if not EMBEDDER_PATH.is_file():
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    ef = TfidfSvdEmbeddingFunction()
    q_emb = ef([query])
    result = collection.query(query_embeddings=q_emb, n_results=n_results)
    documents = result.get("documents") or []
    if not documents or not documents[0]:
        return []
    return list(documents[0])
