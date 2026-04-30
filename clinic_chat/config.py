import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_JSON = PROJECT_ROOT / "knowledge_base" / "documents.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
EMBEDDER_PATH = PROJECT_ROOT / "data" / "tfidf_svd.joblib"
# Dimensão máxima do SVD (ajustada ao nº de documentos em ingestão)
SVD_COMPONENTS = 96

# Nome novo para não colidir com coleções antigas (embeddings ONNX vs TF-IDF+SVD).
COLLECTION_NAME = "clinica_conhecimento_tfidf"
MEMORY_MAX_MESSAGES = 20  # ~10 interações (user + assistant)
RAG_TOP_K = 4

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
