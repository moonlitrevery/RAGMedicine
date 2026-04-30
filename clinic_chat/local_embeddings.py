from __future__ import annotations

import joblib
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from clinic_chat.config import EMBEDDER_PATH, SVD_COMPONENTS


def fit_and_save_pipeline(texts: list[str], dest: Path | None = None) -> Pipeline:
    """Encaixa TF-IDF + SVD em memória; sem downloads de rede."""
    path = dest or EMBEDDER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(texts)
    if n_samples == 0:
        raise ValueError("Lista de textos vazia.")

    max_features = min(512, max(32, sum(len(t) for t in texts) // 2 or 32))
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    # TruncatedSVD: n_components < min(n_samples, n_features)
    upper = min(X.shape[0], X.shape[1])
    n_comp = min(SVD_COMPONENTS, max(1, upper - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X)
    pipe = Pipeline([("tfidf", vec), ("svd", svd)])
    joblib.dump(pipe, path)
    return pipe


def load_pipeline(path: Path | None = None) -> Pipeline:
    p = path or EMBEDDER_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Modelo de embeddings não encontrado: {p}. Execute a ingestão primeiro.")
    return joblib.load(p)


def embed_texts(pipe: Pipeline, texts: list[str]) -> list[list[float]]:
    X = pipe.transform(texts)
    return X.tolist()
