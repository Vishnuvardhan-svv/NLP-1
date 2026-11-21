"""Local Gensim Word2Vec training and mean-pooled document embeddings."""

import re
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_PATH = Path(__file__).resolve().parents[2] / "datasets" / "corpus" / "w2v_training.txt"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.casefold())

def train_word2vec(resume_text: str, job_text: str, sg: int = 1, vector_size: int = 100, seed: int = 17) -> Word2Vec:
    """Train a small model locally on the included synthetic corpus and this analysis."""
    corpus = [tokenize(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line and not line.startswith("SYNTHETIC")]
    corpus += [tokenize(resume_text), tokenize(job_text)]
    corpus = [sentence for sentence in corpus if sentence]
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=5, min_count=1, workers=1, sg=sg, seed=seed, epochs=50)
    return model

def document_embedding(text: str, model: Word2Vec) -> np.ndarray:
    tokens = [token for token in tokenize(text) if token in model.wv]
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean([model.wv[token] for token in tokens], axis=0)

def word2vec_similarity(resume_text: str, job_text: str, sg: int = 1) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    model = train_word2vec(resume_text, job_text, sg=sg)
    left, right = document_embedding(resume_text, model), document_embedding(job_text, model)
    if not np.any(left) or not np.any(right):
        return 0.0
    return float(np.clip(cosine_similarity([left], [right])[0, 0], 0.0, 1.0))
