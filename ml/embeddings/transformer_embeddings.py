import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = os.getenv("TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE = PROJECT_ROOT / "models" / "transformers"
def load_transformer_model(local_files_only: bool | None = None) -> SentenceTransformer:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    if local_files_only is True:
        return SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), local_files_only=True, device="cpu")
    try:
        return SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), local_files_only=True, device="cpu")
    except (OSError, ValueError):
        if local_files_only is True:
            raise
        return SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), local_files_only=False, device="cpu")
def transformer_similarity(resume_text: str, job_text: str, model: SentenceTransformer) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    embeddings = model.encode([resume_text, job_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return float(np.clip(cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0], 0.0, 1.0))
