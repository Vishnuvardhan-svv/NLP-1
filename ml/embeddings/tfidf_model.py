from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def _similarity(vectorizer, resume_text: str, job_text: str) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    try:
        matrix = vectorizer.fit_transform([resume_text, job_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return 0.0
def tfidf_similarity(resume_text: str, job_text: str) -> float:
    return _similarity(TfidfVectorizer(ngram_range=(1, 2), stop_words="english"), resume_text, job_text)
def bow_similarity(resume_text: str, job_text: str) -> float:
    return _similarity(CountVectorizer(stop_words="english"), resume_text, job_text)
