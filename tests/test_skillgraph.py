"""Unit tests using clearly synthetic resume and job-description text."""

from io import BytesIO

import numpy as np
import pytest

from ml.embeddings.tfidf_model import bow_similarity, tfidf_similarity
from ml.embeddings.word2vec_model import document_embedding, train_word2vec, word2vec_similarity
from ml.preprocessing.text_cleaning import clean_text, tokenize
from ml.preprocessing.text_extraction import extract_profile_sections, extract_text
from ml.skill_analysis.job_analysis import extract_job_requirements
from ml.skill_analysis.recommendations import build_roadmap
from ml.skill_analysis.skill_extraction import extract_skills
from ml.skill_analysis.skill_graph import build_skill_graph, related_skills
from ml.skill_analysis.skill_matching import compare_skills
from ml.skill_analysis.skill_normalization import normalize_skill

SYNTHETIC_RESUME = "Synthetic resume: Python, ReactJS, Node, Mongo. Education:\nBSc Computing"
SYNTHETIC_JOB = "Required skills:\nPython, React, Express, Docker, AWS\nPreferred skills:\nMongoDB"

def test_pdf_extraction():
    import pymupdf
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Synthetic PDF resume with Python experience")
    buffer = BytesIO(document.tobytes())
    document.close()
    assert "Python experience" in extract_text("sample.pdf", buffer.getvalue())

def test_profile_section_extraction():
    sections = extract_profile_sections("EDUCATION\nSynthetic BSc\nEXPERIENCE\nSynthetic internship\nPROJECTS\nSynthetic app")
    assert sections["education"] == ["Synthetic BSc"]
    assert sections["experience"] == ["Synthetic internship"]
    assert sections["projects"] == ["Synthetic app"]

def test_text_preprocessing():
    assert clean_text("  Python\u00a0+  SQL! ") == "python + sql"
    assert "python" in tokenize("Python and SQL")

def test_skill_normalization_and_phrase_extraction():
    assert normalize_skill("JS") == "JavaScript"
    assert normalize_skill("ReactJS") == "React"
    assert extract_skills("JS and javascript; REACTJS, React; mongo") == ["JavaScript", "React", "MongoDB"]
    assert extract_skills("") == []

def test_job_required_and_preferred_extraction():
    result = extract_job_requirements(SYNTHETIC_JOB)
    assert result["required"] == ["Python", "React", "Express", "Docker", "AWS"]
    assert result["preferred"] == ["MongoDB"]

def test_skill_match_and_coverage_math():
    result = compare_skills(["Python", "React", "Pandas"], ["Python", "React", "Docker"], ["AWS", "Pandas"])
    assert result["matching"] == ["Python", "React", "Pandas"]
    assert result["missing"] == ["Docker", "AWS"]
    assert result["additional"] == []
    assert result["exact_coverage"] == pytest.approx(60.0)
    assert result["weighted_coverage"] == pytest.approx(62.5)
    assert compare_skills([], [], [])['exact_coverage'] == 0

def test_weighted_coverage_changes_when_importance_changes():
    skills = ["Python"]
    baseline = compare_skills(skills, ["Python", "Docker"], ["React"], required_weight=2, preferred_weight=1)
    changed = compare_skills(skills, ["Python", "Docker"], ["React"], required_weight=4, preferred_weight=1)
    assert baseline["exact_coverage"] == changed["exact_coverage"] == pytest.approx(100 / 3)
    assert changed["weighted_coverage"] > baseline["weighted_coverage"]

def test_bow_and_tfidf_similarity():
    assert 0 < bow_similarity("Python data", "Python engineer") <= 1
    assert 0 < tfidf_similarity("Python data", "Python engineer") <= 1
    assert tfidf_similarity("", "Python") == 0

def test_local_word2vec_training_and_unknown_words():
    model = train_word2vec("Python data science", "Python machine learning", sg=0, vector_size=20)
    vector = document_embedding("completely unknown qqqword", model)
    assert vector.shape == (20,)
    assert np.all(vector == 0)
    assert 0 <= word2vec_similarity("Python data", "Python learning", sg=1) <= 1

def test_transformer_cosine_with_stub_encoder():
    from ml.embeddings.transformer_embeddings import transformer_similarity
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0], [0.8, 0.6]])
    assert transformer_similarity("synthetic resume", "synthetic role", FakeEncoder()) == pytest.approx(0.8)
    assert transformer_similarity("", "role", FakeEncoder()) == 0

def test_graph_and_related_skills():
    graph = build_skill_graph()
    assert graph.has_edge("JavaScript", "React")
    assert graph.has_edge("Python", "Django")
    assert graph.has_edge("Docker", "Kubernetes")
    assert "React" in graph
    related = related_skills(["React"], ["JavaScript"])
    assert any(item["target"] == "React" and item["related"] == "JavaScript" for item in related)

def test_roadmap_order_includes_missing_prerequisites_and_resources():
    roadmap = build_roadmap(["Kubernetes"], [])
    skills = [entry["skill"] for entry in roadmap]
    assert "Docker" in skills and "Kubernetes" in skills
    assert skills.index("Docker") < skills.index("Kubernetes")
    assert roadmap[skills.index("Docker")]["resource"]["source"] == "Docker Docs"

