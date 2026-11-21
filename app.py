"""SkillGraph AI: local-first resume and job skill analysis in Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ml.config import PREFERRED_SKILL_WEIGHT, REQUIRED_SKILL_WEIGHT
from ml.embeddings.tfidf_model import bow_similarity, tfidf_similarity
from ml.embeddings.word2vec_model import word2vec_similarity
from ml.preprocessing.text_extraction import extract_profile_sections, extract_text
from ml.skill_analysis.job_analysis import extract_job_requirements
from ml.skill_analysis.recommendations import build_roadmap
from ml.skill_analysis.skill_extraction import extract_skills
from ml.skill_analysis.skill_graph import build_skill_graph, related_skills
from ml.skill_analysis.skill_matching import compare_skills
from ml.skill_analysis.skill_normalization import load_taxonomy

st.set_page_config(page_title="SkillGraph AI", page_icon="📚", layout="wide")
PAGES = ["Dashboard", "Resume Upload", "Job Description", "Skill Gap Analysis", "Similarity Analysis", "Learning Roadmap"]
st.sidebar.title("SkillGraph AI")
page = st.sidebar.radio("Navigate", PAGES, key="page_nav")
for key, default in {"resume_text": "", "resume_filename": "", "resume_skills": [], "resume_skill_source": None, "job_text": "", "required": [], "preferred": [], "analysis": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def show_list(label: str, values: list[str]) -> None:
    st.write(f"**{label}:** " + (", ".join(values) if values else "None identified"))

@st.cache_data
def cached_taxonomy():
    return load_taxonomy()

def navigate_to(target: str) -> None:
    st.session_state.page_nav = target

@st.cache_resource(show_spinner="Loading local Transformer model…")
def load_transformer(local_files_only: bool = False):
    from ml.embeddings.transformer_embeddings import load_transformer_model
    return load_transformer_model(local_files_only=local_files_only)

if page == "Dashboard":
    st.title("SkillGraph AI")
    st.caption("Interactive overview of the resume and job description in this session")
    analysis = st.session_state.analysis
    candidate_skills = st.session_state.resume_skills or extract_skills(st.session_state.resume_text)
    job_skills = list(dict.fromkeys(st.session_state.required + st.session_state.preferred))
    exact = analysis["exact_coverage"] if analysis else None
    weighted = analysis["weighted_coverage"] if analysis else None
    missing_count = len(analysis["missing"]) if analysis else None
    a, b, c = st.columns(3)
    a.metric("Exact skill coverage", f"{exact:.1f}%" if exact is not None else "—")
    b.metric("Weighted skill coverage", f"{weighted:.1f}%" if weighted is not None else "—")
    c.metric("Missing job skills", missing_count if missing_count is not None else "—")
    if exact is not None:
        st.progress(int(max(0, min(100, round(exact)))), text=f"Exact coverage: {exact:.1f}%")

    st.subheader("Quick actions")
    action1, action2, action3 = st.columns(3)
    action1.button("Upload or review resume", on_click=navigate_to, args=("Resume Upload",), use_container_width=True)
    action2.button("Add or review job", on_click=navigate_to, args=("Job Description",), use_container_width=True)
    action3.button("Run or update analysis", on_click=navigate_to, args=("Skill Gap Analysis",), use_container_width=True, disabled=not bool(st.session_state.resume_text and st.session_state.job_text.strip()))

    st.subheader("Analysis chart")
    chart_mode = st.radio("Choose a chart", ["Skill coverage", "Text similarity"], horizontal=True, key="dashboard_chart_mode")
    if analysis:
        if chart_mode == "Skill coverage":
            chart = pd.Series({"Matched": len(analysis["matching"]), "Missing": len(analysis["missing"]), "Additional": len(analysis["additional"])}, name="Skills")
            st.bar_chart(chart, y_label="Number of skills")
        else:
            similarity = analysis["similarity"]
            values = {"Bag of Words": similarity["bow"] * 100, "TF-IDF": similarity["tfidf"] * 100, "Word2Vec": similarity["word2vec"] * 100}
            if similarity.get("transformer") is not None:
                values["Transformer"] = similarity["transformer"] * 100
            st.bar_chart(pd.Series(values, name="Similarity (%)"), y_label="Similarity (%)")
    else:
        st.info("Add a resume and job description, then run an analysis to populate the charts.")

    taxonomy = cached_taxonomy()
    categories = list(taxonomy)
    selected_category = st.selectbox("Explore a skill category", categories, key="dashboard_skill_category")
    all_category_skills = list(taxonomy[selected_category])
    resume_set = set(candidate_skills)
    required_set = set(analysis["required"] if analysis else st.session_state.required)
    preferred_set = set(analysis["preferred"] if analysis else st.session_state.preferred)
    matching_set = set(analysis["matching"] if analysis else [])
    missing_set = set(analysis["missing"] if analysis else [])
    show_all_skills = st.checkbox("Show all taxonomy skills in this category", value=False)
    visible_skills = all_category_skills if show_all_skills else [skill for skill in all_category_skills if skill in resume_set | required_set | preferred_set]
    if visible_skills:
        rows = []
        for skill in visible_skills:
            if skill in matching_set:
                status = "Matched"
            elif skill in missing_set or skill in required_set | preferred_set:
                status = "Missing from resume"
            elif skill in resume_set:
                status = "Additional resume skill"
            else:
                status = "Not in current inputs"
            rows.append({"Skill": skill, "In resume": skill in resume_set, "Required": skill in required_set, "Preferred": skill in preferred_set, "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No skills from this category appear in the current resume or job description yet.")
    st.caption("All scores are descriptive comparisons, not probabilities of getting hired.")

elif page == "Resume Upload":
    st.title("Resume Upload")
    uploaded = st.file_uploader("Upload a PDF or DOCX resume (10 MB maximum)", type=["pdf", "docx"], max_upload_size=10)
    if uploaded:
        try:
            content = extract_text(uploaded.name, uploaded.getvalue())
            if not content:
                st.warning("No readable text found. Image-only PDFs are not OCR processed.")
            else:
                st.session_state.resume_text = content
                st.session_state.resume_filename = uploaded.name
                st.session_state.analysis = None
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if st.session_state.resume_text:
        st.success(f"Loaded {st.session_state.resume_filename}")
        detected = extract_skills(st.session_state.resume_text)
        taxonomy = sorted({skill for skills in cached_taxonomy().values() for skill in skills})
        if st.session_state.resume_skill_source != st.session_state.resume_text:
            st.session_state.resume_skill_source = st.session_state.resume_text
            st.session_state.resume_skills = detected.copy()
            st.session_state["resume_skill_editor"] = detected.copy()
            st.session_state.analysis = None
        show_list("Skills detected automatically", detected)
        selected_resume_skills = st.multiselect(
            "Confirm or add your resume skills",
            taxonomy,
            key="resume_skill_editor",
            help="If automatic extraction missed skills, select them here. Coverage uses this reviewed list.",
        )
        if selected_resume_skills != st.session_state.resume_skills:
            st.session_state.resume_skills = selected_resume_skills
            st.session_state.analysis = None
        if not selected_resume_skills:
            st.warning("No resume skills are selected. Coverage will be 0% until you select skills from the taxonomy.")
        with st.expander("Education, experience, and project sections (best effort)"):
            for section, lines in extract_profile_sections(st.session_state.resume_text).items():
                st.markdown(f"**{section.title()}**")
                st.write("\n".join(lines) if lines else "No clearly headed section found.")
        with st.expander("Extracted resume text"):
            st.text_area("Resume text", st.session_state.resume_text, height=300, disabled=True)
        if st.button("Clear resume from this session"):
            st.session_state.resume_text = ""
            st.session_state.resume_filename = ""
            st.session_state.resume_skills = []
            st.session_state.resume_skill_source = None
            st.session_state.analysis = None
            st.rerun()
    st.caption("The raw resume stays in this browser session and is not saved to disk.")

elif page == "Job Description":
    st.title("Job Description")
    entered = st.text_area("Paste the job description", value=st.session_state.job_text, height=260, key="job_text_entry")
    if entered != st.session_state.job_text:
        st.session_state.job_text = entered
        st.session_state.analysis = None
    if entered.strip():
        proposed = extract_job_requirements(entered)
        taxonomy = sorted({skill for skills in cached_taxonomy().values() for skill in skills})
        if st.session_state.get("requirements_text") != entered:
            st.session_state.requirements_text = entered
            st.session_state.required = [s for s in proposed["required"] if s in taxonomy]
            st.session_state.preferred = [s for s in proposed["preferred"] if s in taxonomy and s not in st.session_state.required]
            st.session_state["required_editor"] = st.session_state.required.copy()
            st.session_state["preferred_editor"] = st.session_state.preferred.copy()
            st.session_state.analysis = None
        required = st.multiselect("Required skills: review and edit", taxonomy, key="required_editor")
        preferred = st.multiselect("Preferred skills: review and edit", taxonomy, key="preferred_editor")
        changed = required != st.session_state.required or preferred != st.session_state.preferred
        st.session_state.required = required
        st.session_state.preferred = [s for s in preferred if s not in required]
        if changed:
            st.session_state.analysis = None
        st.caption("The local taxonomy proposes skills; review whether each is required or preferred.")
        if not required and not preferred:
            st.warning("No job skills are selected. Select required or preferred skills above; otherwise coverage will be 0%.")
        lines = [line.strip(" •-\t") for line in entered.splitlines() if any(w in line.casefold() for w in ("degree", "bachelor", "master", "qualification", "years of experience"))]
        show_list("Qualification and experience lines", lines)
        keywords = sorted(set(word.casefold().strip(".,;:()[]") for word in entered.split() if len(word.strip(".,;:()[]")) >= 5))
        show_list("Long-form keywords", keywords[:30])
    else:
        st.info("Paste a job description to detect and review skills.")

elif page == "Skill Gap Analysis":
    st.title("Skill Gap Analysis")
    weight_left, weight_right = st.columns(2)
    required_weight = weight_left.slider("Required skill weight", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="required_weight_control")
    preferred_weight = weight_right.slider("Preferred skill weight", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="preferred_weight_control")
    st.caption("Weighted coverage updates automatically when these weights change. Default weights are 2 for required and 1 for preferred skills.")
    if not st.session_state.resume_text:
        st.warning("Upload a resume first.")
    elif not st.session_state.job_text.strip():
        st.warning("Add a job description first.")
    elif st.button("Analyze resume against job", type="primary"):
        with st.spinner("Computing skill coverage and local text similarities…"):
            candidate = st.session_state.resume_skills or extract_skills(st.session_state.resume_text)
            required, preferred = st.session_state.required, st.session_state.preferred
            if not required and not preferred:
                proposed = extract_job_requirements(st.session_state.job_text)
                required, preferred = proposed["required"], proposed["preferred"]
            comparison = compare_skills(candidate, required, preferred, required_weight, preferred_weight)
            graph = build_skill_graph()
            roadmap = build_roadmap(comparison["missing"], candidate)
            scores = {"bow": bow_similarity(st.session_state.resume_text, st.session_state.job_text), "tfidf": tfidf_similarity(st.session_state.resume_text, st.session_state.job_text), "word2vec": word2vec_similarity(st.session_state.resume_text, st.session_state.job_text, sg=1), "transformer": None}
            record = {"resume_filename": st.session_state.resume_filename, "resume_skills": candidate, **comparison, "similarity": scores, "roadmap": roadmap}
            st.session_state.analysis = {**record, "related": related_skills(comparison["missing"], candidate, graph)}
    analysis = st.session_state.analysis
    if analysis:
        if analysis.get("required_weight") != required_weight or analysis.get("preferred_weight") != preferred_weight:
            refreshed = compare_skills(analysis["resume_skills"], analysis["required"], analysis["preferred"], required_weight, preferred_weight)
            for field in ("matching", "missing", "additional", "exact_coverage", "weighted_coverage", "required_weight", "preferred_weight"):
                analysis[field] = refreshed[field]
        if not analysis["required"] and not analysis["preferred"]:
            st.warning("No job skills were selected or detected. Both coverage scores are 0% because there are no job skills in the denominator. Review skills on the Job Description page.")
        st.caption("This analysis remains in the current browser session only.")
        left, right = st.columns(2)
        left.metric("Exact skill coverage", f"{analysis['exact_coverage']:.1f}%")
        right.metric("Weighted skill coverage", f"{analysis['weighted_coverage']:.1f}%")
        st.caption(f"Exact = matched listed skills ÷ all required and preferred skills. Weighted coverage divides matched skill weights by all job skill weights (required {required_weight:g}, preferred {preferred_weight:g}).")
        for key, label in [("required", "Required skills"), ("preferred", "Preferred skills"), ("matching", "Matching skills"), ("missing", "Missing skills"), ("additional", "Additional candidate skills")]:
            show_list(label, analysis[key])
        show_list("Related candidate skills", [f"{r['related']} ({r['relationship']} to {r['target']})" for r in analysis["related"]])

elif page == "Similarity Analysis":
    st.title("Similarity Analysis")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run Skill Gap Analysis first. Transformer weights download on first use and are cached locally.")
    else:
        metrics = analysis["similarity"]
        word2vec_mode = st.selectbox("Word2Vec architecture", ["Skip-gram", "CBOW"], help="Skip-gram predicts context from a target word; CBOW predicts a target from surrounding words.")
        if st.button("Recalculate Word2Vec with selected architecture"):
            metrics["word2vec"] = word2vec_similarity(st.session_state.resume_text, st.session_state.job_text, sg=1 if word2vec_mode == "Skip-gram" else 0)
            analysis["similarity"] = metrics
        if st.button("Calculate local Transformer similarity"):
            try:
                from ml.embeddings.transformer_embeddings import transformer_similarity
                model = load_transformer(local_files_only=False)
                metrics["transformer"] = transformer_similarity(st.session_state.resume_text, st.session_state.job_text, model)
                analysis["similarity"] = metrics
                st.success("Transformer embeddings were generated locally; model weights are now cached.")
            except Exception as exc:
                st.error(f"Could not load the local Transformer. First use needs model-download access; afterward cached inference works offline. Details: {exc}")
        metrics = analysis["similarity"]
        rows = [{"Method": "Bag of Words", "Similarity (0–100%)": 100 * metrics["bow"]}, {"Method": "TF-IDF", "Similarity (0–100%)": 100 * metrics["tfidf"]}, {"Method": "Word2Vec", "Similarity (0–100%)": 100 * metrics["word2vec"]}, {"Method": "Transformer", "Similarity (0–100%)": 100 * metrics["transformer"] if metrics["transformer"] is not None else None}]
        chart = pd.DataFrame(rows)
        st.dataframe(chart, use_container_width=True, hide_index=True)
        available = chart.dropna().set_index("Method")
        if not available.empty:
            st.bar_chart(available)
        with st.expander("How these measures differ"):
            st.write("Bag of Words and TF-IDF compare literal token overlap; TF-IDF reduces common word influence. Word2Vec averages local word vectors trained on the bundled explicitly synthetic demo corpus plus these documents. The pretrained Sentence Transformer embeds whole texts locally and can represent semantic relationships despite wording differences. Cosine similarity is shown on a 0–100 scale. These are not accuracy or hiring probability scores.")
        st.caption("Word2Vec is trained locally on the synthetic demo corpus and the current documents. The architecture can be switched above.")

elif page == "Learning Roadmap":
    st.title("Learning Roadmap")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run Skill Gap Analysis to build a roadmap.")
    elif not analysis["roadmap"]:
        st.success("No missing taxonomy skills were identified.")
    else:
        st.caption("The order follows the curated prerequisite graph; missing prerequisites are included before dependent skills.")
        for item in analysis["roadmap"]:
            with st.expander(f"{item['order']}. {item['skill']} — {item['category']}", expanded=item["order"] == 1):
                st.write(item["reason"])
                show_list("Prerequisites missing from this resume", item["prerequisites"])
                resource = item.get("resource")
                if resource:
                    st.markdown(f"Learning resource: [{resource['title']}]({resource['url']}) — {resource['source']}")
                else:
                    st.write("No curated link for this skill is in the local resource file.")

