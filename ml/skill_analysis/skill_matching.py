def compare_skills(resume_skills: list[str], required: list[str], preferred: list[str], required_weight: float = 2.0, preferred_weight: float = 1.0) -> dict:
    resume = set(resume_skills)
    required = list(dict.fromkeys(required))
    preferred = [skill for skill in dict.fromkeys(preferred) if skill not in required]
    job_skills = required + preferred
    matching = [skill for skill in job_skills if skill in resume]
    missing = [skill for skill in job_skills if skill not in resume]
    weighted_total = len(required) * required_weight + len(preferred) * preferred_weight
    weighted_matched = sum(required_weight if skill in required else preferred_weight for skill in matching)
    total = len(job_skills)
    return {
        "matching": matching,
        "missing": missing,
        "required": required,
        "preferred": preferred,
        "additional": [skill for skill in resume_skills if skill not in set(job_skills)],
        "exact_coverage": (len(matching) / total * 100) if total else 0.0,
        "weighted_coverage": (weighted_matched / weighted_total * 100) if weighted_total else 0.0,
        "required_weight": required_weight,
        "preferred_weight": preferred_weight,
    }
