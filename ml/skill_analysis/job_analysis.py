"""Classify extracted job skills by nearby requirement language."""

import re

from ml.skill_analysis.skill_extraction import extract_skills
from ml.skill_analysis.skill_normalization import load_taxonomy

PREFERRED_MARKERS = re.compile(r"\b(preferred|nice to have|bonus|ideally|desired|plus)\b", re.I)
REQUIRED_MARKERS = re.compile(r"\b(required|must have|must-have|minimum qualifications|requirements)\b", re.I)

def extract_job_requirements(text: str) -> dict[str, list[str]]:
    skills = extract_skills(text)
    lines = text.splitlines()
    required, preferred = [], []
    by_canonical: dict[str, list[str]] = {}
    for category in load_taxonomy().values():
        for canonical, aliases in category.items():
            by_canonical.setdefault(canonical, []).extend([canonical, *aliases])
    modes_by_line = []
    mode = "required"
    for line in lines:
        if PREFERRED_MARKERS.search(line) and not REQUIRED_MARKERS.search(line):
            mode = "preferred"
        elif REQUIRED_MARKERS.search(line) and not PREFERRED_MARKERS.search(line):
            mode = "required"
        modes_by_line.append(mode)
    for skill in skills:
        modes = []
        for line, mode in zip(lines, modes_by_line):
            patterns = by_canonical.get(skill, [skill.casefold()])
            if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", line, re.I) for alias in patterns):
                modes.append(mode)
        if not modes:
            modes = ["required"]
        if "preferred" in modes and "required" not in modes:
            preferred.append(skill)
        else:
            required.append(skill)
    return {"required": required, "preferred": preferred}
