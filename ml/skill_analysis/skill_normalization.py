import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = PROJECT_ROOT / "datasets" / "skill_taxonomy" / "skills.json"

def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, dict[str, list[str]]]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)

def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", value.casefold())

def alias_map(taxonomy: dict[str, dict[str, list[str]]] | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for skills in (taxonomy or load_taxonomy()).values():
        for canonical, aliases in skills.items():
            mapping[normalize_key(canonical)] = canonical
            for alias in aliases:
                mapping[normalize_key(alias)] = canonical
    return mapping

def normalize_skill(skill: str, taxonomy: dict[str, dict[str, list[str]]] | None = None) -> str | None:
    return alias_map(taxonomy).get(normalize_key(skill.strip()))

def category_for(skill: str, taxonomy: dict[str, dict[str, list[str]]] | None = None) -> str | None:
    for category, skills in (taxonomy or load_taxonomy()).items():
        if skill in skills:
            return category
    return None
