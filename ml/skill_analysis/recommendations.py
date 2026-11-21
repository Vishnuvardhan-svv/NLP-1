"""Roadmap ordering from curated prerequisite edges and local resources."""

import json
from pathlib import Path

import networkx as nx

from ml.skill_analysis.skill_graph import build_skill_graph
from ml.skill_analysis.skill_normalization import category_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_PATH = PROJECT_ROOT / "datasets" / "learning_resources" / "resources.json"

def load_resources(path: Path = RESOURCES_PATH) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["skill"]: record for record in records}

def build_roadmap(missing_skills: list[str], candidate_skills: list[str]) -> list[dict]:
    graph = build_skill_graph()
    resources = load_resources()
    candidate_set = set(candidate_skills)
    targets = list(dict.fromkeys(missing_skills))
    needed = set(targets)

    def include_missing_prerequisites(skill: str) -> None:
        if skill not in graph:
            return
        for prerequisite in graph.predecessors(skill):
            if prerequisite not in candidate_set and prerequisite not in needed:
                needed.add(prerequisite)
                include_missing_prerequisites(prerequisite)

    for skill in targets:
        include_missing_prerequisites(skill)
    subgraph = graph.subgraph(needed).copy()
    try:
        ordered = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        ordered = sorted(needed)
    ordered = [skill for skill in ordered if skill not in candidate_set]
    ordered = [skill for skill in ordered if skill in needed]

    roadmap = []
    for index, skill in enumerate(ordered, start=1):
        prereqs = sorted(graph.predecessors(skill)) if skill in graph else []
        resource = resources.get(skill)
        roadmap.append({
            "order": index,
            "skill": skill,
            "category": graph.nodes[skill].get("category", category_for(skill) or "Other") if skill in graph else category_for(skill) or "Other",
            "reason": "Listed as a job requirement" if skill in targets else "Curated prerequisite for a missing job skill",
            "prerequisites": [p for p in prereqs if p not in candidate_set],
            "resource": resource,
        })
    return roadmap
