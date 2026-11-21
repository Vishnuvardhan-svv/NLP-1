import json
from pathlib import Path
import networkx as nx
from ml.skill_analysis.skill_normalization import load_taxonomy

RELATIONSHIPS_PATH = Path(__file__).resolve().parents[2] / "datasets" / "skill_taxonomy" / "skill_relationships.json"

def load_relationships(path: Path = RELATIONSHIPS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def build_skill_graph() -> nx.DiGraph:
    data = load_relationships()
    graph = nx.DiGraph()
    for category, skills in load_taxonomy().items():
        for skill in skills:
            graph.add_node(skill, category=category)
    for skill, details in data.items():
        graph.add_node(skill, category=details.get("category", graph.nodes.get(skill, {}).get("category", "Other")))
        for prerequisite in details.get("prerequisites", []):
            graph.add_edge(prerequisite, skill, relationship="prerequisite")
    return graph

def related_skills(targets: list[str], candidate_skills: list[str], graph: nx.DiGraph | None = None) -> list[dict]:
    graph = graph or build_skill_graph()
    candidates = set(candidate_skills)
    results = []
    for target in targets:
        if target not in graph:
            continue
        neighbors = set(graph.predecessors(target)) | set(graph.successors(target))
        shared = sorted(neighbors & candidates)
        for skill in shared:
            results.append({"target": target, "related": skill, "relationship": "prerequisite or dependent skill"})
    return results
