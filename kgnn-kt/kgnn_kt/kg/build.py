from __future__ import annotations
from typing import Dict, List, Tuple
from .schema import Graph, Node, Edge
from .normalize import normalize_concept
from .prompts import EXTRACT_PROMPT
from .llm import make_provider
import json

def build_kg_from_problems(
    problems: List[dict],
    llm_provider: str = "mock",
    openai_model: str = "gpt-4.1-mini",
) -> Graph:
    provider = make_provider(llm_provider, model=openai_model)
    g = Graph()

    # Add problem nodes
    for p in problems:
        g.nodes.append(Node(node_id=f"problem:{p['problem_id']}", node_type="problem", label=p["problem_id"]))

    concept_map: Dict[str, str] = {}  # normalized_name -> node_id

    def get_concept_node(name: str) -> str:
        norm = normalize_concept(name)
        if norm not in concept_map:
            node_id = f"concept:{norm}"
            concept_map[norm] = node_id
            g.nodes.append(Node(node_id=node_id, node_type="concept", label=norm))
        return concept_map[norm]

    # Extract concepts + connect problems -> concepts
    for p in problems:
        prompt = EXTRACT_PROMPT.format(problem_text=p.get("text", ""), code=p.get("code", ""))
        out = provider.extract(prompt)
        concepts = []
        for k in ["data_structures", "algorithms", "paradigms"]:
            for name in out.get(k, []) or []:
                concepts.append(name)

        pid = f"problem:{p['problem_id']}"
        for c in concepts:
            cid = get_concept_node(c)
            g.edges.append(Edge(src=pid, dst=cid, edge_type="requires"))

    # Add simple hierarchical edges programmatically (placeholder logic)
    # This is where you would add SubClassOf / Requires / Antonym edges from curated taxonomy rules.
    for norm, node_id in concept_map.items():
        if "min-heap" in norm or "max-heap" in norm:
            g.edges.append(Edge(src=node_id, dst=get_concept_node("heap"), edge_type="subclass_of"))
    # Antonym example
    if "recursion" in concept_map and "iteration" in concept_map:
        a = concept_map["recursion"]; b = concept_map["iteration"]
        g.edges.append(Edge(src=a, dst=b, edge_type="antonym"))
        g.edges.append(Edge(src=b, dst=a, edge_type="antonym"))

    return g

def save_graph(graph: Graph, out_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(graph.model_dump_json(indent=2, exclude_none=True))

def load_graph(path: str) -> Graph:
    with open(path, "r", encoding="utf-8") as f:
        return Graph.model_validate_json(f.read())
