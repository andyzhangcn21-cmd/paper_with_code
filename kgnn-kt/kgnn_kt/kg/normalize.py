from __future__ import annotations
import re
from typing import Dict, Optional

# Lightweight, review-friendly normalization. Replace with WordNet/CSO alignment if desired.
ALIASES = {
    "hashmap": "hash table",
    "hash map": "hash table",
    "min heap": "heap",
    "max heap": "heap",
    "dfs": "depth-first search",
    "bfs": "breadth-first search",
}

def normalize_concept(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)
