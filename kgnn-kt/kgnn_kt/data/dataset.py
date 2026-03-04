from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .schema import Problem, Interaction, UserProfile
from .io import read_jsonl

@dataclass
class LoadedData:
    problems: Dict[str, Problem]
    interactions: List[Interaction]
    user_profiles: Dict[str, UserProfile]

def load_data(problems_path: str, interactions_path: str, user_profiles_path: Optional[str] = None) -> LoadedData:
    problems = {p.problem_id: p for p in read_jsonl(problems_path, Problem)}
    interactions = read_jsonl(interactions_path, Interaction)
    profiles = {}
    if user_profiles_path:
        profiles = {u.user_id: u for u in read_jsonl(user_profiles_path, UserProfile)}
    return LoadedData(problems=problems, interactions=interactions, user_profiles=profiles)

class KTExampleDataset(Dataset):
    """Reference dataset: one sample is (user_id, problem_id, label).

    For simplicity, we build a flattened list. In practice, KT often uses sequences;
    this repo keeps the interface clear and extensible.
    """
    def __init__(self, interactions: List[Interaction]):
        self.items = interactions

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return {
            "user_id": it.user_id,
            "problem_id": it.problem_id,
            "y": float(it.correct),
            "timestamp": int(it.timestamp),
        }

def collate_batch(batch: List[dict]) -> dict:
    # ids remain python lists; labels become tensors
    return {
        "user_id": [b["user_id"] for b in batch],
        "problem_id": [b["problem_id"] for b in batch],
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
        "timestamp": torch.tensor([b["timestamp"] for b in batch], dtype=torch.long),
    }
