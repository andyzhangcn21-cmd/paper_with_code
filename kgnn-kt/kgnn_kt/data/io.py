from __future__ import annotations
import json
from typing import Iterable, Type, TypeVar, List
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def read_jsonl(path: str, model: Type[T]) -> List[T]:
    items: List[T] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(model.model_validate(json.loads(line)))
    return items

def write_jsonl(path: str, items: Iterable[BaseModel]) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")
