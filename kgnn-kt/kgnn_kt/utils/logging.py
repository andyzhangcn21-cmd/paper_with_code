from __future__ import annotations
import json, os, time
from dataclasses import dataclass

@dataclass
class JsonlLogger:
    path: str

    def log(self, payload: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = dict(payload)
        payload["_ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
