from __future__ import annotations
from typing import Protocol, Dict, Any, Optional
import json
import os

class LLMProvider(Protocol):
    def extract(self, prompt: str) -> Dict[str, Any]:
        ...

class MockProvider:
    """Deterministic offline provider for tests and artifact review."""
    def extract(self, prompt: str) -> Dict[str, Any]:
        # Extremely simple heuristic to make outputs look reasonable.
        lower = prompt.lower()
        ds, alg, para = [], [], []
        if "heap" in lower:
            ds.append("heap")
        if "hash" in lower:
            ds.append("hash table")
        if "dynamic programming" in lower or "dp" in lower:
            alg.append("dynamic programming")
        if "dijkstra" in lower:
            alg.append("dijkstra's algorithm")
        if "recursion" in lower:
            para.append("recursion")
        return {
            "data_structures": ds,
            "algorithms": alg,
            "paradigms": para,
            "time_complexity": "unknown",
            "space_complexity": "unknown",
        }

class OpenAIChatProvider:
    """Minimal OpenAI-style adapter (intentionally light; swap with your own SDK wrapper).

    Reads:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (optional)
    """
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def extract(self, prompt: str) -> Dict[str, Any]:
        # NOTE: This is a stub intentionally kept small for review.
        # You can implement with your preferred client (openai, httpx, etc.).
        raise NotImplementedError(
            "OpenAIChatProvider.extract is stubbed. Please implement with your SDK of choice."
        )

def make_provider(name: str, model: str = "gpt-4.1-mini") -> LLMProvider:
    name = (name or "mock").lower()
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIChatProvider(model=model)
    raise ValueError(f"Unknown llm provider: {name}")
