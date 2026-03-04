from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

class HFTextEncoder(nn.Module):
    """BERT/CodeBERT encoder wrapper.

    For artifact review, this is a thin wrapper; in production, you may want caching,
    pooling strategies, and batching across the full dataset.
    """
    def __init__(self, model_name: str, out_dim: int = 768):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers is not installed.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = out_dim

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        out = self.model(**batch)
        # Use [CLS] pooled output when available, else mean pool.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        last = out.last_hidden_state  # (B, T, H)
        return last.mean(dim=1)

class StudentEncoder(nn.Module):
    """LSTM student state encoder (paper: 128 hidden units; concat with profile features)."""
    def __init__(self, event_dim: int = 32, profile_dim: int = 32, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=event_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.profile_proj = nn.Linear(profile_dim, profile_dim)
        self.out_proj = nn.Linear(hidden + profile_dim, out_dim)

    def forward(self, events: torch.Tensor, profiles: torch.Tensor) -> torch.Tensor:
        # events: (B, T, event_dim)
        # profiles: (B, profile_dim)
        o, (h, c) = self.lstm(events)
        h_last = h[-1]  # (B, hidden)
        prof = torch.tanh(self.profile_proj(profiles))
        z = torch.cat([h_last, prof], dim=-1)
        return torch.tanh(self.out_proj(z))

class SimpleRGCN(nn.Module):
    """Lightweight RGCN-style layer stack.

    To avoid heavy dependencies, this implements typed-edge message passing in a dense way
    (sufficient for small graphs and for review-friendly code).
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int = 3, num_layers: int = 2):
        super().__init__()
        self.num_relations = num_relations
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.layers.append(_RGCNLayer(d_in, hidden_dim, num_relations=num_relations))

    def forward(self, x: torch.Tensor, adjs: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        # adjs: (R, N, N) binary adjacency per relation type
        h = x
        for layer in self.layers:
            h = layer(h, adjs)
        return h

class _RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.W_rel = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * 0.02)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adjs: torch.Tensor) -> torch.Tensor:
        # adjs: (R, N, N)
        R, N, _ = adjs.shape
        msgs = []
        for r in range(R):
            A = adjs[r]  # (N, N)
            deg = A.sum(dim=-1, keepdim=True).clamp_min(1.0)
            agg = (A @ x) / deg
            msgs.append(agg @ self.W_rel[r])
        m = torch.stack(msgs, dim=0).sum(dim=0)  # (N, out_dim)
        out = m + self.W_self(x)
        return self.act(out)
