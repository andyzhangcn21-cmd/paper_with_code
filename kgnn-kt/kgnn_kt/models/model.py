from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .encoders import HFTextEncoder, StudentEncoder, SimpleRGCN
from .fusion import MultimodalFusion
from .head import PredHead

@dataclass
class KGInputs:
    node_features: torch.Tensor      # (N, D0)
    adjs: torch.Tensor               # (R, N, N)
    problem_node_ids: List[int]      # list index per batch sample -> node index

class KGNNKT(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        code_model_name: str,
        concept_in_dim: int = 100,
        rgcn_hidden: int = 256,
        fused_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_encoder = HFTextEncoder(text_model_name)
        self.code_encoder = HFTextEncoder(code_model_name)

        # student encoder: event vectors are toy here; real pipelines should build richer features
        self.student_encoder = StudentEncoder(event_dim=32, profile_dim=32, hidden=128, out_dim=256)

        self.rgcn = SimpleRGCN(in_dim=concept_in_dim, hidden_dim=rgcn_hidden, num_relations=3, num_layers=2)

        # problem embedding: concat text+code (768+768)
        self.fusion = MultimodalFusion(in_problem=1536, in_student=256, in_graph=256, d=fused_dim)
        self.head = PredHead(in_dim=fused_dim, hidden=128, dropout=dropout)

    def forward(
        self,
        problem_texts: List[str],
        problem_codes: List[str],
        student_events: torch.Tensor,
        student_profiles: torch.Tensor,
        kg: KGInputs,
    ) -> torch.Tensor:
        # Encode problems
        t = self.text_encoder(problem_texts)   # (B, 768)
        c = self.code_encoder(problem_codes)   # (B, 768)
        p = torch.cat([t, c], dim=-1)          # (B, 1536)

        # Encode students
        s = self.student_encoder(student_events, student_profiles)  # (B, 256)

        # Encode KG
        # NOTE: For simplicity, we treat all nodes as having the same features and use dense adjacency.
        node_h = self.rgcn(kg.node_features, kg.adjs)  # (N, 256)

        # Extract per-problem graph feature by indexing the problem node
        idx = torch.tensor(kg.problem_node_ids, device=node_h.device, dtype=torch.long)
        g = node_h.index_select(dim=0, index=idx)  # (B, 256)

        z = self.fusion(p, s, g)  # (B, 256)
        y = self.head(z)          # (B,)
        return y
