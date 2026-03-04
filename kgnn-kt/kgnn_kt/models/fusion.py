from __future__ import annotations
import torch
import torch.nn as nn
import math

class MultimodalFusion(nn.Module):
    """Cross-modal attention gate + LayerNorm (paper: align to 256D then fuse)."""
    def __init__(self, in_problem: int, in_student: int, in_graph: int, d: int = 256):
        super().__init__()
        self.p_proj = nn.Linear(in_problem, d)
        self.s_proj = nn.Linear(in_student, d)
        self.g_proj = nn.Linear(in_graph, d)

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, problem_feat: torch.Tensor, student_feat: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        p = torch.tanh(self.p_proj(problem_feat))
        s = torch.tanh(self.s_proj(student_feat))
        g = torch.tanh(self.g_proj(graph_feat))

        # scalar gate alpha in [0,1] per sample
        q = self.Wq(p)  # (B, d)
        k = self.Wk(s)  # (B, d)
        score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(p.shape[-1])
        alpha = torch.sigmoid(score)  # (B,1)

        z = alpha * p + (1.0 - alpha) * s + g
        return self.ln(z)
