# utils/metrics.py
from __future__ import annotations

from typing import Sequence, Union, Tuple, List
import torch


TensorLike = Union[torch.Tensor, Sequence[float]]


def compute_r_squared(y_true: TensorLike, y_fit: TensorLike) -> float:
    """
    R^2 = 1 - SSE/SST
    Supports torch.Tensor or python list/tuple.
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(list(y_true), dtype=torch.float32)
    if not isinstance(y_fit, torch.Tensor):
        y_fit = torch.tensor(list(y_fit), dtype=torch.float32)

    y_true = y_true.flatten()
    y_fit = y_fit.flatten()

    sse = torch.sum((y_true - y_fit) ** 2)
    mean = torch.mean(y_true)
    sst = torch.sum((y_true - mean) ** 2)

    if float(sst) == 0.0:
        return 0.0

    r2 = 1.0 - sse / sst
    return float(r2)


def detect_change_point(n_values: Sequence[int], ratios: Sequence[float]) -> Tuple[int, float]:
    """
    Find a change point n_c by fitting two separate lines (before/after split)
    and picking the split that minimizes total squared error.

    Returns:
        n_c: change-point location (an element from n_values)
        best_sse: the minimal total SSE achieved
    """
    if len(n_values) != len(ratios):
        raise ValueError("n_values and ratios must have the same length")

    n = torch.tensor(list(n_values), dtype=torch.float32)
    y = torch.tensor(list(ratios), dtype=torch.float32)

    if len(n) < 6:
        # too short to split meaningfully
        return int(n_values[len(n_values) // 2]), float("inf")

    def fit_line(x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y = a*x + b (least squares)
        X = torch.stack([x, torch.ones_like(x)], dim=1)  # [m,2]
        sol = torch.linalg.lstsq(X, t.unsqueeze(1)).solution.squeeze(1)  # [2]
        a, b = sol[0], sol[1]
        return a, b

    best_sse = None
    best_k = None

    # avoid splitting too close to edges
    for k in range(2, len(n) - 2):
        n1, y1 = n[:k], y[:k]
        n2, y2 = n[k:], y[k:]

        a1, b1 = fit_line(n1, y1)
        a2, b2 = fit_line(n2, y2)

        y1_hat = a1 * n1 + b1
        y2_hat = a2 * n2 + b2

        sse = torch.sum((y1 - y1_hat) ** 2) + torch.sum((y2 - y2_hat) ** 2)
        sse_val = float(sse.item())

        if (best_sse is None) or (sse_val < best_sse):
            best_sse = sse_val
            best_k = k

    n_c = int(n_values[best_k]) if best_k is not None else int(n_values[len(n_values) // 2])
    return n_c, float(best_sse if best_sse is not None else float("inf"))
