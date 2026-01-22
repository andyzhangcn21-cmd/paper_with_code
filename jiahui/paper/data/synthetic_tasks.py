# data/synthetic_tasks.py

from dataclasses import dataclass
from typing import List, Tuple, Optional

import math
import torch


# -------------------------
# Exp1: polynomial tasks
# -------------------------
@dataclass
class PolynomialTask:
    """
    One univariate polynomial regression task:
        f(x) = a0 + a1 x + ... + a_d x^d
    coeffs stores [a0, ..., a_d].
    """
    coeffs: torch.Tensor  # shape: [degree + 1]

    def sample_xy(
        self,
        n_points: int,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample n_points (x, y) pairs.
        x: [n_points, 1]
        y: [n_points, 1]
        """
        x = torch.empty(n_points, 1, device=device).uniform_(x_range[0], x_range[1])

        degree = self.coeffs.numel() - 1
        powers = torch.cat([x ** i for i in range(degree + 1)], dim=1)  # [n_points, d+1]

        y = powers @ self.coeffs.to(device).unsqueeze(1)  # [n_points, 1]

        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)

        return x, y

    def make_prompt(
        self,
        n_context: int,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prompt: [1, n_context+1, 2]
          - first n_context tokens: [x_i, y_i]
          - last token: [x_query, 0.0]
        target: [1] (scalar y_query)
        """
        n_total = n_context + 1
        x, y = self.sample_xy(
            n_points=n_total,
            x_range=x_range,
            noise_std=noise_std,
            device=device,
        )

        x_context, x_query = x[:-1], x[-1:]
        y_context, y_query = y[:-1], y[-1:]

        context_tokens = torch.cat([x_context, y_context], dim=1)  # [n,2]
        query_token = torch.cat([x_query, torch.zeros_like(y_query)], dim=1)  # [1,2]

        tokens = torch.cat([context_tokens, query_token], dim=0).unsqueeze(0)  # [1,n+1,2]
        target = y_query.squeeze(-1)  # [1]

        return tokens, target


def generate_polynomial_tasks(
    degree: int = 3,
    num_tasks: int = 100,
    coeff_range: Tuple[float, float] = (-1.0, 1.0),
    device: str = "cpu",
    return_coeffs: bool = False,
) -> Tuple[List[PolynomialTask], Optional[torch.Tensor]]:
    """
    Generate num_tasks independent polynomial tasks.
    Each task has coeffs sampled uniformly from coeff_range.
    """
    tasks: List[PolynomialTask] = []
    low, high = coeff_range
    coeff_list: List[torch.Tensor] = []

    for _ in range(num_tasks):
        coeffs = (high - low) * torch.rand(degree + 1, device=device) + low
        tasks.append(PolynomialTask(coeffs=coeffs))
        coeff_list.append(coeffs.unsqueeze(0))

    if return_coeffs:
        return tasks, torch.cat(coeff_list, dim=0)

    return tasks, None


# -------------------------
# Exp2: GRF-like batch (your current version)
# -------------------------
def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True).transpose(-2, -1)
    xy = x @ y.transpose(-2, -1)
    d2 = x2 + y2 - 2 * xy
    return torch.exp(-0.5 * d2 / (lengthscale**2 + 1e-12))


def make_grf_batch(
    batch_size: int,
    n_context: int,
    x_dim: int = 1,
    lengthscale: float = 0.3,
    noise_std: float = 0.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      prompts: [B, n_context+1, x_dim+1]  (last dim is y slot)
      targets: [B]
    """
    n_total = n_context + 1

    x = torch.randn(batch_size, n_total, x_dim, device=device)

    K = _rbf_kernel(x, x, lengthscale=lengthscale)
    jitter = 1e-4 * torch.eye(n_total, device=device).unsqueeze(0)
    K = K + jitter

    L = torch.linalg.cholesky(K)
    eps = torch.randn(batch_size, n_total, 1, device=device)
    y = L @ eps

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    y_context = y[:, :-1, :]
    y_query = y[:, -1, :].squeeze(-1)

    y_query_slot = torch.zeros(batch_size, 1, 1, device=device)
    y_all = torch.cat([y_context, y_query_slot], dim=1)

    prompts = torch.cat([x, y_all], dim=-1)
    targets = y_query
    return prompts, targets


# -------------------------
# Exp3: sparse linear batch (new, backward-safe)
# -------------------------
def make_sparse_linear_batch(
    batch_size: int,
    n_context: int,
    dim: Optional[int] = None,
    x_dim: Optional[int] = None,
    sparsity: float = 0.1,
    noise_std: float = 0.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse linear regression tasks.
      For each sample in batch, sample a sparse weight vector w (same dim),
      then generate context pairs (x_i, y_i) and a query x_q.

    Output:
      prompts: [B, n_context+1, dim+1]  (last dim is y slot)
      targets: [B]
    Notes:
      - Accepts both 'dim' and 'x_dim' to avoid keyword mismatch errors.
      - This function is additive; does not change Exp1/Exp2 behavior.
    """
    if dim is None and x_dim is None:
        raise ValueError("Please provide dim or x_dim.")
    if dim is None:
        dim = int(x_dim)
    if x_dim is None:
        x_dim = int(dim)

    if dim != x_dim:
        raise ValueError(f"dim ({dim}) and x_dim ({x_dim}) must match for this setup.")

    if not (0.0 < sparsity <= 1.0):
        raise ValueError("sparsity must be in (0, 1].")

    k = max(1, int(round(dim * sparsity)))

    # sample x: [B, n_context+1, dim]
    n_total = n_context + 1
    x = torch.randn(batch_size, n_total, dim, device=device)

    # sample sparse w per sample: [B, dim]
    w = torch.zeros(batch_size, dim, device=device)
    for b in range(batch_size):
        idx = torch.randperm(dim, device=device)[:k]
        # scale to keep y magnitude roughly stable across different k
        w[b, idx] = torch.randn(k, device=device) / math.sqrt(k)

    # y = x @ w + noise
    y = (x * w.unsqueeze(1)).sum(dim=-1)  # [B, n_total]
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    # context y filled, query y slot = 0
    y_context = y[:, :-1]  # [B, n_context]
    y_query = y[:, -1]     # [B]
    y_slot = torch.zeros(batch_size, 1, device=device)

    y_all = torch.cat([y_context, y_slot], dim=1).unsqueeze(-1)  # [B, n_total, 1]
    prompts = torch.cat([x, y_all], dim=-1)  # [B, n_total, dim+1]
    targets = y_query
    return prompts, targets


# -------------------------
# quick sanity (optional)
# -------------------------
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tasks, coeffs = generate_polynomial_tasks(degree=3, num_tasks=2, device=device, return_coeffs=True)
    print("coeffs shape:", coeffs.shape)

    prompt, target = tasks[0].make_prompt(n_context=4, device=device)
    print("prompt shape:", prompt.shape)
    print("target:", target)

    prompts, targets = make_grf_batch(batch_size=2, n_context=8, x_dim=1, device=device)
    print("grf prompts shape:", prompts.shape)
    print("grf targets shape:", targets.shape)

    prompts2, targets2 = make_sparse_linear_batch(batch_size=2, n_context=8, dim=100, sparsity=0.1, device=device)
    print("sparse prompts shape:", prompts2.shape)
    print("sparse targets shape:", targets2.shape)
    print("query token first 6 dims:", prompts2[0, -1, :6])


if __name__ == "__main__":
    main()
