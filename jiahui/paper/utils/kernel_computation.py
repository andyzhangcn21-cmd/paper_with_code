# utils/kernel_computation.py

from typing import Tuple
import torch


def polynomial_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Simple 1D polynomial kernel:
        K(x, z) = (1 + x * z)^degree

    x1: [n, 1]
    x2: [m, 1]
    return: [n, m]
    """
    # x1 @ x2^T = [n, m]
    dot_mat = x1 @ x2.T
    return (1.0 + dot_mat) ** degree


def kernel_predict_from_prompt(
    prompt: torch.Tensor,
    reg_lambda: float,
    kernel_degree: int,
) -> float:
    """
    Given a prompt of shape [1, n_context + 1, 2], where each token is [x, y]:

        - first n_context tokens are (x_i, y_i) as training data
        - last token is (x_query, 0.0) as query (label placeholder)

    We perform 1D polynomial-kernel ridge regression on (x_i, y_i),
    then return the kernel prediction at x_query.

    This is the explicit "reference" kernel method we compare against
    the Transformer.
    """
    # Drop batch dimension: [1, L, 2] -> [L, 2]
    tokens = prompt[0]  # [L, 2]
    all_x = tokens[:, 0:1]  # [L, 1]
    all_y = tokens[:, 1:2]  # [L, 1]

    # Split into context and query
    x_context = all_x[:-1]  # [n, 1]
    y_context = all_y[:-1]  # [n, 1]
    x_query = all_x[-1:]    # [1, 1]

    # Build kernel matrix on context points
    K_xx = polynomial_kernel(x_context, x_context, degree=kernel_degree)  # [n, n]

    n = x_context.size(0)
    eye = torch.eye(n, dtype=K_xx.dtype, device=K_xx.device)

    # Solve (K + λI) α = y  for α
    K_reg = K_xx + reg_lambda * eye
    alpha = torch.linalg.solve(K_reg, y_context)  # [n, 1]

    # Kernel between query point and context points: k(x_query, x_i)
    k_qx = polynomial_kernel(x_context, x_query, degree=kernel_degree)  # [n, 1]

    # Prediction: ŷ = k_qx^T α
    y_pred = (k_qx.T @ alpha).squeeze()  # scalar tensor

    # Return Python float
    return float(y_pred.item())

@torch.no_grad()
def attention_smoother_from_prompt(
    prompt: torch.Tensor,
    x_proj: torch.nn.Linear,
    pos_encoding: Optional[torch.nn.Module] = None,
    q_proj: Optional[torch.nn.Linear] = None,
    k_proj: Optional[torch.nn.Linear] = None,
    y_scale: Optional[float] = None,
    y_bias: float = 0.0,
) -> float:
    tokens = prompt[0]
    x = tokens[:, 0:1]
    y = tokens[:, 1:2]

    y_ctx = y[:-1]

    h = x_proj(x.unsqueeze(0))
    if pos_encoding is not None:
        h = pos_encoding(h)
    h = h[0]

    k = h[:-1, :]
    q = h[-1:, :]

    if q_proj is not None:
        q = q_proj(q.unsqueeze(0)).squeeze(0)
    if k_proj is not None:
        k = k_proj(k.unsqueeze(0)).squeeze(0)

    d = q.size(-1)
    logits = (k @ q.transpose(0, 1)).squeeze(-1) / (d ** 0.5)
    w = torch.softmax(logits, dim=0).unsqueeze(1)

    y_hat = (w * y_ctx).sum(dim=0).squeeze()
    if y_scale is not None:
        y_hat = y_scale * y_hat + float(y_bias)
    return float(y_hat.item())


@torch.no_grad()
def theory_kernel_predict_from_prompt(
    prompt: torch.Tensor,
    x_proj: torch.nn.Linear,
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    y_to_v: torch.nn.Linear,
    readout: torch.nn.Linear,
    pos_encoding: Optional[torch.nn.Module] = None,
) -> float:
    s = float((readout.weight @ y_to_v.weight).squeeze().item())
    b = float(readout.bias.squeeze().item()) if readout.bias is not None else 0.0
    return attention_smoother_from_prompt(
        prompt=prompt,
        x_proj=x_proj,
        pos_encoding=pos_encoding,
        q_proj=q_proj,
        k_proj=k_proj,
        y_scale=s,
        y_bias=b,
    )
