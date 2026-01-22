from __future__ import annotations

import os
import sys
import math
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data.synthetic_tasks import make_sparse_linear_batch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_pub_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.0,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, path_png: str) -> None:
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight")
    if path_png.lower().endswith(".png"):
        fig.savefig(path_png[:-4] + ".pdf", bbox_inches="tight")


def beautify_ax(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)


class TinyICLTransformer(nn.Module):
    def __init__(self, token_dim: int, d_model: int, n_heads: int, mlp_hidden: int, max_len: int):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.max_len = max_len
        self.register_buffer("_mask", torch.empty(0), persistent=False)

    def _causal_mask(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._mask.numel() == 0 or self._mask.size(0) < L:
            m = torch.full((L, L), float("-inf"))
            m = torch.triu(m, diagonal=1)
            self._mask = m
        return self._mask[:L, :L].to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence too long: L={L} > max_len={self.max_len}")

        h = self.in_proj(x)
        idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = h + self.pos(idx)

        mask = self._causal_mask(L, x.device, h.dtype)
        a, w = self.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)

        h = self.ln1(h + a)
        h = self.ln2(h + self.mlp(h))

        y = self.out_proj(h[:, -1, :]).squeeze(-1)
        if return_attn:
            return y, w
        return y


@torch.no_grad()
def attn_stats(attn_w: torch.Tensor) -> Tuple[float, float]:
    w = attn_w[:, :, -1, :-1]  
    w = w.mean(dim=1)        
    w = torch.clamp(w, min=1e-12)
    w = w / w.sum(dim=-1, keepdim=True)

    ent = -(w * torch.log(w)).sum(dim=-1)
    top1 = w.max(dim=-1).values
    return float(ent.mean().item()), float(top1.mean().item())


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) <= 1:
        return float(xs[0]), 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def plot_err(
    x: List[int],
    y: List[float],
    yerr: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: str,
    y_log: bool = False,
) -> None:
    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o",
        capsize=3,
        elinewidth=1.2,
        markeredgewidth=0.0,
        label="mean ± SE",
    )
    ax.plot(x, y, linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y_log:
        ax.set_yscale("log")

    beautify_ax(ax)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    save_fig(fig, out_png)
    plt.close(fig)


def train_one_n(
    x_dim: int,
    sparsity: float,
    noise_std: float,
    n_ctx: int,
    seed: int,
    device: str,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    grad_clip: float,
    d_model: int,
    n_heads: int,
    mlp_hidden: int,
    eval_batches: int,
    eval_size: int,
) -> Tuple[float, float, float]:
    set_seed(seed)

    token_dim = x_dim + 1
    max_len = n_ctx + 1

    model = TinyICLTransformer(token_dim, d_model, n_heads, mlp_hidden, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(steps):
        x, y = make_sparse_linear_batch(
            batch_size=batch_size,
            dim=x_dim,
            sparsity=sparsity,
            n_context=n_ctx,
            noise_std=noise_std,
            device=device,
        )
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    model.eval()
    mse_list: List[float] = []
    ent_list: List[float] = []
    top1_list: List[float] = []

    for _ in range(eval_batches):
        x, y = make_sparse_linear_batch(
            batch_size=eval_size,
            dim=x_dim,
            sparsity=sparsity,
            n_context=n_ctx,
            noise_std=noise_std,
            device=device,
        )
        pred, w = model(x, return_attn=True)
        mse = torch.mean((pred - y) ** 2).item()
        ent, top1 = attn_stats(w)
        mse_list.append(float(mse))
        ent_list.append(ent)
        top1_list.append(top1)

    return float(np.mean(mse_list)), float(np.mean(ent_list)), float(np.mean(top1_list))


def main() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    set_pub_style()

    device = get_device()
    print("script:", os.path.abspath(__file__))
    print("cwd:", os.getcwd())
    print("device:", device)

    x_dim = 100
    sparsity = 0.1
    noise_std = 0.0

    steps = 4000
    lr = 8e-4
    batch_size = 32
    grad_clip = 1.0

    d_model = 128
    n_heads = 4
    mlp_hidden = 256

    eval_size = 512
    eval_batches = 8

    n_list = [2, 4, 8, 16, 32, 64]
    seeds = [0, 1, 2, 3, 4]

    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    mse_by_n: Dict[int, List[float]] = {n: [] for n in n_list}
    ent_by_n: Dict[int, List[float]] = {n: [] for n in n_list}
    top1_by_n: Dict[int, List[float]] = {n: [] for n in n_list}

    for n_ctx in n_list:
        print(f"\n=== n_context = {n_ctx} ===")
        for sd in seeds:
            mse, ent, top1 = train_one_n(
                x_dim, sparsity, noise_std, n_ctx, sd, device,
                steps=steps, lr=lr, batch_size=batch_size, grad_clip=grad_clip,
                d_model=d_model, n_heads=n_heads, mlp_hidden=mlp_hidden,
                eval_batches=eval_batches, eval_size=eval_size,
            )
            mse_by_n[n_ctx].append(mse)
            ent_by_n[n_ctx].append(ent)
            top1_by_n[n_ctx].append(top1)
            print(f"  seed={sd}  eval_mse={mse:.6f}  attn_entropy={ent:.4f}  attn_top1={top1:.4f}")

    n_plot: List[int] = []
    mse_m: List[float] = []
    mse_se: List[float] = []
    ent_m: List[float] = []
    ent_se: List[float] = []
    top1_m: List[float] = []
    top1_se: List[float] = []

    for n_ctx in n_list:
        m1, s1 = mean_std(mse_by_n[n_ctx])
        m2, s2 = mean_std(ent_by_n[n_ctx])
        m3, s3 = mean_std(top1_by_n[n_ctx])

        se1 = s1 / math.sqrt(len(seeds)) if len(seeds) > 1 else 0.0
        se2 = s2 / math.sqrt(len(seeds)) if len(seeds) > 1 else 0.0
        se3 = s3 / math.sqrt(len(seeds)) if len(seeds) > 1 else 0.0

        n_plot.append(n_ctx)
        mse_m.append(m1); mse_se.append(se1)
        ent_m.append(m2); ent_se.append(se2)
        top1_m.append(m3); top1_se.append(se3)

    csv_path = os.path.join(out_dir, "exp3_phase_transition_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n,eval_mse_mean,eval_mse_se,attn_entropy_mean,attn_entropy_se,attn_top1_mean,attn_top1_se\n")
        for i in range(len(n_plot)):
            f.write(
                f"{n_plot[i]},"
                f"{mse_m[i]:.10e},{mse_se[i]:.10e},"
                f"{ent_m[i]:.10e},{ent_se[i]:.10e},"
                f"{top1_m[i]:.10e},{top1_se[i]:.10e}\n"
            )
    print("\nsaved:", csv_path)

    plot_err(
        n_plot, mse_m, mse_se,
        xlabel="Context length n",
        ylabel="Query MSE",
        title="Sparse linear tasks: error vs context length",
        out_png=os.path.join(out_dir, "exp3_mse_vs_n.png"),
        y_log=True,
    )

    plot_err(
        n_plot, ent_m, ent_se,
        xlabel="Context length n",
        ylabel="Attention entropy (query→context)",
        title="Attention concentration vs context length",
        out_png=os.path.join(out_dir, "exp3_attn_entropy_vs_n.png"),
        y_log=False,
    )

    plot_err(
        n_plot, top1_m, top1_se,
        xlabel="Context length n",
        ylabel="Attention top-1 mass (query→context)",
        title="Attention sparsity proxy vs context length",
        out_png=os.path.join(out_dir, "exp3_attn_top1_vs_n.png"),
        y_log=False,
    )

    print("saved figures to:", out_dir)


if __name__ == "__main__":
    main()
