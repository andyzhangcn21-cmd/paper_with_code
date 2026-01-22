

import os
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


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


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
    xy = x @ y.transpose(-2, -1)
    d2 = x2 + y2 - 2.0 * xy
    return torch.exp(-0.5 * d2 / (lengthscale * lengthscale + 1e-12))


def make_gp_batch(
    batch_size: int,
    n_context: int,
    x_dim: int,
    lengthscale: float,
    device: str,
    *,
    jitter: float = 1e-6,
    max_tries: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_total = n_context + 1
    x = torch.randn(batch_size, n_total, x_dim, device=device, dtype=torch.float64)

    K = rbf_kernel(x, x, lengthscale=lengthscale).to(torch.float64)
    K = 0.5 * (K + K.transpose(-2, -1))

    eye = torch.eye(n_total, device=device, dtype=torch.float64).unsqueeze(0)
    cur_jit = float(jitter)

    L = None
    for _ in range(max_tries):
        K_try = K + cur_jit * eye
        L_try, info = torch.linalg.cholesky_ex(K_try)
        if int(info.max().item()) == 0:
            L = L_try
            break
        cur_jit *= 10.0

    if L is None:
        cur_jit = max(cur_jit, 1e-3)
        L = torch.linalg.cholesky(K + cur_jit * eye)

    eps = torch.randn(batch_size, n_total, 1, device=device, dtype=torch.float64)
    y = L @ eps

    y_ctx = y[:, :-1, :]
    y_q = y[:, -1, 0].to(torch.float32)

    y_slot = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.float64)
    y_all = torch.cat([y_ctx, y_slot], dim=1)

    prompts = torch.cat([x, y_all], dim=-1).to(torch.float32)
    targets = y_q
    return prompts, targets


def gp_posterior_mean_and_var(
    prompts: torch.Tensor,
    lengthscale: float,
    *,
    ridge: float = 1e-6,
    use_double: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompts.dim() != 3:
        raise ValueError(f"prompts must be [B, L, D], got {tuple(prompts.shape)}")

    device = prompts.device
    dtype = torch.float64 if use_double else prompts.dtype
    p = prompts.to(dtype=dtype)

    x = p[..., :-1]
    y = p[..., -1:]

    x_ctx = x[:, :-1, :]
    y_ctx = y[:, :-1, :]
    x_q = x[:, -1:, :]

    n = x_ctx.size(1)
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)

    K_xx = rbf_kernel(x_ctx, x_ctx, lengthscale=lengthscale)
    K_xx_reg = K_xx + ridge * eye

    k_qx = rbf_kernel(x_ctx, x_q, lengthscale=lengthscale)
    k_qq = rbf_kernel(x_q, x_q, lengthscale=lengthscale).squeeze(-1)

    alpha = torch.linalg.solve(K_xx_reg, y_ctx)
    v = torch.linalg.solve(K_xx_reg, k_qx)

    mean = (k_qx.transpose(-2, -1) @ alpha).squeeze(-1).squeeze(-1)
    var = (k_qq - (k_qx.transpose(-2, -1) @ v).squeeze(-1)).squeeze(-1)

    var = torch.clamp(var, min=0.0)
    return mean.to(torch.float32), var.to(torch.float32)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def _err_bar(std: np.ndarray, n_seeds: int, use_se: bool) -> np.ndarray:
    if use_se and n_seeds > 1:
        return std / np.sqrt(n_seeds)
    return std


def save_csv_simple(
    path: str,
    n_list: List[int],
    mse_mean: List[float],
    mse_std: List[float],
    theory_mean: List[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("n,mse_mean,mse_std,theory_var_mean\n")
        for n, m, s, t in zip(n_list, mse_mean, mse_std, theory_mean):
            f.write(f"{n},{m:.10e},{s:.10e},{t:.10e}\n")


def make_plots_exp2(
    n_list: List[int],
    mse_mean: List[float],
    mse_std: List[float],
    theory_mean: List[float],
    out_dir: str,
    *,
    n_seeds: int,
    use_se: bool = True,
) -> None:
    set_pub_style()

    n = np.asarray(n_list, dtype=float)
    mse_m = np.asarray(mse_mean, dtype=float)
    mse_s = np.asarray(mse_std, dtype=float)
    theo_m = np.asarray(theory_mean, dtype=float)

    yerr = _err_bar(mse_s, n_seeds=n_seeds, use_se=use_se)
    err_name = "SE" if (use_se and n_seeds > 1) else "std"

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(n, mse_m, yerr=yerr, fmt="o", capsize=3, label=f"Empirical MSE (± {err_name})")
    ax.plot(n, theo_m, label="Theory (posterior variance)")
    ax.set_yscale("log")
    ax.set_xlabel("Context length n")
    ax.set_ylabel("Query MSE")
    ax.set_title("GP tasks: Empirical MSE vs Theory")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    beautify_ax(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "exp2_mse_vs_n.png"))
    plt.close(fig)


    inv_n = 1.0 / n
    a, b, r2 = _fit_line(inv_n, mse_m)
    fit_y = a * inv_n + b

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(inv_n, mse_m, yerr=yerr, fmt="o", capsize=3, label=f"Empirical MSE (± {err_name})")
    ax.plot(inv_n, theo_m, label="Theory (posterior variance)")
    ax.plot(inv_n, fit_y, linestyle="--", label=f"Fit: a*(1/n)+b  (R$^2$={r2:.3f})")
    ax.set_xlabel("1 / n")
    ax.set_ylabel("Query MSE")
    ax.set_title("Empirical vs Theory (and 1/n fit)")
    beautify_ax(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "exp2_mse_vs_inv_n.png"))
    plt.close(fig)

    print("\n[fit] mse ~= a*(1/n) + b")
    print(f"  a = {a:.6e}")
    print(f"  b = {b:.6e}")
    print(f"  R2 = {r2:.6f}")


    ratio = mse_m / np.clip(theo_m, 1e-12, None)
    ratio_err = yerr / np.clip(theo_m, 1e-12, None)

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(n, ratio, yerr=ratio_err, fmt="o", capsize=3, label=f"MSE / Theory (± {err_name})")
    ax.axhline(1.0, linestyle="--", linewidth=1.6, label="Ideal (=1)")
    ax.set_xlabel("Context length n")
    ax.set_ylabel("MSE / theory")
    ax.set_title("Diagnostic: deviation from theory")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    beautify_ax(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "exp2_ratio_mse_over_theory.png"))
    plt.close(fig)


def main() -> None:
    print("### EXP2 FILE ###")
    print(os.path.abspath(__file__))

    device = get_device()
    print("device:", device)

    x_dim = 1
    lengthscale = 0.3
    ridge = 1e-6
    n_list = [2, 4, 8, 16, 32, 64]

    seeds = [0, 1, 2, 3, 4]
    batch_size = 256
    num_batches = 32

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    mse_by_n: Dict[int, List[float]] = {n: [] for n in n_list}
    theo_by_n: Dict[int, List[float]] = {n: [] for n in n_list}

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== seed {seed} ===")

        for n_ctx in n_list:
            mse_acc = 0.0
            theo_acc = 0.0

            for _ in range(num_batches):
                prompts, targets = make_gp_batch(
                    batch_size=batch_size,
                    n_context=n_ctx,
                    x_dim=x_dim,
                    lengthscale=lengthscale,
                    device=device,
                    jitter=1e-6,
                )

                mean, var = gp_posterior_mean_and_var(
                    prompts,
                    lengthscale=lengthscale,
                    ridge=ridge,
                    use_double=True,
                )

                err2 = (mean - targets.to(mean.device)) ** 2
                mse_acc += float(err2.mean().item())
                theo_acc += float(var.mean().item())

            mse = mse_acc / float(num_batches)
            theo = theo_acc / float(num_batches)

            mse_by_n[n_ctx].append(mse)
            theo_by_n[n_ctx].append(theo)

            diff = abs(mse - theo)
            print(f"n={n_ctx:>3d}  mse={mse:.6e}  theory_var={theo:.6e}  |diff|={diff:.2e}")

    mse_mean: List[float] = []
    mse_std: List[float] = []
    theo_mean: List[float] = []

    print("\n=== summary (mean ± std over seeds) ===")
    for n_ctx in n_list:
        m = float(np.mean(mse_by_n[n_ctx]))
        s = float(np.std(mse_by_n[n_ctx], ddof=1)) if len(mse_by_n[n_ctx]) > 1 else 0.0
        t = float(np.mean(theo_by_n[n_ctx]))

        mse_mean.append(m)
        mse_std.append(s)
        theo_mean.append(t)

        print(f"n={n_ctx:>3d}  mse={m:.6e} ± {s:.2e}   theory={t:.6e}")

    csv_path = os.path.join(results_dir, "exp2_gp_mse_vs_theory.csv")
    save_csv_simple(csv_path, n_list, mse_mean, mse_std, theo_mean)
    print("\nsaved:", csv_path)

    make_plots_exp2(
        n_list=n_list,
        mse_mean=mse_mean,
        mse_std=mse_std,
        theory_mean=theo_mean,
        out_dir=results_dir,
        n_seeds=len(seeds),
        use_se=True,
    )
    print("saved figures to:", results_dir)


if __name__ == "__main__":
    main()
