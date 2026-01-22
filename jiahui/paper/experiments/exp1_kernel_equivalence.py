
import os
import sys
import random
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.synthetic_tasks import generate_polynomial_tasks
from utils.seed import set_global_seed


def pick_device() -> str:
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
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def beautify_ax(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)


def save_fig(fig: plt.Figure, path_png: str) -> None:
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight")
    if path_png.lower().endswith(".png"):
        fig.savefig(path_png[:-4] + ".pdf", bbox_inches="tight")



def poly_k(x1: torch.Tensor, x2: torch.Tensor, deg: int) -> torch.Tensor:
    return (1.0 + (x1 @ x2.T)) ** deg


@torch.no_grad()
def krr_from_prompt(prompt: torch.Tensor, deg: int, lam: float) -> torch.Tensor:
    if prompt.dim() != 3 or prompt.size(0) != 1 or prompt.size(-1) != 2:
        raise ValueError(f"need prompt [1,L,2], got {tuple(prompt.shape)}")

    dev = prompt.device
    t = prompt[0].to(dtype=torch.float64)
    x = t[:, 0:1]
    y = t[:, 1:2]

    x_ctx = x[:-1]
    y_ctx = y[:-1]
    x_q = x[-1:]

    n = x_ctx.size(0)
    K = poly_k(x_ctx, x_ctx, deg) + lam * torch.eye(n, dtype=torch.float64, device=dev)
    alpha = torch.linalg.solve(K, y_ctx)
    kq = poly_k(x_ctx, x_q, deg)
    y_hat = (kq.T @ alpha).squeeze()
    return y_hat.to(dtype=torch.float32)


class KRRClosedForm(nn.Module):
    def __init__(self, deg: int = 3, lam: float = 1e-4):
        super().__init__()
        self.deg = int(deg)
        self.lam = float(lam)
        self._dummy = nn.Parameter(torch.zeros(()))

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        return krr_from_prompt(prompt, self.deg, self.lam)



def row_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / (e.sum(dim=-1, keepdim=True) + 1e-9)


def vec_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max()
    e = torch.exp(x)
    return e / (e.sum() + 1e-9)



class IterMixAttn(nn.Module):
    def __init__(self, d: int = 64, mix_steps: int = 2):
        super().__init__()
        self.d = int(d)
        self.mix_steps = int(mix_steps)

        self.x_in = nn.Linear(1, d, bias=True)

        self.qc = nn.Linear(d, d, bias=False)
        self.kc = nn.Linear(d, d, bias=False)
        self.vc = nn.Linear(d, d, bias=False)

        self.y_in = nn.Linear(1, d, bias=False)

        self.qq = nn.Linear(d, d, bias=False)
        self.kq = nn.Linear(d, d, bias=False)
        self.vq = nn.Linear(d, d, bias=False)

        self.out = nn.Linear(d, 1, bias=True)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        if prompt.dim() != 3 or prompt.size(0) != 1 or prompt.size(-1) != 2:
            raise ValueError(f"need prompt [1,L,2], got {tuple(prompt.shape)}")

        t = prompt[0]
        x = t[:, 0:1]
        y = t[:, 1:2]

        h = self.x_in(x)
        ctx = h[:-1]
        qh = h[-1:]
        y_ctx = y[:-1]

        mix = ctx
        y_feat = self.y_in(y_ctx)

        for _ in range(self.mix_steps):
            q = self.qc(mix)
            k = self.kc(mix)
            s = (q @ k.T) / (self.d ** 0.5)
            a = row_softmax(s)
            v = self.vc(mix) + y_feat
            mix = mix + (a @ v)

        q2 = self.qq(qh).squeeze(0)
        k2 = self.kq(mix)
        s2 = (k2 @ q2) / (self.d ** 0.5)
        a2 = vec_softmax(s2)
        z = (a2.unsqueeze(0) @ self.vq(mix)).squeeze(0)
        return self.out(z).squeeze()


def train_model(
    model: nn.Module,
    tasks,
    dev: str,
    n_ctx: int,
    steps: int,
    lr: float,
    log_every: int = 400,
    clip: float = 1.0,
) -> nn.Module:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    acc = 0.0
    for it in range(1, steps + 1):
        task = random.choice(tasks)
        prompt, target = task.make_prompt(n_context=n_ctx, device=dev)
        prompt = prompt.to(dev)

        pred = model(prompt).view(1)
        tgt = target.to(dev).view(1)

        loss = loss_fn(pred, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()

        acc += float(loss.item())
        if it % log_every == 0:
            print(f"[train {it}] loss={acc / log_every:.6f}")
            acc = 0.0

    return model


@torch.no_grad()
def test_against_krr(
    model: nn.Module,
    tasks,
    dev: str,
    n_ctx: int,
    deg: int,
    lam: float,
) -> Tuple[float, List[float], List[float]]:
    model.eval()
    ys_m: List[float] = []
    ys_k: List[float] = []

    for task in tasks:
        prompt, _ = task.make_prompt(n_context=n_ctx, device=dev)
        prompt = prompt.to(dev)

        y_m = float(model(prompt).detach().cpu().item())
        y_k = float(krr_from_prompt(prompt, deg, lam).detach().cpu().item())

        ys_m.append(y_m)
        ys_k.append(y_k)

    mae = float(np.mean([abs(a - b) for a, b in zip(ys_m, ys_k)]))
    return mae, ys_m, ys_k


def save_scatter(y_ref: List[float], y_pred: List[float], title: str, path: str) -> None:
    set_pub_style()

    yr = np.array(y_ref, dtype=float)
    yp = np.array(y_pred, dtype=float)

    lo = float(min(yr.min(), yp.min()))
    hi = float(max(yr.max(), yp.max()))
    span = hi - lo + 1e-12
    lo -= 0.06 * span
    hi += 0.06 * span

    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(yr, yp, s=18, alpha=0.65, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("KRR (reference)")
    ax.set_ylabel("Model")
    ax.set_title(title)

    mae = float(np.mean(np.abs(yp - yr)))
    corr = float(np.corrcoef(yr, yp)[0, 1]) if len(yr) > 1 else 1.0
    ax.text(
        0.04, 0.96,
        f"MAE={mae:.6f}\nPearson r={corr:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    beautify_ax(ax)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def run_seed(
    seed: int,
    dev: str,
    deg: int,
    lam: float,
    n_train_tasks: int,
    n_test_tasks: int,
    n_ctx: int,
    train_steps: int,
    lr: float,
    mix_steps: int,
) -> Tuple[float, float]:
    print(f"\n=== seed {seed} | device {dev} ===")
    set_global_seed(seed)

    train_tasks, _ = generate_polynomial_tasks(
        degree=deg, num_tasks=n_train_tasks, device=dev, return_coeffs=False
    )
    test_tasks, _ = generate_polynomial_tasks(
        degree=deg, num_tasks=n_test_tasks, device=dev, return_coeffs=False
    )

    demo_p, demo_y = train_tasks[0].make_prompt(n_context=n_ctx, device=dev)
    print("demo prompt:", tuple(demo_p.shape), " demo target:", float(demo_y))

    out_dir = os.path.join(ROOT, "results")

    krr_mod = KRRClosedForm(deg=deg, lam=lam).to(dev)
    mae_a, ya, yk = test_against_krr(krr_mod, test_tasks, dev, n_ctx, deg, lam)
    print(f"(A) sanity MAE (closed_form vs KRR) = {mae_a:.12e}")
    save_scatter(
        yk, ya,
        f"Exp1(A) sanity (seed {seed})",
        os.path.join(out_dir, f"exp1_A_sanity_seed_{seed}.png"),
    )

    model = IterMixAttn(d=64, mix_steps=mix_steps).to(dev)
    print(f"[info] learned model mix_steps = {mix_steps}")
    model = train_model(
        model, train_tasks, dev, n_ctx, steps=train_steps, lr=lr, log_every=400, clip=1.0
    )

    mae_b, yb, yk2 = test_against_krr(model, test_tasks, dev, n_ctx, deg, lam)
    print(f"(B) learned MAE (iter-mix model vs KRR) = {mae_b:.6f}")
    save_scatter(
        yk2, yb,
        f"Exp1(B) learned (seed {seed})",
        os.path.join(out_dir, f"exp1_B_learned_seed_{seed}.png"),
    )

    return mae_a, mae_b


def main():
    print("### EXP1 FILE ###")
    print(__file__)

    dev = pick_device()

    seeds = [0, 1, 2, 3, 4]
    deg = 3
    lam = 1e-4
    n_ctx = 4
    n_train_tasks = 200
    n_test_tasks = 50

    train_steps = 12000
    lr = 3e-4

    mix_steps = 2

    maes_a = []
    maes_b = []

    for s in seeds:
        a, b = run_seed(
            seed=s,
            dev=dev,
            deg=deg,
            lam=lam,
            n_train_tasks=n_train_tasks,
            n_test_tasks=n_test_tasks,
            n_ctx=n_ctx,
            train_steps=train_steps,
            lr=lr,
            mix_steps=mix_steps,
        )
        maes_a.append(a)
        maes_b.append(b)

    a_arr = np.array(maes_a, dtype=float)
    b_arr = np.array(maes_b, dtype=float)

    print("\n=== summary (mean ± std over seeds) ===")
    for i, s in enumerate(seeds):
        print(f"seed {s}: A={a_arr[i]:.3e} | B={b_arr[i]:.6f}")

    print(f"A (sanity) mean±std = {float(a_arr.mean()):.3e} ± {float(a_arr.std(ddof=0)):.3e}")
    print(f"B (learned) mean±std = {float(b_arr.mean()):.6f} ± {float(b_arr.std(ddof=0)):.6f}")


if __name__ == "__main__":
    main()
