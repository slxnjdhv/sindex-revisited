"""Data figures for the paper (matplotlib; generated from the run files).

    cd sindex_research && .venv/bin/python paper/make_figures.py

Method diagrams are drawn in draw.io (paper/diagrams/*.drawio).
Palette: validated categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper"))
from hallu.metrics import _batched_auroc  # noqa: E402
from hallu.scores import sindex_original, sindex_pp  # noqa: E402
import make_tables as T  # noqa: E402

FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def theory():
    c = np.linspace(0.6, 1.0, 81)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for p, ls, tag in (([0.2] * 5, "-", "5 clusters of 2"), ([0.5] * 2, "--", "2 clusters of 5")):
        a = [sindex_original(p, [x] * len(p)) for x in c]
        b = [sindex_pp(p, [x] * len(p)) for x in c]
        ax.plot(c, a, color=ORANGE, lw=2, ls=ls)
        ax.plot(c, b, color=BLUE, lw=2, ls=ls)
        ax.text(0.605, a[0] + (0.035 if p[0] == 0.2 else -0.09), f"SINdex, {tag}", color=INK, fontsize=7)
        ax.text(0.605, b[0] + 0.03, f"SINdex++, {tag}", color=INK, fontsize=7)
    ax.set_xlabel("Within-cluster cohesion $\\bar c$ (all clusters)")
    ax.set_ylabel("Score")
    ax.invert_xaxis()
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Looser clusters (right) should raise the score", fontsize=8, color=INK, loc="left")
    ax.set_xlim(1.0, 0.6)
    fig.tight_layout()
    fig.savefig(FIG / "theory.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def label_rates():
    rows = []
    for (m, d) in T.RUNS:
        S, L = T.DATA[(m, d)]
        e, _ = T.yx(S, L, "embedding", "sindex")
        v, _ = T.yx(S, L, T.MAIN, "sindex")
        rows.append((f"{T.MODEL_NAME[m]}\n{T.DS_NAME[d]}", e.mean() * 100, v.mean() * 100))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = np.arange(len(rows))[::-1]
    for yi, (_, e, v) in zip(y, rows):
        ax.plot([v, e], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([r[2] for r in rows], y, s=36, color=BLUE, zorder=3, edgecolor="white", lw=1.5, label="Validated labels (3-judge majority)")
    ax.scatter([r[1] for r in rows], y, s=36, color=ORANGE, zorder=3, edgecolor="white", lw=1.5, label="Paper's rule (embedding cos > 0.95)")
    for yi, (_, e, v) in zip(y, rows):
        ax.text(e + 2.5, yi, f"{e:.0f}%", va="center", fontsize=7, color=INK)
        ax.text(v - 2.5, yi, f"{v:.0f}%", va="center", ha="right", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(-12, 112)
    ax.set_xlabel("Measured hallucination rate (%)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.25), frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "label_rates.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def forest():
    rng = np.random.default_rng(0)
    rows, boots = [], []
    for k in T.RUNS:
        S, L = T.DATA[k]
        yv, a = T.yx(S, L, T.MAIN, "nli:sindex_pp")
        _, b = T.yx(S, L, T.MAIN, "sindex")
        idx = rng.integers(0, len(yv), (4000, len(yv)))
        dd = _batched_auroc(yv[idx], a[idx]) - _batched_auroc(yv[idx], b[idx])
        d0 = _batched_auroc(yv[None], a[None])[0] - _batched_auroc(yv[None], b[None])[0]
        boots.append(dd)
        rows.append((f"{T.MODEL_NAME[k[0]]} / {T.DS_NAME[k[1]]} (n+={int(yv.sum())})", d0,
                     np.nanpercentile(dd, 2.5), np.nanpercentile(dd, 97.5)))
    m = np.nanmean(np.vstack(boots), axis=0)
    rows.append(("Pooled (6 runs)", float(np.mean([r[1] for r in rows])), np.percentile(m, 2.5), np.percentile(m, 97.5)))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    for yi, (lab, d, lo, hi) in zip(y, rows):
        pooled = lab.startswith("Pooled")
        ax.plot([lo, hi], [yi, yi], color=INK if pooled else BLUE, lw=2, solid_capstyle="round")
        ax.scatter([d], [yi], s=44 if pooled else 30, marker="D" if pooled else "o",
                   color=INK if pooled else BLUE, zorder=3, edgecolor="white", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("ΔAUROC over SINdex")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "forest.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    theory()
    label_rates()
    forest()
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
