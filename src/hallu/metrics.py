"""Evaluation metrics and statistics.

* AUROC (paper's metric), with 95% bootstrap CI (paper: 2,000 resamples)
* Paired bootstrap test between two scores (paper compares SINdex vs SE)
* AURAC: area under the accuracy-rejection curve (Farquhar et al. 2024)
* Rejection accuracy at fixed coverage
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _clean(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s))
    return y[ok].astype(int), s[ok]


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """y=1 hallucination, s higher = more uncertain."""
    y, s = _clean(y, s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _batched_auroc(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Row-wise AUROC via Mann-Whitney U with tie-averaged ranks.
    Identical to sklearn's roc_auc_score, but vectorised over bootstrap rows.
    Rows with only one class give NaN."""
    from scipy.stats import rankdata

    R = rankdata(S, axis=1)
    n1 = Y.sum(axis=1).astype(float)
    n0 = Y.shape[1] - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = ((R * Y).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    auc[(n1 == 0) | (n0 == 0)] = np.nan
    return auc


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(y, s, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    y, s = _clean(y, s)
    n = len(y)
    idx = _boot_idx(n, n_boot, seed)
    vals = _batched_auroc(y[idx], s[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:  # only one class present: AUROC undefined
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(n)}
    return {"auroc": auroc(y, s), "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))), "n": int(n)}


def paired_bootstrap(y, s_a, s_b, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Is AUROC(a) > AUROC(b)? One-sided p-value = share of resamples where a <= b."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(s_a, dtype=float)
    b = np.asarray(s_b, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    y, a, b = y[ok].astype(int), a[ok], b[ok]
    n = len(y)
    idx = _boot_idx(n, n_boot, seed)
    Y = y[idx]
    diffs = _batched_auroc(Y, a[idx]) - _batched_auroc(Y, b[idx])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"delta": float("nan"), "p_value": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"delta": auroc(y, a) - auroc(y, b), "p_value": float(np.mean(diffs <= 0)),
            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}


def rejection_accuracy_curve(y, s, steps: int = 20):
    """Accuracy of kept answers when rejecting the most uncertain fraction."""
    y, s = _clean(y, s)
    order = np.argsort(s)  # most confident first
    correct = 1 - y[order]
    n = len(y)
    fracs = np.linspace(1.0 / steps, 1.0, steps)
    accs = [float(correct[: max(1, int(round(f * n)))].mean()) for f in fracs]
    return fracs, np.asarray(accs)


def aurac(y, s, steps: int = 100) -> float:
    fracs, accs = rejection_accuracy_curve(y, s, steps)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(accs, fracs) / (fracs[-1] - fracs[0])) if len(fracs) > 1 else float(accs[0])


def accuracy_at_coverage(y, s, coverage: float = 0.8) -> float:
    y, s = _clean(y, s)
    order = np.argsort(s)
    k = max(1, int(round(coverage * len(y))))
    return float((1 - y[order][:k]).mean())
