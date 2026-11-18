"""N3: sample-efficient scoring with early stopping.

The expensive part of sampling-based detection is the P LLM calls, not the
clustering. Here we add samples one at a time and stop once the score has
stabilised. Because all P samples are cached on disk, this is simulated
offline: we report the score at the stopping point and how many samples it
needed, then compare AUROC vs average samples used.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .cluster import agglomerative_labels, cosine_similarity_matrix
from .scores import cluster_cohesion, cluster_proportions, sindex_pp


def early_stop_score(
    emb: np.ndarray,
    distance_threshold: float = 0.05,
    min_samples: int = 3,
    patience: int = 2,
    tol: float = 0.05,
    unanimous_stop: bool = True,
    score_fn: Callable = None,
    lam: float = 1.0,
) -> Dict[str, float]:
    """Return {'score', 'n_used'}.

    Stopping rules (whichever first):
      * unanimous_stop: the first `min_samples` answers all fall in one cluster
        -> confident, stop immediately;
      * the score changes by < tol for `patience` consecutive added samples.
    """
    emb = np.asarray(emb)
    P = emb.shape[0]
    full_sim = cosine_similarity_matrix(emb)

    def score_at(k: int) -> float:
        sim = full_sim[:k, :k]
        labels = agglomerative_labels(sim, distance_threshold)
        if score_fn is not None:
            return float(score_fn(labels, sim))
        p = cluster_proportions(labels)
        c = cluster_cohesion(labels, sim)
        return sindex_pp(p, c, lam=lam)

    if P <= min_samples:
        return {"score": score_at(P), "n_used": float(P)}

    k = min_samples
    prev = score_at(k)
    if unanimous_stop:
        if len(np.unique(agglomerative_labels(full_sim[:k, :k], distance_threshold))) == 1:
            return {"score": prev, "n_used": float(k)}
    stable = 0
    while k < P:
        k += 1
        cur = score_at(k)
        stable = stable + 1 if abs(cur - prev) < tol else 0
        prev = cur
        if stable >= patience:
            break
    return {"score": prev, "n_used": float(k)}
