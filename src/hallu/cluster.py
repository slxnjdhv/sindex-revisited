"""Semantic clustering of sampled LLM answers.

Reproduces the SINdex clustering step (Abdaljalil et al., IEEE OJCS 2026):
hierarchical agglomerative clustering, average linkage, cosine distance,
fixed distance threshold 0.05 (= cosine similarity 0.95).

Extensions:
  * N2 contradiction-aware clustering: pairs flagged as contradicting get
    distance 1.0 before linkage, so they cannot be merged cheaply.
  * N4 adaptive threshold: per-question cut chosen from the dendrogram.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Rows of `emb` need not be normalised."""
    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)
    return sim


def _distance_matrix(sim: np.ndarray) -> np.ndarray:
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry against float noise
    np.fill_diagonal(dist, 0.0)
    return dist


def apply_contradictions(
    sim: np.ndarray,
    texts: Sequence[str],
    contradicts: Callable[[str, str], bool],
    band: tuple = (0.85, 1.0),
) -> np.ndarray:
    """N2: for pairs whose similarity lies in `band`, ask `contradicts(a, b)`.

    Only borderline pairs are checked, so the expensive checker (e.g. an NLI
    model) runs on a small fraction of the P*(P-1)/2 pairs. Contradicting pairs
    get similarity -1 (distance 2 -> effectively never merged under average
    linkage at small thresholds).
    """
    sim = sim.copy()
    lo, hi = band
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            if lo <= sim[i, j] <= hi and texts[i].strip() != texts[j].strip():
                if contradicts(texts[i], texts[j]):
                    sim[i, j] = sim[j, i] = -1.0
    return sim


def agglomerative_labels(sim: np.ndarray, distance_threshold: float = 0.05) -> np.ndarray:
    """Cluster labels (0..k-1) using average linkage on cosine distance."""
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    condensed = squareform(_distance_matrix(sim), checks=False)
    Z = linkage(condensed, method="average")
    raw = fcluster(Z, t=distance_threshold, criterion="distance")
    return _relabel(raw)


def adaptive_threshold(
    sim: np.ndarray,
    lo: float = 0.02,
    hi: float = 0.30,
    fallback: float = 0.05,
) -> float:
    """N4: choose a per-question distance threshold from the dendrogram.

    Takes the merge heights inside [lo, hi] and cuts in the middle of the
    largest gap between consecutive heights (the most "natural" break).
    If all merges are below lo (all answers agree), or there is no gap in the
    window, returns `fallback`.
    """
    n = sim.shape[0]
    if n < 3:
        return fallback
    condensed = squareform(_distance_matrix(sim), checks=False)
    heights = np.sort(linkage(condensed, method="average")[:, 2])
    pts = np.concatenate([[lo], heights[(heights > lo) & (heights < hi)], [hi]])
    if len(pts) < 3:
        return fallback
    gaps = np.diff(pts)
    k = int(np.argmax(gaps))
    return float((pts[k] + pts[k + 1]) / 2.0)


def cluster_answers(
    emb: np.ndarray,
    texts: Optional[Sequence[str]] = None,
    distance_threshold: float = 0.05,
    adaptive: bool = False,
    contradicts: Optional[Callable[[str, str], bool]] = None,
    contradiction_band: tuple = (0.85, 1.0),
    sim: Optional[np.ndarray] = None,
):
    """Full clustering step. Returns (labels, sim_matrix, threshold_used).

    `sim` returned is the *original* cosine similarity (used for cohesion),
    not the contradiction-modified one.
    """
    base_sim = cosine_similarity_matrix(emb) if sim is None else sim
    work = base_sim
    if contradicts is not None and texts is not None:
        work = apply_contradictions(base_sim, texts, contradicts, contradiction_band)
    thr = adaptive_threshold(work, fallback=distance_threshold) if adaptive else distance_threshold
    labels = agglomerative_labels(work, thr)
    return labels, base_sim, thr


def _relabel(raw: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to 0..k-1 in order of first appearance."""
    mapping = {}
    out = np.empty(len(raw), dtype=int)
    for i, r in enumerate(raw):
        if r not in mapping:
            mapping[r] = len(mapping)
        out[i] = mapping[r]
    return out
