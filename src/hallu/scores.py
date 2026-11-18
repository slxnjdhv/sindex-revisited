"""Uncertainty scores for hallucination detection. Higher = more likely hallucination.

Original SINdex (eq. 6-9 of the paper):
    p'_i   = p_i * mean_pairwise_cos(C_i)
    SINdex = -sum_i p'_i log p'_i

Known issues (see README "Findings"):
  * p' does not sum to 1, so the paper's majorization / Schur-concavity
    argument (eq. 10-11) does not apply. When cluster shares are below 1/e,
    looser clusters *decrease* SINdex.
  * mean pairwise cosine of a singleton cluster is 0/0 (undefined).

SINdex++ (proposed, N1):
    SINdex++ = H(p) + lam * sum_i p_i * (1 - c_i)
  - equals H(p) when every cluster is perfectly coherent (c_i = 1)
  - strictly increasing as any cluster's cohesion c_i decreases (p_i > 0, lam > 0)
  - always >= H(p) >= 0
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

SINGLETON_POLICIES = ("one", "zero_dispersion", "nan")


def cluster_proportions(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0)
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    return counts / counts.sum()


def cluster_cohesion(
    labels: np.ndarray, sim: np.ndarray, singleton: str = "one"
) -> np.ndarray:
    """Mean pairwise cosine similarity inside each cluster (eq. 9).

    singleton policy (the paper does not define |C|=1):
      "one" / "zero_dispersion": treat a single point as perfectly coherent (c=1)
      "nan": return NaN (lets you detect how often it happens)
    """
    if singleton not in SINGLETON_POLICIES:
        raise ValueError(f"singleton must be one of {SINGLETON_POLICIES}")
    labels = np.asarray(labels)
    out = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) == 1:
            out.append(np.nan if singleton == "nan" else 1.0)
            continue
        block = sim[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        out.append(float(np.mean(block[iu])))
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def entropy(p: Sequence[float]) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def sindex_original(p: Sequence[float], cohesion: Sequence[float]) -> float:
    """Exact reproduction of the paper's SINdex (eq. 6-7)."""
    adj = np.asarray(p, dtype=float) * np.asarray(cohesion, dtype=float)
    return entropy(adj)


def sindex_pp(p: Sequence[float], cohesion: Sequence[float], lam: float = 1.0) -> float:
    """Proposed SINdex++ = H(p) + lam * sum p_i (1 - c_i)."""
    p = np.asarray(p, dtype=float)
    c = np.asarray(cohesion, dtype=float)
    return entropy(p) + lam * float((p * (1.0 - c)).sum())


# ---------------------------------------------------------------------------
# Black-box baselines from Lin, Trivedi & Sun (2023),
# "Generating with Confidence: Uncertainty Quantification for Black-box LLMs".
# Here the similarity graph W is the embedding cosine similarity clipped to [0,1].
# ---------------------------------------------------------------------------
def _graph(sim: np.ndarray) -> np.ndarray:
    W = np.clip(np.asarray(sim, dtype=float), 0.0, 1.0)
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 1.0)
    return W


def lin_degree(sim: np.ndarray) -> float:
    """U_Deg = trace(mI - D) / m^2 = 1 - sum(W)/m^2."""
    W = _graph(sim)
    m = W.shape[0]
    return float(1.0 - W.sum() / (m * m))


def lin_eigv(sim: np.ndarray) -> float:
    """U_EigV = sum_k max(0, 1 - lambda_k) of the normalised graph Laplacian."""
    W = _graph(sim)
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(W.shape[0]) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    eig = np.linalg.eigvalsh((L + L.T) / 2.0)
    return float(np.maximum(0.0, 1.0 - eig).sum())


def lin_eccentricity(sim: np.ndarray, eigv_threshold: float = 0.9) -> float:
    """U_Ecc: spread of answers in the spectral embedding.

    Uses eigenvectors of the normalised Laplacian whose eigenvalue is below
    `eigv_threshold` (as in Lin et al.'s reference code). A fixed k would pick
    arbitrary vectors from the degenerate eigenspace when all answers agree.
    """
    W = _graph(sim)
    m = W.shape[0]
    if m < 2:
        return 0.0
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(m) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    vals, vecs = np.linalg.eigh((L + L.T) / 2.0)
    k = max(1, int(np.sum(vals < eigv_threshold)))
    V = vecs[:, :k]
    centred = V - V.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centred, axis=1).sum())


def naive_entropy(logliks: Optional[Sequence[float]], lengths: Optional[Sequence[int]] = None) -> float:
    """Predictive entropy estimate from sampled-sequence log-likelihoods
    (length-normalised if lengths given). Gray-box: needs token log-probs."""
    if logliks is None or len(logliks) == 0:
        return float("nan")
    ll = np.asarray(logliks, dtype=float)
    if lengths is not None:
        ln = np.maximum(np.asarray(lengths, dtype=float), 1.0)
        ll = ll / ln
    return float(-ll.mean())


def all_scores(
    labels: np.ndarray,
    sim: np.ndarray,
    lam: float = 1.0,
    singleton: str = "one",
    logliks: Optional[Sequence[float]] = None,
    lengths: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    p = cluster_proportions(labels)
    c = cluster_cohesion(labels, sim, singleton=singleton)
    c_filled = np.where(np.isnan(c), 1.0, c)
    return {
        "sindex": sindex_original(p, c_filled),
        "sindex_pp": sindex_pp(p, c_filled, lam=lam),
        "cluster_entropy": entropy(p),  # discrete semantic entropy w/ embedding clusters
        "num_clusters": float(len(p)),
        "lin_degree": lin_degree(sim),
        "lin_eigv": lin_eigv(sim),
        "lin_ecc": lin_eccentricity(sim),
        "naive_entropy": naive_entropy(logliks, lengths),
        "n_singletons": float(int(np.sum(np.bincount(labels) == 1))) if len(labels) else 0.0,
    }
