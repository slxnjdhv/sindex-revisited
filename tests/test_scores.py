import math

import numpy as np
import pytest

from hallu.scores import (all_scores, cluster_cohesion, cluster_proportions, entropy, lin_degree,
                          lin_eccentricity, lin_eigv, sindex_original, sindex_pp)


def test_entropy_basics():
    assert entropy([1.0]) == pytest.approx(0.0)
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2))


def test_sindex_reduces_to_entropy_when_perfectly_coherent():
    p = [0.5, 0.3, 0.2]
    assert sindex_original(p, [1, 1, 1]) == pytest.approx(entropy(p))
    assert sindex_pp(p, [1, 1, 1]) == pytest.approx(entropy(p))


def test_original_sindex_flaw_documented():
    """Paper eq.11 claims SINdex >= H(p) and rises as coherence falls.
    With 5 clusters of share 0.2 (< 1/e) it falls instead."""
    p = [0.2] * 5
    tight = sindex_original(p, [1.0] * 5)
    loose = sindex_original(p, [0.8] * 5)
    assert loose < tight            # counter-example to the paper's claim
    assert loose < entropy(p)       # violates SINdex >= H(p)


def test_sindex_pp_monotone_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = rng.integers(1, 11)
        p = rng.dirichlet(np.ones(k))
        c = rng.uniform(0.5, 1.0, k)
        j = rng.integers(0, k)
        c2 = c.copy()
        c2[j] = max(0.0, c[j] - rng.uniform(0.01, 0.3))  # make one cluster looser
        assert sindex_pp(p, c2) > sindex_pp(p, c)
        assert sindex_pp(p, c) >= entropy(p) - 1e-12


def test_singleton_cohesion_policy():
    labels = np.array([0, 0, 1])
    sim = np.array([[1, .9, .1], [.9, 1, .1], [.1, .1, 1.]])
    assert cluster_cohesion(labels, sim, "one").tolist() == pytest.approx([0.9, 1.0])
    assert math.isnan(cluster_cohesion(labels, sim, "nan")[1])


def test_proportions():
    assert cluster_proportions(np.array([0, 0, 1, 2])).tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_lin_baselines_order():
    same = np.ones((5, 5))
    diff = np.eye(5)
    assert lin_degree(same) == pytest.approx(0.0)
    assert lin_degree(diff) > lin_degree(same)
    assert lin_eigv(diff) > lin_eigv(same)
    assert lin_eccentricity(same) == pytest.approx(0.0, abs=1e-9)
    two = np.kron(np.eye(2), np.ones((3, 3)))  # two clean clusters
    assert lin_eccentricity(two) > lin_eccentricity(same)


def test_all_scores_keys():
    sim = np.eye(4)
    s = all_scores(np.array([0, 1, 2, 3]), sim, logliks=[-1, -2, -3, -4], lengths=[1, 1, 1, 1])
    for k in ("sindex", "sindex_pp", "cluster_entropy", "lin_degree", "lin_eigv", "naive_entropy"):
        assert k in s and np.isfinite(s[k])
    assert s["n_singletons"] == 4
