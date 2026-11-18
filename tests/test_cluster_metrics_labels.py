import numpy as np
import pytest

from hallu.cluster import adaptive_threshold, agglomerative_labels, cluster_answers, cosine_similarity_matrix
from hallu.contradiction import negation_conflict, numeric_conflict
from hallu.labels import contains_gold, exact_match, label_record, max_f1, normalize_answer
from hallu.metrics import aurac, auroc, bootstrap_ci, paired_bootstrap
from hallu.sequential import early_stop_score


def _two_groups(n1=5, n2=5, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([1.0, 0, 0]) + noise * rng.standard_normal((n1, 3))
    b = np.array([0, 1.0, 0]) + noise * rng.standard_normal((n2, 3))
    return np.vstack([a, b])


def test_clustering_finds_two_groups():
    E = _two_groups()
    labels = agglomerative_labels(cosine_similarity_matrix(E), 0.05)
    assert len(set(labels)) == 2
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1


def test_single_and_empty():
    assert agglomerative_labels(np.ones((1, 1))).tolist() == [0]


def test_adaptive_threshold_in_range():
    E = _two_groups(noise=0.05)
    t = adaptive_threshold(cosine_similarity_matrix(E))
    assert 0.0 < t < 0.5


def test_contradiction_splits_cluster():
    E = np.array([[1.0, 0.0], [0.999, 0.01]])
    texts = ["born in 1879", "born in 1897"]
    lab_plain, _, _ = cluster_answers(E, texts)
    lab_ck, _, _ = cluster_answers(E, texts, contradicts=numeric_conflict)
    assert len(set(lab_plain)) == 1 and len(set(lab_ck)) == 2


def test_rules():
    assert numeric_conflict("1879", "in 1897")
    assert not numeric_conflict("1,000", "1000")
    assert negation_conflict("It is Earth", "It is not Earth")


def test_labels():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Paris", ["Paris"])
    assert contains_gold("The capital is Paris.", ["paris"])
    assert max_f1("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)
    rec = {"question": "q", "answers": ["Paris"], "most_likely": {"text": "London"}}
    assert label_record(rec, ["ensemble", "em"]) == {"ensemble": 1, "em": 1}
    # regression: shared generic token must not make a wrong answer "correct"
    rec2 = {"question": "q", "answers": ["Mount Chimborazo"], "most_likely": {"text": "Mount Everest"}}
    assert label_record(rec2, ["ensemble"]) == {"ensemble": 1}
    rec3 = {"question": "q", "answers": ["Ebony"], "most_likely": {"text": "Book, Ebony"}}
    assert label_record(rec3, ["ensemble"]) == {"ensemble": 0}


def test_metrics():
    y = np.array([0, 0, 1, 1] * 25)
    perfect = y + 0.0
    rand = np.random.default_rng(0).random(len(y))
    assert auroc(y, perfect) == pytest.approx(1.0)
    ci = bootstrap_ci(y, perfect, n_boot=200)
    assert ci["lo"] == pytest.approx(1.0)
    pb = paired_bootstrap(y, perfect, rand, n_boot=200)
    assert pb["delta"] > 0 and pb["p_value"] < 0.05
    assert aurac(y, perfect) > aurac(y, -perfect)


def test_early_stop_unanimous_uses_min_samples():
    E = np.tile([1.0, 0.0, 0.0], (10, 1))
    r = early_stop_score(E, min_samples=3)
    assert r["n_used"] == 3 and r["score"] == pytest.approx(0.0)


def test_early_stop_disagreement_uses_more():
    E = np.eye(10)
    r = early_stop_score(E, min_samples=3, patience=2, tol=1e-9)
    assert r["n_used"] == 10
