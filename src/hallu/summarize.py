"""Summarise several runs into paper-style tables.

Usage:
    python -m hallu.summarize runs/colab_qwen7b_triviaqa_n400 runs/colab_qwen7b_nq_n400 ...
    python -m hallu.summarize runs/*_n400 --labels judge --csv results_table.csv

Prints, per run: AUROC (95% CI) for the main scores, the paired difference vs
original SINdex with p-value, label-agreement statistics and the
hallucination rate under each labelling method.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

MAIN_SCORES = [
    ("sindex", "SINdex (paper)"),
    ("sindex_pp", "SINdex++ (N1)"),
    ("nli:sindex_pp", "+NLI veto (N2)"),
    ("adaptive_nli:sindex_pp", "+NLI +adaptive"),
    ("adaptive:sindex_pp", "adaptive thr. (N4)"),
    ("early_stop:sindex_pp", "early stop (N3)"),
    ("cluster_entropy", "cluster entropy"),
    ("lin_eigv", "Lin EigV"),
    ("lin_degree", "Lin Degree"),
    ("naive_entropy", "naive entropy"),
]


def load(run: Path):
    rep = json.loads((run / "report.json").read_text())
    labels = [json.loads(l) for l in open(run / "labels.jsonl")] if (run / "labels.jsonl").exists() else []
    return rep, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", default="judge", help="label set for the main table")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)

    rows = []
    for r in a.runs:
        run = Path(r)
        rep, labels = load(run)
        res = rep["results"].get(a.labels)
        if res is None:
            print(f"{run.name}: no '{a.labels}' labels; available {list(rep['results'])}")
            continue
        print(f"\n=== {run.name}  (n={rep['n_questions']}, labels={a.labels}, "
              f"hallucination rate {res['hallucination_rate']:.3f}, "
              f"early-stop samples {rep.get('early_stop_mean_samples', float('nan')):.2f})")
        paired = res.get("paired_vs_sindex", {})
        for key, name in MAIN_SCORES:
            s = res["scores"].get(key)
            if not s:
                continue
            p = paired.get(key)
            extra = f"  d={p['delta']:+.3f} p={p['p_value']:.3f}" if p else ""
            print(f"  {name:22s} {s['auroc']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]{extra}")
            rows.append({"run": run.name, "labels": a.labels, "score": name, "auroc": s["auroc"],
                         "lo": s["lo"], "hi": s["hi"], "delta_vs_sindex": p["delta"] if p else None,
                         "p_value": p["p_value"] if p else None})
        rates = {k: v["hallucination_rate"] for k, v in rep["results"].items()}
        print("  hallucination rate by label:", {k: round(v, 3) for k, v in rates.items()})
        if labels:
            ms = [m for m in ["ensemble", "embedding", "em", "f1", "judge"] if m in labels[0]]
            ag = []
            for x, y in itertools.combinations(ms, 2):
                ok = [l for l in labels if l[x] is not None and l[y] is not None]
                ag.append(f"{x}/{y}={np.mean([l[x] == l[y] for l in ok]):.2f}")
            print("  label agreement:", ", ".join(ag))
        if "embedding" in rep["results"] and a.labels in rep["results"]:
            e, j = rep["results"]["embedding"], rep["results"][a.labels]
            ge = e["scores"]["sindex"]["auroc"] - e["scores"]["cluster_entropy"]["auroc"]
            gj = j["scores"]["sindex"]["auroc"] - j["scores"]["cluster_entropy"]["auroc"]
            print(f"  SINdex lead over cluster entropy: paper labels {ge:+.3f} vs {a.labels} labels {gj:+.3f}")
    if a.csv and rows:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {a.csv}")


if __name__ == "__main__":
    main()
