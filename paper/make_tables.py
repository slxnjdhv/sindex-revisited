"""Generate every number used in the paper from the run files.

    cd sindex_research && .venv/bin/python paper/make_tables.py

Writes paper/tables/*.tex (included by main.tex) and paper/tables/numbers.json
(every scalar quoted in the text, so prose and tables cannot drift apart).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hallu.metrics import _batched_auroc, bootstrap_ci, paired_bootstrap  # noqa: E402
from hallu.scores import cluster_proportions, sindex_original  # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {(m, d): ROOT / "runs" / f"colab_{m}_{d}_n400"
        for m in ("qwen7b", "llama8b") for d in ("triviaqa", "nq", "squad")}
MODEL_NAME = {"qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B"}
DS_NAME = {"triviaqa": "TriviaQA", "nq": "NQ", "squad": "SQuAD"}
MAIN = "judges_majority"
METHODS = [
    ("sindex", "SINdex~\\cite{abdaljalil2026sindex}"),
    ("sindex_pp", "SINdex++ (ours, C1)"),
    ("nli:sindex_pp", "SINdex++ + NLI veto (ours, C2)"),
    ("adaptive:sindex_pp", "SINdex++ + adaptive cut"),
    ("early_stop:sindex_pp", "SINdex++ + early stop"),
    ("cluster_entropy", "Cluster entropy"),
    ("lin_eigv", "EigV~\\cite{lin2024generating}"),
    ("lin_degree", "Degree~\\cite{lin2024generating}"),
    ("lin_ecc", "Eccentricity~\\cite{lin2024generating}"),
    ("naive_entropy", "Predictive entropy (gray-box)"),
]
N = {}


def sg(x: float) -> str:
    """Signed 3-decimal number without a negative zero."""
    x = round(float(x), 3) + 0.0
    return f"{x:+.3f}"


def load(run: Path):
    S, L = {}, {}
    for line in open(run / "scores.jsonl"):
        r = json.loads(line)
        S.setdefault(r["id"], r)
    for line in open(run / "labels.jsonl"):
        r = json.loads(line)
        L.setdefault(r["id"], r)
    return S, L


def yx(S, L, lab, key):
    ids = [i for i in S if i in L and L[i].get(lab) is not None]
    return np.array([L[i][lab] for i in ids]), np.array([S[i][key] for i in ids], dtype=float)


DATA = {k: load(p) for k, p in RUNS.items()}


# ---------------------------------------------------------------- main table
def main_table():
    cols = list(RUNS)
    res = {}
    for k in cols:
        S, L = DATA[k]
        for key, _ in METHODS:
            y, s = yx(S, L, MAIN, key)
            res[(key, k)] = bootstrap_ci(y, s, n_boot=2000)["auroc"]
            if key != "sindex":
                _, b = yx(S, L, MAIN, "sindex")
                res[("p", key, k)] = paired_bootstrap(y, s, b, n_boot=2000)["p_value"]
    best = {k: max(res[(m, k)] for m, _ in METHODS if m != "naive_entropy") for k in cols}
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{AUROC for hallucination detection with validated labels (majority of three LLM judges). "
        "400 questions per dataset (TriviaQA: 388 unique), $P{=}10$ samples. Bold: best black-box score per column. "
        "$^{\\dagger}$: better than SINdex, one-sided paired bootstrap $p<0.05$ (2{,}000 resamples).}",
        "\\label{tab:main}\\small",
        "\\begin{tabular}{l" + "c" * len(cols) + "}\\toprule",
        " & \\multicolumn{3}{c}{" + MODEL_NAME["qwen7b"] + "} & \\multicolumn{3}{c}{" + MODEL_NAME["llama8b"] + "}\\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Method & " + " & ".join(DS_NAME[d] for _, d in cols) + "\\\\\\midrule",
    ]
    for key, name in METHODS:
        cells = []
        for k in cols:
            v = res[(key, k)]
            c = f"{v:.3f}"
            if key != "naive_entropy" and abs(v - best[k]) < 1e-9:
                c = f"\\textbf{{{c}}}"
            if key != "sindex" and res[("p", key, k)] < 0.05 and v > res[("sindex", k)]:
                c += "$^{\\dagger}$"
            cells.append(c)
        if key == "cluster_entropy":
            lines.append("\\midrule")
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (OUT / "main.tex").write_text("\n".join(lines))
    for k in cols:
        N[f"auroc_sindex_{k[0]}_{k[1]}"] = round(res[("sindex", k)], 3)
        N[f"auroc_nli_{k[0]}_{k[1]}"] = round(res[("nli:sindex_pp", k)], 3)
    N["n_runs_nli_better"] = sum(res[("nli:sindex_pp", k)] > res[("sindex", k)] for k in cols)


# ---------------------------------------------------------------- pooled
def pooled(a, b, lab, keys, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    per, boots = [], []
    for k in keys:
        S, L = DATA[k]
        y, sa = yx(S, L, lab, a)
        _, sb = yx(S, L, lab, b)
        idx = rng.integers(0, len(y), (B, len(y)))
        boots.append(_batched_auroc(y[idx], sa[idx]) - _batched_auroc(y[idx], sb[idx]))
        per.append(_batched_auroc(y[None], sa[None])[0] - _batched_auroc(y[None], sb[None])[0])
    m = np.nanmean(np.vstack(boots), axis=0)
    return float(np.mean(per)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(np.mean(m <= 0))


def pooled_table():
    keys = list(RUNS)
    rows = [
        ("NLI veto (C2) $-$ SINdex", "nli:sindex_pp", "sindex", MAIN, "nli"),
        ("SINdex++ (C1) $-$ SINdex", "sindex_pp", "sindex", MAIN, "pp"),
        ("SINdex $-$ clust.\\ ent.\\ (paper labels)", "sindex", "cluster_entropy", "embedding", "lead_emb"),
        ("SINdex $-$ clust.\\ ent.\\ (valid.\\ labels)", "sindex", "cluster_entropy", MAIN, "lead_maj"),
        ("SINdex $-$ EigV~\\cite{lin2024generating}", "sindex", "lin_eigv", MAIN, "vs_eigv"),
        ("Adaptive cut $-$ SINdex", "adaptive:sindex_pp", "sindex", MAIN, "adaptive"),
        ("Early stop $-$ SINdex", "early_stop:sindex_pp", "sindex", MAIN, "early"),
    ]
    lines = ["\\begin{table}[t]\\centering",
             "\\caption{Pooled comparison over 2 models $\\times$ 3 datasets: mean $\\Delta$AUROC with 95\\% CI from a "
             "stratified paired bootstrap (4{,}000 resamples per run) and one-sided $p$-value for $\\Delta\\le 0$. "
             "Validated labels unless stated.}",
             "\\label{tab:pooled}\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{lcc}\\toprule", "Comparison & $\\Delta$AUROC [95\\% CI] & $p$\\\\\\midrule"]
    for name, a, b, lab, tag in rows:
        d, lo, hi, p = pooled(a, b, lab, keys)
        ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"{name} & ${sg(d)}$ [${sg(lo)}$, ${sg(hi)}$] & {ptxt}\\\\")
        N[f"pooled_{tag}"] = [round(d, 3), round(lo, 3), round(hi, 3), round(p, 4)]
    d, lo, hi, p = pooled("nli:sindex_pp", "sindex", "consensus", keys)
    N["pooled_nli_consensus"] = [round(d, 3), round(lo, 3), round(hi, 3), round(p, 4)]
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    (OUT / "pooled.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------- label effect
def label_effect_table():
    lines = ["\\begin{table}[t]\\centering",
             "\\caption{Effect of the labelling rule. Hallucination rate of the most-likely answer under the paper's "
             "embedding rule vs.\\ validated labels, and SINdex's AUROC lead over plain cluster entropy under each.}",
             "\\label{tab:labeleffect}\\small\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{llcccc}\\toprule",
             " & & \\multicolumn{2}{c}{Halluc.\\ rate} & \\multicolumn{2}{c}{SINdex $-$ clust.\\ ent.}\\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "Model & Data & paper & valid. & paper & valid.\\\\\\midrule"]
    for (m, d), _ in RUNS.items():
        S, L = DATA[(m, d)]
        out = []
        for lab in ("embedding", MAIN):
            y, _ = yx(S, L, lab, "sindex")
            out.append(y.mean())
        for lab in ("embedding", MAIN):
            y, a = yx(S, L, lab, "sindex")
            _, b = yx(S, L, lab, "cluster_entropy")
            out.append(_batched_auroc(y[None], a[None])[0] - _batched_auroc(y[None], b[None])[0])
        lines.append(f"{MODEL_NAME[m]} & {DS_NAME[d]} & {out[0]*100:.0f}\\% & {out[1]*100:.0f}\\% & "
                     f"${sg(out[2])}$ & ${sg(out[3])}$\\\\")
        N[f"rate_emb_{m}_{d}"] = round(out[0] * 100)
        N[f"rate_maj_{m}_{d}"] = round(out[1] * 100)
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    (OUT / "labeleffect.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------- label validity
def label_validity_table():
    items = {it["key"]: it for it in json.loads((ROOT / "annotation/items.json").read_text())}
    ref = {r["key"]: r["verdict"] for r in csv.DictReader(open(ROOT / "annotation/annotations_claude.csv"))}
    human = {r["key"]: r["verdict"] for r in csv.DictReader(open(ROOT / "annotation_check/annotations.csv"))}
    labs = {}
    for (m, d), (S, L) in DATA.items():
        if m != "qwen7b":
            continue
        for i, r in L.items():
            labs[f"{d}::{i}"] = r
    keys = [k for k in ref if ref[k] in ("correct", "wrong") and items[k]["subset"] == "random" and k in labs]
    hum = [int(ref[k] == "wrong") for k in keys]

    def kappa(a, b):
        n = len(a)
        po = sum(x == y for x, y in zip(a, b)) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe)

    rows = [("embedding", "Embedding cosine $>0.95$ (SINdex~\\cite{abdaljalil2026sindex})"),
            ("em", "Exact match"), ("f1", "Token F1 $\\ge 0.5$"), ("ensemble", "Exact match $\\lor$ contains"),
            ("judge_qwen7b", "Judge: Qwen2.5-7B"), ("judge_qwen14b", "Judge: Qwen2.5-14B"),
            ("judge_granite8b", "Judge: Granite-3.1-8B"), ("judges_majority", "Majority of 3 judges (ours)")]
    lines = ["\\begin{table}[t]\\centering",
             f"\\caption{{Accuracy of labelling rules against the reference annotation (random subset, $n={len(keys)}$; "
             "Qwen2.5-7B answers on all three datasets). $\\kappa$: Cohen's kappa.}",
             "\\label{tab:labels}\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lcc}\\toprule", "Labelling rule & Accuracy & $\\kappa$\\\\\\midrule"]
    for col, name in rows:
        pairs = [(h, labs[k][col]) for h, k in zip(hum, keys) if labs[k].get(col) is not None]
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        acc = sum(x == y for x, y in pairs) / len(pairs)
        kp = kappa(a, b)
        lines.append(f"{name} & {acc:.3f} & {kp:.2f}\\\\")
        N[f"labacc_{col}"] = [round(acc, 3), round(kp, 2)]
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    (OUT / "labels.tex").write_text("\n".join(lines))
    hk = [k for k in human if human[k] in ("correct", "wrong") and ref.get(k) in ("correct", "wrong")]
    N["human_n"] = len(hk)
    N["human_agree"] = sum(human[k] == ref[k] for k in hk)
    N["ref_n_items"] = len(ref)
    N["ref_counts"] = {v: sum(x == v for x in ref.values()) for v in ("correct", "wrong", "unsure")}
    N["ref_random_n"] = len(keys)
    # false-hallucination rate of the embedding rule among truly correct answers
    corr = [k for k in keys if ref[k] == "correct"]
    N["emb_false_halluc"] = [sum(labs[k]["embedding"] == 1 for k in corr), len(corr)]


# ---------------------------------------------------------------- SINdex++ diagnostics
def diagnostics():
    S, L = DATA[("qwen7b", "triviaqa")]
    rows = list(S.values())
    N["n1_differs"] = sum(abs(r["sindex_pp"] - r["sindex"]) > 1e-9 for r in rows)
    N["n1_flaw_cases"] = sum(r["sindex"] < r["cluster_entropy"] - 1e-9 for r in rows)
    N["n1_rows"] = len(rows)
    for k in RUNS:
        rep = json.loads((RUNS[k] / "report.json").read_text())
        N[f"early_samples_{k[0]}_{k[1]}"] = round(rep.get("early_stop_mean_samples", float("nan")), 2)
    # counterexample
    N["cex"] = {c: round(sindex_original([0.2] * 5, [c] * 5), 3) for c in (1.0, 0.9, 0.8)}
    N["cex2"] = {c: round(sindex_original([0.5, 0.5], [c, c]), 3) for c in (1.0, 0.9, 0.8)}
    _ = cluster_proportions


if __name__ == "__main__":
    main_table()
    pooled_table()
    label_effect_table()
    label_validity_table()
    diagnostics()
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    print(json.dumps(N, indent=1))
