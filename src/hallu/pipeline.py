"""End-to-end pipeline: generate -> score -> label -> evaluate.

Usage:
    python -m hallu.pipeline --config configs/smoke.yaml --stage all
    python -m hallu.pipeline --config configs/llama3_triviaqa.yaml --stage generate
Stages read/write files under cfg.output_dir so each can be re-run alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from . import data as data_mod
from .cluster import cluster_answers, cosine_similarity_matrix
from .contradiction import make_checker
from .embed import answer_strings, embed
from .generate import generate_dataset
from .labels import label_record
from .metrics import accuracy_at_coverage, aurac, bootstrap_ci, paired_bootstrap
from .scores import all_scores
from .sequential import early_stop_score


def load_cfg(path: str) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def out_dir(cfg: Dict) -> Path:
    d = Path(cfg.get("output_dir", "runs/default"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(p: Path) -> List[Dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: List[Dict]) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- stages
def stage_generate(cfg: Dict) -> None:
    d = cfg["data"]
    rows = data_mod.load(d["name"], n=d.get("n"), seed=cfg.get("seed", 0), split=d.get("split", "validation"),
                         path=d.get("path"))
    generate_dataset(rows, cfg, out_dir(cfg) / "generations.jsonl")


def _variants(cfg: Dict) -> List[Dict]:
    """Clustering variants to score. Each has a name prefix for its score columns."""
    s = cfg["scoring"]
    base = {"name": "base", "adaptive": False, "contradiction": None}
    vs = [base]
    for v in s.get("variants", []):
        vs.append({"name": v["name"], "adaptive": v.get("adaptive", False),
                   "contradiction": v.get("contradiction")})
    return vs


def stage_score(cfg: Dict) -> None:
    s = cfg["scoring"]
    recs = read_jsonl(out_dir(cfg) / "generations.jsonl")
    emb_model = s.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    device = s.get("device")
    thr = float(s.get("distance_threshold", 0.05))
    lam = float(s.get("lam", 1.0))
    checkers = {}
    rows = []
    t0 = time.time()
    for rec in recs:
        texts = [x["text"] for x in rec["samples"]]
        strs = answer_strings(rec["question"], texts, s.get("string_mode", "question_sep"))
        E = embed(strs, emb_model, device)
        sim = cosine_similarity_matrix(E)
        row = {"id": rec["id"]}
        for v in _variants(cfg):
            ck = v["contradiction"]
            if ck and ck not in checkers:
                checkers[ck] = make_checker(ck, device)
            labels, _, used = cluster_answers(E, texts, distance_threshold=thr, adaptive=v["adaptive"],
                                              contradicts=checkers.get(ck), sim=sim)
            sc = all_scores(labels, sim, lam=lam, singleton=s.get("singleton", "one"),
                            logliks=[x.get("loglik") for x in rec["samples"]],
                            lengths=[x.get("length") for x in rec["samples"]])
            if v["name"] != "base":
                sc = {k: sc[k] for k in ("sindex", "sindex_pp", "cluster_entropy", "num_clusters")}
            for k, val in sc.items():
                row[k if v["name"] == "base" else f"{v['name']}:{k}"] = val
            row[f"{v['name']}:threshold"] = used
        es = s.get("early_stop")
        if es:
            r = early_stop_score(E, distance_threshold=thr, min_samples=es.get("min_samples", 3),
                                 patience=es.get("patience", 2), tol=es.get("tol", 0.05), lam=lam)
            row["early_stop:sindex_pp"] = r["score"]
            row["early_stop:n_used"] = r["n_used"]
        rows.append(row)
    write_jsonl(out_dir(cfg) / "scores.jsonl", rows)
    print(f"[score] {len(rows)} questions scored in {time.time() - t0:.1f}s")


def stage_label(cfg: Dict) -> None:
    L = cfg["labels"]
    recs = read_jsonl(out_dir(cfg) / "generations.jsonl")
    methods = L.get("methods", ["ensemble", "embedding"])
    emb_model = L.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_fn = (lambda t: embed(t, emb_model, L.get("device"))) if "embedding" in methods else None
    judge = None
    if "judge" in methods and L.get("judge_model"):
        from .generate import HFGenerator, VLLMGenerator

        if L.get("judge_backend", "hf") == "vllm":
            judge = VLLMGenerator(L["judge_model"], quantization=L.get("judge_quantization"),
                                  dtype=L.get("judge_dtype", "auto"),
                                  gpu_memory_utilization=L.get("gpu_memory_utilization", 0.85),
                                  max_model_len=L.get("max_model_len", 2048))
        else:
            judge = HFGenerator(L["judge_model"], device=L.get("device", "auto"))
    rows = []
    for rec in recs:
        lab = label_record(rec, methods, f1_threshold=L.get("f1_threshold", 0.5), embed_fn=embed_fn,
                           emb_threshold=L.get("embedding_threshold", 0.95), judge=judge)
        rows.append({"id": rec["id"], "pred": rec["most_likely"]["text"], "gold": rec["answers"][:3], **lab})
    write_jsonl(out_dir(cfg) / "labels.jsonl", rows)
    for m in methods:
        vals = [r[m] for r in rows if r[m] is not None]
        if vals:
            print(f"[label] {m}: hallucination rate {np.mean(vals):.3f} (n={len(vals)})")
    if "embedding" in methods and "ensemble" in methods:
        agree = np.mean([r["embedding"] == r["ensemble"] for r in rows])
        print(f"[label] agreement embedding-vs-ensemble labels: {agree:.3f}")


def stage_evaluate(cfg: Dict) -> Dict:
    E = cfg.get("evaluation", {})
    d = out_dir(cfg)
    scores = {r["id"]: r for r in read_jsonl(d / "scores.jsonl")}
    labels = {r["id"]: r for r in read_jsonl(d / "labels.jsonl")}
    ids = [i for i in scores if i in labels]
    score_keys = [k for k in scores[ids[0]] if k != "id" and not k.endswith(":threshold")
                  and not k.endswith("n_used") and k != "n_singletons"]
    label_keys = [k for k in labels[ids[0]] if k not in ("id", "pred", "gold")]
    n_boot = int(E.get("n_boot", 2000))
    report = {"n_questions": len(ids), "config": cfg.get("_config_path"), "results": {}}
    for lk in label_keys:
        y = np.array([np.nan if labels[i][lk] is None else labels[i][lk] for i in ids], dtype=float)
        if np.all(np.isnan(y)) or len(np.unique(y[~np.isnan(y)])) < 2:
            print(f"[evaluate] skipping labels '{lk}': only one class present (AUROC undefined)")
            continue
        res = {"hallucination_rate": float(np.nanmean(y)), "scores": {}}
        for sk in score_keys:
            s = np.array([scores[i][sk] for i in ids], dtype=float)
            if np.all(np.isnan(s)):
                continue
            ci = bootstrap_ci(y, s, n_boot=n_boot, seed=cfg.get("seed", 0))
            ci["aurac"] = aurac(y, s)
            ci["acc@80%"] = accuracy_at_coverage(y, s, 0.8)
            res["scores"][sk] = ci
        ref = E.get("compare_against", "sindex")
        res["paired_vs_" + ref] = {}
        for sk in E.get("compare", ["sindex_pp", "cluster_entropy"]):
            if sk in scores[ids[0]] and ref in scores[ids[0]]:
                a = [scores[i][sk] for i in ids]
                b = [scores[i][ref] for i in ids]
                res["paired_vs_" + ref][sk] = paired_bootstrap(y, a, b, n_boot=n_boot, seed=cfg.get("seed", 0))
        report["results"][lk] = res
    if any("early_stop:n_used" in scores[i] for i in ids):
        report["early_stop_mean_samples"] = float(np.mean([scores[i]["early_stop:n_used"] for i in ids]))
    (d / "report.json").write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(rep: Dict) -> None:
    print(f"\n=== Report: {rep['n_questions']} questions ({rep['config']}) ===")
    if "early_stop_mean_samples" in rep:
        print(f"early stopping used {rep['early_stop_mean_samples']:.2f} samples on average")
    for lk, res in rep["results"].items():
        print(f"\n-- labels: {lk}  (hallucination rate {res['hallucination_rate']:.3f})")
        print(f"{'score':34s} {'AUROC':>7s} {'95% CI':>17s} {'AURAC':>7s} {'acc@80%':>8s}")
        for sk, r in sorted(res["scores"].items(), key=lambda kv: -np.nan_to_num(kv[1]["auroc"])):
            print(f"{sk:34s} {r['auroc']:7.3f} [{r['lo']:.3f}, {r['hi']:.3f}] {r['aurac']:7.3f} {r['acc@80%']:8.3f}")
        for k, v in res.items():
            if k.startswith("paired_vs_"):
                for sk, pr in v.items():
                    print(f"  {sk} vs {k[10:]}: dAUROC={pr['delta']:+.3f} "
                          f"CI[{pr['lo']:+.3f},{pr['hi']:+.3f}] p={pr['p_value']:.3f}")


STAGES = {"generate": stage_generate, "score": stage_score, "label": stage_label, "evaluate": stage_evaluate}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all"] + list(STAGES))
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    for name in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"\n##### stage: {name}")
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
