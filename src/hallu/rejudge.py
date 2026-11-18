"""Re-label existing runs with several LLM judges + majority vote (no regeneration).

Usage (Colab):
    !python -m hallu.rejudge /content/drive/MyDrive/sindex_runs

Finds every run folder under ROOT whose name matches --pattern, then:
  1. for each judge model, starts a fresh process (vLLM frees GPU memory only on
     exit), loads the model once and judges the most-likely answer of every run;
     writes <run>/judge_<name>.jsonl with the raw replies;
  2. merges the verdicts into <run>/labels.jsonl as columns judge_<name>
     (1 = hallucination), plus
       judges_majority : majority vote of the judges
       consensus       : judges_majority AND ensemble (EM/contains) both say wrong
  3. re-runs the evaluate stage for each run.

Judge set (all fit an L4 / A100 one at a time):
  qwen7b    Qwen/Qwen2.5-7B-Instruct         same family as the generator
  qwen14b   Qwen/Qwen2.5-14B-Instruct-AWQ    larger
  granite8b ibm-granite/granite-3.1-8b-instruct  different family (no access request needed)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

JUDGES: Dict[str, Dict] = {
    "qwen7b": {"model": "Qwen/Qwen2.5-7B-Instruct"},
    "qwen14b": {"model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "quantization": "awq"},
    "granite8b": {"model": "ibm-granite/granite-3.1-8b-instruct"},
}


def read_jsonl(p: Path) -> List[Dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: List[Dict]) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_runs(root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and fnmatch.fnmatch(p.name, pattern) and (p / "generations.jsonl").exists())


def _unique(recs: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def worker(name: str, runs: List[Path], dtype: str) -> None:
    """Load one judge model and label all runs."""
    from .generate import VLLMGenerator
    from .labels import parse_yes_no
    from .prompts import judge_prompt

    spec = JUDGES[name]
    gen = VLLMGenerator(spec["model"], quantization=spec.get("quantization"), dtype=dtype,
                        gpu_memory_utilization=0.85, max_model_len=2048)
    for run in runs:
        recs = _unique(read_jsonl(run / "generations.jsonl"))
        prompts = [judge_prompt(r["question"], r["answers"], r["most_likely"]["text"]) for r in recs]
        replies = gen.complete_batch(prompts, max_new_tokens=8)
        rows = []
        for r, raw in zip(recs, replies):
            ok = parse_yes_no(raw)
            rows.append({"id": r["id"], "raw": raw, "label": None if ok is None else int(not ok)})
        write_jsonl(run / f"judge_{name}.jsonl", rows)
        n_none = sum(x["label"] is None for x in rows)
        rate = sum(x["label"] == 1 for x in rows) / max(1, len(rows) - n_none)
        print(f"[{name}] {run.name}: {len(rows)} judged, hallucination rate {rate:.3f}, unparsed {n_none}", flush=True)


def merge(runs: List[Path], names: List[str]) -> None:
    for run in runs:
        labels = _unique(read_jsonl(run / "labels.jsonl"))
        verdicts = {}
        for n in names:
            f = run / f"judge_{n}.jsonl"
            if f.exists():
                verdicts[n] = {r["id"]: r["label"] for r in read_jsonl(f)}
        for row in labels:
            votes = []
            for n, v in verdicts.items():
                row[f"judge_{n}"] = v.get(row["id"])
                if row[f"judge_{n}"] is not None:
                    votes.append(row[f"judge_{n}"])
            row["judges_majority"] = None if not votes else int(sum(votes) * 2 > len(votes))
            ens = row.get("ensemble")
            row["consensus"] = (None if row["judges_majority"] is None or ens is None
                                else int(row["judges_majority"] == 1 and ens == 1))
        write_jsonl(run / "labels.jsonl", labels)
        print(f"[merge] {run.name}: judges {list(verdicts)} -> labels.jsonl", flush=True)
        _agreement(labels, [f"judge_{n}" for n in verdicts] + ["judges_majority", "ensemble", "embedding"])


def _agreement(rows: List[Dict], cols: List[str]) -> None:
    import itertools

    for a, b in itertools.combinations([c for c in cols if c in rows[0]], 2):
        ok = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        if ok:
            agree = sum(r[a] == r[b] for r in ok) / len(ok)
            print(f"    agree {a:18s} {b:18s} {agree:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder that contains run folders (e.g. .../sindex_runs)")
    ap.add_argument("--pattern", default="*_n400_P10")
    ap.add_argument("--judges", default="qwen7b,qwen14b,granite8b")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args(argv)

    runs = find_runs(Path(a.root), a.pattern)
    if not runs:
        sys.exit(f"no runs matching {a.pattern} under {a.root}")
    if a.worker:
        worker(a.worker, runs, a.dtype)
        return

    names = [n.strip() for n in a.judges.split(",") if n.strip()]
    print("runs:", [r.name for r in runs], "\njudges:", names, flush=True)
    for n in names:
        print(f"\n===== judge {n}: {JUDGES[n]['model']}", flush=True)
        cmd = [sys.executable, "-m", "hallu.rejudge", a.root, "--pattern", a.pattern,
               "--worker", n, "--dtype", a.dtype]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"!! judge {n} failed (exit {res.returncode}); continuing without it", flush=True)
    merge(runs, names)
    if not a.skip_eval:
        for run in runs:
            print(f"\n===== evaluate {run.name}", flush=True)
            subprocess.run([sys.executable, "-m", "hallu.pipeline", "--config", str(run / "config.yaml"),
                            "--stage", "evaluate"])


if __name__ == "__main__":
    main()
