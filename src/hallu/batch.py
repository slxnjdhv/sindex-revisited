"""Run a full experiment for one generator model on several datasets, then re-judge.

Usage (Colab, one line):
    !python -m hallu.batch unsloth/Meta-Llama-3.1-8B-Instruct /content/drive/MyDrive/sindex_runs

For each dataset it writes <root>/<model>_<dataset>_n<N>_P<P>/config.yaml (same
settings as the notebook's section 7), runs generate -> score -> label ->
evaluate as separate processes (vLLM frees GPU memory only on exit), then runs
hallu.rejudge on just these runs (3 LLM judges + majority vote) and re-evaluates.
Resumable: generation skips questions already in generations.jsonl.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def make_config(model: str, dataset: str, out: Path, n: int, P: int, dtype: str, quantization) -> dict:
    return {
        "seed": 0,
        "output_dir": str(out),
        "data": {"name": dataset, "n": n},
        "generation": {
            "backend": "vllm", "model": model, "quantization": quantization, "dtype": dtype,
            "num_samples": P, "temperature": 1.0, "low_temperature": 0.1, "max_new_tokens": 64,
            "batch_size": 32, "gpu_memory_utilization": 0.85, "max_model_len": 2048, "use_context": True,
        },
        "scoring": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cuda",
            "distance_threshold": 0.05, "string_mode": "question_sep", "singleton": "one", "lam": 1.0,
            "variants": [
                {"name": "adaptive", "adaptive": True},
                {"name": "rules", "contradiction": "rules"},
                {"name": "nli", "contradiction": "nli"},
                {"name": "adaptive_nli", "adaptive": True, "contradiction": "nli"},
            ],
            "early_stop": {"min_samples": 3, "patience": 2, "tol": 0.05},
        },
        "labels": {
            # LLM judges are added afterwards by hallu.rejudge
            "methods": ["ensemble", "embedding", "em", "f1"], "f1_threshold": 0.5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_threshold": 0.95,
            "device": "cuda",
        },
        "evaluation": {
            "n_boot": 2000, "compare_against": "sindex",
            "compare": ["sindex_pp", "cluster_entropy", "adaptive:sindex_pp", "nli:sindex_pp",
                        "adaptive_nli:sindex_pp", "early_stop:sindex_pp", "lin_eigv"],
        },
    }


def run(cmd) -> int:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("root")
    ap.add_argument("--datasets", default="triviaqa,nq,squad")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--judges", default="qwen7b,qwen14b,granite8b")
    ap.add_argument("--skip-rejudge", action="store_true")
    a = ap.parse_args(argv)

    tag = a.model.split("/")[-1]
    root = Path(a.root)
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        out = root / f"{tag}_{ds}_n{a.n}_P{a.P}"
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(make_config(a.model, ds, out, a.n, a.P, a.dtype, a.quantization),
                                           sort_keys=False))
        print(f"\n########## {tag} / {ds} -> {out}", flush=True)
        for stage in ("generate", "score", "label", "evaluate"):
            rc = run([sys.executable, "-m", "hallu.pipeline", "--config", str(cfg_path), "--stage", stage])
            if rc != 0:
                print(f"!! {ds}: stage {stage} failed (exit {rc}); skipping rest of this dataset", flush=True)
                break
    if not a.skip_rejudge:
        print(f"\n########## re-judging {tag} runs", flush=True)
        run([sys.executable, "-m", "hallu.rejudge", str(root), "--pattern", f"{tag}_*_n{a.n}_P{a.P}",
             "--judges", a.judges])


if __name__ == "__main__":
    main()
