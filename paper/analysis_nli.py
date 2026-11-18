"""Cost and examples of the NLI veto, recomputed locally from cached generations.

    cd sindex_research && .venv/bin/python paper/analysis_nli.py

Outputs paper/tables/nli_cost.json and paper/tables/nli_examples.json.
Uses the same settings as the experiments: MiniLM on "q [SEP] a", band [0.85, 1],
cross-encoder/nli-deberta-v3-small, threshold 0.5, HAC average linkage at 0.05.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hallu.cluster import agglomerative_labels, apply_contradictions, cosine_similarity_matrix  # noqa: E402
from hallu.contradiction import NLIChecker  # noqa: E402
from hallu.embed import answer_strings, embed  # noqa: E402

OUT = ROOT / "paper" / "tables"
RUNS = {d: ROOT / "runs" / f"colab_qwen7b_{d}_n400" for d in ("triviaqa", "nq", "squad")}


def main():
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    nli = NLIChecker(device=dev)
    calls = {"n": 0}

    def counting(a, b):
        calls["n"] += 1
        return nli(a, b)

    stats, examples = {}, []
    for d, run in RUNS.items():
        gens = {}
        for line in open(run / "generations.jsonl"):
            r = json.loads(line)
            gens.setdefault(r["id"], r)
        labels = {}
        for line in open(run / "labels.jsonl"):
            r = json.loads(line)
            labels.setdefault(r["id"], r)
        tot_pairs = band_pairs = vetoed = changed = 0
        t_emb = t_nli = 0.0
        for qid, g in gens.items():
            texts = [s["text"] for s in g["samples"]]
            t0 = time.time()
            E = embed(answer_strings(g["question"], texts), device=dev)
            sim = cosine_similarity_matrix(E)
            t_emb += time.time() - t0
            n = len(texts)
            iu = np.triu_indices(n, 1)
            tot_pairs += len(iu[0])
            band_pairs += int(sum(1 for i, j in zip(*iu) if 0.85 <= sim[i, j] <= 1.0 and texts[i].strip() != texts[j].strip()))
            before = calls["n"]
            t0 = time.time()
            sim2 = apply_contradictions(sim, texts, counting, (0.85, 1.0))
            t_nli += time.time() - t0
            v = int(((sim2 == -1) & (sim != -1))[iu].sum())
            vetoed += v
            assert calls["n"] - before >= 0
            lab0 = agglomerative_labels(sim, 0.05)
            lab1 = agglomerative_labels(sim2, 0.05)
            k0, k1 = len(set(lab0)), len(set(lab1))
            if k1 != k0:
                changed += 1
                lab = labels.get(qid, {})
                examples.append({"dataset": d, "id": qid, "question": g["question"], "gold": g["answers"][:3],
                                 "most_likely": g["most_likely"]["text"], "samples": texts,
                                 "clusters_before": k0, "clusters_after": k1, "vetoed_pairs": v,
                                 "halluc_majority": lab.get("judges_majority")})
        nq = len(gens)
        stats[d] = {"questions": nq, "pairs": tot_pairs, "band_pairs": band_pairs,
                    "band_frac": round(band_pairs / tot_pairs, 4), "vetoed_pairs": vetoed,
                    "questions_changed": changed, "questions_changed_frac": round(changed / nq, 4),
                    "nli_seconds_per_question": round(t_nli / nq, 4),
                    "embed_seconds_per_question": round(t_emb / nq, 4), "device": dev}
        print(d, stats[d], flush=True)
    stats["nli_calls_total"] = calls["n"]
    (OUT / "nli_cost.json").write_text(json.dumps(stats, indent=1))
    (OUT / "nli_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    print("examples:", len(examples))


if __name__ == "__main__":
    main()
