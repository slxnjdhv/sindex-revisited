"""Dataset loaders. Every loader returns a list of dicts:
    {"id": str, "question": str, "answers": [str, ...], "context": str | None}

Datasets used by the SINdex paper: TriviaQA, NQ, SQuAD, BioASQ.
Extra: `jsonl` for any custom/Indic dataset in the same format.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


def _dedupe(rows: List[Dict]) -> List[Dict]:
    """TriviaQA rc.nocontext repeats some question_ids (one row per evidence doc)."""
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def _sample(rows: List[Dict], n: Optional[int], seed: int) -> List[Dict]:
    rows = _dedupe(rows)
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_triviaqa(split: str = "validation") -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    rows = []
    for ex in ds:
        ans = [ex["answer"]["value"]] + list(ex["answer"].get("aliases", []))
        rows.append({"id": ex["question_id"], "question": ex["question"],
                     "answers": list(dict.fromkeys(a for a in ans if a)), "context": None})
    return rows


def load_nq(split: str = "validation") -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split=split)
    return [{"id": f"nq-{i}", "question": ex["question"], "answers": list(ex["answer"]), "context": None}
            for i, ex in enumerate(ds)]


def load_squad(split: str = "validation") -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split=split)
    return [{"id": ex["id"], "question": ex["question"],
             "answers": list(dict.fromkeys(ex["answers"]["text"])), "context": ex["context"]}
            for ex in ds]


def load_bioasq(path: str) -> List[Dict]:
    """BioASQ Task B training JSON (e.g. training11b.json). Requires free
    registration at bioasq.org, so it is loaded from a local file."""
    data = json.loads(Path(path).read_text())
    rows = []
    for q in data["questions"]:
        ans = []
        exact = q.get("exact_answer")
        if isinstance(exact, str):
            ans.append(exact)
        elif isinstance(exact, list):
            for e in exact:
                ans.extend(e if isinstance(e, list) else [e])
        if not ans:
            ideal = q.get("ideal_answer")
            ans.extend(ideal if isinstance(ideal, list) else [ideal] if ideal else [])
        if ans:
            rows.append({"id": q.get("id", str(len(rows))), "question": q["body"],
                         "answers": [str(a) for a in ans], "context": None})
    return rows


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                ex = json.loads(line)
                ex.setdefault("id", str(i))
                ex.setdefault("context", None)
                if isinstance(ex.get("answers"), str):
                    ex["answers"] = [ex["answers"]]
                rows.append(ex)
    return rows


def load(name: str, n: Optional[int] = None, seed: int = 0, split: str = "validation",
         path: Optional[str] = None) -> List[Dict]:
    name = name.lower()
    if name == "triviaqa":
        rows = load_triviaqa(split)
    elif name in ("nq", "nq_open"):
        rows = load_nq(split)
    elif name == "squad":
        rows = load_squad(split)
    elif name == "bioasq":
        if not path:
            raise ValueError("bioasq needs data.path pointing to the BioASQ JSON file")
        rows = load_bioasq(path)
    elif name == "jsonl":
        if not path:
            raise ValueError("jsonl needs data.path")
        rows = load_jsonl(path)
    else:
        raise ValueError(f"unknown dataset {name}")
    return _sample(rows, n, seed)
