"""Correctness labels for the most-likely answer. label = 1 means HALLUCINATION.

Methods:
  * "embedding" - the paper's method: cos(emb(pred), emb(gold)) > 0.95 for any gold
                  answer (limitations #8, #9: strict, and shares the detector's embedder)
  * "em"        - SQuAD exact match after normalisation
  * "f1"        - SQuAD token F1 >= f1_threshold
  * "contains"  - normalised gold string contained in prediction (good for short QA)
  * "judge"     - LLM-as-judge (yes/no), see prompts.JUDGE_TEMPLATE
  * "ensemble"  - correct if em OR contains (cheap default; F1 excluded, see label_record)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def token_f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def max_f1(pred: str, golds: List[str]) -> float:
    return max((token_f1(pred, g) for g in golds), default=0.0)


def contains_gold(pred: str, golds: List[str]) -> bool:
    p = f" {normalize_answer(pred)} "
    return any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g))


def embedding_correct(pred: str, golds: List[str], embed_fn, threshold: float = 0.95) -> bool:
    vecs = embed_fn([pred] + list(golds))
    sims = vecs[1:] @ vecs[0]
    return bool(np.max(sims) > threshold)


def parse_yes_no(text: str) -> Optional[bool]:
    """Robust yes/no parse of a judge reply.

    The first version used text.startswith("yes"), which treats replies such as
    "**Yes**", "Response: Yes" or '"yes"' as NO. Here we take the first
    standalone yes/no word. Returns None if neither appears.
    """
    m = re.search(r"\b(yes|no)\b", text.lower())
    if not m:
        return None
    return m.group(1) == "yes"


def judge_correct(question: str, pred: str, golds: List[str], generator) -> Optional[bool]:
    from .prompts import judge_prompt

    out = generator.complete(judge_prompt(question, golds, pred), max_new_tokens=8)
    return parse_yes_no(out)


def label_record(rec: Dict, methods: List[str], f1_threshold: float = 0.5, embed_fn=None,
                 emb_threshold: float = 0.95, judge=None) -> Dict[str, Optional[int]]:
    pred = rec["most_likely"]["text"]
    golds = rec["answers"]
    out: Dict[str, Optional[int]] = {}
    for m in methods:
        if m == "em":
            ok = exact_match(pred, golds)
        elif m == "f1":
            ok = max_f1(pred, golds) >= f1_threshold
        elif m == "contains":
            ok = contains_gold(pred, golds)
        elif m == "ensemble":
            # F1 is deliberately NOT part of the ensemble: "Mount Everest" vs gold
            # "Mount Chimborazo" shares one token and scores F1 = 0.5 (found in the
            # first Colab run). Use the LLM judge for paraphrases that EM/contains miss.
            ok = exact_match(pred, golds) or contains_gold(pred, golds)
        elif m == "embedding":
            if embed_fn is None:
                raise ValueError("embedding label needs embed_fn")
            ok = embedding_correct(pred, golds, embed_fn, emb_threshold)
        elif m == "judge":
            if judge is None:
                out[m] = None
                continue
            ok = judge_correct(rec["question"], pred, golds, judge)
            if ok is None:
                out[m] = None
                continue
        else:
            raise ValueError(f"unknown label method {m}")
        out[m] = int(not ok)  # 1 = hallucination
    return out
