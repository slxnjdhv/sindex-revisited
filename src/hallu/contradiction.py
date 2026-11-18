"""N2: contradiction checkers used on borderline answer pairs only.

Embedding similarity is weak on numbers, dates and negation
("born in 1879" vs "born in 1897" can have cosine > 0.95). These checkers
veto merges for such pairs.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")
_NEG = re.compile(r"\b(not|no|never|isn't|wasn't|aren't|don't|doesn't|didn't|cannot|can't|none)\b", re.I)


def numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def numeric_conflict(a: str, b: str) -> bool:
    """True if both answers contain numbers and the number sets differ."""
    na, nb = numbers_in(a), numbers_in(b)
    return bool(na) and bool(nb) and na != nb


def negation_conflict(a: str, b: str) -> bool:
    """True if exactly one answer contains a negation word."""
    return bool(_NEG.search(a)) != bool(_NEG.search(b))


def rule_checker(a: str, b: str) -> bool:
    return numeric_conflict(a, b) or negation_conflict(a, b)


class NLIChecker:
    """Cross-encoder NLI contradiction check (lazy-loaded).

    Default model is small (~140M params) so it stays cheap; it only runs on
    borderline pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None,
                 threshold: float = 0.5):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name, device=device)
        self.threshold = threshold
        cfg = self.model.model.config
        id2label = {int(k): v.lower() for k, v in cfg.id2label.items()}
        self.contra_idx = next(i for i, v in id2label.items() if "contra" in v)

    def __call__(self, a: str, b: str) -> bool:
        import numpy as np

        logits = self.model.predict([(a, b), (b, a)])
        logits = np.asarray(logits)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return bool(probs[:, self.contra_idx].max() > self.threshold)


def make_checker(kind: Optional[str], device: Optional[str] = None) -> Optional[Callable[[str, str], bool]]:
    if kind in (None, "", "none"):
        return None
    if kind == "rules":
        return rule_checker
    if kind == "nli":
        return NLIChecker(device=device)
    if kind == "rules+nli":
        nli = NLIChecker(device=device)
        return lambda a, b: rule_checker(a, b) or nli(a, b)
    raise ValueError(f"unknown contradiction checker: {kind}")
