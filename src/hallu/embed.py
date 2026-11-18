"""Sentence embeddings (sentence-transformers), lazy-loaded and cached per model."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_CACHE: Dict[str, object] = {}


def get_model(name: str, device: Optional[str] = None):
    key = f"{name}@{device}"
    if key not in _CACHE:
        from sentence_transformers import SentenceTransformer

        _CACHE[key] = SentenceTransformer(name, device=device)
    return _CACHE[key]


def embed(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
          device: Optional[str] = None, batch_size: int = 64) -> np.ndarray:
    model = get_model(model_name, device)
    return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=False))


def answer_strings(question: str, answers: List[str], mode: str = "question_sep") -> List[str]:
    """Paper eq. 2: s_i = q [SEP] g_i. mode='answer_only' is the ablation (limitation #11)."""
    if mode == "question_sep":
        return [f"{question} [SEP] {a}" for a in answers]
    if mode == "answer_only":
        return list(answers)
    raise ValueError(mode)
