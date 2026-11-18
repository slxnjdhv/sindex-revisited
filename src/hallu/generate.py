"""LLM answer generation with on-disk caching.

Backends:
  * "hf"   - Hugging Face transformers (CPU / Apple MPS / CUDA). Good for small tests.
  * "vllm" - fast batched generation on an NVIDIA GPU. Use for real experiments.

Per question we store one low-temperature "most likely" answer (temp 0.1, as in
the paper) and P high-temperature samples (temp 1.0), with sequence
log-likelihoods for gray-box baselines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .prompts import qa_prompt


def clean_answer(text: str) -> str:
    """Keep the first line; drop trailing junk."""
    text = text.strip()
    for stop in ["\n", "Question:", "Context:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def pick_device(pref: Optional[str] = None) -> str:
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFGenerator:
    def __init__(self, model_name: str, device: str = "auto", use_chat_template: bool = True,
                 dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch_dtype = {"auto": (torch.float16 if self.device != "cpu" else torch.float32),
                       "float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.use_chat = use_chat_template and getattr(self.tok, "chat_template", None)

    def _encode(self, prompt: str):
        if self.use_chat:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        return self.tok(text, return_tensors="pt").to(self.device)

    def sample(self, prompt: str, n: int, temperature: float, max_new_tokens: int, seed: int = 0) -> List[Dict]:
        torch = self.torch
        torch.manual_seed(seed)
        enc = self._encode(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                num_return_sequences=n, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tok.pad_token_id,
            )
        trans = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        res = []
        for i in range(n):
            ids = gen[i]
            mask = ids != self.tok.pad_token_id
            if self.tok.eos_token_id is not None:
                eos_pos = (ids == self.tok.eos_token_id).nonzero()
                if len(eos_pos):
                    mask[eos_pos[0, 0] + 1:] = False
            ll = float(trans[i][mask].float().sum().item())
            text = self.tok.decode(ids[mask], skip_special_tokens=True)
            res.append({"text": clean_answer(text), "raw": text, "loglik": ll, "length": int(mask.sum().item())})
        return res

    def complete(self, prompt: str, max_new_tokens: int = 8) -> str:
        """Greedy completion (used for the LLM judge)."""
        torch = self.torch
        enc = self._encode(prompt)
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def complete_batch(self, prompts: List[str], max_new_tokens: int = 8) -> List[str]:
        return [self.complete(p, max_new_tokens) for p in prompts]


class VLLMGenerator:
    def __init__(self, model_name: str, use_chat_template: bool = True, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096, quantization: Optional[str] = None, dtype: str = "auto"):
        from vllm import LLM

        # dtype "half" is needed on GPUs without bfloat16 (e.g. Colab T4)
        self.llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, quantization=quantization, dtype=dtype)
        self.tok = self.llm.get_tokenizer()
        self.use_chat = use_chat_template and getattr(self.tok, "chat_template", None)

    def _fmt(self, prompt: str) -> str:
        if self.use_chat:
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=False, add_generation_prompt=True)
        return prompt

    def sample_batch(self, prompts: List[str], n: int, temperature: float, max_new_tokens: int,
                     seed: int = 0) -> List[List[Dict]]:
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
                            logprobs=0, seed=seed)
        outs = self.llm.generate([self._fmt(p) for p in prompts], sp)
        res = []
        for o in outs:
            res.append([{"text": clean_answer(c.text), "raw": c.text,
                         "loglik": float(c.cumulative_logprob or 0.0), "length": len(c.token_ids)}
                        for c in o.outputs])
        return res

    def sample(self, prompt, n, temperature, max_new_tokens, seed=0):
        return self.sample_batch([prompt], n, temperature, max_new_tokens, seed)[0]

    def complete(self, prompt: str, max_new_tokens: int = 8) -> str:
        return self.complete_batch([prompt], max_new_tokens)[0]

    def complete_batch(self, prompts: List[str], max_new_tokens: int = 8) -> List[str]:
        from vllm import SamplingParams

        outs = self.llm.generate([self._fmt(p) for p in prompts],
                                 SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        return [o.outputs[0].text for o in outs]


def make_generator(cfg: Dict):
    g = cfg["generation"]
    if g["backend"] == "hf":
        return HFGenerator(g["model"], device=g.get("device", "auto"),
                           use_chat_template=g.get("chat_template", True), dtype=g.get("dtype", "auto"))
    if g["backend"] == "vllm":
        return VLLMGenerator(g["model"], use_chat_template=g.get("chat_template", True),
                             gpu_memory_utilization=g.get("gpu_memory_utilization", 0.85),
                             max_model_len=g.get("max_model_len", 4096),
                             quantization=g.get("quantization"), dtype=g.get("dtype", "auto"))
    raise ValueError(f"unknown backend {g['backend']}")


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def generate_dataset(rows: List[Dict], cfg: Dict, out_path: Path, log=print) -> None:
    """Generate (resumable) and append one JSON line per question to out_path."""
    g = cfg["generation"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    log(f"[generate] {len(done)} cached, {len(todo)} to do -> {out_path}")
    if not todo:
        return
    gen = make_generator(cfg)
    P = int(g["num_samples"])
    bs = int(g.get("batch_size", 16)) if g["backend"] == "vllm" else 1
    seed = int(cfg.get("seed", 0))
    with open(out_path, "a") as f:
        for start in range(0, len(todo), bs):
            chunk = todo[start:start + bs]
            prompts = [qa_prompt(r["question"], r.get("context") if g.get("use_context", True) else None)
                       for r in chunk]
            if g["backend"] == "vllm":
                ml = gen.sample_batch(prompts, 1, g["low_temperature"], g["max_new_tokens"], seed)
                sm = gen.sample_batch(prompts, P, g["temperature"], g["max_new_tokens"], seed + 1)
            else:
                ml = [gen.sample(prompts[0], 1, g["low_temperature"], g["max_new_tokens"], seed)]
                sm = [gen.sample(prompts[0], P, g["temperature"], g["max_new_tokens"], seed + 1)]
            for r, m, s in zip(chunk, ml, sm):
                rec = {"id": r["id"], "question": r["question"], "answers": r["answers"],
                       "most_likely": m[0], "samples": s, "model": g["model"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            log(f"[generate] {min(start + bs, len(todo))}/{len(todo)}")
