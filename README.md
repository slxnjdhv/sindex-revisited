# SINdex reproduction + SINdex++ extensions

Research code for building on:

> S. Abdaljalil, P. Sharma, E. Serpedin, R. Atat, H. Kurban,
> "SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs,"
> *IEEE Open Journal of the Computer Society*, 2026. DOI 10.1109/OJCS.2026.3697236
> (arXiv 2503.05980). No official code was released; `src/hallu` re-implements it.

Target venue for the follow-up: IEEE OJCS (APC covered under the SIU–IEEE agreement).

## Pipeline

```
generate  ->  score  ->  label  ->  evaluate
(LLM, P samples)   (cluster + scores)   (correct / hallucinated)   (AUROC, CI, tests)
```

All intermediate results are JSONL files in `output_dir`, so every stage can be re-run alone.
Generation is resumable: re-running skips questions already in `generations.jsonl`.

| File | What it does |
|---|---|
| `src/hallu/data.py` | TriviaQA, NQ-Open, SQuAD (Hugging Face), BioASQ (local JSON), any `jsonl` (e.g. Hindi/Marathi QA) |
| `src/hallu/generate.py` | 1 low-temp answer (T=0.1) + P samples (T=1.0) with log-likelihoods. Backends: `hf` (laptop/MPS/CUDA) and `vllm` (GPU) |
| `src/hallu/cluster.py` | Paper's clustering (average-linkage HAC, cosine distance 0.05) + **N2** contradiction veto + **N4** adaptive threshold |
| `src/hallu/scores.py` | Original **SINdex**, proposed **SINdex++**, cluster entropy, Lin et al. (2023) Degree / EigV / Eccentricity, naive entropy |
| `src/hallu/sequential.py` | **N3** early stopping (fewer LLM samples), simulated offline from cached samples |
| `src/hallu/contradiction.py` | Rules (numbers, negation) and NLI cross-encoder checkers for N2 |
| `src/hallu/labels.py` | Paper's embedding label, EM, token-F1, contains, ensemble, LLM-judge (**N5**) |
| `src/hallu/metrics.py` | AUROC + 2,000-sample bootstrap CI, paired bootstrap test, AURAC, accuracy@coverage |
| `src/hallu/pipeline.py` | CLI tying everything together |

## Setup

### Laptop (Mac M1 / CPU) - for development and small tests
```bash
cd sindex_research
python3 -m venv .venv
.venv/bin/pip install -e ".[models,dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m hallu.pipeline --config configs/smoke.yaml --stage all
```
`configs/smoke.yaml` uses Qwen2.5-0.5B on 12 NQ questions (~5 min). Its numbers are meaningless; it only proves the pipeline runs.

### Google Colab (easiest way to get a GPU)
1. Upload `sindex_research.zip` (one folder up from this README) to the top level of *My Drive*.
2. Open `colab_sindex.ipynb` in Colab (File → Upload notebook) and set the runtime to a GPU (A100/L4 best; T4 → preset `t4_qwen_awq`).
3. Run the cells top to bottom. Start with `PRESET = 'quick_test'`, then `repro_llama2` to reproduce the paper.
Results go to `My Drive/sindex_runs/<run name>/`, and generation resumes after a disconnect.

### GPU server (>=24 GB) - for real experiments
```bash
pip install -e ".[models,gpu,dev]"
huggingface-cli login            # Llama models need an approved HF access request
python -m hallu.pipeline --config configs/repro_llama3_triviaqa.yaml --stage generate
python -m hallu.pipeline --config configs/repro_llama3_triviaqa.yaml --stage score
python -m hallu.pipeline --config configs/repro_llama3_triviaqa.yaml --stage label
python -m hallu.pipeline --config configs/repro_llama3_triviaqa.yaml --stage evaluate
```
Run the stages as **separate commands** on GPU: vLLM keeps GPU memory until the process exits,
and the `label` stage may load an LLM judge.

Copy the config per (model, dataset). Suggested grid:
models `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-9b-it`,
`mistralai/Mistral-7B-Instruct-v0.3` x datasets `triviaqa`, `nq`, `squad`, `bioasq`.

## Research contributions mapped to code

| ID | Idea | Where | Config knob |
|---|---|---|---|
| N1 | **SINdex++** = H(p) + λ·Σ pᵢ(1 − cᵢ): provably monotone, ≥ H(p) | `scores.sindex_pp` | `scoring.lam` |
| N2 | Contradiction-aware clustering on borderline pairs only | `cluster.apply_contradictions`, `contradiction.py` | `variants[].contradiction: rules / nli / rules+nli` |
| N3 | Early stopping: fewer LLM calls | `sequential.early_stop_score` | `scoring.early_stop` |
| N4 | Per-question adaptive threshold (largest dendrogram gap) | `cluster.adaptive_threshold` | `variants[].adaptive: true` |
| N5 | Rigorous evaluation: gold-based labels, LLM judge, CIs, paired tests, AURAC, more baselines | `labels.py`, `metrics.py` | `labels.methods`, `evaluation` |
| Abl. | Embed answer only instead of `q [SEP] a` (limitation #11) | `embed.answer_strings` | `scoring.string_mode: answer_only` |
| Abl. | Singleton-cluster handling (paper leaves it undefined) | `scores.cluster_cohesion` | `scoring.singleton` |

## Findings so far

1. **The paper's theoretical claim (eq. 10-11) is false in general.** p' = p·c does not sum to 1, so the
   majorization/Schur-concavity argument does not apply. With 5 clusters of share 0.2, SINdex goes
   1.609 → 1.466 as cohesion drops 1.0 → 0.8 (it should rise). Test: `tests/test_scores.py::test_original_sindex_flaw_documented`.
   SINdex++ is monotone in every one of 2,000 random tests (`test_sindex_pp_monotone_random`).
   Practical impact is probably small at the 0.95 threshold, because cohesion inside clusters stays high. **Measure it in the experiments.**
2. **The paper's embedding labels mislabel correct answers.** Smoke run example: "Star Wars: The Clone Wars has 6 seasons."
   vs gold "6" is labelled a hallucination by cosine > 0.95, but correct by EM/contains/F1.
   In the smoke run, embedding and ensemble labels agreed on only 83% of questions.
3. At the 0.95 threshold, sentence-length answers are almost all singleton clusters, so SINdex ≈ log(P).
   Check the `n_singletons` column on real models. If this holds, it supports N4 (adaptive threshold).

## First real run: Qwen2.5-7B-Instruct, TriviaQA, 388 unique questions, P=10 (Colab L4, 2026-11-18)
Files: `runs/colab_qwen7b_triviaqa_n400/` (also in Drive `sindex_runs/`). One model and one dataset, so treat this as preliminary.

| Score | AUROC, LLM-judge labels | AUROC, paper's embedding labels | Δ vs SINdex (judge), p |
|---|---|---|---|
| SINdex (paper) | 0.798 | 0.807 | — |
| SINdex++ (N1) | 0.798 | 0.808 | +0.000, p=0.35 |
| **NLI contradiction check (N2)** | **0.817** | **0.824** | **+0.019, p=0.001** |
| Adaptive + NLI | 0.815 | 0.821 | +0.017, p=0.03 |
| Adaptive threshold (N4) | 0.780 | 0.801 | −0.017, p=0.97 |
| Early stop (N3), 5.7/10 samples | 0.766 | 0.765 | −0.032, p=0.99 |
| Cluster entropy (no cohesion) | 0.785 | 0.762 | −0.012, p=0.96 |
| Lin et al. EigV (not compared in paper) | 0.795 | 0.807 | −0.002, p=0.65 |
| Naive entropy (gray-box) | 0.779 | 0.737 | — |

- **N2 works.** The NLI veto improves AUROC significantly under all 5 label types (+0.013 to +0.019).
- **N1 is a theory fix with no measurable AUROC effect.** It changes the score on 260/400 questions, and the paper's monotonicity flaw shows up on 62 of them, but the AUROC change is +0.001.
- **The paper's labels are wrong about correctness.** They mark 73% of answers as hallucinations; the LLM judge marks 47%. They disagree on 28% of questions, and 108 answers the judge accepts are marked as hallucinations (e.g. "Turin (Torino)" vs gold "Turin"). The AUROC ranking changes less than expected, **but SINdex's advantage over plain cluster entropy drops from +0.045 (significant) to +0.012 (not significant) under judge labels.**
- **SINdex ≈ Lin et al. (2023) EigV**, a baseline the paper did not compare against.
- **N3 saves 43% of LLM calls but costs 0.03 AUROC** at the current settings. It needs tuning on a validation split.
- Caveats: the judge is Qwen2.5-7B, the same model that generated the answers (possible self-bias). There is one model and one dataset so far. TriviaQA has duplicate question IDs (12 here); the loader now removes them.

## Three datasets: Qwen2.5-7B-Instruct, 400 questions each, P=10 (2026-11-18)
Runs: `runs/colab_qwen7b_{triviaqa,nq,squad}_n400/`; table CSV: `runs/table_qwen7b_judge.csv`.
Re-create any table with `python -m hallu.summarize runs/colab_qwen7b_*_n400 --labels judge`.

AUROC with **LLM-judge labels**:

| Score | TriviaQA | NQ | SQuAD |
|---|---|---|---|
| SINdex (paper) | 0.798 | 0.790 | 0.551 |
| SINdex++ (N1) | 0.798 | 0.790 | 0.551 |
| **+NLI veto (N2)** | **0.817** (p=.001) | **0.811** (p=.013) | 0.551 |
| adaptive thr. (N4) | 0.780 | 0.768 | 0.550 |
| early stop (N3) | 0.766 | 0.795 | 0.548 |
| cluster entropy | 0.785 | 0.786 | 0.568 |
| Lin EigV | 0.795 | 0.789 | 0.553 |

Hallucination rate under the paper's embedding labels vs the judge: TriviaQA 73% vs 47%, **NQ 98% vs 71%, SQuAD 69% vs 9%**.
SINdex's lead over plain cluster entropy, paper's labels vs judge labels: TriviaQA +0.045 → +0.012, NQ +0.047 → +0.004, **SQuAD +0.087 → −0.017**.

With **consensus labels** (hallucination only if the judge AND EM/contains both say wrong):
the N2 gain holds on TriviaQA (+0.013, p=.022) and NQ (+0.018, p=.035).
SQuAD then has only 7/400 hallucinations (1.8%), so AUROC there is meaningless.

What this means:
1. **N2 (NLI veto) is the main positive result.** It gives +0.013 to +0.022 AUROC, significant on both closed-book datasets and under every label type.
2. **The paper's labels drive much of its reported advantage.** On SQuAD their labels call 69% of answers hallucinations, and a manual spot-check found all 10 sampled "hallucinations" were correct (e.g. "Möngke Khan died in 1259." vs gold "1259"). Under correct labels SINdex does no better than plain cluster entropy, and no better than Lin et al.'s EigV.
3. **N1 is correct in theory but changes nothing measurable.** N4 and N3 need tuning on a validation split.
4. **The LLM judge also makes errors.** It marked "Gold" (gold "Gold") and "Westwood One" as wrong. Use a stronger or different judge model and hand-check ~200 labels before writing the paper.

## Better labels: three LLM judges + majority vote (2026-11-18)
`python -m hallu.rejudge <runs root>` re-labels existing runs without regenerating answers. It uses three judges, each loaded in its own process:
Qwen2.5-7B (same family as the generator), Qwen2.5-14B-AWQ (larger) and IBM Granite-3.1-8B (different family).
It also uses a v2 judge prompt that accepts correct answers given as full sentences.
Each run's `labels.jsonl` gains the columns `judge_qwen7b`, `judge_qwen14b`, `judge_granite8b`, `judges_majority` and `consensus` (majority AND EM/contains).

- The v2 prompt fixed the v1 errors: "Gold" vs gold "Gold", "Westwood One" and "Denmark … 1972" are now judged correct.
- The judges agree with each other 93–96% of the time (all three agree on 91–94% of questions). They agree with the paper's embedding labels on only 66–70% (TriviaQA/NQ) and 35% (SQuAD).
- Results with **majority-vote labels** (`runs/table_qwen7b_judges_majority.csv`):

| Score | TriviaQA | NQ | SQuAD (4% halluc.) |
|---|---|---|---|
| SINdex (paper) | 0.808 | 0.779 | 0.649 |
| SINdex++ (N1) | 0.808 | 0.779 | 0.649 |
| **+NLI veto (N2)** | **0.826** (+0.018, p=.002) | **0.797** (+0.018, p=.037) | 0.640 |
| cluster entropy | 0.798 | 0.779 | 0.655 |
| Lin EigV | 0.805 | 0.778 | 0.652 |

SINdex's lead over plain cluster entropy, paper's labels vs majority labels: +0.045 → +0.010 (TriviaQA), +0.047 → 0.000 (NQ), +0.087 → −0.006 (SQuAD).

## Human check (needed before writing)
`annotation/annotate.html` holds 201 items: 50 random questions per dataset, plus 17 per dataset where the judges and EM/contains disagree. It is blind: no model label is shown.
1. Open it in a browser by double-clicking it in Finder. Keys: **C** = correct, **W** = wrong, **U** = unsure, arrows to move. Progress autosaves in the browser.
2. Click **Export CSV** and save the file as `annotation/annotations.csv`.
3. Run `python -m hallu.human_eval score annotation/annotations.csv runs/colab_qwen7b_*_n400`. It prints accuracy, Cohen's κ, precision and recall for every label method.
   Report the **random** subset as the unbiased accuracy of each label method. Ideally a second person annotates too, so you can report agreement between annotators.

## Label accuracy against careful annotation (2026-11-18)
**Not a human annotation.** All 201 items were annotated by Claude (an AI assistant), blind to every model label, using the same rule as the page.
The result is `annotation/annotations_claude.csv`: 131 correct, 56 wrong, 14 unsure. In a paper, describe this as an LLM-assisted reference annotation.
It must be confirmed by at least one human (see below) before any number is reported as "human".

Random subset (n=146 decided; this is the unbiased estimate):

| Label method | Accuracy | Cohen's κ |
|---|---|---|
| Paper's embedding labels (cos > 0.95) | **0.568** | **0.260** |
| Exact match | 0.589 | 0.288 |
| Token F1 ≥ 0.5 | 0.719 | 0.472 |
| EM / contains ensemble | 0.945 | 0.884 |
| Judge Qwen2.5-7B (v2 prompt) | 0.966 | 0.926 |
| Judge Qwen2.5-14B | 0.925 | 0.842 |
| Judge Granite-3.1-8B | 0.986 | 0.970 |
| **Majority of 3 judges** | **0.973** | **0.940** |
| Consensus (majority AND ensemble) | 0.979 | 0.955 |

The paper's labelling is close to chance (κ = 0.26). Of the 94 answers in the random subset that are actually correct, it marks 63 (67%) as hallucinations.
On the 41 "disagreement" items (deliberately hard), the majority vote reaches 0.68 accuracy and the judges tend to over-call hallucinations; the true rate there is 10%.

**Human confirmation, done 2026-11-18.** One human annotator labelled 40 random items blind (`annotation_check/annotations.csv`).
They agreed with Claude's annotation on **40/40 items (Cohen's κ = 1.00; 95% Clopper–Pearson lower bound on agreement 0.91)**.
On those 40 human labels: paper's embedding labels 0.606 accuracy (κ 0.32), majority of 3 judges 1.00, EM/contains 0.94.
For the paper: "A 201-item reference set was annotated with LLM assistance; a human annotator independently re-labelled a random 40-item subset with perfect agreement (κ = 1.00)."
A second human annotator on the same 40 items would further strengthen this.

How it was run: `annotation_check/annotate.html` has 40 random items that Claude already decided. Annotate them, export, then run:
`python -m hallu.human_eval agree annotation_check/annotations.csv annotation/annotations_claude.csv`
Report this agreement (κ) alongside the table above.

## Second generator: Llama-3.1-8B-Instruct (2026-11-18)
Weights: `unsloth/Meta-Llama-3.1-8B-Instruct`, an ungated copy of Meta's Llama-3.1-8B-Instruct released under the Llama 3.1 Community License (cite as Meta's model).
Run with `python -m hallu.batch unsloth/Meta-Llama-3.1-8B-Instruct <runs root>`. Results are in `runs/colab_llama8b_*_n400/` and `runs/table_llama8b_judges_majority.csv`.

AUROC with majority-of-3-judges labels:

| Score | TriviaQA | NQ | SQuAD (5% halluc.) |
|---|---|---|---|
| SINdex (paper) | 0.858 | 0.705 | 0.799 |
| SINdex++ (N1) | 0.858 | 0.705 | 0.799 |
| +NLI veto (N2) | 0.868 (p=.023) | 0.716 (p=.087) | 0.836 (p=.063) |
| cluster entropy | 0.851 | 0.706 | 0.770 |
| Lin EigV | 0.847 | 0.691 | 0.789 |

Hallucination rate, paper's labels vs majority labels: TriviaQA 58% vs 28%, NQ 94% vs 56%, SQuAD 51% vs 5%.
Llama gives shorter answers than Qwen, so the paper's labelling is less wrong here, but it still roughly doubles the hallucination rate.

### Pooled over 2 models × 3 datasets (6 runs; stratified paired bootstrap, 4,000 resamples; majority labels)

| Comparison | Mean ΔAUROC [95% CI] | p (one-sided) |
|---|---|---|
| **N2 NLI veto − SINdex** | **+0.014 [+0.005, +0.025]** | **0.0005** |
| N1 SINdex++ − SINdex | +0.000 [−0.001, +0.001] | 0.47 |
| SINdex − cluster entropy, **paper's labels** | **+0.040 [+0.030, +0.050]** | <0.001 |
| SINdex − cluster entropy, **majority labels** | +0.007 [−0.009, +0.023] | 0.21 |
| SINdex − Lin EigV (2023 baseline) | +0.006 [−0.003, +0.015] | 0.08 |
| N3 early stop − SINdex | −0.020 [−0.047, +0.003] | (worse) |
| N4 adaptive threshold − SINdex | −0.020 [−0.031, −0.007] | (worse) |

With strict consensus labels: N2 +0.017 (p=.002); SINdex − cluster entropy +0.009 (p=.25).

**Paper-level conclusions**
1. The **NLI veto on borderline pairs (N2)** is a robust improvement: positive in 5 of 6 runs, pooled +0.014 AUROC, p < 0.001.
2. **SINdex's reported advantage comes mostly from its labels.** Over plain cluster entropy, its lead is +0.040 with the paper's embedding labels and +0.007 (not significant) with validated labels. It is statistically tied with Lin et al.'s EigV, which the paper does not compare against.
3. The **paper's labelling is unreliable**: 57% accuracy, κ = 0.26, against a 201-item reference set whose 40-item subset a human confirmed with κ = 1.00.
4. **SINdex++ (N1)** fixes the paper's incorrect monotonicity proof but has no measurable effect on AUROC.
5. **Negative results:** the adaptive threshold (N4) and early stopping (N3) both reduce AUROC at the tested settings. Report them honestly; tuning them on a validation split is future work.

## Next steps (in order)
1. Reproduce: Llama-2-7b-chat on TriviaQA with `labels.methods: [embedding]`. Match the paper's Table 1 within ±2 AUROC points.
2. Switch to modern models and gold-based labels. Report how much the labels change the conclusions.
3. Tune `lam`, the thresholds and the early-stop parameters on a **validation split**. Report results on a separate test split.
4. Add the NLI semantic-entropy baseline (Farquhar et al. 2024 code: `github.com/jlko/semantic_uncertainty`).
5. Have a person check about 200 labels to measure how accurate the LLM judge is.
6. Optional: Hindi/Marathi QA via `data.name: jsonl`, with a multilingual embedder (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `intfloat/multilingual-e5-base`).
