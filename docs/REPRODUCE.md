# Reproduction guide

Run commands from the repository root. Activate the environment described in the README. Paths below are examples to replace with locally prepared data or model exports. None of these commands requires an experiment-tracking account.

## Pretraining (Section 4, Tables 1–2)

```bash
DCLM_STAGE1=/path/to/shard01 DCLM_STAGE2=/path/to/shards02-10 \
  bash scripts/run_main.sh 390m both
```

Repeat with `180m`, `250m`, `510m`, `750m`, and `1b` for Table 1, and `gdn-400m`/`gla-400m` for Table 2. GDN/GLA retain the 390M KDA token budgets rather than rescaling to their parameter counts. The default output is `outputs/bld/<model>-<method>-seed0-stage<1|2>/`.

Both stages use a repeating Attention–Recurrent–Recurrent–Recurrent pattern, tied embeddings, and NoPE. Stage 1 freezes entire recurrent blocks for BLD, including mixer, MLP, and normalization; attention blocks, embeddings, and final normalization remain trainable. Stage 2 updates all parameters. Both methods restart optimizer/scheduler state at the stage boundary.

| Size | Stage 1 updates | Stage 2 updates | Stage 1 warmup | Stage 2 warmup/decay |
| --- | ---: | ---: | ---: | ---: |
| 180M | 458 | 2,289 | 23 | 229 |
| 250M | 1,068 | 5,341 | 53 | 534 |
| 390M | 1,831 | 9,155 | 92 | 916 |
| 510M | 2,747 | 13,733 | 137 | 1,373 |
| 750M | 4,064 | 20,319 | 198 | 2,032 |
| 1B | 6,104 | 30,518 | 305 | 3,052 |

Stage 1 uses 262,144 tokens per update, AdamW at 1e-4, linear warmup, and constant LR thereafter. Stage 2 uses 524,288 tokens per update, Muon at 0.02 plus auxiliary AdamW at 0.001, with warmup, stable LR, and cosine decay to 5% of the peak LR. The JSON files preserve the paper runs' actual update counts; budgets are approximate multiples of non-embedding parameters.

For an interrupted stage, call `train.py` with the same resolved arguments plus `--resume_path <ckpt_directory> --resume_step <update>`. This restores optimizer, scheduler, RNG, and rank-specific dataloader state. The shell launcher starts fresh stages and does not automatically resume interrupted runs.

## Spaced Retrieval and attribution (Section 3, Appendix A)

```bash
# Baseline, paired gaps 64/8192:
python -m experiments.retrieval.run --output outputs/retrieval/baseline
# Freeze complete KDA blocks after 175 updates:
python -m experiments.retrieval.run --output outputs/retrieval/frozen --freeze-step 175
# Resume their updates after update 500, retaining their paused Adam moments:
python -m experiments.retrieval.run --output outputs/retrieval/unfrozen \
  --freeze-step 175 --unfreeze-step 500
# Change the share of short-gap examples:
python -m experiments.retrieval.run --output outputs/retrieval/short10 --short-percent 10
# All freeze timings, mixture ratios, and three seeds:
bash scripts/run_retrieval.sh
```

The model has four blocks in KDA–Attention–KDA–Attention order, hidden width 256, FFN width 512, and a twelve-symbol vocabulary. There are 50,000 distinct training keys and 2,000 held-out attribution pairs. Original and counterfactual held-out keys are excluded from training. Training uses 32 examples per update and five answer-token losses. Exact integer quotas implement the short/long mixture; both gap losses accumulate before the optimizer update.

`core.py` retains the original data generation and signed bidirectional mixer-output intervention. It replaces only the selected layer's mixer output at the prediction position and averages the two directions. Scores remain signed and are not normalized to sum to one. `eval_<step>.json` contains accuracy and per-layer score summaries; `attribution_<step>.pt` retains individual effects. Checkpoints preserve Adam moments, scheduler, and RNG. `--resume` accepts a checkpoint from the same seed/data/mixture and can branch into another freezing schedule.

```bash
python -m experiments.plot attribution \
  --input outputs/retrieval/baseline outputs/retrieval/frozen \
  --labels Vanilla Frozen --output outputs/figures/attribution.pdf
```

## Gradient amplification (Section 3.3.2, Figure 5)

```bash
python -m experiments.retrieval.gradients --output outputs/gradients/seed0
bash scripts/run_gradients.sh
python -m experiments.plot gradients --input outputs/gradients/freeze175 \
  --output outputs/figures/gradients.pdf
```

The gradient experiment freezes only the two KDA mixers, as specified in Appendix I.1. Baseline, frozen, and unfrozen branches share initialization, batches, optimizer states at branching, and the baseline's per-update clipping factor. Backward passes include all parameters before frozen updates are masked. The diagnostic measures the norm of the mean gradient over 32 fixed training examples on the first answer token at gap 8192. The default freezes after 175 updates, branches to unfreezing after 225, and ends at 425. The script runs all three displayed timings and ten seeds. Plotting takes the geometric mean of seed-paired ratios per attention mixer, then the arithmetic mean over the two attention mixers.

## CSR, Recall, WikiText-2, and RULER (Tables 1–2, Appendix E)

```bash
bash scripts/run_eval.sh outputs/bld/390m-bld-seed0-stage2 outputs/eval/390m-bld \
  --ruler-data /path/to/ruler
# Individual benchmark groups:
python -m evaluation.run --model /path/to/model --output outputs/eval/lm --benchmark lm
python -m evaluation.run --model /path/to/model --output outputs/eval/ruler \
  --benchmark ruler --ruler-data /path/to/ruler
```

CSR averages ARC-Easy, ARC-Challenge, HellaSwag, PIQA (`acc_norm`), and BoolQ, WinoGrande, LAMBADA (`acc`). Recall averages the `contains` metric on Based SWDE, FDA, SQuAD, TriviaQA, DROP, and NQ-2048. WikiText reports `word_perplexity`. Evaluation uses zero demonstrations, greedy generation, and no chat template. The six Based task definitions are included under `evaluation/tasks/`; other tasks use lm-evaluation-harness. The paper development environment used harness 0.4.11.dev0. The clean release environment pins public harness 0.4.13, with all 14 task registrations checked. Sample records and task-level outputs are saved for auditing. `--limit` creates a development subset and must be omitted for paper scores.

RULER expects `<ruler-data>/qwen3/<length>/<task>/test.jsonl`, using the raw NVIDIA/RULER schema (`input`, `answer_prefix`, `outputs`). It evaluates 1K/2K/4K/8K/16K contexts and averages 57 task–length entries, excluding CWE and the unsupported short QA combinations. The original task matching and result schema are retained.

The generation wrapper is included; point it at a separately obtained NVIDIA/RULER checkout and its required source resources:

```bash
RULER_SOURCE=/path/to/RULER python -m evaluation.prepare_ruler \
  --tokenizer-name qwen3 --tokenizer-path Qwen/Qwen3-0.6B \
  --out-root datasets/ruler
```

## LongPPL (Appendix E)

Create a JSON model list such as `[ {"name": "390m-bld", "path": "outputs/bld/390m-bld-seed0-stage2"} ]`. Obtain the GovReport test split distributed through LongBench and a local Qwen3-8B reference model separately.

```bash
python -m evaluation.longppl prepare --models models.json --arrow /path/to/gov_report.arrow \
  --output outputs/longppl-data --documents 50 --seed 42
python -m evaluation.longppl run --models models.json --data outputs/longppl-data \
  --reference /path/to/Qwen3-8B --reference-label Qwen3-8B --output outputs/longppl
```

The scorer retains reusable reference annotations, character-span alignment, and pooled selected-token perplexity. Defaults are short-context K=4096, stride=1024, maximum context 32768, alpha=2, beta=-2. Run all compared checkpoints against the same documents and reference annotations. The deterministic preparation reconstructs the documented sampling protocol; original corpora and document snapshots are not redistributed.

## Context utilization (Figure 6, Appendix G)

Prepare local `Books3.jsonl` and `BookCorpus2.jsonl` files with a `text` field. The scorer takes the first 50 nonempty documents per corpus and scores up to 16,384 prediction positions, using full prefixes and 512-token bins.

```bash
python -m evaluation.position_loss --model /path/to/model \
  --books /path/to/Books3.jsonl /path/to/BookCorpus2.jsonl --output outputs/position/bld
python -m experiments.plot position --input outputs/position/vanilla outputs/position/bld \
  --labels Vanilla BLD --output outputs/figures/position.pdf
```

Use the same source documents for all models. Book corpora are not included in this repository.

## Retrieval-head identification and masking (Figure 7, Appendix I)

```bash
python -m experiments.heads.identify --model /path/to/510m-bld --booksum /path/to/booksum \
  --output outputs/heads/bld-alpha
python -m experiments.heads.evaluate --model /path/to/510m-bld --booksum /path/to/booksum \
  --alpha outputs/heads/bld-alpha/duo_alpha_manifest.json --output outputs/heads/bld-eval
python -m experiments.plot heads --input outputs/heads/vanilla-eval outputs/heads/bld-eval \
  --labels Vanilla BLD --output outputs/figures/heads.pdf
```

Identify each model's own 12 KV-head gates for 1,000 updates on 16K BookSum examples, with ten 32-word passkeys, sink=128, recent=256, answer-token hidden-state MSE, and 0.05 mean absolute alpha. Base weights remain frozen and KDA layers retain their original computation. Alpha begins at one and is clamped to [0,1].

Masking ranks KV heads by alpha and replaces full attention with streaming attention in increasing-alpha order. The same 64 separately generated documents are used across models and budgets. Each question is decoded independently from the shared document cache; prior answers are excluded. Figure 7 uses first-generated-token accuracy; exact passkey accuracy is also retained.

## Scaling fits (Figure 8)

```bash
python -m experiments.scaling --input results_table.csv --output outputs/scaling.json
```

Fits use unrounded non-embedding parameter counts and `loss = E + A*N^(-alpha)`. `results_table.csv` is a small transcription of the paper experiment summaries, not a fresh evaluation run.

## Checks

```bash
python -m unittest discover -s tests -v
```

CPU checks cover model parameter counts, full-block freezing, optimizer/scheduler behavior, data separation, task aggregation, and launch command construction. Training kernels and complete benchmark runs require allocated GPUs and local datasets.
