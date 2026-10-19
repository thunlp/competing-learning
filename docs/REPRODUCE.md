# Reproduction guide

Run commands from the repository root. Activate the environment described in the README. Paths below are examples to replace with locally prepared data or model exports. None of these commands requires an experiment-tracking account.

## Pretraining (Section 4, Tables 1–2)

```bash
DCLM_STAGE1=/path/to/shard01 DCLM_STAGE2=/path/to/shards02-10 \
  bash scripts/run_main.sh 390m both
```

Repeat with `180m`, `250m`, `510m`, `750m`, and `1b` for Table 1, and `gdn-400m`/`gla-400m` for Table 2. GDN/GLA retain the 390M KDA token budgets rather than rescaling to their parameter counts. The default output is `outputs/bld/<model>-<method>-seed0-stage<1|2>/`.

All models use a repeating Attention–Recurrent–Recurrent–Recurrent pattern, tied embeddings, and NoPE. During early pretraining, BLD freezes entire recurrent blocks, including mixer, MLP, and normalization; attention blocks, embeddings, and final normalization remain trainable. All parameters are then updated throughout the remainder of pretraining. Both methods reset optimizer and schedule state when early pretraining ends, continuing from the current weights. The launcher names `stage1` and `stage2` refer to early and remaining pretraining, respectively.

| Size | Early updates | Remaining updates | Early warmup | Remaining warmup/decay |
| --- | ---: | ---: | ---: | ---: |
| 180M | 458 | 2,289 | 23 | 229 |
| 250M | 1,068 | 5,341 | 53 | 534 |
| 390M | 1,831 | 9,155 | 92 | 916 |
| 510M | 2,747 | 13,733 | 137 | 1,373 |
| 750M | 4,064 | 20,319 | 198 | 2,032 |
| 1B | 6,104 | 30,518 | 305 | 3,052 |

Early pretraining uses 262,144 tokens per update, AdamW at 1e-4, linear warmup, and constant LR thereafter. Remaining pretraining uses 524,288 tokens per update, Muon at 0.02 plus auxiliary AdamW at 0.001, with warmup, stable LR, and cosine decay to 5% of the peak LR. The JSON files preserve the paper runs' actual update counts; budgets are approximate multiples of non-embedding parameters.

For an interrupted run, call `train.py` with the same resolved arguments plus `--resume_path <ckpt_directory> --resume_step <update>`. This restores optimizer, scheduler, RNG, and rank-specific dataloader state. The shell launcher starts each pretraining portion with fresh optimizer and schedule state and does not automatically resume interrupted runs.

<a id="section-3"></a>
<a id="analysis-experiments"></a>

## Analysis Experiments

Use one allocated GPU and the installed training environment. All synthetic data are generated locally; these experiments train from scratch without an external tokenizer, pretrained weights, or DCLM data. Each training command requires a new output directory.

### Spaced Retrieval

[Data generation](../experiments/retrieval/core.py) constructs five-digit keys with a delimiter, a gap of 64 or 8192 spaces, and a retrieval query. The model has four blocks in KDA–Attention–KDA–Attention order, hidden width 256, FFN width 512, and a twelve-symbol vocabulary; see [the configuration](../configs/model/retrieval.json). There are 50,000 distinct training keys and 2,000 held-out attribution pairs per gap. Original and counterfactual held-out keys are excluded from training.

[The training entry point](../experiments/retrieval/run.py) defaults to 1,500 AdamW updates, 32 examples per update, and a 50:50 short/long-gap mixture. The learning rate warms up for 100 updates to 3e-4 and decays to 1e-5. Cross-entropy is computed only on the five answer tokens. Exact integer quotas implement each requested mixture, and the short/long losses accumulate before one optimizer update.

### Measuring layer retrieval ability

```bash
python -m experiments.retrieval.run --output outputs/retrieval/baseline
python -m experiments.plot attribution --input outputs/retrieval/baseline \
  --labels Vanilla --gap 8192 --output outputs/figures/baseline.pdf
```

`core.causal_contributions` implements the intervention in Figure 1 and Equations (6)–(7). For a paired input with correct digit `a` and counterfactual digit `b`, the answer margin is `m(z; a, b) = z[a] - z[b]`. Replace only the selected layer's mixer output at the prediction position with that from its counterfactual input, before residual addition. The layer's score is the drop in this margin, averaged over both replacement directions and all held-out pairs. The residual input, MLP, downstream layers, and model weights retain their normal computation.

Each pair probes one answer position, balanced across all five positions. Scores are signed logit-margin changes, retain negative values, and are not normalized shares. In Figures 2/4, a KDA curve averages layers 0 and 2; an attention curve averages layers 1 and 3. The plot command shows these family means for each input run. Use `--gap 64` for the corresponding short-gap analysis.

`core.native_accuracy` separately performs greedy autoregressive generation of all five answer digits. Its `first_accuracy`, `character_accuracy`, and `exact_accuracy` fields report first-digit, per-digit, and whole-key accuracy as fractions in [0, 1]. These accuracy fields and the signed retrieval ability score measure different quantities.

### Freezing and unfreezing recurrent blocks

```bash
# Freeze entire KDA blocks after 175 completed updates:
python -m experiments.retrieval.run --output outputs/retrieval/frozen --freeze-step 175
# Resume their updates after update 500:
python -m experiments.retrieval.run --output outputs/retrieval/unfrozen \
  --freeze-step 175 --unfreeze-step 500
python -m experiments.plot attribution \
  --input outputs/retrieval/baseline outputs/retrieval/frozen outputs/retrieval/unfrozen \
  --labels Vanilla Frozen Unfrozen --gap 8192 --output outputs/figures/interventions.pdf
```

`--freeze-step s` starts masking updates to entire KDA blocks 0 and 2 at update `s + 1`, including their mixers, MLPs, and normalization. Other components remain trainable, and gradients still propagate through frozen blocks. `--unfreeze-step 500` resumes KDA updates at 501. Optimizer moments, the learning-rate schedule, random state, and data order continue through the intervention; unfreezing does not restart warmup. `--resume <checkpoint.pt>` resumes a saved run with the same seed, data seed, model configuration, and mixture and can branch into another freezing schedule.

### Attention gradient dynamics

```bash
# One seed, freezing after 175 and unfreezing after 225 updates:
python -m experiments.retrieval.gradients --output outputs/gradients/example \
  --freeze-step 175 --unfreeze-step 225 --steps 425
# Paper sweep: freeze after 100/150/175 updates, ten seeds per timing:
bash scripts/run_gradients.sh
python -m experiments.plot gradients --input outputs/gradients/freeze175 \
  --labels freeze175 --output outputs/figures/gradients.pdf
```

[The gradient experiment](../experiments/retrieval/gradients.py) freezes only the two KDA **mixers**, following Appendix I.1. Baseline, frozen, and unfrozen branches share initialization, batches, optimizer states at branching, and the baseline's per-update clipping factor. Full backward passes precede masking of frozen updates; unfreezing resumes paused Adam moments.

`probe` measures unclipped gradients of first-answer-token cross-entropy on 32 fixed training examples at gap 8192. For each attention mixer, it takes the L2 norm of the **mean gradient vector** over examples, then divides by the never-frozen baseline norm at the same step and seed. The plot takes the geometric mean of these paired ratios across seeds separately for each attention mixer, then the arithmetic mean over the two mixers. Ratios above 1 indicate amplification relative to the baseline. No temporal smoothing is applied. Change `--input` to `freeze100` or `freeze150` for the other timings.

### Short- and long-distance mixtures

```bash
# One mixture: 10% short-gap and 90% long-gap examples:
python -m experiments.retrieval.run --output outputs/retrieval/short10 --short-percent 10
# Complete Figure 3 grid:
for seed in 0 1 2; do
  for percent in {0..100..10}; do
    python -m experiments.retrieval.run \
      --output "outputs/retrieval/seed${seed}/short${percent}" \
      --seed "$seed" --short-percent "$percent"
  done
done
python -m experiments.plot mixture --input outputs/retrieval --labels Hybrid \
  --gap 8192 --step 1500 --output outputs/figures/mixtures.pdf
```

`--short-percent` controls the fraction of training examples with gap 64; all remaining examples use gap 8192. Thus 0 is pure long-gap training and 100 is pure short-gap training. Both gaps are evaluated for every mixture. Figure 3 uses final-checkpoint scores: sum the two layer scores within each mixer family **before** averaging over seeds, as specified in Appendix A.3. `experiments.plot.mixture_scores` implements this aggregation from `seed*/short*/eval_1500.json`; `--gap 64` selects short-gap evaluation. The plot averages the available seeds at each mixture, so complete the three-seed grid for the paper setting.

### Full sweep and output fields

```bash
# Figures 2–4: baseline, all freeze/unfreeze timings, and all mixtures; seeds 0/1/2:
bash scripts/run_retrieval.sh
# Figure 5: paired gradient experiments, ten seeds per timing:
bash scripts/run_gradients.sh
```

The full retrieval sweep writes `outputs/retrieval/seed<seed>/{baseline,freeze<step>,unfreeze<step>,short<percent>}/`. The gradient sweep writes `outputs/gradients/freeze<step>/seed<seed>/`. Set `OUTPUT` to change either sweep's output root and `PYTHON` to select its interpreter. Use either the full retrieval sweep or the separate mixture grid above when populating the same output root.

| File | Contents and metric fields |
| --- | --- |
| `protocol.json` | Resolved experiment arguments for each retrieval run. |
| `eval_<step>.json` | `contribution[gap][layer]` contains the signed retrieval score's `mean`, standard error `se`, and pair count `n`; `native[gap]` contains greedy accuracy fields. Evaluation defaults to every 25 updates and the final update. |
| `attribution_<step>.pt` | Per-pair signed effects in `[gap]["effect"][layer]`, auxiliary probability changes, probed positions, and same-input replacement checks. |
| `checkpoint_<step>.pt` | Model, optimizer, scheduler, RNG states, update, and experiment arguments for resumption. |
| `gradients.jsonl` | One record per measured update with `seed`, `step`, raw `norms[branch][layer]`, and baseline-normalized `ratios[branch][layer]`; attention layer IDs are `"1"` and `"3"`. |

## CSR, Recall, WikiText-2, and RULER (Tables 1–2, Appendix E)

```bash
bash scripts/run_eval.sh outputs/bld/390m-bld-seed0-stage2 outputs/eval/390m-bld \
  --ruler-data /path/to/ruler
# Individual benchmark groups:
python -m evaluation.run --model /path/to/model --output outputs/eval/lm --benchmark lm
python -m evaluation.run --model /path/to/model --output outputs/eval/ruler \
  --benchmark ruler --ruler-data /path/to/ruler
```

CSR averages ARC-Easy, ARC-Challenge, HellaSwag, PIQA (`acc_norm`), and BoolQ, WinoGrande, LAMBADA (`acc`). Recall averages the `contains` metric on Based SWDE, FDA, SQuAD, TriviaQA, DROP, and NQ-2048. WikiText reports `word_perplexity`. Evaluation uses zero demonstrations, greedy generation, and no chat template. The six Based task definitions are included under `evaluation/tasks/`; other tasks use lm-evaluation-harness 0.4.13. Task-level outputs are saved alongside aggregate results. `--limit` creates a development subset and must be omitted for paper scores.

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
