# Evaluation

Evaluation uses the same Python and CUDA environment as training. The Conda recipe includes `lm-evaluation-harness`, the included Based tasks' metric dependencies, and the packages used to prepare RULER data. When extending an existing training environment, install `requirements-eval.txt` as well. The CUDA extensions are installed with `scripts/install_cuda.sh` after PyTorch; see [Environment](ENVIRONMENT.md).

Model directories must contain the exported weights, configuration, tokenizer, and custom modeling files. `scripts/run_main.sh` produces these exports. `scripts/run_eval.sh` runs CSR, Recall, WikiText-2, and RULER; LongPPL has a separate entry point because it also requires a reference model.

## CSR, Recall, and WikiText-2

Download the datasets and metric resources before allocating a GPU:

```bash
python -m evaluation.prepare_data --benchmark lm
```

This uses the pinned evaluation framework's own task definitions, downloads their datasets into the Hugging Face cache, loads the evaluation splits, and caches the `squad_v2` metric used by the included SQuAD and TriviaQA tasks. For a subset, pass task names, for example `--tasks arc_easy,piqa`.

| Tasks | Dataset sources |
| --- | --- |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc`, subsets `ARC-Easy` / `ARC-Challenge` |
| HellaSwag | `Rowan/hellaswag` |
| PIQA | `baber/piqa` |
| BoolQ | `aps/super_glue`, subset `boolq` |
| WinoGrande | `allenai/winogrande`, subset `winogrande_xl` |
| LAMBADA | `cimec/lambada` |
| Based Recall | `hazyresearch/based-swde-v2`, `based-fda`, `based-squad`, `based_triviaqa`, `based_drop`, and `based_nq_2048` under the same namespace |
| WikiText-2 | `EleutherAI/wikitext_document_level`, subset `wikitext-2-raw-v1` |

The preparation command saves dataset identifiers, document counts, and fingerprints to `datasets/evaluation-manifest.json`. The data remain in the Hugging Face cache. To run these benchmark groups:

```bash
bash scripts/run_eval.sh /path/to/model outputs/eval/lm --benchmark lm
```

Evaluation also loads the task data and metric resources before loading model weights, so missing downloads fail before model initialization. `--limit 1` can check a small subset of each task; omit it for reported scores.

## RULER

Prepare the upstream generator, source corpora, and NLTK resources:

```bash
python -m evaluation.prepare_data --benchmark ruler
```

The command obtains [NVIDIA/RULER](https://github.com/NVIDIA/RULER) at the commit pinned in `evaluation/prepare_data.py`, downloads the Paul Graham essay corpus using the upstream script, and downloads SQuAD v2 and HotpotQA using the upstream source URLs. It also prepares NLTK `punkt` and `punkt_tab`. The Python dependencies include `wonderwords`, `tenacity`, `html2text`, and `beautifulsoup4`; NeMo and vLLM are not used by this Hugging Face evaluation path.

The default generator checkout is `third_party/RULER`. `RULER_SOURCE` selects another checkout; `--ruler-revision` explicitly selects its full commit ID. An existing checkout with a different revision is left untouched and reported as an error. Source-file hashes and the generator revision are recorded in the preparation manifest. `--tasks` can prepare only selected task resources.

Generate data with the same tokenizer used by the evaluated models:

```bash
python -m evaluation.prepare_ruler \
  --tokenizer-name qwen3 --tokenizer-path Qwen/Qwen3-0.6B \
  --num-samples 100 --out-root datasets/ruler
```

The tokenizer is downloaded through Transformers, or `--tokenizer-path` can point to an exported model's tokenizer directory. Outputs use the layout `datasets/ruler/qwen3/<length>/<task>/test.jsonl`. The default grid has 57 task–length combinations at 1K, 2K, 4K, 8K, and 16K tokens, with 100 examples per combination. CWE is excluded; QA-1 starts at 2K and QA-2 at 4K. Git LFS is not required for this grid. Cached files with fewer examples than requested are regenerated.

For a small data-generation check:

```bash
python -m evaluation.prepare_data --benchmark ruler --tasks niah_single_1
python -m evaluation.prepare_ruler --tokenizer-name qwen3 \
  --tokenizer-path Qwen/Qwen3-0.6B --tasks niah_single_1 \
  --lengths 1024 --num-samples 1 --out-root datasets/ruler-check
```

Use the complete grid for evaluation:

```bash
bash scripts/run_eval.sh /path/to/model outputs/eval/all --ruler-data datasets/ruler
# RULER only:
bash scripts/run_eval.sh /path/to/model outputs/eval/ruler \
  --benchmark ruler --ruler-data datasets/ruler
```

## Other evaluation inputs

- **LongPPL** uses the GovReport test data distributed with [LongBench](https://huggingface.co/datasets/zai-org/LongBench) and a separately downloaded [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) reference model. The scorer loads model and tokenizer files from their supplied directories.
- **Context utilization** reads Books3 and BookCorpus2 JSONL files containing a `text` field. These corpus files are user-provided inputs.
- **Retrieval-head analysis** reads [BookSum](https://huggingface.co/datasets/kmfoda/booksum) chapters with a `chapter` field. Pass the prepared dataset directory to `--booksum`.

For LongPPL, download the source data and reference model explicitly:

```bash
hf download zai-org/LongBench data.zip --repo-type dataset --local-dir datasets/longbench
hf download Qwen/Qwen3-8B --local-dir outputs/models/Qwen3-8B
```

Convert GovReport to the Arrow input consumed by `evaluation.longppl prepare`:

```python
import json
from pathlib import Path
from zipfile import ZipFile
from datasets import Dataset

with ZipFile("datasets/longbench/data.zip") as archive:
    member = next(name for name in archive.namelist() if Path(name).name == "gov_report.jsonl")
    rows = [json.loads(line) for line in archive.open(member) if line.strip()]
Dataset.from_list(rows).save_to_disk("datasets/gov-report")
```

The Arrow file is `datasets/gov-report/data-00000-of-00001.arrow`. Use it with the LongPPL commands in [the reproduction guide](REPRODUCE.md). That guide also describes context utilization and retrieval-head evaluation.

## Caches and offline use

`HF_HOME` selects the Hugging Face cache root; `HF_DATASETS_CACHE` and `HF_EVALUATE_CACHE` can override dataset and metric caches. `NLTK_DATA` selects NLTK resources, defaulting to `datasets/nltk_data` for RULER preparation.

For offline evaluation, prepare the datasets and metric resources first, preserve the complete cache, and generate all RULER JSONL files. Then set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `HF_EVALUATE_OFFLINE=1`. LongPPL also requires the local reference weights and tokenizer. Network access is required when downloading uncached resources; the evaluation scripts do not redirect requests to a mirror.
