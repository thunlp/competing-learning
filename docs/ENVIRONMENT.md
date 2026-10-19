# Environment

Use Linux x86_64, Python 3.11, and an NVIDIA GPU supporting BF16. Full pretraining uses 16,384-token sequences; model configurations and GPU counts are listed in the README.

## Installation

```bash
conda env create -f environment.yml
conda activate bld
bash scripts/install_cuda.sh
python -m pip check
```

`environment.yml` installs Python, pip, curl, and the training, evaluation, and plotting dependencies from `requirements-lock.txt`. Conda packages use conda-forge. Install the CUDA extensions after PyTorch.

## Core dependencies

| Component | Version |
| --- | --- |
| Python | 3.11.14 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.51.0 |
| Triton | 3.4.0 |
| Flash Linear Attention | 0.4.2 |
| FlashAttention | 2.8.3 |
| causal-conv1d | 1.7.0 |
| lm-evaluation-harness | 0.4.13 |

The complete dependency versions are pinned in `requirements-lock.txt`. The smaller `requirements.txt`, `requirements-cuda.txt`, and `requirements-eval.txt` groups are also available for manual installation.

Evaluation data preparation also uses NLTK, Wonderwords, Tenacity, Beautiful Soup, and html2text, all included in the lock file. NLTK tokenizer data and benchmark datasets are downloaded separately with the commands in the [evaluation guide](EVALUATION.md).

## CUDA extensions

`scripts/install_cuda.sh` downloads official FlashAttention and causal-conv1d wheels and verifies their SHA-256 digests. These wheels target Linux x86_64, Python 3.11, and PyTorch 2.8.0 with CUDA 12 and the C++11 ABI. `PYTHON` selects the interpreter, and `WHEEL_DIR` selects the download cache, which defaults to `outputs/wheels`.

For another Python, PyTorch, or ABI combination, install PyTorch first and build the pinned extensions from source:

```bash
python -m pip install -r requirements-cuda.txt --no-build-isolation
```

Source builds require an NVIDIA CUDA toolkit with `nvcc` and a C++ compiler. Prebuilt wheels use the CUDA runtime supplied with PyTorch. Use an NVIDIA driver compatible with that runtime.

## Training options

| Variable | Purpose |
| --- | --- |
| `DCLM_STAGE1` | DCLM parquet directory for early pretraining. |
| `DCLM_STAGE2` | Disjoint DCLM parquet directory for remaining pretraining. |
| `TOKENIZER` | Tokenizer identifier or directory. |
| `NPROC` | Number of GPU processes. |
| `OUTPUT_DIR` | Root directory for experiment outputs. |
| `PYTHON` | Python interpreter for the launch scripts. |
| `MASTER_PORT` | Port for distributed training. |
| `CONFIG_DIR` | Configuration tree; defaults to `configs/`. |

## Checks

Run the CPU checks:

```bash
python -m unittest discover -s tests -v
```

On an allocated GPU, check the pretraining launcher with generated data:

```bash
bash scripts/smoke_test.sh --output outputs/smoke --method bld --steps 1
```

See [the reproduction guide](REPRODUCE.md) for the full experiment commands and evaluation protocols.
