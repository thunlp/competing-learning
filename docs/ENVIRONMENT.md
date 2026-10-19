# Environment

The source release targets Linux x86_64, Python 3.11, and NVIDIA GPUs supporting BF16. The paper configurations use 16,384-token sequences; GPU counts are listed in the README. The small smoke test uses one GPU and 128-token sequences.

## Recorded runtime

| Component | Version |
| --- | --- |
| Smoke-test GPU | NVIDIA A800-SXM4-80GB |
| Smoke-test NVIDIA driver | 610.43.02 |
| Python | 3.11.14 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime used by PyTorch | 12.8 |
| cuDNN | 9.10.2 |
| Transformers | 4.51.0 |
| Accelerate | 1.12.0 |
| Datasets | 4.5.0 |
| Flash Linear Attention | 0.4.2 |
| FlashAttention | 2.8.3 |
| causal-conv1d | 1.7.0 |
| Triton | 3.4.0 |
| torchdata | 0.11.0 |
| NumPy | 2.4.2 |
| safetensors | 0.7.0 |
| typed-argument-parser | 1.11.0 |
| einops | 0.8.2 |
| FLA core | 0.4.2 |
| lm-evaluation-harness (clean environment) | 0.4.13 |

`requirements-lock.txt` pins all 109 Python dependencies for training, evaluation, and plotting, including transitive packages. `environment.yml` installs this lock file. The two CUDA extensions are installed afterward so PyTorch is available first. The smaller `requirements.txt`, `requirements-cuda.txt`, and `requirements-eval.txt` groups remain available for manual installs. The clean environment uses the public lm-evaluation-harness 0.4.13 release and the six included Based tasks; no local harness checkout is required.

## Installation order

```bash
conda env create -f environment.yml
conda activate bld
bash scripts/install_cuda.sh
python -m pip check
```

The environment YAML installs Python, pip, curl, and the complete locked Python requirements using only conda-forge for Conda packages. The CUDA installer obtains the official Python 3.11/PyTorch 2.8/CUDA 12/C++11 ABI wheels from the upstream GitHub API and checks their published SHA-256 digests. It uses curl over IPv4 with resumable byte ranges; rerun the command after a connection failure. It requires no GitHub account or API key. `PYTHON` selects the interpreter and `WHEEL_DIR` selects its download cache (default `outputs/wheels`). This avoids relying on `nvcc` or a source build for the recorded Linux x86_64 runtime.

For another Python/PyTorch/ABI combination, use `pip install -r requirements-cuda.txt --no-build-isolation` after installing PyTorch. Source builds need an NVIDIA CUDA toolkit with `nvcc`, a C++ compiler, and the pinned packaging/ninja/setuptools/wheel helpers. The CUDA runtime reported by PyTorch and the separately installed CUDA toolkit are distinct components. Ordinary GitHub release-download connections timed out on the validation host; the official API endpoint used by the installer was reachable.

The recorded runtime uses CUDA 12.8; use a compatible NVIDIA driver. A local `nvcc` executable is not required to run already-built extensions. Triton compiles its kernels on first use, so the first smoke test can take several minutes. Keep the active environment on `PATH` for Slurm jobs; no named cluster environment is hard-coded in the release scripts.

## Local settings

Set `DCLM_STAGE1` for early pretraining data, `DCLM_STAGE2` for remaining pretraining data, and optionally `TOKENIZER` for real training. `PYTHON` selects an interpreter, `NPROC` controls the GPU process count, `OUTPUT_DIR` selects the output root, and `MASTER_PORT` selects the distributed port. `CONFIG_DIR` can point to a separate configuration tree for small experiments; the default is `configs/`.

The smoke test creates a local tokenizer and two synthetic parquet views, then calls `scripts/run_main.sh` with a separate configuration tree. It retains the real AdamW-to-Muon transition and BLD block freezing. Results and data caches are written under its output directory. Compiled kernels are reused from `outputs/.kernel-cache/` by default; existing `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` settings are honored. All of these generated files are excluded from the source archive.

No tracking service, API key, account configuration, or external data download is required by the smoke test. Full evaluation requires separately prepared benchmark data and a reference model for LongPPL.

## Clean-environment checks

On 2026-10-19, an independent Conda prefix was created without cloning the development environment. Public PyPI and upstream CUDA distributions were installed into it; cached distribution files were checked against official SHA-256 digests. No installed packages were copied from the development environment. The final Conda recipe resolved successfully, and `pip check` reported no broken requirements. All 111 Python distributions came from the new environment; user-site packages were disabled and `PYTHONPATH` was cleared. All 14 evaluation tasks registered with the pinned public harness release.

Run CPU checks with a writable local dataset cache:

```bash
PYTHONNOUSERSITE=1 HF_HOME="$PWD/outputs/cpu-check/hf" \
  HF_DATASETS_CACHE="$PWD/outputs/cpu-check/datasets" \
  python -m unittest discover -s tests -v
```

All CPU release checks passed in the clean environment. For the GPU check, use:

```bash
TMPDIR=/tmp PYTHONNOUSERSITE=1 bash scripts/smoke_test.sh \
  --output outputs/clean-smoke --method bld --steps 1
```

Keep `TMPDIR` short for PyTorch data-loader workers. A deeply nested temporary path exceeded the Unix socket path limit during validation; a short node-local `/tmp` directory resolved it.

The command completed successfully on one A800 with new kernel caches. The 1,254,787-parameter KDA hybrid used a local tokenizer, synthetic text, and 128-token contexts:

| Pretraining portion | Updates | Loss | Gradient norm |
| --- | ---: | ---: | ---: |
| Early pretraining, AdamW | 1 | 5.0312 | 12.1 |
| Remaining pretraining, Muon + AdamW | 1 | 5.0000 | 16.2 |

Both checkpoints and model exports were saved. Frozen recurrent weights matched initialization exactly during early pretraining, attention weights changed, and recurrent weights changed after unfreezing. Continuation from the learned weights with a fresh optimizer passed, and the exported final model reloaded and produced finite GPU logits. The first update took 384.6 seconds including cold compilation; the second took 10.9 seconds including additional optimizer-specific compilation. These are execution checks, not paper benchmark results.

## GPU smoke-test result

Validated on 2026-10-19 with the runtime and A800 GPU listed above. A 1,254,787-parameter KDA hybrid used generated text, a local tokenizer, and 128-token contexts. The actual `run_main.sh` launcher completed the following optimizer updates:

| Method | Pretraining portion | Updates | First loss | Final loss |
| --- | --- | ---: | ---: | ---: |
| Vanilla | Early pretraining, AdamW | 2 | 5.0312 | 4.6562 |
| Vanilla | Remaining pretraining, Muon + AdamW | 2 | 4.9375 | 4.3125 |
| BLD | Early pretraining, AdamW, recurrent blocks frozen | 2 | 5.0312 | 4.8438 |
| BLD | Remaining pretraining, Muon + AdamW, all blocks trainable | 2 | 4.9688 | 4.3125 |

All eight gradient norms were finite and nonzero. Both pretraining portions saved checkpoints and model exports; both final models reloaded and produced finite GPU logits. BLD recurrent weights matched initialization exactly during early pretraining, its attention weights changed, and recurrent weights changed after unfreezing. The paper configuration files were unchanged.

The first update took approximately 410 seconds including cold compilation/autotuning; cached updates took approximately 0.54–0.67 seconds. These synthetic-run losses and timings validate execution only; they are not paper benchmark results. The CPU release suite also passed.
