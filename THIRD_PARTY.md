# Third-party code

- Training framework: adapted from [tiny-pretrainer](https://github.com/chen-yingfa/tiny-pretrainer), with PyTorch, Transformers, and Accelerate.
- Recurrent kernels and model components: [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). Retained source headers and the installed distribution's MIT notice are included in `licenses/flash-linear-attention.txt`.
- Muon implementation: adapted from [KellerJordan/muon](https://github.com/KellerJordan/Muon), MIT. Attribution is retained in `optim/muon.py`; the notice is in `licenses/muon.txt`.
- Evaluation: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The included Based task definitions come from the experiment environment. Its MIT notice is included in `licenses/lm-evaluation-harness.txt`.
- RULER generation: the wrapper invokes a separately installed [NVIDIA/RULER](https://github.com/NVIDIA/RULER) checkout. Dataset resources remain subject to their respective source licenses.

Dataset examples, pretrained weights, tokenizer assets, and source-corpus snapshots are obtained separately from their providers. The source release does not include those assets.
