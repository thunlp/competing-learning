"""Options shared by the paper's pretraining experiments."""

from tap import Tap


class Args(Tap):
    model_name: str = 'kda'
    model_config: str
    train_config: str
    tok_path: str
    data_path: str
    output_dir: str = 'outputs'
    proj_name: str = 'bld'
    run_name: str = 'experiment'
    init_from: str = 'scratch'
    resume_path: str | None = None
    resume_step: int | None = None
    pretrained_path: str | None = None
    freeze_layers: list[int] | None = None
    grad_ckpt: int = 1
    use_cce_loss: int = 0
    cce_loss_impl: str = 'torch_compile'
    seed: int | None = None
    data_name: str | None = None
    max_len: int | None = None
    batch_size: int | None = None
    grad_accum_steps: int | None = None
    optimizer: str | None = None
    lr_scheduler: str | None = None
    lr: float | None = None
    min_lr: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    weight_decay: float | None = None
    grad_clip: float | None = None
    n_train_steps: int | None = None
    n_warmup_steps: int | None = None
    n_drop_steps: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    eval_interval: int = 1000
    n_workers: int = 1
    prefetch_factor: int = 2
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float | None = None
    muon_ns_steps: int = 5
