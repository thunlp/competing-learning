from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from arguments import Args

from .config import KDAConfig
from .model import KDAForCausalLM, KDAModel

AutoConfig.register(KDAConfig.model_type, KDAConfig, exist_ok=True)
AutoModel.register(KDAConfig, KDAModel, exist_ok=True)
AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM, exist_ok=True)
KDAConfig.register_for_auto_class()
KDAModel.register_for_auto_class("AutoModel")
KDAForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["KDAConfig", "KDAForCausalLM", "KDAModel"]


def get_config(
    path: str | None = None,
    args: Args | None = None,
) -> KDAConfig:
    """Load a KDA config and apply compatible command-line overrides."""
    if path is not None:
        model_config = KDAConfig.from_json_file(path)
    else:
        model_config = KDAConfig()

    if args is not None:
        for key, value in args.as_dict().items():
            if hasattr(model_config, key) and value is not None:
                setattr(model_config, key, value)
    return model_config


def get_model(
    config_path: str | None = None,
    args: Args | None = None,
) -> KDAForCausalLM:
    model_config = get_config(path=config_path, args=args)
    model = KDAForCausalLM(model_config)

    if args is not None and args.init_from != "scratch":
        print(f"Loading model parameters from: {args.init_from}")
        state_dict = load_file(args.init_from)
        state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in state_dict.items()
        }

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embeddings.weight"]

        for key, value in model.named_parameters():
            if key in state_dict and value.dtype != state_dict[key].dtype:
                value.data = value.data.to(state_dict[key].dtype)

        if state_dict["lm_head.weight"].shape != model.lm_head.weight.shape:
            model.lm_head = nn.Linear(
                model_config.hidden_size,
                state_dict["lm_head.weight"].shape[0],
                bias=False,
            )
            print(
                "Warning: lm_head weight shape mismatch, re-initialize lm_head layer: "
                f"{state_dict['lm_head.weight'].shape} vs {model.lm_head.weight.shape}",
            )

        model.load_state_dict(state_dict, strict=True)

    if args is not None:
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()

        if args.use_cce_loss:
            model.config.fuse_cross_entropy = True
            if args.cce_loss_impl is not None:
                model.config.cce_loss_impl = args.cce_loss_impl

    return model
