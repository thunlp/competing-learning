import json

from .model import GatedDeltaNetForCausalLM
from .config import GatedDeltaNetConfig
from arguments import Args

from safetensors.torch import load_file
from torch import nn

def get_config(path: str | None = None, args: Args | None = None) -> GatedDeltaNetConfig:
    """
    Load from `args.model_config` if it exists, then override the
    attributes with command line arguments.
    """
    if path is not None:
        model_config = GatedDeltaNetConfig.from_json_file(path)
    else:
        model_config = GatedDeltaNetConfig()  # use default values

    if args is not None:
        # Override values in the config from the command line arguments
        for k, v in args.as_dict().items():
            if hasattr(model_config, k) and v is not None:
                setattr(model_config, k, v)
    return model_config


def get_model(
    config_path: str | None = None,
    args: Args | None = None,
) -> GatedDeltaNetForCausalLM:
    # init a new model from scratch
    model_config = get_config(path=config_path, args=args)
    model = GatedDeltaNetForCausalLM(model_config)

    if args.init_from != "scratch":
        print(f"Loading model parameters from: {args.init_from}")
        state_dict = load_file(args.init_from)
        state_dict = {key.replace('_orig_mod.', ''): val for key, val in state_dict.items()}

        if 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['model.embeddings.weight']

        # Convert params dtype
        for key, value in model.named_parameters():
            if key in state_dict and value.dtype != state_dict[key].dtype:
                value.data = value.data.to(state_dict[key].dtype)

        if state_dict['lm_head.weight'].shape != model.lm_head.weight.shape:
            model.lm_head = nn.Linear(model_config.hidden_size, state_dict['lm_head.weight'].shape[0], bias=False)
            print(f"Warning: lm_head weight shape mismatch, re-initialize lm_head layer: {state_dict['lm_head.weight'].shape} vs {model.lm_head.weight.shape}")

        model.load_state_dict(state_dict, strict=True)

    if args is not None:
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            # model.training = True
            # model.gradient_checkpointing = True

        if args.use_cce_loss:
            model.config.fuse_cross_entropy = True
            if args.cce_loss_impl is not None:
                model.config.cce_loss_impl = args.cce_loss_impl

    return model
