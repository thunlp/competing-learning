"""Hybrid architectures used in the paper."""

from importlib import import_module


def get_model(accelerator, args):
    families = {'kda': 'kda', 'gdn': 'gated_deltanet', 'gla': 'gla'}
    family = import_module(f'modeling.{families[args.model_name]}')
    model = family.get_model(config_path=args.model_config, args=args)
    total = sum(p.numel() for p in model.parameters())
    non_embedding = sum(p.numel() for n, p in model.named_parameters() if 'embed' not in n)
    accelerator.print(f'Parameters: {total:,}; non-embedding: {non_embedding:,}')
    return model.to(accelerator.device)
