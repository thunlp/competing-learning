from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator, DataLoaderConfiguration
from torchdata.stateful_dataloader import StatefulDataLoader

from arguments import Args
from data.dclm import build_dataset
from optim.lr_scheduler import WSDScheduler
from optim.muon import build_muon_with_aux_adam

@dataclass(frozen=True, eq=False)
class LayerFreezeState:
    layer_path: str | None
    layer_indices: tuple[int, ...]
    newly_frozen_parameters: tuple[nn.Parameter, ...] = field(repr=False)
    previously_frozen_parameters: tuple[nn.Parameter, ...] = field(repr=False)
    total_numel: int
    trainable_numel: int
    frozen_numel: int

    @property
    def newly_frozen_numel(self) -> int:
        return sum(parameter.numel() for parameter in self.newly_frozen_parameters)


@dataclass(frozen=True)
class LayerRestoreSummary:
    tensor_count: int
    numel: int
    preexisting_refrozen_tensor_count: int


def _validate_freeze_layers(freeze_layers: list[int] | None) -> tuple[int, ...]:
    if freeze_layers is None:
        return ()
    if not isinstance(freeze_layers, list):
        raise TypeError(
            "`freeze_layers` must be a JSON/CLI array of integer layer indices."
        )

    invalid_indices = [
        index
        for index in freeze_layers
        if isinstance(index, bool) or not isinstance(index, int)
    ]
    if invalid_indices:
        raise TypeError(
            "`freeze_layers` must contain only integers; "
            f"received invalid values: {invalid_indices!r}."
        )
    if any(index < 0 for index in freeze_layers):
        raise ValueError(
            "`freeze_layers` does not accept negative indices; "
            f"received: {freeze_layers!r}."
        )
    if len(set(freeze_layers)) != len(freeze_layers):
        raise ValueError(
            "`freeze_layers` contains duplicate indices: "
            f"{freeze_layers!r}."
        )
    return tuple(freeze_layers)


def _get_decoder_layers(
    model: nn.Module,
    option_name: str = "freeze_layers",
) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []

    inner_model = getattr(model, "model", None)
    config = getattr(model, "config", None)
    has_encoder_decoder_stacks = (
        isinstance(inner_model, nn.Module)
        and hasattr(inner_model, "encoder")
        and hasattr(inner_model, "decoder")
    )
    if getattr(config, "is_encoder_decoder", False) or has_encoder_decoder_stacks:
        raise ValueError(
            f"`{option_name}` only supports decoder-only models; "
            f"{type(model).__name__} is an encoder-decoder model."
        )

    model_layers = getattr(inner_model, "layers", None)
    if isinstance(model_layers, nn.ModuleList):
        candidates.append(("model.layers", model_layers))

    backbone = getattr(model, "backbone", None)
    backbone_layers = getattr(backbone, "layers", None)
    if isinstance(backbone_layers, nn.ModuleList):
        candidates.append(("backbone.layers", backbone_layers))

    root_layers = getattr(model, "layers", None)
    if isinstance(root_layers, nn.ModuleList):
        candidates.append(("layers", root_layers))

    if not candidates:
        raise ValueError(
            f"`{option_name}` only supports decoder-only models whose main layer "
            "container is `model.layers`, `backbone.layers`, or `layers`; "
            f"{type(model).__name__} has none of these."
        )
    if len(candidates) > 1:
        paths = ", ".join(path for path, _ in candidates)
        raise ValueError(
            "Could not determine a unique decoder-only layer container; "
            f"found: {paths}."
        )
    return candidates[0]


def _parameter_numel(model: nn.Module) -> tuple[int, int, int]:
    parameters = tuple(model.parameters())
    total_numel = sum(parameter.numel() for parameter in parameters)
    trainable_numel = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return total_numel, trainable_numel, total_numel - trainable_numel


def freeze_model_layers(
    model: nn.Module,
    freeze_layers: list[int] | None,
) -> LayerFreezeState:
    layer_indices = _validate_freeze_layers(freeze_layers)
    if not layer_indices:
        total_numel, trainable_numel, frozen_numel = _parameter_numel(model)
        return LayerFreezeState(
            layer_path=None,
            layer_indices=(),
            newly_frozen_parameters=(),
            previously_frozen_parameters=(),
            total_numel=total_numel,
            trainable_numel=trainable_numel,
            frozen_numel=frozen_numel,
        )

    layer_path, layers = _get_decoder_layers(model)
    out_of_range = [index for index in layer_indices if index >= len(layers)]
    if out_of_range:
        raise ValueError(
            f"`freeze_layers` indices {out_of_range!r} are out of range for "
            f"`{layer_path}` with {len(layers)} layers."
        )

    selected_parameters: list[nn.Parameter] = []
    seen_parameter_ids: set[int] = set()
    for index in layer_indices:
        layer_parameters = tuple(layers[index].parameters())
        if not layer_parameters:
            raise ValueError(
                f"Selected layer `{layer_path}.{index}` has no parameters to freeze."
            )
        for parameter in layer_parameters:
            parameter_id = id(parameter)
            if parameter_id not in seen_parameter_ids:
                selected_parameters.append(parameter)
                seen_parameter_ids.add(parameter_id)

    newly_frozen_parameters = tuple(
        parameter for parameter in selected_parameters if parameter.requires_grad
    )
    previously_frozen_parameters = tuple(
        parameter for parameter in selected_parameters if not parameter.requires_grad
    )
    for parameter in newly_frozen_parameters:
        parameter.requires_grad_(False)

    total_numel, trainable_numel, frozen_numel = _parameter_numel(model)
    if trainable_numel == 0:
        for parameter in newly_frozen_parameters:
            parameter.requires_grad_(True)
        raise ValueError(
            "`freeze_layers` would leave the model with no trainable parameters."
        )

    return LayerFreezeState(
        layer_path=layer_path,
        layer_indices=layer_indices,
        newly_frozen_parameters=newly_frozen_parameters,
        previously_frozen_parameters=previously_frozen_parameters,
        total_numel=total_numel,
        trainable_numel=trainable_numel,
        frozen_numel=frozen_numel,
    )


def restore_frozen_parameters(
    freeze_state: LayerFreezeState,
) -> LayerRestoreSummary:
    preexisting_refrozen_tensor_count = sum(
        parameter.requires_grad
        for parameter in freeze_state.previously_frozen_parameters
    )
    for parameter in freeze_state.newly_frozen_parameters:
        parameter.requires_grad_(True)
    for parameter in freeze_state.previously_frozen_parameters:
        parameter.requires_grad_(False)

    return LayerRestoreSummary(
        tensor_count=len(freeze_state.newly_frozen_parameters),
        numel=freeze_state.newly_frozen_numel,
        preexisting_refrozen_tensor_count=preexisting_refrozen_tensor_count,
    )



def load_train_config(args: Args):
    with open(args.train_config) as handle:
        config = json.load(handle)
    for name, value in config.items():
        if not hasattr(args, name):
            raise ValueError(f"Unknown training option: {name}")
        if getattr(args, name) is None:
            setattr(args, name, value)


def get_accelerator(args: Args) -> Accelerator:
    return Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            non_blocking=True,
            use_stateful_dataloader=True,
        ),
        step_scheduler_with_optimizer=False,
    )


def get_dataloaders(args, tokenizer, accelerator):
    if args.data_name != "dclm":
        raise ValueError("This release uses DCLM pretraining data.")
    dataset = build_dataset(tokenizer, args.data_path, max_len=args.max_len)
    options = dict(num_workers=args.n_workers, pin_memory=True)
    if args.n_workers > 0:
        options.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    return StatefulDataLoader(dataset, batch_size=args.batch_size, **options), None


def prepare_optimizers(model: nn.Module, args: Args):
    if args.optimizer == "muon":
        optimizer = build_muon_with_aux_adam(
            model=model,
            adam_lr=args.lr,
            adam_betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_weight_decay=args.muon_weight_decay,
            muon_ns_steps=args.muon_ns_steps,
        )
    elif args.optimizer == "adamw":
        groups = []
        for no_decay in (False, True):
            params = [p for p in model.parameters() if p.requires_grad
                      and bool(getattr(p, "_no_weight_decay", False)) == no_decay]
            if params:
                groups.append(dict(params=params, weight_decay=0.0 if no_decay else args.weight_decay))
        optimizer = AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return optimizer, WSDScheduler(
        optimizer=optimizer,
        lr=args.lr,
        min_lr=args.min_lr,
        n_decay_iters=args.n_drop_steps,
        n_warmup_iters=args.n_warmup_steps,
        n_train_iters=args.n_train_steps,
    )
