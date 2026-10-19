"""DuoAttention alpha-only distillation utilities and trainer."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor, nn



DUO_ALPHA_FILENAME = "duo_alpha.safetensors"
DUO_ALPHA_MANIFEST_FILENAME = "duo_alpha_manifest.json"


@dataclass(frozen=True)
class DuoFreezeSummary:
    """Parameter counts after making only DuoAttention gates trainable."""

    alpha_tensors: int
    alpha_numel: int
    total_numel: int

    @property
    def frozen_numel(self) -> int:
        return self.total_numel - self.alpha_numel


@dataclass(frozen=True)
class DuoDistillationLossOutput:
    """Components of the official DuoAttention identification objective."""

    total: Tensor
    distill: Tensor
    regularization: Tensor
    num_answer_tokens: int


def _is_duo_alpha_name(name: str) -> bool:
    return name == "duo_alpha" or name.endswith(".duo_alpha")


def _named_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    named = tuple(model.named_parameters())
    seen_names: set[str] = set()
    for name, parameter in named:
        if not isinstance(name, str) or not name:
            raise ValueError("Every model parameter must have a non-empty string name.")
        if name in seen_names:
            raise ValueError(f"Duplicate parameter name: {name!r}.")
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"Named parameter {name!r} is not an nn.Parameter.")
        seen_names.add(name)
    return named


def _live_duo_alpha_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    named_alpha = [
        (name, parameter)
        for name, parameter in _named_parameters(model)
        if _is_duo_alpha_name(name)
    ]
    if not named_alpha:
        raise ValueError("The model has no parameters named `duo_alpha`.")

    seen_parameter_ids: set[int] = set()
    for name, parameter in named_alpha:
        if id(parameter) in seen_parameter_ids:
            raise ValueError(
                f"Duo alpha parameter {name!r} aliases another duo alpha parameter.",
            )
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"Duo alpha parameter {name!r} must be float32; "
                f"received {parameter.dtype}.",
            )
        if parameter.numel() == 0:
            raise ValueError(f"Duo alpha parameter {name!r} must not be empty.")
        seen_parameter_ids.add(id(parameter))
    return tuple(parameter for _, parameter in named_alpha)


def freeze_for_duo_distillation(model: nn.Module) -> DuoFreezeSummary:
    """Freeze a model transactionally, leaving only FP32 ``duo_alpha`` trainable."""

    named = _named_parameters(model)
    named_alpha = [
        (name, parameter)
        for name, parameter in named
        if _is_duo_alpha_name(name)
    ]
    if not named_alpha:
        raise ValueError("The model has no parameters named `duo_alpha`.")

    seen_alpha_ids: set[int] = set()
    for name, parameter in named_alpha:
        if id(parameter) in seen_alpha_ids:
            raise ValueError(
                f"Duo alpha parameter {name!r} aliases another duo alpha parameter.",
            )
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"Duo alpha parameter {name!r} must be float32; "
                f"received {parameter.dtype}.",
            )
        if parameter.numel() == 0:
            raise ValueError(f"Duo alpha parameter {name!r} must not be empty.")
        parameter._no_weight_decay = True
        seen_alpha_ids.add(id(parameter))

    alpha_ids = {id(parameter) for _, parameter in named_alpha}
    for _, parameter in named:
        parameter.requires_grad_(id(parameter) in alpha_ids)

    total_numel = sum(parameter.numel() for _, parameter in named)
    alpha_numel = sum(parameter.numel() for _, parameter in named_alpha)
    return DuoFreezeSummary(
        alpha_tensors=len(named_alpha),
        alpha_numel=alpha_numel,
        total_numel=total_numel,
    )


def _validate_l1_weight(l1_weight: float) -> float:
    if isinstance(l1_weight, bool) or not isinstance(l1_weight, Real):
        raise TypeError("`l1_weight` must be a real number.")
    weight = float(l1_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("`l1_weight` must be finite and nonnegative.")
    return weight


def duo_distillation_loss(
    last_hidden_state: Tensor,
    labels: Tensor,
    alpha_parameters: Iterable[Tensor] | Tensor,
    l1_weight: float,
) -> DuoDistillationLossOutput:
    """Compute answer-token hidden-state MSE plus mean absolute alpha."""

    weight = _validate_l1_weight(l1_weight)
    if not isinstance(last_hidden_state, Tensor):
        raise TypeError("`last_hidden_state` must be a tensor.")
    if not last_hidden_state.is_floating_point():
        raise TypeError("`last_hidden_state` must be floating-point.")
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "`last_hidden_state` must have shape [2B, T, D]; "
            f"received {tuple(last_hidden_state.shape)}.",
        )
    doubled_batch, sequence_length, hidden_size = last_hidden_state.shape
    if doubled_batch == 0 or doubled_batch % 2 != 0:
        raise ValueError(
            "`last_hidden_state` batch dimension must be a positive even number.",
        )
    if sequence_length == 0 or hidden_size == 0:
        raise ValueError("`last_hidden_state` sequence and hidden dimensions must be nonzero.")

    if not isinstance(labels, Tensor):
        raise TypeError("`labels` must be a tensor.")
    expected_label_shape = (doubled_batch // 2, sequence_length)
    if labels.ndim != 2 or tuple(labels.shape) != expected_label_shape:
        raise ValueError(
            f"`labels` must have shape {expected_label_shape}; "
            f"received {tuple(labels.shape)}.",
        )

    answer_mask = labels.ne(-100)
    num_answer_tokens = int(answer_mask.sum().item())
    if num_answer_tokens == 0:
        raise ValueError("`labels` must contain at least one answer token.")

    if isinstance(alpha_parameters, Tensor):
        alphas = (alpha_parameters,)
    else:
        try:
            alphas = tuple(alpha_parameters)
        except TypeError as exception:
            raise TypeError("`alpha_parameters` must be an iterable of tensors.") from exception
    if not alphas:
        raise ValueError("`alpha_parameters` must contain at least one tensor.")
    for index, alpha in enumerate(alphas):
        if not isinstance(alpha, Tensor):
            raise TypeError(f"Alpha parameter {index} is not a tensor.")
        if alpha.dtype != torch.float32:
            raise TypeError(
                f"Alpha parameter {index} must be float32; received {alpha.dtype}.",
            )
        if alpha.numel() == 0:
            raise ValueError(f"Alpha parameter {index} must not be empty.")
        if alpha.device != last_hidden_state.device:
            raise ValueError(
                "All alpha parameters must be on the same device as "
                "`last_hidden_state`.",
            )

    batch_size = doubled_batch // 2
    teacher = last_hidden_state[:batch_size].detach().float()
    student = last_hidden_state[batch_size:].float()
    per_token_mse = (teacher - student).square().mean(dim=-1)
    distill = per_token_mse[answer_mask.to(device=per_token_mse.device)].mean()

    flattened_alpha = torch.cat(tuple(alpha.reshape(-1) for alpha in alphas))
    regularization = flattened_alpha.abs().mean()
    total = distill + weight * regularization
    return DuoDistillationLossOutput(
        total=total,
        distill=distill,
        regularization=regularization,
        num_answer_tokens=num_answer_tokens,
    )


def collect_duo_alpha_state(model: nn.Module) -> dict[str, Tensor]:
    """Collect full-name keyed, portable FP32 snapshots of every duo alpha."""

    state: dict[str, Tensor] = {}
    seen_parameter_ids: set[int] = set()
    for name, parameter in _named_parameters(model):
        if not _is_duo_alpha_name(name):
            continue
        if name in state:
            raise ValueError(f"Duplicate duo alpha parameter name: {name!r}.")
        if id(parameter) in seen_parameter_ids:
            raise ValueError(
                f"Duo alpha parameter {name!r} aliases another duo alpha parameter.",
            )
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"Duo alpha parameter {name!r} must be float32; "
                f"received {parameter.dtype}.",
            )
        if parameter.numel() == 0:
            raise ValueError(f"Duo alpha parameter {name!r} must not be empty.")
        state[name] = (
            parameter.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        seen_parameter_ids.add(id(parameter))

    if not state:
        raise ValueError("The model has no parameters named `duo_alpha`.")
    return state


def _temporary_path(directory: Path, *, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".duo_alpha_tmp_",
        suffix=suffix,
        dir=directory,
    )
    os.close(descriptor)
    return Path(raw_path)


def _publish_without_overwrite(temporary_path: Path, target_path: Path) -> None:
    try:
        os.link(temporary_path, target_path)
    except FileExistsError as exception:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target_path}") from exception
    temporary_path.unlink()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_duo_alpha_artifacts(
    model: nn.Module,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically publish portable alpha tensors and a JSON manifest."""

    if not isinstance(metadata, Mapping):
        raise TypeError("`metadata` must be a mapping.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensor_path = output_path / DUO_ALPHA_FILENAME
    manifest_path = output_path / DUO_ALPHA_MANIFEST_FILENAME
    existing = [
        path
        for path in (tensor_path, manifest_path)
        if os.path.lexists(path)
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing DuoAttention artifacts: "
            + ", ".join(str(path) for path in existing),
        )

    state = collect_duo_alpha_state(model)
    layers: list[dict[str, Any]] = []
    for name, tensor in state.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Duo alpha parameter {name!r} contains non-finite values.")
        layers.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "values": tensor.reshape(-1).tolist(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
            },
        )

    manifest = {
        "schema_version": 1,
        "tensor_file": DUO_ALPHA_FILENAME,
        "metadata": dict(metadata),
        "layers": layers,
    }
    try:
        manifest_text = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exception:
        raise ValueError("`metadata` must be JSON serializable with finite values.") from exception

    tensor_temporary_path: Path | None = None
    manifest_temporary_path: Path | None = None
    try:
        tensor_temporary_path = _temporary_path(output_path, suffix=".safetensors")
        save_file(state, str(tensor_temporary_path))
        with tensor_temporary_path.open("rb") as tensor_file:
            os.fsync(tensor_file.fileno())

        manifest_temporary_path = _temporary_path(output_path, suffix=".json")
        with manifest_temporary_path.open("w", encoding="utf-8") as manifest_file:
            manifest_file.write(manifest_text)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        _publish_without_overwrite(tensor_temporary_path, tensor_path)
        tensor_temporary_path = None
        _publish_without_overwrite(manifest_temporary_path, manifest_path)
        manifest_temporary_path = None
        _fsync_directory(output_path)
    finally:
        if tensor_temporary_path is not None:
            tensor_temporary_path.unlink(missing_ok=True)
        if manifest_temporary_path is not None:
            manifest_temporary_path.unlink(missing_ok=True)


def _validated_batch(batch: object) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("Each DuoAttention batch must be a mapping.")
    forbidden = {"attention_mask", "cu_seqlens"}.intersection(batch)
    if forbidden:
        raise ValueError(
            "DuoAttention distillation forbids batch fields: "
            + ", ".join(sorted(forbidden)),
        )
    missing = {"input_ids", "labels"}.difference(batch)
    if missing:
        raise ValueError(
            "DuoAttention batches require fields: " + ", ".join(sorted(missing)),
        )

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if not isinstance(input_ids, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("`input_ids` and `labels` must be tensors.")
    if input_ids.ndim != 2:
        raise ValueError(
            f"`input_ids` must have shape [B, T]; received {tuple(input_ids.shape)}.",
        )
    if labels.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError(
            "`labels` must have the same [B, T] shape as `input_ids`; "
            f"received {tuple(labels.shape)} and {tuple(input_ids.shape)}.",
        )
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("DuoAttention batches must not be empty.")
    return input_ids, labels


def _duo_forward(model: nn.Module, input_ids: Tensor) -> Tensor:
    duplicated_input_ids = torch.cat((input_ids, input_ids), dim=0)
    outputs = model(
        input_ids=duplicated_input_ids,
        return_last_hidden_state_only=True,
        duo_mode="two_way",
        use_cache=False,
        return_dict=True,
    )
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if not isinstance(last_hidden_state, Tensor):
        raise TypeError(
            "The model must return an object with tensor `last_hidden_state`.",
        )
    return last_hidden_state
