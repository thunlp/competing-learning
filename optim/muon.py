"""Muon optimizer integration.

Adapted from https://github.com/KellerJordan/muon (MIT License).
"""

from __future__ import annotations

import torch
from torch import nn


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Approximate matrix orthogonalization with quintic Newton-Schulz steps."""
    assert grad.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        quintic = b * gram + c * gram @ gram
        x = a * x + quintic @ x

    if transposed:
        x = x.mT
    return x


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    original_shape = update.shape
    if update.ndim > 2:
        update = update.reshape(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape)


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias_corrected_avg = exp_avg / (1 - betas[0] ** step)
    bias_corrected_sq = exp_avg_sq / (1 - betas[1] ** step)
    return bias_corrected_avg / (bias_corrected_sq.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrix weights with AdamW-style updates for all others."""

    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError(
                    "MuonWithAuxAdam param groups must set use_muon"
                )
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"],
                    key=lambda param: param.numel(),
                    reverse=True,
                )
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(
                        param.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    param.mul_(
                        1 - group["lr"] * group["weight_decay"]
                    )
                    param.add_(update, alpha=-group["lr"])
            else:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        param.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    param.mul_(
                        1 - group["lr"] * group["weight_decay"]
                    )
                    param.add_(update, alpha=-group["lr"])

        return loss


def _is_embedding_or_head(name: str) -> bool:
    return "embed" in name or "emb." in name or name == "lm_head.weight"


def build_muon_with_aux_adam(
    model: nn.Module,
    adam_lr: float,
    adam_betas: tuple[float, float],
    weight_decay: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float | None = None,
    muon_ns_steps: int = 5,
) -> MuonWithAuxAdam:
    muon_params = []
    muon_no_decay_params = []
    adam_params = []
    adam_no_decay_params = []
    seen_param_ids = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))
        use_muon = param.ndim >= 2 and not _is_embedding_or_head(name)
        no_weight_decay = getattr(param, "_no_weight_decay", False)
        if use_muon and no_weight_decay:
            muon_no_decay_params.append(param)
        elif use_muon:
            muon_params.append(param)
        elif no_weight_decay:
            adam_no_decay_params.append(param)
        else:
            adam_params.append(param)

    if not muon_params and not muon_no_decay_params:
        raise ValueError(
            "Muon optimizer selected, but no hidden matrix parameters were found"
        )

    effective_muon_weight_decay = (
        weight_decay
        if muon_weight_decay is None
        else muon_weight_decay
    )
    param_groups = []
    for params, group_weight_decay in (
        (muon_params, effective_muon_weight_decay),
        (muon_no_decay_params, 0.0),
    ):
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "use_muon": True,
                "lr": muon_lr,
                "momentum": muon_momentum,
                "weight_decay": group_weight_decay,
                "ns_steps": muon_ns_steps,
            }
        )
    for params, group_weight_decay in (
        (adam_params, weight_decay),
        (adam_no_decay_params, 0.0),
    ):
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "use_muon": False,
                "lr": adam_lr,
                "betas": adam_betas,
                "weight_decay": group_weight_decay,
            }
        )

    return MuonWithAuxAdam(param_groups)
