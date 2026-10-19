"""Standalone tensor operators used by KDA DuoAttention layers.

The attention helpers implement raw causal sink-plus-recent attention and
per-head branch mixing.  ``kda_recent_affine`` is deliberately narrower than
a complete streaming KDA implementation: it returns only the contribution of
recent value writes.  A caller that enables KDA sinks must compute the sink
contribution separately (the layer uses a masked-value ``chunk_kda`` scan)
and add it to this result.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Integral, Real
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


__all__ = [
    "kda_recent_affine",
    "make_duo_alpha",
    "mix_attention_heads",
    "mix_kda_heads",
    "streaming_attention",
    "validate_duo_hyperparameters",
]


_KDAToken: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_FLEX_BLOCK_MASK_CACHE: dict[tuple[str, int, int, int], object] = {}
_COMPILED_KDA_APPEND_BLOCK = None
_COMPILED_KDA_SLIDE_BLOCK = None
_COMPILED_KDA_QUERY_LAG_BLOCK = None


def _validate_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"`{name}` must be nonnegative; got {value}.")
    return value


def _validate_positive_integer(name: str, value: int) -> int:
    value = _validate_nonnegative_integer(name, value)
    if value == 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _validate_finite_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"`{name}` must be finite; got {value}.")
    return value


def _validate_alpha_init(alpha_init: float) -> float:
    alpha_init = _validate_finite_real("alpha_init", alpha_init)
    if not 0.0 <= alpha_init <= 1.0:
        raise ValueError(
            "`alpha_init` must lie in the closed interval [0, 1]; "
            f"got {alpha_init}.",
        )
    return alpha_init


def _validate_inverse_epsilon(inverse_epsilon: float) -> float:
    inverse_epsilon = _validate_finite_real(
        "inverse_epsilon",
        inverse_epsilon,
    )
    if inverse_epsilon <= 0.0:
        raise ValueError(
            "`inverse_epsilon` must be positive; "
            f"got {inverse_epsilon}.",
        )
    return inverse_epsilon


def validate_duo_hyperparameters(
    sink_size: int,
    recent_size: int,
    alpha_init: float,
    *,
    inverse_epsilon: float = 1e-4,
    rebase_interval: int = 128,
) -> None:
    """Validate shared DuoAttention window, gate, and KDA settings.

    Integer arguments reject booleans.  Real-valued arguments must be finite;
    ``alpha_init`` is inclusive on both endpoints.
    """

    _validate_nonnegative_integer("sink_size", sink_size)
    _validate_positive_integer("recent_size", recent_size)
    _validate_alpha_init(alpha_init)
    _validate_inverse_epsilon(inverse_epsilon)
    _validate_nonnegative_integer("rebase_interval", rebase_interval)


def make_duo_alpha(
    num_heads: int,
    alpha_init: float,
    *,
    device: torch.device | str | int | None = None,
) -> nn.Parameter:
    """Create a one-dimensional FP32 DuoAttention gate parameter."""

    num_heads = _validate_positive_integer("num_heads", num_heads)
    alpha_init = _validate_alpha_init(alpha_init)
    alpha = nn.Parameter(
        torch.full(
            (num_heads,),
            alpha_init,
            dtype=torch.float32,
            device=device,
        ),
    )
    # Accelerate's init_empty_weights wrapper reconstructs meta parameters
    # using every item in Parameter.__dict__ as constructor kwargs. Defer this
    # optimizer marker until after registration for meta-device parameters.
    if alpha.device.type != "meta":
        alpha._no_weight_decay = True
    return alpha


def _validate_dense_float_tensor(
    name: str,
    tensor: torch.Tensor,
    ndim: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor.")
    if tensor.ndim != ndim:
        raise ValueError(
            f"`{name}` must have {ndim} dimensions; "
            f"got shape {tuple(tensor.shape)}.",
        )
    if tensor.layout is not torch.strided:
        raise ValueError(f"`{name}` must be a strided dense tensor.")
    if not tensor.is_floating_point():
        raise TypeError(
            f"`{name}` must have a floating-point dtype; got {tensor.dtype}.",
        )
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(
            f"`{name}` must have no empty dimensions; got {tuple(tensor.shape)}.",
        )


def _validate_same_device(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> torch.device:
    first_name, first_tensor = named_tensors[0]
    for name, tensor in named_tensors[1:]:
        if tensor.device != first_tensor.device:
            raise ValueError(
                f"`{name}` and `{first_name}` must be on the same device; "
                f"got {tensor.device} and {first_tensor.device}.",
            )
    return first_tensor.device


def _validate_mix_inputs(
    full: torch.Tensor,
    streaming: torch.Tensor,
    alpha: torch.Tensor,
) -> int:
    _validate_dense_float_tensor("full", full, 4)
    _validate_dense_float_tensor("streaming", streaming, 4)
    _validate_dense_float_tensor("alpha", alpha, 1)
    if full.shape != streaming.shape:
        raise ValueError(
            "`full` and `streaming` must have identical [B, T, H, D] shapes; "
            f"got {tuple(full.shape)} and {tuple(streaming.shape)}.",
        )
    if full.dtype != streaming.dtype:
        raise TypeError(
            "`full` and `streaming` must have the same dtype; "
            f"got {full.dtype} and {streaming.dtype}.",
        )
    _validate_same_device(
        (("full", full), ("streaming", streaming), ("alpha", alpha)),
    )
    if alpha.dtype != torch.float32:
        raise TypeError(
            "`alpha` must have dtype torch.float32 so its gradient remains FP32; "
            f"got {alpha.dtype}.",
        )
    return full.shape[2]


def mix_attention_heads(
    full: torch.Tensor,
    streaming: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Mix raw attention heads, with one FP32 gate per contiguous KV group.

    ``full`` and ``streaming`` have shape ``[B, T, Hq, D]`` and
    ``alpha`` has shape ``[Hkv]``.  Each KV gate is repeated over its
    contiguous GQA query-head group.  The gate is cast for the arithmetic,
    while autograd accumulates its gradient into the original FP32 tensor.
    """

    num_query_heads = _validate_mix_inputs(full, streaming, alpha)
    num_kv_heads = alpha.numel()
    if num_query_heads % num_kv_heads:
        raise ValueError(
            "The number of query heads must be an integer multiple of "
            "the number of alpha/KV heads; "
            f"got Hq={num_query_heads}, Hkv={num_kv_heads}.",
        )
    gate = alpha.repeat_interleave(num_query_heads // num_kv_heads)
    gate = gate.to(dtype=full.dtype).view(1, 1, num_query_heads, 1)
    return gate * full + (1.0 - gate) * streaming


def mix_kda_heads(
    full: torch.Tensor,
    streaming: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Mix raw KDA outputs using one FP32 gate per recurrent/value head."""

    num_heads = _validate_mix_inputs(full, streaming, alpha)
    if alpha.numel() != num_heads:
        raise ValueError(
            "`alpha` must have one value per KDA head; "
            f"expected {num_heads}, got {alpha.numel()}.",
        )
    gate = alpha.to(dtype=full.dtype).view(1, 1, num_heads, 1)
    return gate * full + (1.0 - gate) * streaming


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int]:
    _validate_dense_float_tensor("q", q, 4)
    _validate_dense_float_tensor("k", k, 4)
    _validate_dense_float_tensor("v", v, 4)
    _validate_same_device((("q", q), ("k", k), ("v", v)))
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            "`q`, `k`, and `v` must have the same dtype; "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}.",
        )

    batch_size, seq_len, num_query_heads, head_dim = q.shape
    if k.shape != v.shape:
        raise ValueError(
            "`k` and `v` must have identical [B, T, Hkv, D] shapes; "
            f"got {tuple(k.shape)} and {tuple(v.shape)}.",
        )
    if k.shape[:2] != (batch_size, seq_len) or k.shape[3] != head_dim:
        raise ValueError(
            "`q`, `k`, and `v` must share B, T, and D; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.",
        )
    num_kv_heads = k.shape[2]
    if num_query_heads % num_kv_heads:
        raise ValueError(
            "`q` heads must be an integer multiple of KV heads for GQA; "
            f"got Hq={num_query_heads}, Hkv={num_kv_heads}.",
        )
    return seq_len, num_query_heads, num_kv_heads


def _streaming_mask(
    seq_len: int,
    sink_size: int,
    recent_size: int,
    device: torch.device,
) -> torch.Tensor:
    query_index = torch.arange(seq_len, device=device)[:, None]
    key_index = torch.arange(seq_len, device=device)[None, :]
    causal = key_index <= query_index
    return causal & (
        (key_index < sink_size)
        | (key_index >= query_index - recent_size + 1)
    )


def _get_flex_block_mask(
    *,
    seq_len: int,
    sink_size: int,
    recent_size: int,
    device: torch.device,
):
    from torch.nn.attention.flex_attention import create_block_mask

    cache_key = (str(device), seq_len, sink_size, recent_size)
    cached = _FLEX_BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def mask_mod(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch_index, head_index
        causal = key_index <= query_index
        return causal & (
            (key_index < sink_size)
            | (key_index >= query_index - recent_size + 1)
        )

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    _FLEX_BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


def streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_size: int,
    recent_size: int,
    backend: str = "torch",
) -> torch.Tensor:
    """Return raw causal sink-union-recent attention as ``[B, T, Hq, D]``.

    Inputs use ``[B, T, H, D]`` layout and may use contiguous-group GQA.
    The ``torch`` backend uses SDPA with an explicit boolean mask.  The
    ``flex`` backend requires CUDA and uses a FlexAttention block mask.
    """

    sink_size = _validate_nonnegative_integer("sink_size", sink_size)
    recent_size = _validate_positive_integer("recent_size", recent_size)
    if not isinstance(backend, str):
        raise TypeError("`backend` must be a string.")
    if backend not in {"torch", "flex"}:
        raise ValueError(
            "`backend` must be either 'torch' or 'flex'; "
            f"got {backend!r}.",
        )
    seq_len, _, _ = _validate_attention_inputs(q, k, v)

    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    if backend == "torch":
        mask = _streaming_mask(
            seq_len,
            sink_size,
            recent_size,
            q.device,
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
    else:
        if q.device.type != "cuda":
            raise RuntimeError(
                "`backend='flex'` requires q, k, and v on a CUDA device.",
            )
        try:
            from torch.nn.attention.flex_attention import (
                flex_attention,
            )
        except ImportError as exc:
            raise RuntimeError(
                "`backend='flex'` requires PyTorch FlexAttention support.",
            ) from exc

        block_mask = _get_flex_block_mask(
            seq_len=seq_len,
            sink_size=sink_size,
            recent_size=recent_size,
            device=q.device,
        )
        output = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous()


def _validate_scale(scale: float | None, key_dim: int) -> float:
    if scale is None:
        return key_dim**-0.5
    scale = _validate_finite_real("scale", scale)
    if scale <= 0.0:
        raise ValueError(f"`scale` must be positive; got {scale}.")
    return scale


def _validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    for name, tensor, ndim in (
        ("q", q, 4),
        ("k", k, 4),
        ("v", v, 4),
        ("g", g, 4),
        ("beta", beta, 3),
    ):
        _validate_dense_float_tensor(name, tensor, ndim)
    _validate_same_device(
        (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)),
    )

    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(
            "`q`, `k`, and `g` must have the same [B, T, H, K] shape; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, g={tuple(g.shape)}.",
        )
    batch_size, seq_len, num_heads, key_dim = q.shape
    if v.shape[:3] != (batch_size, seq_len, num_heads):
        raise ValueError(
            "`v` must have shape [B, T, H, V] matching q/k heads; "
            f"got q={tuple(q.shape)}, v={tuple(v.shape)}.",
        )
    if beta.shape != (batch_size, seq_len, num_heads):
        raise ValueError(
            "`beta` must have shape [B, T, H]; "
            f"expected {(batch_size, seq_len, num_heads)}, "
            f"got {tuple(beta.shape)}.",
        )
    return batch_size, seq_len, num_heads, key_dim, v.shape[-1]


def _append_kda_affine(
    p: torch.Tensor,
    r: torch.Tensor,
    token: _KDAToken,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``(A, B)`` on the left of the current affine map."""

    k_i, v_i, g_i, beta_i = token
    decay = g_i.exp()
    beta_k = beta_i.unsqueeze(-1) * k_i

    decayed_p = decay.unsqueeze(-1) * p
    projected_p = torch.einsum("bhk,bhkj->bhj", k_i, decayed_p)
    next_p = decayed_p - beta_k.unsqueeze(-1) * projected_p.unsqueeze(-2)

    decayed_r = decay.unsqueeze(-1) * r
    projected_r = torch.einsum("bhk,bhkv->bhv", k_i, decayed_r)
    next_r = decayed_r - beta_k.unsqueeze(-1) * projected_r.unsqueeze(-2)
    next_r = next_r + beta_k.unsqueeze(-1) * v_i.unsqueeze(-2)
    return next_p, next_r


def _remove_oldest_kda_affine(
    p: torch.Tensor,
    r: torch.Tensor,
    oldest: _KDAToken,
    inverse_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the rightmost/oldest affine factor using its structured inverse."""

    k_old, v_old, g_old, beta_old = oldest

    # A_old^-1 = D_old^-1
    #              (I + beta / (1 - beta ||k||^2) k k^T).
    # Since the oldest factor is rightmost in P, deletion is P @ A_old^-1.
    p_dinv = p * (-g_old).exp().unsqueeze(-2)
    denominator = 1.0 - beta_old * k_old.square().sum(dim=-1)
    epsilon = denominator.new_tensor(inverse_epsilon)
    signed_epsilon = torch.where(
        denominator < 0.0,
        -epsilon,
        epsilon,
    )
    safe_denominator = torch.where(
        denominator.abs() < epsilon,
        signed_epsilon,
        denominator,
    )
    coefficient = beta_old / safe_denominator
    p_dinv_k = torch.einsum("bhij,bhj->bhi", p_dinv, k_old)
    suffix_p = p_dinv + coefficient[..., None, None] * (
        p_dinv_k.unsqueeze(-1) * k_old.unsqueeze(-2)
    )

    # R = P_suffix B_old + R_suffix and
    # B_old = beta_old k_old v_old^T.
    suffix_p_k = torch.einsum("bhij,bhj->bhi", suffix_p, k_old)
    transformed_write = (
        (beta_old.unsqueeze(-1) * suffix_p_k).unsqueeze(-1)
        * v_old.unsqueeze(-2)
    )
    suffix_r = r - transformed_write
    return suffix_p, suffix_r


def _kda_append_block(
    p: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for offset in range(q.shape[1]):
        token = (k[:, offset], v[:, offset], g[:, offset], beta[:, offset])
        p, r = _append_kda_affine(p, r, token)
        outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                q[:, offset] * scale,
                r,
            ),
        )
    return p, r, torch.stack(outputs, dim=1)


def _kda_slide_block(
    p: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    old_k: torch.Tensor,
    old_v: torch.Tensor,
    old_g: torch.Tensor,
    old_beta: torch.Tensor,
    scale: torch.Tensor,
    inverse_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for offset in range(q.shape[1]):
        oldest = (
            old_k[:, offset],
            old_v[:, offset],
            old_g[:, offset],
            old_beta[:, offset],
        )
        p, r = _remove_oldest_kda_affine(
            p,
            r,
            oldest,
            inverse_epsilon,
        )
        token = (k[:, offset], v[:, offset], g[:, offset], beta[:, offset])
        p, r = _append_kda_affine(p, r, token)
        outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                q[:, offset] * scale,
                r,
            ),
        )
    return p, r, torch.stack(outputs, dim=1)


def _kda_query_lag_block(
    z: torch.Tensor,
    output: torch.Tensor,
    k_span: torch.Tensor,
    v_span: torch.Tensor,
    g_span: torch.Tensor,
    beta_span: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one fixed-width band of recent writes for query blocks."""

    query_length = z.shape[1]
    lag_count = k_span.shape[1] - query_length + 1
    for lag_offset in range(lag_count):
        source_offset = lag_count - 1 - lag_offset
        source_slice = slice(source_offset, source_offset + query_length)
        k_i = k_span[:, source_slice]
        v_i = v_span[:, source_slice]
        g_i = g_span[:, source_slice]
        beta_i = beta_span[:, source_slice]
        coefficient = beta_i * (z * k_i).sum(dim=-1)
        output = output + coefficient.unsqueeze(-1) * v_i
        z = g_i.exp() * (z - coefficient.unsqueeze(-1) * k_i)
    return z, output


def _compiled_kda_block(kind: str):
    global _COMPILED_KDA_APPEND_BLOCK
    global _COMPILED_KDA_QUERY_LAG_BLOCK
    global _COMPILED_KDA_SLIDE_BLOCK

    if kind == "append":
        if _COMPILED_KDA_APPEND_BLOCK is None:
            _COMPILED_KDA_APPEND_BLOCK = torch.compile(
                _kda_append_block,
                fullgraph=True,
                dynamic=False,
            )
        return _COMPILED_KDA_APPEND_BLOCK
    if kind == "slide":
        if _COMPILED_KDA_SLIDE_BLOCK is None:
            _COMPILED_KDA_SLIDE_BLOCK = torch.compile(
                _kda_slide_block,
                fullgraph=True,
                dynamic=False,
            )
        return _COMPILED_KDA_SLIDE_BLOCK
    if kind == "query_lag":
        if _COMPILED_KDA_QUERY_LAG_BLOCK is None:
            _COMPILED_KDA_QUERY_LAG_BLOCK = torch.compile(
                _kda_query_lag_block,
                fullgraph=True,
                dynamic=False,
            )
        return _COMPILED_KDA_QUERY_LAG_BLOCK
    raise ValueError(f"Unknown KDA block kind: {kind!r}.")


def _run_kda_block(kind: str, *inputs):
    operator = _compiled_kda_block(kind)
    if torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad
        for tensor in inputs
    ):
        return checkpoint(
            operator,
            *inputs,
            use_reentrant=False,
        )
    return operator(*inputs)


def _pad_time_block(
    tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    missing = block_size - tensor.shape[1]
    if missing <= 0:
        return tensor
    padding = tensor.new_zeros(
        tensor.shape[0],
        missing,
        *tensor.shape[2:],
    )
    return torch.cat((tensor, padding), dim=1)


def _rebuild_kda_affine(
    window: deque[_KDAToken],
    identity: torch.Tensor,
    zero_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompose a window oldest-to-newest without inverse deletion."""

    p = identity
    r = zero_state
    for token in window:
        p, r = _append_kda_affine(p, r, token)
    return p, r


def _rebuild_kda_affine_cuda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    start: int,
    end: int,
    identity: torch.Tensor,
    zero_state: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p, r = identity, zero_state
    position = start
    while position < end:
        block_end = min(position + block_size, end)
        actual_size = block_end - position
        block_inputs = tuple(
            _pad_time_block(tensor[:, position:block_end], block_size)
            for tensor in (q, k, v, g, beta)
        )
        p, r, _ = _run_kda_block(
            "append",
            p,
            r,
            *block_inputs,
            scale,
        )
        position += actual_size
    return p, r


def _zero_pad_time(
    tensor: torch.Tensor,
    *,
    left: int = 0,
    right: int = 0,
) -> torch.Tensor:
    pieces = []
    if left:
        pieces.append(
            tensor.new_zeros(
                tensor.shape[0],
                left,
                *tensor.shape[2:],
            ),
        )
    pieces.append(tensor)
    if right:
        pieces.append(
            tensor.new_zeros(
                tensor.shape[0],
                right,
                *tensor.shape[2:],
            ),
        )
    return torch.cat(pieces, dim=1) if len(pieces) > 1 else tensor


def _kda_recent_query_cuda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recent_size: int,
    start_index: int,
    scale: float,
    block_size: int,
    zero_anchor: torch.Tensor,
) -> torch.Tensor:
    """Evaluate exact recent KDA writes without inverse window deletion."""

    batch_size, seq_len, num_heads, _ = q.shape
    value_dim = v.shape[-1]
    active_length = seq_len - start_index
    if active_length == 0:
        return v.new_zeros(
            batch_size,
            seq_len,
            num_heads,
            value_dim,
        ) + zero_anchor

    effective_window = min(recent_size, active_length)
    right_padding = (-active_length) % block_size
    padded_length = active_length + right_padding
    q_active = _zero_pad_time(q[:, start_index:], right=right_padding)
    history_padding = effective_window - 1
    history_tensors = tuple(
        _zero_pad_time(
            tensor[:, start_index:],
            left=history_padding,
            right=right_padding,
        )
        for tensor in (k, v, g, beta)
    )
    scale_tensor = q.new_tensor(scale)
    active_outputs = []
    for output_start in range(0, padded_length, block_size):
        z = q_active[:, output_start : output_start + block_size] * scale_tensor
        block_output = v.new_zeros(
            batch_size,
            block_size,
            num_heads,
            value_dim,
        )
        lag_start = 0
        while lag_start < effective_window:
            lag_count = min(block_size, effective_window - lag_start)
            source_start = (
                history_padding
                + output_start
                - lag_start
                - lag_count
                + 1
            )
            source_end = source_start + block_size + lag_count - 1
            lag_inputs = tuple(
                tensor[:, source_start:source_end]
                for tensor in history_tensors
            )
            z, block_output = _run_kda_block(
                "query_lag",
                z,
                block_output,
                *lag_inputs,
            )
            lag_start += lag_count
        active_outputs.append(block_output)

    active_output = torch.cat(active_outputs, dim=1)[:, :active_length]
    if start_index:
        prefix = v.new_zeros(
            batch_size,
            start_index,
            num_heads,
            value_dim,
        )
        output = torch.cat((prefix, active_output), dim=1)
    else:
        output = active_output
    return output + zero_anchor


def _kda_recent_affine_cuda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recent_size: int,
    start_index: int,
    scale: float,
    inverse_epsilon: float,
    rebase_interval: int,
    block_size: int,
    identity: torch.Tensor,
    zero_state: torch.Tensor,
    zero_output: torch.Tensor,
    unsafe_times: set[int],
) -> torch.Tensor:
    scale_tensor = q.new_tensor(scale)
    outputs: list[torch.Tensor] = []
    if start_index:
        outputs.append(
            zero_output.unsqueeze(1).expand(
                -1,
                start_index,
                -1,
                -1,
            ),
        )

    p, r = identity, zero_state
    position = start_index
    warmup_end = min(start_index + recent_size, q.shape[1])
    while position < warmup_end:
        block_end = min(position + block_size, warmup_end)
        actual_size = block_end - position
        block_inputs = tuple(
            _pad_time_block(tensor[:, position:block_end], block_size)
            for tensor in (q, k, v, g, beta)
        )
        p, r, block_output = _run_kda_block(
            "append",
            p,
            r,
            *block_inputs,
            scale_tensor,
        )
        outputs.append(block_output[:, :actual_size])
        position = block_end

    eviction_count = 0
    unsafe_eviction_positions = sorted(
        token_index + recent_size
        for token_index in unsafe_times
        if token_index >= start_index
        and token_index + recent_size < q.shape[1]
    )
    unsafe_cursor = 0

    while position < q.shape[1]:
        while (
            unsafe_cursor < len(unsafe_eviction_positions)
            and unsafe_eviction_positions[unsafe_cursor] < position
        ):
            unsafe_cursor += 1
        next_unsafe = (
            unsafe_eviction_positions[unsafe_cursor]
            if unsafe_cursor < len(unsafe_eviction_positions)
            else q.shape[1]
        )

        if position == next_unsafe:
            p, r = _rebuild_kda_affine_cuda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                start=position - recent_size + 1,
                end=position,
                identity=identity,
                zero_state=zero_state,
                scale=scale_tensor,
                block_size=block_size,
            )
            block_inputs = tuple(
                _pad_time_block(tensor[:, position:position + 1], block_size)
                for tensor in (q, k, v, g, beta)
            )
            p, r, block_output = _run_kda_block(
                "append",
                p,
                r,
                *block_inputs,
                scale_tensor,
            )
            outputs.append(block_output[:, :1])
            position += 1
            eviction_count += 1
            unsafe_cursor += 1
        else:
            actual_size = min(
                block_size,
                q.shape[1] - position,
                next_unsafe - position,
            )
            if rebase_interval:
                actual_size = min(
                    actual_size,
                    rebase_interval
                    - eviction_count % rebase_interval,
                )
            block_end = position + actual_size
            current_inputs = tuple(
                _pad_time_block(tensor[:, position:block_end], block_size)
                for tensor in (q, k, v, g, beta)
            )
            old_start = position - recent_size
            old_end = block_end - recent_size
            old_inputs = tuple(
                _pad_time_block(tensor[:, old_start:old_end], block_size)
                for tensor in (k, v, g, beta)
            )
            p, r, block_output = _run_kda_block(
                "slide",
                p,
                r,
                *current_inputs,
                *old_inputs,
                scale_tensor,
                inverse_epsilon,
            )
            outputs.append(block_output[:, :actual_size])
            position = block_end
            eviction_count += actual_size

        if (
            rebase_interval
            and eviction_count
            and eviction_count % rebase_interval == 0
            and position < q.shape[1]
        ):
            p, r = _rebuild_kda_affine_cuda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                start=max(start_index, position - recent_size),
                end=position,
                identity=identity,
                zero_state=zero_state,
                scale=scale_tensor,
                block_size=block_size,
            )

    return torch.cat(outputs, dim=1)


def kda_recent_affine(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recent_size: int,
    start_index: int = 0,
    scale: float | None = None,
    inverse_epsilon: float = 1e-4,
    rebase_interval: int = 1024,
    checkpoint_block_size: int = 64,
) -> torch.Tensor:
    """Return only KDA's recent-window ``B`` contribution.

    Inputs follow the KDA kernel layout: ``q``, ``k``, and ``g`` are
    ``[B, T, H, K]``, ``v`` is ``[B, T, H, V]``, and ``beta`` is
    ``[B, T, H]``.  The caller must supply already L2-normalized ``q`` and
    ``k``.  Affine window maps are maintained in FP32 without detaching any
    token, so gradients flow through ``q/k/v/g/beta``.

    Outputs before ``start_index`` are zero.  Thereafter, only writes at or
    after ``start_index`` and inside the current recent window are included.
    This is not complete sink-enabled streaming KDA: the layer computes sink
    state with a separate masked-value chunk scan and adds it to this result.

    CUDA uses an exact inverse-free reverse-query scan, checkpointed in fixed
    blocks. ``inverse_epsilon`` and ``rebase_interval`` apply to the eager
    fallback, where the interval counts left evictions and zero disables
    rebuilding.
    """

    shape = _validate_kda_inputs(q, k, v, g, beta)
    batch_size, seq_len, num_heads, key_dim, value_dim = shape
    recent_size = _validate_positive_integer("recent_size", recent_size)
    start_index = _validate_nonnegative_integer("start_index", start_index)
    if start_index > seq_len:
        raise ValueError(
            f"`start_index` must not exceed sequence length {seq_len}; "
            f"got {start_index}.",
        )
    scale = _validate_scale(scale, key_dim)
    inverse_epsilon = _validate_inverse_epsilon(inverse_epsilon)
    rebase_interval = _validate_nonnegative_integer(
        "rebase_interval",
        rebase_interval,
    )
    checkpoint_block_size = _validate_nonnegative_integer(
        "checkpoint_block_size",
        checkpoint_block_size,
    )

    # Explicitly disable surrounding autocast: P and R must remain FP32 even
    # when this operator is called from a mixed-precision model forward.
    with torch.autocast(device_type=q.device.type, enabled=False):
        q_work = q.float()
        k_work = k.float()
        v_work = v.float()
        g_work = g.float()
        beta_work = beta.float()

        identity = torch.eye(
            key_dim,
            dtype=torch.float32,
            device=q.device,
        ).view(1, 1, key_dim, key_dim)
        identity = identity.expand(
            batch_size,
            num_heads,
            key_dim,
            key_dim,
        )
        zero_state = torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
        p = identity
        r = zero_state
        window: deque[_KDAToken] = deque()
        outputs: list[torch.Tensor] = []
        eviction_count = 0

        # Keep an explicit zero-valued dependency on every input.  In the
        # all-prefix case (start_index == T), autograd still returns defined
        # zero gradients for q/k/v/g/beta instead of reporting unused inputs.
        zero_anchor = (
            q_work.sum()
            + k_work.sum()
            + v_work.sum()
            + g_work.sum()
            + beta_work.sum()
        ) * 0.0
        zero_output = torch.zeros(
            batch_size,
            num_heads,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        ) + zero_anchor

        # Structured left deletion requires A_old to be safely invertible.
        # Resolve all unsafe token positions once to avoid a device sync at
        # every eviction. When such a token reaches the left edge, rebuild the
        # remaining window exactly instead of pretending a clamped denominator
        # is an inverse.
        if q.device.type == "cuda" and checkpoint_block_size:
            output = _kda_recent_query_cuda(
                q=q_work,
                k=k_work,
                v=v_work,
                g=g_work,
                beta=beta_work,
                recent_size=recent_size,
                start_index=start_index,
                scale=scale,
                block_size=checkpoint_block_size,
                zero_anchor=zero_anchor,
            )
            return output.to(v.dtype)

        denominator = 1.0 - beta_work * k_work.square().sum(dim=-1)
        inverse_decay = (-g_work).exp()
        unsafe = (
            ~torch.isfinite(denominator)
            | (denominator.abs() < inverse_epsilon)
            | ~torch.isfinite(inverse_decay).all(dim=-1)
        )
        unsafe_times = set(
            unsafe.any(dim=(0, 2)).nonzero().flatten().detach().cpu().tolist()
        )

        for index in range(seq_len):
            if index < start_index:
                outputs.append(zero_output)
                continue

            if len(window) == recent_size:
                oldest = window.popleft()
                oldest_index = index - recent_size
                if oldest_index in unsafe_times:
                    p, r = _rebuild_kda_affine(
                        window,
                        identity,
                        zero_state,
                    )
                else:
                    p, r = _remove_oldest_kda_affine(
                        p,
                        r,
                        oldest,
                        inverse_epsilon,
                    )
                eviction_count += 1

            token = (
                k_work[:, index],
                v_work[:, index],
                g_work[:, index],
                beta_work[:, index],
            )
            window.append(token)
            p, r = _append_kda_affine(p, r, token)

            if (
                rebase_interval
                and eviction_count
                and eviction_count % rebase_interval == 0
            ):
                p, r = _rebuild_kda_affine(
                    window,
                    identity,
                    zero_state,
                )

            output = torch.einsum(
                "bhk,bhkv->bhv",
                q_work[:, index] * scale,
                r,
            )
            outputs.append(output)

        return torch.stack(outputs, dim=1).to(v.dtype)
