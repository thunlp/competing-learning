from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.utils import (
    get_unpad_data,
    index_first_axis,
    pad_input,
    unpad_input,
)
from fla.modules import RMSNorm
from fla.ops.utils.index import prepare_lens_from_mask

from .duo import (
    make_duo_alpha,
    mix_attention_heads,
    streaming_attention,
    validate_duo_hyperparameters,
)

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via "
        "`pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None


class NoPEAttention(nn.Module):
    """Repository hybrid attention without positional encoding."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_output_gate: bool = False,
        logit_scaling_a: float | None = None,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_output_gate = use_output_gate
        self.logit_scaling_a = logit_scaling_a
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via "
                "`pip install flash-attn --no-build-isolation` first",
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.qkv_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_dim,
            bias=self.qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_dim,
            bias=self.qkv_bias,
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.use_output_gate:
            self.z_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=self.qkv_bias,
            )

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.max_position_embeddings = None
        self.window_size = None
        if kwargs.get("window_size") is not None:
            raise NotImplementedError(
                "`NoPEAttention` does not support positional encoding. "
                "Please remove `window_size`.",
            )

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_output_gate:
            attn_output = attn_output * torch.sigmoid(self.z_proj(hidden_states))
        return attn_output

    def enable_duo_attention(
        self,
        *,
        sink_size: int,
        recent_size: int,
        alpha_init: float,
        backend: str = "flex",
    ) -> None:
        """Inject one FP32 DuoAttention gate per KV head."""

        validate_duo_hyperparameters(sink_size, recent_size, alpha_init)
        if backend not in {"flex", "torch"}:
            raise ValueError(
                "`backend` must be either 'flex' or 'torch'; "
                f"received {backend!r}.",
            )
        if "duo_alpha" in self._parameters:
            raise RuntimeError(
                f"DuoAttention is already enabled for layer {self.layer_idx}.",
            )
        self.register_parameter(
            "duo_alpha",
            make_duo_alpha(
                self.num_kv_heads,
                alpha_init,
                device=self.q_proj.weight.device,
            ),
        )
        self.duo_alpha._no_weight_decay = True
        self.duo_sink_size = int(sink_size)
        self.duo_recent_size = int(recent_size)
        self.duo_streaming_backend = backend

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        alpha = self._parameters.get("duo_alpha")
        if alpha is not None:
            if alpha.dtype != torch.float32:
                alpha.data = alpha.data.float()
                if alpha.grad is not None:
                    alpha.grad.data = alpha.grad.data.float()
            alpha._no_weight_decay = True
        return module

    def _duo_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        if "duo_alpha" not in self._parameters:
            raise RuntimeError(
                "DuoAttention mode was requested before alpha injection.",
            )
        if mode not in {"full", "mixed", "two_way"}:
            raise ValueError(
                "`duo_mode` must be 'full', 'mixed', or 'two_way'; "
                f"received {mode!r}.",
            )

        full_output = flash_attn_func(
            q,
            k,
            v,
            causal=True,
            window_size=(-1, -1),
        )
        if mode == "full":
            return full_output

        if mode == "two_way":
            if q.shape[0] % 2:
                raise ValueError(
                    "DuoAttention two-way forward requires an even batch size; "
                    f"received {q.shape[0]}.",
                )
            student_start = q.shape[0] // 2
        else:
            student_start = 0

        streaming_output = streaming_attention(
            q[student_start:],
            k[student_start:],
            v[student_start:],
            sink_size=self.duo_sink_size,
            recent_size=self.duo_recent_size,
            backend=self.duo_streaming_backend,
        )
        mixed_output = mix_attention_heads(
            full_output[student_start:],
            streaming_output,
            self.duo_alpha,
        )
        if mode == "mixed":
            return mixed_output
        return torch.cat((full_output[:student_start], mixed_output), dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape "
                "[batch_size, seq_len] for padding purposes (0 indicating "
                "padding). Arbitrary attention masks of shape "
                "[batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(
            self.q_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.head_dim,
        )
        k = rearrange(
            self.k_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.head_dim,
        )
        v = rearrange(
            self.v_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.head_dim,
        )

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get("cu_seqlens")

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = (
                    seqlen_offset
                    + prepare_lens_from_mask(attention_mask)
                    - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.logit_scaling_a is not None:
            a = self.logit_scaling_a
            offset = (
                seqlen_offset
                if isinstance(seqlen_offset, int)
                else seqlen_offset.unsqueeze(-1)
            )
            positions = torch.arange(
                q_len,
                device=q.device,
                dtype=q.dtype,
            ) + offset
            scaling = torch.log(positions + a) / math.log(a)
            if scaling.dim() == 1:
                scaling = scaling.unsqueeze(0)
            q = q * scaling.unsqueeze(-1).unsqueeze(-1)

        duo_mode = kwargs.get("duo_mode")
        if duo_mode is not None:
            if attention_mask is not None:
                raise ValueError(
                    "DuoAttention distillation requires fixed-length examples "
                    "without an attention mask.",
                )
            if past_key_values is not None or use_cache:
                raise ValueError(
                    "DuoAttention distillation does not support KV caching.",
                )
            if cu_seqlens is not None:
                raise ValueError(
                    "DuoAttention distillation does not support packed "
                    "`cu_seqlens` batches.",
                )
            output = self._duo_attention_forward(
                q,
                k,
                v,
                mode=duo_mode,
            )
            output = output.reshape(batch_size, q_len, -1)
            output = self._apply_output_gate(output, hidden_states)
            output = self.o_proj(output)
            return output, None, past_key_values

        if past_key_values is not None:
            cache_has_content = (
                past_key_values.get_seq_length(self.layer_idx) > 0
            )
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs={"window_size": self.window_size},
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(
                    k,
                    "... (h d) -> ... h d",
                    d=self.head_dim,
                )
                v = rearrange(
                    v,
                    "... (h d) -> ... h d",
                    d=self.head_dim,
                )

        # Accelerate may cast cache state to FP32 while the current query
        # remains BF16/FP16. FlashAttention requires Q/K/V to share a dtype.
        if k.dtype != q.dtype:
            k = k.to(dtype=q.dtype)
        if v.dtype != q.dtype:
            v = v.to(dtype=q.dtype)

        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            if q_len in (1, k.shape[1]):
                q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                    q,
                    (k, v),
                    attention_mask,
                    q_len,
                )
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
            else:
                indices_k, cu_seqlens_k, max_seqlen_k = get_unpad_data(
                    attention_mask,
                )
                query_mask = attention_mask[:, -q_len:]
                indices_q, cu_seqlens_q, max_seqlen_q = get_unpad_data(
                    query_mask,
                )
                q = index_first_axis(
                    rearrange(q, "b s ... -> (b s) ..."),
                    indices_q,
                )
                k, v = (
                    index_first_axis(
                        rearrange(state, "b s ... -> (b s) ..."),
                        indices_k,
                    )
                    for state in (k, v)
                )
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            )
            output = pad_input(output, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            if cu_seqlens.ndim != 1:
                raise ValueError(
                    "`cu_seqlens` must be one-dimensional for NoPEAttention; "
                    f"received shape {tuple(cu_seqlens.shape)}.",
                )
            segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(segment_lengths.max().item())
            output = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            output = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            )
        output = output.reshape(batch_size, q_len, -1)
        output = self._apply_output_gate(output, hidden_states)
        output = self.o_proj(output)

        attentions = None if not output_attentions else None
        return output, attentions, past_key_values
