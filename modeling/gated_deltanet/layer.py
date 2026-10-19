# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import logging

from fla.layers.gated_deltanet import GatedDeltaNet as _UpstreamGatedDeltaNet
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.modules.l2norm import l2_norm
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class GatedDeltaNetRoPE(_UpstreamGatedDeltaNet):
    """A thin wrapper around `fla.layers.gated_deltanet.GatedDeltaNet` that
    optionally applies Rotary Position Embedding (RoPE) on the query/key
    tensors before the gated delta rule kernel, and/or replaces the
    data-driven forget-gate decay with a fixed per-head decay schedule, and/or
    caps the per-token decay factor.

    When ``use_rope=False``, ``use_fixed_decay=False`` and
    ``decay_clip=None`` (defaults), the layer is identical to the upstream
    implementation (delegates to ``super().forward``). When ``use_rope=True``,
    RoPE is applied right after the short-conv / projection step (and after
    optional GVA head-repeat), followed by an explicit L2 normalization on
    q/k; the kernel is then called with ``use_qk_l2norm_in_kernel=False`` to
    avoid double-normalizing the rotated tensors.

    When ``use_fixed_decay=True``, the per-token / per-head log-decay ``g``
    is replaced by the fixed HypeNet / Lightning Attention schedule
    ``gamma_h = exp(-2**(-8*h/H))`` for ``h = 1..H``, where ``H = num_v_heads``;
    ``a_proj``, ``A_log`` and ``dt_bias`` are then unused at runtime (kept on
    the module purely for checkpoint compatibility, they receive no gradient).

    The behavior mirrors the reference implementation in
    ``halo/modeling/hypenet/gdn.py`` (chenyingfa@hypenet).
    """

    def __init__(
        self,
        *args,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        use_fixed_decay: bool = False,
        decay_clip: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_rope = use_rope
        self.rope_theta = float(rope_theta)
        self.rope_max_position_embeddings = max_position_embeddings
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                dim=self.head_k_dim,
                base=self.rope_theta,
            )

        # Fixed (non-data-driven) forget-gate decay schedule from HypeNet /
        # Lightning Attention:
        #   gamma_h = exp(-2**(-8*h/H)),  h = 1, ..., H = num_v_heads
        # The kernel API expects the *log*-decay `g` (it applies `exp(g)` per
        # step internally), and the data-driven path feeds `g` in fp32, so we
        # do the same: store only the integer head indices here and recompute
        # `g = -2**(-8*h/H)` in fp32 inside `forward`.
        #
        # We deliberately store the head index as a `long` buffer (not float):
        # `model.to(dtype)` casts floating-point buffers (so a float buffer
        # would be down-cast to bf16, losing precision for H where 2**(-8h/H)
        # is not a power of two, e.g. H=32), but it leaves integer buffers
        # untouched. Recomputing in fp32 from the exact integer indices keeps
        # full precision regardless of the model's compute dtype.
        self.use_fixed_decay = use_fixed_decay
        if self.use_fixed_decay:
            head_idx = torch.arange(1, self.num_v_heads + 1, dtype=torch.long)
            # persistent=False: regenerated from the formula every __init__,
            # no need to keep it in the state dict.
            self.register_buffer('fixed_decay_head_idx', head_idx, persistent=False)

            # With a fixed decay, the data-driven decay parameters (`a_proj`,
            # `A_log`, `dt_bias`) no longer participate in the forward. Disable
            # their gradients so DDP does not raise "parameters that were not
            # used in producing loss" (they are kept in the module/state dict
            # for checkpoint compatibility, but receive no gradient/update).
            self.a_proj.weight.requires_grad_(False)
            self.A_log.requires_grad_(False)
            self.dt_bias.requires_grad_(False)

        if decay_clip is not None:
            decay_clip = float(decay_clip)
            if not 0.0 < decay_clip <= 1.0:
                raise ValueError("`decay_clip` must be greater than 0 and at most 1.")
        self.decay_clip = math.log(decay_clip) if decay_clip is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: 'Cache | None' = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        # Fast path: with no custom decay behavior or RoPE we are identical to
        # the upstream `GatedDeltaNet`, so just delegate.
        if not self.use_rope and not self.use_fixed_decay and self.decay_clip is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is None and attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(
                lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads),
                (q, k),
            )

        if self.use_rope:
            seqlen_offset = 0
            max_seqlen = q.shape[1]
            if past_key_values is not None:
                if last_state is not None and '_rope_seq_offset' in last_state:
                    seqlen_offset = last_state['_rope_seq_offset']
                else:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset
                # Per-sequence offset adjustment for batches with padding.
                # Only apply on decode steps (last_state exists) — on the
                # first prefill the base offset is 0 and cu_seqlens already
                # gives correct per-sequence positioning; applying the
                # adjustment there would produce negative offsets for padded
                # (shorter) sequences.
                if attention_mask is not None and last_state is not None:
                    seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                    max_seqlen = q.shape[1] + max(seqlen_offset)
            if self.rope_max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.rope_max_position_embeddings)

            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
            q = l2_norm(q)
            k = l2_norm(k)
            use_qk_l2norm_in_kernel = False
        else:
            # No RoPE: q/k are not rotated and have not been L2-normed yet, so
            # let the kernel do the L2-norm internally (same as upstream).
            use_qk_l2norm_in_kernel = True

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        if self.use_fixed_decay:
            # Recompute the per-head log-decay in fp32 from the exact integer
            # head indices:  g_h = -2**(-8*h/H),  h = 1..H = num_v_heads.
            # This is the *log*-decay (pre-exp), matching the kernel API: the
            # data-driven branch below also passes a fp32 log-space `g`, and
            # the kernel applies `exp(g)` per step internally.
            h_idx = self.fixed_decay_head_idx.to(torch.float32)  # (H,) fp32
            log_decay = -torch.pow(2.0, -8.0 * h_idx / float(self.num_v_heads))
            # Broadcast (H,) -> (B, T, H). `.contiguous()` materializes real
            # strides for the Triton kernels.
            g = log_decay.expand(hidden_states.shape[0], hidden_states.shape[1], -1).contiguous()
        else:
            g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if self.decay_clip is not None:
            g = g.clamp(max=self.decay_clip)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            state = past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )
            if self.use_rope:
                prev_offset = last_state.get('_rope_seq_offset', 0) if last_state is not None else 0
                state['_rope_seq_offset'] = prev_offset + q_len

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class NoPEAttention(nn.Module):

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
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_output_gate = use_output_gate

        self.logit_scaling_a = logit_scaling_a
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.use_output_gate:
            self.z_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)
        
        self.max_position_embeddings = None
        self.window_size = None
        if kwargs.get('window_size') is not None:
            raise NotImplementedError(
                "`NoPEAttention` does not support positional encoding. "
                "Please remove `window_size`."
            )

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_output_gate:
            attn_output = attn_output * torch.sigmoid(self.z_proj(hidden_states))
        return attn_output

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
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.logit_scaling_a is not None:
            a = self.logit_scaling_a
            # t: position indices starting from 0; scaling factor = log_a(t + a)
            # Using change-of-base: log_a(x) = ln(x) / ln(a)
            offset = seqlen_offset if isinstance(seqlen_offset, int) else seqlen_offset.unsqueeze(-1)
            t = torch.arange(q_len, device=q.device, dtype=q.dtype) + offset
            scaling = torch.log(t + a) / math.log(a)  # (B, T) or (T,)
            if scaling.dim() == 1:
                scaling = scaling.unsqueeze(0)
            # shape: (B, T, 1, 1) for broadcast over heads and head_dim
            q = q * scaling.unsqueeze(-1).unsqueeze(-1)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Contains at least one padding token in the sequence
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
                # Cached chunked prefill has a multi-token query covering the
                # tail of a longer K/V timeline. FLA's generic helper supports
                # only q_len == k_len or q_len == 1, so unpad the query tail
                # and complete key timeline independently.
                indices_k, cu_seqlens_k, max_seqlen_k = get_unpad_data(attention_mask)
                query_mask = attention_mask[:, -q_len:]
                indices_q, cu_seqlens_q, max_seqlen_q = get_unpad_data(query_mask)
                q = index_first_axis(
                    rearrange(q, 'b s ... -> (b s) ...'),
                    indices_q,
                )
                k, v = (
                    index_first_axis(
                        rearrange(state, 'b s ... -> (b s) ...'),
                        indices_k,
                    )
                    for state in (k, v)
                )
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
        o = o.reshape(batch_size, q_len, -1)
        o = self._apply_output_gate(o, hidden_states)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
