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
