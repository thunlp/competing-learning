# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.kda import KimiDeltaAttention as _UpstreamKimiDeltaAttention
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RotaryEmbedding
from fla.modules.l2norm import l2_norm
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.utils.index import prepare_lens_from_mask

from .duo import (
    kda_recent_affine,
    make_duo_alpha,
    mix_kda_heads,
    validate_duo_hyperparameters,
)

if TYPE_CHECKING:
    from fla.models.utils import Cache


class KimiDeltaAttentionRoPE(_UpstreamKimiDeltaAttention):
    """KDA with the same optional extensions used by the local GDN layer.

    With all options disabled this class delegates to FLA's
    :class:`KimiDeltaAttention` unchanged. The custom path can apply RoPE to
    query/key tensors, replace KDA's learned channel-wise decay with a fixed
    per-head schedule, and cap the per-token decay factor.

    KDA normally learns a separate decay for every key channel. Fixed decay
    intentionally broadcasts the scalar head schedule across those channels.
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
            if self.head_k_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires an even per-head key dimension, got {self.head_k_dim}",
                )
            self.rotary = RotaryEmbedding(
                dim=self.head_k_dim,
                base=self.rope_theta,
            )

        self.use_fixed_decay = use_fixed_decay
        if self.use_fixed_decay:
            head_idx = torch.arange(1, self.num_v_heads + 1, dtype=torch.long)
            self.register_buffer("fixed_decay_head_idx", head_idx, persistent=False)

            # The learned channel-wise gate is not used by the fixed schedule.
            # Keep the tensors for checkpoint compatibility but exclude them
            # from optimization and DDP unused-parameter checks.
            for parameter in self.f_proj.parameters():
                parameter.requires_grad_(False)
            self.A_log.requires_grad_(False)
            self.dt_bias.requires_grad_(False)

        if decay_clip is not None:
            decay_clip = float(decay_clip)
            if not 0.0 < decay_clip <= 1.0:
                raise ValueError("`decay_clip` must be greater than 0 and at most 1.")
        self.decay_clip = math.log(decay_clip) if decay_clip is not None else None

    def enable_duo_attention(
        self,
        *,
        sink_size: int,
        recent_size: int,
        alpha_init: float,
        kda_use_sink: bool,
        inverse_epsilon: float = 1e-4,
        rebase_interval: int = 1024,
        scan_block_size: int = 64,
    ) -> None:
        """Inject one FP32 DuoAttention gate per recurrent/value head."""

        validate_duo_hyperparameters(
            sink_size,
            recent_size,
            alpha_init,
            inverse_epsilon=inverse_epsilon,
            rebase_interval=rebase_interval,
        )
        if not isinstance(kda_use_sink, bool):
            raise TypeError(
                "`kda_use_sink` must be a bool; "
                f"received {kda_use_sink!r}.",
            )
        if isinstance(scan_block_size, bool) or not isinstance(
            scan_block_size,
            int,
        ):
            raise TypeError("`scan_block_size` must be an integer.")
        if scan_block_size < 0:
            raise ValueError("`scan_block_size` must be nonnegative.")
        if "duo_alpha" in self._parameters:
            raise RuntimeError(
                f"DuoAttention is already enabled for layer {self.layer_idx}.",
            )
        self.register_parameter(
            "duo_alpha",
            make_duo_alpha(
                self.num_v_heads,
                alpha_init,
                device=self.q_proj.weight.device,
            ),
        )
        self.duo_alpha._no_weight_decay = True
        self.duo_sink_size = int(sink_size)
        self.duo_recent_size = int(recent_size)
        self.duo_kda_use_sink = kda_use_sink
        self.duo_inverse_epsilon = float(inverse_epsilon)
        self.duo_rebase_interval = int(rebase_interval)
        self.duo_scan_block_size = scan_block_size

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

    def _duo_kda_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
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

        full_output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
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

        q_student = q[student_start:]
        k_student = k[student_start:]
        v_student = v[student_start:]
        g_student = g[student_start:]
        beta_student = beta[student_start:]
        full_student = full_output[student_start:]
        seq_len = q_student.shape[1]

        recent_start = 0
        if self.duo_kda_use_sink and self.duo_sink_size > 0:
            sink_end = min(self.duo_sink_size, seq_len)
            if sink_end < seq_len:
                sink_values = torch.cat(
                    (
                        v_student[:, :sink_end],
                        torch.zeros_like(v_student[:, sink_end:]),
                    ),
                    dim=1,
                )
                sink_output, _ = chunk_kda(
                    q=q_student,
                    k=k_student,
                    v=sink_values,
                    g=g_student,
                    beta=beta_student,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=False,
                )
                sink_output = sink_output.float()
            else:
                sink_output = full_student.float()
            recent_start = sink_end
        else:
            sink_output = torch.zeros_like(
                full_student,
                dtype=torch.float32,
            )

        recent_output = kda_recent_affine(
            q_student,
            k_student,
            v_student,
            g_student,
            beta_student,
            recent_size=self.duo_recent_size,
            start_index=recent_start,
            inverse_epsilon=self.duo_inverse_epsilon,
            rebase_interval=self.duo_rebase_interval,
            checkpoint_block_size=self.duo_scan_block_size,
        )
        streaming_output = (
            sink_output + recent_output.float()
        ).to(full_student.dtype)
        mixed_output = mix_kda_heads(
            full_student,
            streaming_output,
            self.duo_alpha,
        )
        if mode == "mixed":
            return mixed_output
        return torch.cat((full_output[:student_start], mixed_output), dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        duo_mode = kwargs.get("duo_mode")
        if (
            duo_mode is None
            and not self.use_rope
            and not self.use_fixed_decay
            and self.decay_clip is None
        ):
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
        if duo_mode is not None:
            if attention_mask is not None:
                raise ValueError(
                    "DuoAttention distillation requires fixed-length examples "
                    "without an attention mask.",
                )
            if past_key_values is not None or use_cache:
                raise ValueError(
                    "DuoAttention distillation does not support recurrent caching.",
                )
            if kwargs.get("cu_seqlens") is not None:
                raise ValueError(
                    "DuoAttention distillation does not support packed "
                    "`cu_seqlens` batches.",
                )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        indices = None
        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."),
                indices,
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
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

        q, k = (
            rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim)
            for x in (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.use_rope:
            seqlen_offset = 0
            max_seqlen = q.shape[1]
            if past_key_values is not None:
                if last_state is not None and "_rope_seq_offset" in last_state:
                    seqlen_offset = last_state["_rope_seq_offset"]
                else:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset
                if attention_mask is not None and last_state is not None:
                    seqlen_offset = (
                        seqlen_offset
                        + prepare_lens_from_mask(attention_mask)
                        - attention_mask.shape[-1]
                    )
                    max_seqlen = q.shape[1] + max(seqlen_offset)
            if self.rope_max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.rope_max_position_embeddings)

            q, k = self.rotary(
                q,
                k,
                seqlen_offset=seqlen_offset,
                max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
            )
            q, k = l2_norm(q), l2_norm(k)
            use_qk_l2norm_in_kernel = False
        else:
            use_qk_l2norm_in_kernel = True

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self.use_fixed_decay:
            head_idx = self.fixed_decay_head_idx.to(torch.float32)
            log_decay = -torch.pow(
                2.0,
                -8.0 * head_idx / float(self.num_v_heads),
            )
            g = log_decay.view(1, 1, self.num_v_heads, 1).expand(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_v_heads,
                self.head_k_dim,
            ).contiguous()
        else:
            g = rearrange(
                self.f_proj(hidden_states),
                "... (h d) -> ... h d",
                h=self.num_heads,
                d=self.head_k_dim,
            )
            g = -self.A_log.float().exp().view(1, 1, self.num_heads, 1) * F.softplus(
                g.float()
                + self.dt_bias.view(1, 1, self.num_heads, self.head_k_dim),
            )

        if self.num_v_heads > self.num_heads:
            group_size = self.num_v_heads // self.num_heads
            q, k = (
                repeat(x, "... h d -> ... (h g) d", g=group_size)
                for x in (q, k)
            )
            if not self.use_fixed_decay:
                g = repeat(g, "... h d -> ... (h g) d", g=group_size)
            beta = repeat(beta, "... h -> ... (h g)", g=group_size)

        if self.decay_clip is not None:
            g = g.clamp(max=self.decay_clip)

        if duo_mode is not None:
            if use_qk_l2norm_in_kernel:
                q, k = l2_norm(q), l2_norm(k)
            o = self._duo_kda_forward(
                q,
                k,
                v,
                g,
                beta,
                mode=duo_mode,
            )
            recurrent_state = None
        else:
            recurrent_state = last_state["recurrent_state"] if last_state is not None else None
            if mode == "chunk":
                o, recurrent_state = chunk_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                )
            elif mode == "fused_recurrent":
                o, recurrent_state = fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            state = past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(
                    (conv_state_q, conv_state_k, conv_state_v)
                    if self.use_short_conv
                    else None
                ),
                layer_idx=self.layer_idx,
                offset=q_len,
            )
            if self.use_rope:
                previous_offset = (
                    last_state.get("_rope_seq_offset", 0)
                    if last_state is not None
                    else 0
                )
                state["_rope_seq_offset"] = previous_offset + q_len

        o = self.o_norm(
            o,
            rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.head_v_dim,
            ),
        )
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
