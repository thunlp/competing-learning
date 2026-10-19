from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.kda import KimiDeltaAttention
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as KDAMLP
from fla.modules.l2warp import l2_warp

from .attention import NoPEAttention
from .config import KDAConfig
from .duo import validate_duo_hyperparameters
from .layer import KimiDeltaAttentionRoPE

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class KDABlock(GradientCheckpointingLayer):
    def __init__(self, config: KDAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )
        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.attn = NoPEAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=config.max_position_embeddings,
                logit_scaling_a=config.attn.get("logit_scaling_a", None),
                use_output_gate=getattr(config, "attn_output_gate", False),
                layer_idx=layer_idx,
            )
        else:
            self.attn = KimiDeltaAttentionRoPE(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                use_rope=getattr(config, "kda_use_rope", False),
                rope_theta=getattr(config, "kda_rope_theta", 10000.0),
                max_position_embeddings=(
                    getattr(config, "kda_max_position_embeddings", None)
                    or config.max_position_embeddings
                ),
                use_fixed_decay=getattr(config, "kda_use_fixed_decay", False),
                decay_clip=getattr(config, "decay_clip", None),
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.mlp = KDAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        if getattr(self.config, "duo_attention_only", False) and not isinstance(self.attn, NoPEAttention):
            kwargs.pop("duo_mode", None)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


class KDAPreTrainedModel(PreTrainedModel):
    config_class = KDAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KDABlock"]
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, KimiDeltaAttention) and next(module.parameters()).device.type != "meta":
            with torch.no_grad():
                module.A_log.copy_(nn.init.uniform_(module.A_log, a=1, b=16).log())
                dt = torch.exp(
                    nn.init.uniform_(module.dt_bias)
                    * (math.log(0.1) - math.log(0.001))
                    + math.log(0.001),
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                module.dt_bias.copy_(inv_dt)
                module.dt_bias._is_hf_initialized = True
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            parameter = None
            if hasattr(module, "o_proj"):
                parameter = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                parameter = module.down_proj.weight
            if parameter is not None:
                if prenorm_residual_strategy == "rescale":
                    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                    with torch.no_grad():
                        parameter /= math.sqrt(
                            num_residuals_per_layer * self.config.num_hidden_layers,
                        )
                elif prenorm_residual_strategy == "zero":
                    nn.init.zeros_(parameter)
                else:
                    raise ValueError(
                        f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}",
                    )


class KDAModel(KDAPreTrainedModel):
    def __init__(self, config: KDAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [KDABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)],
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )

        if getattr(config, "duo_attention", False):
            self.enable_duo_attention(
                attention_only=getattr(config, "duo_attention_only", False),
                sink_size=config.duo_sink_size,
                recent_size=config.duo_recent_size,
                alpha_init=config.duo_alpha_init,
                kda_use_sink=config.duo_kda_use_sink,
                attention_backend=config.duo_streaming_backend,
                inverse_epsilon=config.duo_kda_inverse_epsilon,
                rebase_interval=config.duo_kda_rebase_interval,
                scan_block_size=getattr(
                    config,
                    "duo_kda_scan_block_size",
                    64,
                ),
            )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def enable_duo_attention(
        self,
        *,
        sink_size: int,
        recent_size: int,
        alpha_init: float = 1.0,
        kda_use_sink: bool = True,
        attention_backend: str = "flex",
        inverse_epsilon: float = 1e-4,
        rebase_interval: int = 1024,
        scan_block_size: int = 64,
        attention_only: bool = False,
    ) -> None:
        """Inject gates into every recurrent and hybrid-attention layer."""

        validate_duo_hyperparameters(
            sink_size,
            recent_size,
            alpha_init,
            inverse_epsilon=inverse_epsilon,
            rebase_interval=rebase_interval,
        )
        if not isinstance(kda_use_sink, bool):
            raise TypeError("`kda_use_sink` must be a bool.")
        if attention_backend not in {"flex", "torch"}:
            raise ValueError(
                "`attention_backend` must be either 'flex' or 'torch'.",
            )
        if isinstance(scan_block_size, bool) or not isinstance(
            scan_block_size,
            int,
        ):
            raise TypeError("`scan_block_size` must be an integer.")
        if scan_block_size < 0:
            raise ValueError("`scan_block_size` must be nonnegative.")
        if any(
            "duo_alpha" in layer.attn._parameters
            for layer in self.layers
        ):
            raise RuntimeError("DuoAttention has already been enabled.")
        for layer in self.layers:
            if isinstance(layer.attn, NoPEAttention):
                layer.attn.enable_duo_attention(
                    sink_size=sink_size,
                    recent_size=recent_size,
                    alpha_init=alpha_init,
                    backend=attention_backend,
                )
            elif not attention_only:
                layer.attn.enable_duo_attention(
                    sink_size=sink_size,
                    recent_size=recent_size,
                    alpha_init=alpha_init,
                    kda_use_sink=kda_use_sink,
                    inverse_epsilon=inverse_epsilon,
                    rebase_interval=rebase_interval,
                    scan_block_size=scan_block_size,
                )

        self.config.duo_attention_only = bool(attention_only)
        self.config.duo_attention = True
        self.config.duo_sink_size = int(sink_size)
        self.config.duo_recent_size = int(recent_size)
        self.config.duo_alpha_init = float(alpha_init)
        self.config.duo_kda_use_sink = kda_use_sink
        self.config.duo_streaming_backend = attention_backend
        self.config.duo_kda_inverse_epsilon = float(inverse_epsilon)
        self.config.duo_kda_rebase_interval = int(rebase_interval)
        self.config.duo_kda_scan_block_size = scan_block_size

    def named_duo_alpha_parameters(self):
        for layer_index, layer in enumerate(self.layers):
            alpha = layer.attn._parameters.get("duo_alpha")
            if alpha is not None:
                yield f"layers.{layer_index}.attn.duo_alpha", alpha

    @torch.no_grad()
    def clamp_duo_alpha_(self) -> None:
        for _, alpha in self.named_duo_alpha_parameters():
            alpha.clamp_(0.0, 1.0)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn("`KDAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                item
                for item in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                ]
                if item is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class KDAForCausalLM(KDAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = KDAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def enable_duo_attention(self, **kwargs) -> None:
        self.model.enable_duo_attention(**kwargs)

    def named_duo_alpha_parameters(self):
        for name, parameter in self.model.named_duo_alpha_parameters():
            yield f"model.{name}", parameter

    @torch.no_grad()
    def clamp_duo_alpha_(self) -> None:
        self.model.clamp_duo_alpha_()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if "past_key_values" in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates "
                    f"`past_key_values`, which is not supported for {self.__class__.__name__}. "
                    "Try another generation strategy instead. For the available generation "
                    "strategies, check https://huggingface.co/docs/transformers/en/"
                    "generation_strategies#decoding-strategies",
                )
            raise

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        return_last_hidden_state_only: bool = False,
        return_loss_only: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if return_last_hidden_state_only:
            return outputs

        hidden_states = outputs[0]
        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(
                hidden_states
                if logits_to_keep is None
                else hidden_states[:, -logits_to_keep:],
            )
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(
                        use_l2warp=self.config.use_l2warp,
                    )
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], criterion.ignore_index),
                ),
                1,
            )
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(
                    hidden_states,
                    labels,
                    self.lm_head.weight,
                    self.lm_head.bias,
                )
            else:
                loss = criterion(
                    logits.view(labels.numel(), -1),
                    labels.view(-1),
                )
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if return_loss_only:
            if loss is None:
                raise ValueError("`return_loss_only=True` requires `labels`.")
            # Returning only the scalar prevents Accelerate's mixed-precision
            # wrapper from materializing an FP32 copy of the full 16K logits.
            return (loss,)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
