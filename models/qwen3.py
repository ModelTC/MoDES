import torch
from loguru import logger
from transformers.processing_utils import Unpack
from typing import Any, Optional, Union

try:
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeCausalLMOutputWithPast,
        Qwen3VLMoeModelOutputWithPast,
    )
    from transformers.utils import (
        auto_docstring,
        is_torchdynamo_compiling,
    )
    from transformers.utils.generic import check_model_inputs
    from transformers.utils import TransformersKwargs
    from transformers.masking_utils import create_causal_mask
except Exception:
    logger.warning("Qwen3VLMoeForConditionalGeneration is not available.")
    TransformersKwargs = None
    Qwen3VLMoeModelOutputWithPast = None
    Qwen3VLMoeCausalLMOutputWithPast = None

    # placeholder for decorator
    def check_model_inputs():
        def wrapper(*args, **kwargs):
            return None

        return wrapper


from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from MLLMExpertSkip.models.utils import (
    apply_tensor_scale,
    apply_scaler_scale,
    threshold_and_mask,
)
import os
import pickle

SKIP_EXP_COUNT = 0
TOTAL_EXP_COUNT = 0


def experts_forward(
    self,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    router_indices: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through MoE experts with routing.

    Args:
        hidden_states: (batch_size * token_num, hidden_size).
        routing_weights: (batch_size * token_num, top_k) routing weights.
        router_indices: (batch_size * token_num, top_k) expert indices.

    Returns:
        Tensor of shape (batch_size * token_num, hidden_size), weighted expert outputs.
    """
    # hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
    next_states = torch.zeros_like(
        hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
    )
    expert_mask = torch.nn.functional.one_hot(
        router_indices, num_classes=self.num_experts
    )
    expert_mask = expert_mask.permute(2, 1, 0)
    # we sum on the top_k and on the sequence length to get which experts
    # are hit this time around
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit[:]:
        _, token_idx = torch.where(expert_mask[expert_idx[0]])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ self.gate_up_proj[expert_idx]
        gate, up = gate_up.chunk(2, dim=-1)
        gated_output = up * self.act_fn(gate)
        out = gated_output @ self.down_proj[expert_idx]
        weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
    return next_states


def mlp_forward(
    self,
    hidden_states: torch.Tensor,
    moe_layer_skip_flag: Optional[bool] = False,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """MoE MLP forward with optional layer skip and tau-based expert skipping.

    Args:
        hidden_states: Input tensor (batch, seq, hidden_size).
        moe_layer_skip_flag: If True, skip experts for the given modality (layer importance).
        skip_modality: 'text' or 'visual' when moe_layer_skip_flag is True.
        enable_tau_skip: If True, apply tau-based expert skipping.
        tau: Dict with 'text' and 'visual' thresholds.

    Returns:
        Output tensor of same shape as hidden_states.
    """
    global SKIP_EXP_COUNT, TOTAL_EXP_COUNT
    orig_shape = hidden_states.shape
    # batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    skip_mask = None
    if moe_layer_skip_flag or (
        hasattr(self, "moe_padding_mask") and self.moe_padding_mask is not None
    ):
        orig_hidden_states = hidden_states.clone().view(-1, hidden_states.shape[-1])
        if moe_layer_skip_flag:
            skip_mask = (
                self.moe_text_mask if skip_modality == "text" else self.moe_media_mask
            )
        else:
            skip_mask = torch.ones_like(self.moe_text_mask, dtype=torch.bool)
        if hasattr(self, "moe_padding_mask") and self.moe_padding_mask is not None:
            skip_mask = skip_mask & self.moe_padding_mask
        skip_mask = ~skip_mask.to(hidden_states.device).view(-1)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[skip_mask, :]
    TOTAL_EXP_COUNT += hidden_states.shape[0] * self.top_k

    router_logits = self.gate(hidden_states)
    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=-1, dtype=torch.float
    )
    routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)

    # import ipdb; ipdb.set_trace()
    if hasattr(self, "gate_dict") and self.gate_dict is not None:
        assert enable_tau_skip is False, "tau skip is not supported for layer gate"
        routing_weights.mul_(self.valid_expert_mask.to(routing_weights.device))
        SKIP_EXP_COUNT += (
            hidden_states.shape[0] * self.top_k - (routing_weights > 0).sum().item()
        )

    router_weights = torch.zeros_like(router_logits).scatter_(
        1, router_indices, routing_weights
    )
    # We employ simulated expert skip here
    if enable_tau_skip:
        new_router_weights = router_weights.clone()
        self.gate.moe_text_mask = self.gate.moe_text_mask.to(router_weights.device)
        self.gate.moe_media_mask = self.gate.moe_media_mask.to(router_weights.device)

        if hasattr(self.gate, "text_layer_importance"):
            new_router_weights = apply_scaler_scale(
                self.gate.text_layer_importance,
                new_router_weights,
                self.gate.moe_text_mask,
            )
            new_router_weights = apply_scaler_scale(
                self.gate.visual_layer_importance,
                new_router_weights,
                self.gate.moe_media_mask,
            )
            to_zero = (self.gate.moe_text_mask & (new_router_weights < tau["text"])) | (
                self.gate.moe_media_mask & (new_router_weights < tau["visual"])
            )
            router_weights.masked_fill_(to_zero, 0.0)
            SKIP_EXP_COUNT += (
                hidden_states.shape[0] * self.top_k - (router_weights > 0).sum().item()
            )

    # hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
    routed_out = self.experts(hidden_states, router_weights, router_indices)
    if skip_mask is not None:
        orig_hidden_states[skip_mask, :] = routed_out
        routed_out = orig_hidden_states.view(*orig_shape)
    else:
        routed_out = routed_out.view(*orig_shape)
    return routed_out


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    moe_layer_skip_flag: Optional[bool] = False,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[float] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states.to(self.input_layernorm.weight.device)
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if hasattr(self.mlp, "top_k"):
        hidden_states = self.mlp(
            hidden_states,
            moe_layer_skip_flag=moe_layer_skip_flag,
            skip_modality=skip_modality,
            enable_tau_skip=enable_tau_skip,
            tau=tau,
        )
    else:
        hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


@check_model_inputs()
def language_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    moe_layer_skip: Optional[int] = None,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[float] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    r"""
    visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
        The mask of the visual positions.
    deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
        The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
        The feature is extracted from the different visual encoder layers, and fed to the decoder
        hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
    """
    # import ipdb; ipdb.set_trace()
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            moe_layer_skip_flag=True if layer_idx == moe_layer_skip else False,
            skip_modality=skip_modality,
            enable_tau_skip=enable_tau_skip,
            tau=tau,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(
            len(deepstack_visual_embeds)
        ):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@check_model_inputs()
def moe_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    moe_layer_skip: Optional[int] = None,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[float] = None,
    enable_load_layer_importance: Optional[bool] = None,
    layer_importance_path: Optional[str] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLMoeModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(
                visual_pos_masks.sum(), img_embed.shape[-1]
            ).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask
            if not isinstance(attention_mask, dict)
            else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(
                attention_mask_tensor[:, 0], dim1=1, dim2=2
            )
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        # import ipdb; ipdb.set_trace()
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # import ipdb; ipdb.set_trace()
    moe_media_mask = None
    moe_text_mask = None
    moe_other_mask = None
    expert_range = None
    moe_padding_mask = None
    if (
        enable_tau_skip
        or moe_layer_skip is not None
        or (
            hasattr(self.language_model.layers[0].mlp, "gate_dict")
            and self.language_model.layers[0].mlp.gate_dict is not None
        )
    ):
        moe_media_mask = (
            visual_pos_masks.view(-1).to(input_ids.device)
            if visual_pos_masks is not None
            else torch.zeros_like(input_ids)
            .view(-1)
            .to(device=input_ids.device, dtype=torch.bool)
        )
        moe_text_mask = ~torch.isin(
            input_ids, self.special_token_id_tensor.to(input_ids.device)
        ).view(-1)
        moe_other_mask = ~(moe_text_mask | moe_media_mask)
        expert_range = torch.arange(
            self.config.text_config.num_experts_per_tok, device=input_ids.device
        )[None, :]
        # consider decoding stage!!!
        if input_ids.shape[-1] == 1:
            moe_padding_mask = None
        else:
            moe_padding_mask = (
                (~attention_mask.to(torch.bool)).view(-1)
                if attention_mask is not None
                else None
            )

    self.layer_importance = None
    for idx, layer in enumerate(self.language_model.layers):
        if (
            idx not in self.config.text_config.mlp_only_layers
            and self.config.text_config.num_experts > 0
            and (idx + 1) % self.config.text_config.decoder_sparse_step == 0
        ) and (
            moe_layer_skip is not None
            or enable_tau_skip
            or (hasattr(layer.mlp, "gate_dict") and layer.mlp.gate_dict is not None)
        ):
            layer.mlp.moe_text_mask = (
                moe_text_mask[:, None] if moe_text_mask is not None else None
            )
            layer.mlp.moe_media_mask = (
                moe_media_mask[:, None] if moe_media_mask is not None else None
            )
            layer.mlp.gate.moe_text_index = layer.mlp.moe_text_mask.squeeze(-1).nonzero(
                as_tuple=True
            )[0][:, None]
            layer.mlp.gate.moe_media_index = layer.mlp.moe_media_mask.squeeze(
                -1
            ).nonzero(as_tuple=True)[0][:, None]

            if enable_tau_skip:
                layer.mlp.gate.experts_len = layer.mlp.experts.num_experts
                layer.mlp.moe_padding_mask = (
                    moe_padding_mask[:, None] if moe_padding_mask is not None else None
                )
                if moe_padding_mask is not None:
                    # import ipdb; ipdb.set_trace()
                    layer.mlp.gate.moe_text_mask = layer.mlp.moe_text_mask[
                        ~layer.mlp.moe_padding_mask
                    ][:, None]
                    layer.mlp.gate.moe_media_mask = layer.mlp.moe_media_mask[
                        ~layer.mlp.moe_padding_mask
                    ][:, None]
                else:
                    layer.mlp.gate.moe_text_mask = layer.mlp.moe_text_mask
                    layer.mlp.gate.moe_media_mask = layer.mlp.moe_media_mask
                if (
                    enable_load_layer_importance
                    and layer_importance_path is not None
                    and not hasattr(layer.mlp.gate, "text_layer_importance")
                ):
                    if self.layer_importance is None:
                        logger.info(
                            f"Load layer importance from {layer_importance_path}"
                        )
                        with open(layer_importance_path, "rb") as f:
                            self.layer_importance = pickle.load(f)
                    layer.mlp.gate.text_layer_importance = self.layer_importance[idx][
                        "text"
                    ]
                    layer.mlp.gate.visual_layer_importance = self.layer_importance[idx][
                        "visual"
                    ]

            if hasattr(layer.mlp, "gate_dict") and layer.mlp.gate_dict is not None:
                layer.mlp.moe_other_mask = (
                    moe_other_mask[:, None] if moe_other_mask is not None else None
                )
                layer.mlp.moe_other_mask = (
                    moe_other_mask[:, None] if moe_other_mask is not None else None
                )
                layer.mlp.text_allowed = expert_range < layer.mlp.gate_dict["text"]
                layer.mlp.visual_allowed = expert_range < layer.mlp.gate_dict["visual"]
                layer.mlp.valid_expert_mask = (
                    (layer.mlp.moe_text_mask & layer.mlp.text_allowed)
                    | (layer.mlp.moe_media_mask & layer.mlp.visual_allowed)
                    | layer.mlp.moe_other_mask
                )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
        **kwargs,
    )

    return Qwen3VLMoeModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


@check_model_inputs()
def vl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    moe_layer_skip: Optional[int] = None,
    skip_modality: Optional[str] = None,
    enable_tau_skip: bool = False,
    tau: Optional[float] = None,
    enable_load_layer_importance: bool = False,
    layer_importance_path: Optional[str] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
        enable_load_layer_importance=enable_load_layer_importance,
        layer_importance_path=layer_importance_path,
        **kwargs,
    )

    hidden_states = outputs[0]

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
        )

    aux_loss = None

    return Qwen3VLMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
    )


def load_model(
    model_path: str,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    layer_gate_dict: dict = None,
):
    """Load Qwen3-VL-MoE model with MoDES expert skipping support.

    Args:
        model_path: Path or HF model id for Qwen3-VL-MoE.
        attn_implementation: 'flash_attention_2', 'sdpa', or 'eager'.
        trust_remote_code: Allow custom model code.
        torch_dtype: Model dtype (default bfloat16).
        device_map: Device map for model (default 'auto').
        layer_gate_dict: Optional per-layer expert config for gate_dict.

    Returns:
        Tuple of (model, processor).
    """
    if layer_gate_dict is not None:
        logger.info(f"layer_gate_dict: {layer_gate_dict}")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    model.eval()
    idx = 0
    config = model.config.text_config
    model.forward = vl_forward.__get__(model)
    # model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    model.model.forward = moe_forward.__get__(model.model)
    model.model.language_model.forward = language_forward.__get__(
        model.model.language_model
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    # import ipdb; ipdb.set_trace()
    special_token_id_list = processor.tokenizer.all_special_ids
    model.model.special_token_id_tensor = torch.tensor(special_token_id_list)
    for layer in model.model.language_model.layers:
        layer.forward = decoder_layer_forward.__get__(layer)
        if (
            idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (idx + 1) % config.decoder_sparse_step == 0
        ):
            layer.mlp.forward = mlp_forward.__get__(layer.mlp)
            layer.mlp.experts.forward = experts_forward.__get__(layer.mlp.experts)
            if layer_gate_dict is not None:
                layer.mlp.gate_dict = layer_gate_dict[
                    idx
                ]  # {"text": ..., "visiual": ...}
                layer.mlp.experts.gate_dict = layer.mlp.gate_dict
        layer.mlp.layer_idx = idx
        idx += 1
    return model, processor
