import torch
from typing import Optional, Tuple, List, Union
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
import warnings
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from torch.nn import CrossEntropyLoss
from loguru import logger
from MoDES.models.utils import (
    apply_tensor_scale,
    apply_scaler_scale,
    threshold_and_mask,
)
import torch.nn.functional as F
import os
import pickle

HS_DICT = None
FREQ_DICT = None
TEXT_FREQ_DICT = None
VISUAL_FREQ_DICT = None
TOPK_DICT = None
SKIP_EXP_COUNT = 0
TOTAL_EXP_COUNT = 0


def save_states(
    self,
    hidden_states,
    attention_mask,
    name,
) -> None:
    global HS_DICT
    if hasattr(self, "layer_to_save") and (
        self.layer_to_save == "all" or self.layer_idx in self.layer_to_save
    ):
        self.input_ids = self.input_ids.to(hidden_states.device)
        for shift, hs_per_prompt in enumerate(hidden_states):  # [seq, dim]
            idx = self.start_idx + shift
            input_ids_per_prompt = self.input_ids[idx]
            if hs_per_prompt.shape[0] > 1:  # prefill
                current_attention_mask = (
                    attention_mask[shift] if attention_mask is not None else None
                )
                first_non_pad_idx = (
                    (current_attention_mask == 1).nonzero(as_tuple=True)[0][0].item()
                    if current_attention_mask is not None
                    else 0
                )
                effective_hs = hs_per_prompt[first_non_pad_idx:]
                effective_input_ids = input_ids_per_prompt[first_non_pad_idx:]
                image_token_mask = (
                    effective_input_ids == self.media_placeholder_token_id
                )
                image_hidden_states = effective_hs[image_token_mask]
                last_image_idx = image_token_mask.nonzero(as_tuple=True)[0][-1].item()
                text_hidden_states = effective_hs[last_image_idx + 2 : -4]
                system_hidden_states = effective_hs[3:8]  # for chat model
                HS_DICT[self.layer_idx][name][idx] = {
                    "visual": image_hidden_states.to("cpu"),
                    "text": text_hidden_states.to("cpu"),
                    "system": system_hidden_states.to("cpu"),
                }
            else:
                # import ipdb; ipdb.set_trace()
                # assert os.path.exists(save_path)
                if input_ids_per_prompt.item() in [
                    self.eos_token_id,
                    self.pad_token_id,
                    163586,
                ]:
                    continue
                HS_DICT[self.layer_idx][name][idx]["text"] = torch.cat(
                    [
                        HS_DICT[self.layer_idx][name][idx]["text"],
                        hs_per_prompt.to("cpu"),
                    ],
                    dim=-2,
                )


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    moe_layer_skip_flag = kwargs.get("moe_layer_skip_flag", False)
    skip_modality = kwargs.get("skip_modality", "all")
    skip_expert_idx = kwargs.get("skip_expert_idx", None)
    batch_idx = kwargs.get("batch_idx", None)
    enable_tau_skip = kwargs.get("enable_tau_skip", False)
    tau = kwargs.get("tau", {})

    global HS_DICT
    if hasattr(self, "layer_idx") and HS_DICT.get(self.layer_idx) is None:
        HS_DICT[self.layer_idx] = {
            "pre_attn_norm": {},
            "post_attn_norm": {},
            "pre_ffn_norm": {},
            "post_ffn_norm": {},
            "output": {},
        }

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    # replacing forward will drop original hooks, we need to move
    # input to the same device as self
    residual = hidden_states.to(self.input_layernorm.weight.device)

    save_states(self, residual, attention_mask, "pre_attn_norm")

    hidden_states = self.input_layernorm(hidden_states)

    save_states(self, hidden_states, attention_mask, "post_attn_norm")

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )

    # save_states(self, hidden_states, attention_mask, "post_attn") pre_ffn_norm - pre_attn_norm

    hidden_states = residual + hidden_states

    save_states(self, hidden_states, attention_mask, "pre_ffn_norm")

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    save_states(self, hidden_states, attention_mask, "post_ffn_norm")
    # print(f"moe_layer_skip_flag: {moe_layer_skip_flag}, skip_modality: {skip_modality}")
    if enable_tau_skip and hasattr(self.mlp, "num_experts_per_tok"):
        hidden_states = self.mlp(
            hidden_states,
            enable_tau_skip=enable_tau_skip,
            tau=tau,
        )
    elif not moe_layer_skip_flag and skip_expert_idx is None:
        hidden_states = self.mlp(hidden_states)
    elif skip_modality == "all":
        assert (
            False
        ), 'Not implement skip_expert_idx != None or moe_layer_skip_flag == True when skip_modality == "all"'
    else:
        # import ipdb; ipdb.set_trace()
        decoding_stage = hidden_states.shape[1] == 1
        assert not decoding_stage, "Not implement decoding_stage"
        hidden_states = self.mlp(
            hidden_states,
            moe_layer_skip_flag=moe_layer_skip_flag,
            skip_modality=skip_modality,
            skip_expert_idx=skip_expert_idx,
            batch_idx=batch_idx,
        )

    # save_states(self, hidden_states, attention_mask, "post_ffn") # output - pre_ffn_norm

    hidden_states = residual + hidden_states

    save_states(self, hidden_states, attention_mask, "output")

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    moe_layer_skip: int = -1,
    skip_modality: str | None = None,
    skip_expert_idx: int | None = None,
    enable_tau_skip: bool = False,
    tau: dict = {},
) -> Union[Tuple, BaseModelOutputWithPast]:
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
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # import ipdb; ipdb.set_trace()
        layer_outputs = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            moe_layer_skip_flag=(
                layer_idx == moe_layer_skip if skip_expert_idx is None else False
            ),
            skip_modality=skip_modality,
            skip_expert_idx=skip_expert_idx if layer_idx == moe_layer_skip else None,
            enable_tau_skip=enable_tau_skip,
            tau=tau,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def language_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    moe_layer_skip: int = -1,
    skip_modality: str | None = None,
    skip_expert_idx: int | None = None,
    enable_tau_skip: bool = False,
    tau: dict = {},
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

    >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    # import ipdb; ipdb.set_trace()
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

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        skip_expert_idx=skip_expert_idx,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    # logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

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


def vl_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
    image_grid_hws: Optional[torch.LongTensor] = None,
    start_idx: int = 0,
    moe_layer_skip: int = -1,
    skip_modality: str | None = None,
    skip_expert_idx: int | None = None,
    record_mask: bool = False,
    enable_tau_skip: bool = False,
    enable_load_expert_importance: bool = False,
    enable_load_layer_importance: bool = False,
    tau: dict = {},
    expert_importance_path: str | None = None,
    layer_importance_path: str | None = None,
) -> Union[tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:
    ```python
    >>> from PIL import Image

    >>> # generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> # decode
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ```"""
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

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None and pixel_values.size(0) > 0:
        pixel_values = pixel_values.to(self.vision_tower.dtype)

        image_features: torch.Tensor = self._extract_image_features(
            pixel_values, image_grid_hws
        )  # image_token_nums = [(h1 * w1) / (kernel_size ** 2) + ... ] / patch_size ** 2
        inputs_embeds = inputs_embeds.to(image_features[0].dtype)
        inputs_embeds = self._merge_with_image_features(
            inputs_embeds, input_ids, image_features
        )

    # help decoder_layer to get image_features
    moe_media_mask = None
    moe_text_mask = None
    moe_other_mask = None
    expert_range = None
    moe_padding_mask = None
    if (
        hasattr(self.language_model.model.layers[1].mlp, "gate_dict")
        or hasattr(self.language_model.model.layers[1].mlp, "freq_save_dir")
        or hasattr(self.language_model.model.layers[1].mlp.gate, "topk_save_dir")
        or moe_layer_skip != -1
        or record_mask
        or enable_tau_skip
    ):
        moe_text_mask = ~torch.isin(
            input_ids, self.special_token_id_tensor.to(input_ids.device)
        ).view(-1)
        moe_media_mask = (input_ids == self.config.media_placeholder_token_id).view(-1)
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

    global TEXT_FREQ_DICT, VISUAL_FREQ_DICT
    for idx, layer in enumerate(self.language_model.model.layers):
        layer.input_ids = input_ids
        layer.start_idx = start_idx
        if (
            self.config.text_config.n_routed_experts is not None
            and idx >= self.config.text_config.first_k_dense_replace
            and idx % self.config.text_config.moe_layer_freq == 0
        ) and (
            record_mask  # for expert importance
        ):
            if hasattr(layer.mlp, "moe_text_mask_list"):
                layer.mlp.moe_text_mask_list.append(moe_text_mask[:, None])
                layer.mlp.moe_media_mask_list.append(moe_media_mask[:, None])
            else:
                layer.mlp.moe_text_mask_list = [moe_text_mask[:, None]]
                layer.mlp.moe_media_mask_list = [moe_media_mask[:, None]]

    self.layer_importance = None
    self.expert_importance = None
    for idx, layer in enumerate(self.language_model.model.layers):
        layer.input_ids = input_ids
        layer.start_idx = start_idx
        if (
            self.config.text_config.n_routed_experts is not None
            and idx >= self.config.text_config.first_k_dense_replace
            and idx % self.config.text_config.moe_layer_freq == 0
        ) and (
            (hasattr(layer.mlp, "gate_dict") and layer.mlp.gate_dict is not None)
            or (
                hasattr(layer.mlp, "freq_save_dir")
                and layer.mlp.freq_save_dir is not None
            )
            or (
                hasattr(layer.mlp.gate, "topk_save_dir")
                and layer.mlp.gate.topk_save_dir is not None
            )
            or moe_layer_skip != -1
            or enable_tau_skip
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
                layer.mlp.gate.experts_len = len(layer.mlp.experts)
                layer.mlp.moe_padding_mask = (
                    moe_padding_mask[:, None] if moe_padding_mask is not None else None
                )  # for calculate skip ratio
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
                if (
                    enable_load_expert_importance
                    and expert_importance_path is not None
                    and not hasattr(layer.mlp.gate, "text_expert_importance")
                ):
                    if self.expert_importance is None:
                        logger.info(
                            f"Load expert importance from {expert_importance_path}"
                        )
                        with open(expert_importance_path, "rb") as f:
                            self.expert_importance = pickle.load(f)
                    layer_expert_importance = self.expert_importance[idx]
                    text_expert_importance = torch.zeros(
                        self.config.text_config.n_routed_experts,
                        device=input_ids.device,
                        dtype=torch.bfloat16,
                    )
                    visual_expert_importance = torch.zeros(
                        self.config.text_config.n_routed_experts,
                        device=input_ids.device,
                        dtype=torch.bfloat16,
                    )
                    for expert_idx in range(self.config.text_config.n_routed_experts):
                        text_expert_importance[expert_idx] = layer_expert_importance[
                            expert_idx
                        ]["text"]
                        visual_expert_importance[expert_idx] = layer_expert_importance[
                            expert_idx
                        ]["visual"]
                    layer.mlp.gate.text_expert_importance = text_expert_importance
                    layer.mlp.gate.visual_expert_importance = visual_expert_importance

            if hasattr(layer.mlp, "gate_dict") and layer.mlp.gate_dict is not None:
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
                text_expert = layer.mlp.gate_dict.get(
                    "text_expert_num", self.config.text_config.n_routed_experts
                )
                visual_expert = layer.mlp.gate_dict.get(
                    "visual_expert_num", self.config.text_config.n_routed_experts
                )
                assert (
                    layer.mlp.gate_dict["text"]
                    <= text_expert
                    <= self.config.text_config.n_routed_experts
                ), f"text_expert: {text_expert}, n_routed_experts: {self.config.text_config.n_routed_experts}, gate_dict: {layer.mlp.gate_dict}"
                assert (
                    layer.mlp.gate_dict["visual"]
                    <= visual_expert
                    <= self.config.text_config.n_routed_experts
                ), f"visual_expert: {visual_expert}, n_routed_experts: {self.config.text_config.n_routed_experts}, gate_dict: {layer.mlp.gate_dict}"
                if TEXT_FREQ_DICT is not None:
                    text_freq = torch.from_numpy(TEXT_FREQ_DICT[idx])
                    layer.mlp.gate.text_low_freq_index = text_freq.argsort(
                        descending=True
                    )[text_expert:]
                if VISUAL_FREQ_DICT is not None:
                    visual_freq = torch.from_numpy(VISUAL_FREQ_DICT[idx])
                    layer.mlp.gate.visual_low_freq_index = visual_freq.argsort(
                        descending=True
                    )[visual_expert:]
            layer.mlp.input_ids = input_ids

    global HS_DICT
    if HS_DICT is None:
        HS_DICT = {}
    global FREQ_DICT
    if FREQ_DICT is None:
        FREQ_DICT = {}
    global TOPK_DICT
    if TOPK_DICT is None:
        TOPK_DICT = {}

    # import ipdb; ipdb.set_trace()

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        skip_expert_idx=skip_expert_idx,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][
                shift_attention_mask.to(logits.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][
                shift_attention_mask.to(labels.device) != 0
            ].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def load_model_save(layer_to_save: list, save_dir: str, model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.forward = vl_forward.__get__(model)
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    model.language_model.forward = language_model_forward.__get__(model.language_model)
    model.language_model.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_language.__get__(model.language_model)
    )
    model.language_model.model.forward = model_forward.__get__(
        model.language_model.model
    )
    idx = 0
    for layer in model.language_model.model.layers:
        layer.forward = decoder_layer_forward.__get__(layer)
        layer.layer_to_save = layer_to_save
        layer.save_dir = save_dir
        layer.layer_idx = idx
        layer.media_placeholder_token_id = model.config.media_placeholder_token_id
        layer.pad_token_id = model.config.text_config.pad_token_id
        layer.eos_token_id = model.config.text_config.eos_token_id
        idx += 1
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def moe_forward(self, hidden_states, **kwargs):
    skip_modality = kwargs.get("skip_modality", None)
    moe_layer_skip_flag = kwargs.get("moe_layer_skip_flag", False)
    skip_expert_idx = kwargs.get("skip_expert_idx", None)
    batch_idx = kwargs.get("batch_idx", None)
    enable_tau_skip = kwargs.get("enable_tau_skip", False)
    tau = kwargs.get("tau", {})
    assert not (
        skip_expert_idx and moe_layer_skip_flag
    ), "Not support skipping expert and layer at the same time."
    identity = hidden_states  # [bs, seq, dim]
    orig_shape = hidden_states.shape
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

    topk_idx, topk_weight, aux_loss = self.gate(
        hidden_states, tau=tau, enable_tau_skip=enable_tau_skip
    )
    # [bs * seq, 6], [bs * seq, 6]
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    assert self.training is False, "Gate reduction only support inference"

    y = self.moe_infer(
        hidden_states,
        topk_idx,
        topk_weight,
        skip_expert_idx=skip_expert_idx,
        skip_modality=skip_modality,
        batch_idx=batch_idx,
    )

    if skip_mask is not None:
        orig_hidden_states[skip_mask, :] = y
        y = orig_hidden_states.view(*orig_shape)
    else:
        y = y.view(*orig_shape)

    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y


def gate_forward(self, hidden_states, **kwargs):
    if len(hidden_states.shape) == 2:
        bsz, h = hidden_states.shape
        seq_len = 1
    else:
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), self.weight.type(torch.float32), None
    )  # [bs * seq, 64]
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(
            f"insupportable scoring function for MoE gating: {self.scoring_func}"
        )

    # select top-k experts
    if self.topk_method == "noaux_tc":
        assert not self.training
        scores_for_choice = scores.view(
            bsz * seq_len, -1
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        # import ipdb; ipdb.set_trace()
        if (
            hasattr(self, "gate_dict")
            and self.gate_dict is not None
            and hasattr(self, "text_low_freq_index")
            and hasattr(self, "visual_low_freq_index")
        ):
            tmp_scores[self.moe_text_index, self.text_low_freq_index] = 0.0
            tmp_scores[self.moe_media_index, self.visual_low_freq_index] = 0.0

        _, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1, sorted=True
        )  # important!!!
        topk_weight = scores.gather(1, topk_idx)
    elif self.topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    else:
        raise NotImplementedError(
            f"insupportable TopK function for MoE gating: {self.topk_method}"
        )

    # norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator

    if hasattr(self, "topk_save_dir") and self.topk_save_dir is not None:
        # import ipdb

        # ipdb.set_trace()
        global TOPK_DICT
        if TOPK_DICT != None and TOPK_DICT.get(self.layer_idx, None) is None:
            TOPK_DICT[self.layer_idx] = {
                "text": None,
                "visual": None,
                "text_num": 0,
                "visual_num": 0,
            }

        self.moe_text_index = self.moe_text_index.to("cpu")
        self.moe_media_index = self.moe_media_index.to("cpu")
        text_mask_topk_weight = topk_weight[:, self.topk - 1 : self.topk][
            self.moe_text_index.view(-1), :
        ].flatten()
        visual_mask_topk_weight = topk_weight[:, self.topk - 1 : self.topk][
            self.moe_media_index.view(-1), :
        ].flatten()
        text_mask_topk_idx = topk_idx[:, self.topk - 1 : self.topk][
            self.moe_text_index.view(-1), :
        ].flatten()
        visual_mask_topk_idx = topk_idx[:, self.topk - 1 : self.topk][
            self.moe_media_index.view(-1), :
        ].flatten()
        visual_output_tensor = torch.zeros(
            self.n_routed_experts,
            dtype=visual_mask_topk_weight.dtype,
            device=visual_mask_topk_weight.device,
        )
        text_output_tensor = torch.zeros(
            self.n_routed_experts,
            dtype=text_mask_topk_weight.dtype,
            device=text_mask_topk_weight.device,
        )
        visual_num_output_tensor = torch.zeros(
            self.n_routed_experts,
            dtype=visual_mask_topk_idx.dtype,
            device=visual_mask_topk_weight.device,
        )
        text_num_output_tensor = torch.zeros(
            self.n_routed_experts,
            dtype=text_mask_topk_idx.dtype,
            device=text_mask_topk_weight.device,
        )
        text_output_tensor.scatter_add_(0, text_mask_topk_idx, text_mask_topk_weight)
        visual_output_tensor.scatter_add_(
            0, visual_mask_topk_idx, visual_mask_topk_weight
        )
        text_num_output_tensor.scatter_add_(
            0, text_mask_topk_idx, torch.ones_like(text_mask_topk_idx)
        )
        visual_num_output_tensor.scatter_add_(
            0, visual_mask_topk_idx, torch.ones_like(visual_mask_topk_idx)
        )
        if TOPK_DICT[self.layer_idx]["text"] is None:
            TOPK_DICT[self.layer_idx]["text"] = text_output_tensor.to("cpu")
        else:
            TOPK_DICT[self.layer_idx]["text"] += text_output_tensor.to("cpu")
        if TOPK_DICT[self.layer_idx]["visual"] is None:
            TOPK_DICT[self.layer_idx]["visual"] = visual_output_tensor.to("cpu")
        else:
            TOPK_DICT[self.layer_idx]["visual"] += visual_output_tensor.to("cpu")
        TOPK_DICT[self.layer_idx]["text_num"] += text_num_output_tensor.to("cpu")
        TOPK_DICT[self.layer_idx]["visual_num"] += visual_num_output_tensor.to("cpu")

    enable_tau_skip = kwargs.get("enable_tau_skip", False)
    tau = kwargs.get("tau", {})
    # import ipdb; ipdb.set_trace()
    if enable_tau_skip:
        new_weight = topk_weight.clone()
        self.moe_text_mask = self.moe_text_mask.to(topk_weight.device)
        self.moe_media_mask = self.moe_media_mask.to(topk_weight.device)
        if hasattr(self, "text_expert_importance"):
            new_weight = apply_tensor_scale(
                self.text_expert_importance,
                topk_idx,
                new_weight,
                self.moe_text_mask,
            )
            new_weight = apply_tensor_scale(
                self.visual_expert_importance,
                topk_idx,
                new_weight,
                self.moe_media_mask,
            )
        # normalize new_weight
        new_weight.div_(
            new_weight.sum(dim=1, keepdim=True) + 1e-20
        )  # align with main branch `no_exp`

        if hasattr(self, "text_layer_importance"):
            new_weight = apply_scaler_scale(
                self.text_layer_importance, new_weight, self.moe_text_mask
            )
            new_weight = apply_scaler_scale(
                self.visual_layer_importance, new_weight, self.moe_media_mask
            )

        topk_weight, topk_idx = threshold_and_mask(
            new_weight,
            topk_weight,
            topk_idx,
            self.moe_text_mask,
            tau["text"],
            self.experts_len,
        )
        topk_weight, topk_idx = threshold_and_mask(
            new_weight,
            topk_weight,
            topk_idx,
            self.moe_media_mask,
            tau["visual"],
            self.experts_len,
        )

    topk_weight = (
        topk_weight * self.routed_scaling_factor
    )  # must multiply the scaling factor

    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = None

    return topk_idx, topk_weight, aux_loss


@torch.no_grad()
def moe_infer(self, x, topk_ids, topk_weight, **kwargs):
    # import ipdb; ipdb.set_trace()
    skip_expert_idx = kwargs.get("skip_expert_idx", None)
    skip_modality = kwargs.get("skip_modality", None)
    batch_idx = kwargs.get("batch_idx", None)

    if skip_expert_idx is not None:
        if batch_idx is not None:
            skip_mask = (
                self.moe_text_mask_list[batch_idx]
                if skip_modality == "text"
                else self.moe_media_mask_list[batch_idx]
            )
        else:
            skip_mask = (
                self.moe_text_mask if skip_modality == "text" else self.moe_media_mask
            )
        # if skip_modality != "text":
        #     import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        target_mask = (topk_ids == skip_expert_idx) & skip_mask.to(topk_ids.device)
        topk_weight.mul_(~target_mask)
        topk_ids[target_mask] = len(self.experts)

    if hasattr(self, "gate_dict") and self.gate_dict is not None:
        topk_weight.mul_(self.valid_expert_mask.to(topk_weight.device))
        # renorm_weights = topk_weight.sum(dim=1, keepdim=True)
        # topk_weight.div_(renorm_weights + 1e-20).mul_(
        #     self.gate.routed_scaling_factor
        # )  # numerical precision
        topk_ids[~self.valid_expert_mask] = len(self.experts)

    global SKIP_EXP_COUNT, TOTAL_EXP_COUNT
    TOTAL_EXP_COUNT += topk_ids.numel()
    SKIP_EXP_COUNT += (topk_ids == len(self.experts)).sum().item()

    idxs = topk_ids.view(-1).argsort()  # [bs * seq, 6]
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts) + 1))
    src = torch.ones_like(topk_ids, dtype=cnts.dtype, device=cnts.device)
    cnts.scatter_add_(1, topk_ids, src)
    tokens_per_expert = cnts.sum(dim=0)
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    global FREQ_DICT
    if FREQ_DICT != None and FREQ_DICT.get(self.layer_idx, None) is None:
        FREQ_DICT[self.layer_idx] = {
            "text": None,
            "visual": None,
        }

    if hasattr(self, "freq_save_dir") and self.freq_save_dir is not None:
        assert not hasattr(self, "gate_dict"), "Gate reduction only support inference"
        text_mask_topk_ids = topk_ids[self.moe_text_mask, :]
        visual_mask_topk_ids = topk_ids[self.moe_media_mask, :]
        text_freq = text_mask_topk_ids.new_zeros(len(self.experts), dtype=torch.int32)
        visual_freq = visual_mask_topk_ids.new_zeros(
            len(self.experts), dtype=torch.int32
        )
        # import ipdb; ipdb.set_trace()
        text_freq.scatter_add_(
            0,
            text_mask_topk_ids.view(-1),
            torch.ones_like(text_mask_topk_ids.view(-1), dtype=torch.int32),
        )
        visual_freq.scatter_add_(
            0,
            visual_mask_topk_ids.view(-1),
            torch.ones_like(visual_mask_topk_ids.view(-1), dtype=torch.int32),
        )
        if FREQ_DICT[self.layer_idx]["text"] is None:
            FREQ_DICT[self.layer_idx]["text"] = text_freq
            FREQ_DICT[self.layer_idx]["visual"] = visual_freq
        else:
            FREQ_DICT[self.layer_idx]["text"] += text_freq
            FREQ_DICT[self.layer_idx]["visual"] += visual_freq

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        if i == len(self.experts):
            outputs.append(tokens_for_this_expert)
            break
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    # import ipdb; ipdb.set_trace()
    return final_out


def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    attention_mask=None,
    image_grid_hws=None,
    cache_position=None,
    **kwargs,
):
    # import ipdb; ipdb.set_trace()
    model_inputs = self.language_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        **kwargs,
    )

    # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
    # Otherwise we need pixel values to be passed to model
    if cache_position[0] == 0:
        model_inputs["pixel_values"] = pixel_values
        model_inputs["image_grid_hws"] = image_grid_hws

    # get forward model input name
    forward_input_names = self.forward.__code__.co_varnames
    for key, value in kwargs.items():
        if key in forward_input_names and key not in model_inputs:
            model_inputs[key] = value

    return model_inputs


def prepare_inputs_for_generation_language(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_seq_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )

    # get forward model input name
    forward_input_names = self.forward.__code__.co_varnames
    for key, value in kwargs.items():
        if key in forward_input_names and key not in model_inputs:
            model_inputs[key] = value

    return model_inputs


def load_model(
    model_path: str,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    layer_gate_dict: dict = None,
):
    """Load Kimi-VL model with MoDES expert skipping support.

    Args:
        model_path: Path or HF model id for Kimi-VL.
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    model.eval()
    idx = 0
    config = model.config.text_config
    model.forward = vl_forward.__get__(model)
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    model.language_model.forward = language_model_forward.__get__(model.language_model)
    model.language_model.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_language.__get__(model.language_model)
    )
    model.language_model.model.forward = model_forward.__get__(
        model.language_model.model
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    special_token_id_list = processor.tokenizer.all_special_ids
    model.special_token_id_tensor = torch.tensor(special_token_id_list)
    if layer_gate_dict is not None:
        path = layer_gate_dict.get(
            "freq_path", "storage/data/freq/gqa/Kimi-VL-A3B-Instruct/plot"
        )
        if os.path.exists(os.path.join(path, "text_counts.pt")):
            global TEXT_FREQ_DICT, VISUAL_FREQ_DICT
            TEXT_FREQ_DICT = torch.load(
                os.path.join(path, "text_counts.pt"), weights_only=False
            )
            VISUAL_FREQ_DICT = torch.load(
                os.path.join(path, "visual_counts.pt"), weights_only=False
            )
    for layer in model.language_model.model.layers:
        layer.forward = decoder_layer_forward.__get__(layer)
        if (
            config.n_routed_experts is not None
            and idx >= config.first_k_dense_replace
            and idx % config.moe_layer_freq == 0
        ):
            layer.mlp.forward = moe_forward.__get__(layer.mlp)
            layer.mlp.moe_infer = moe_infer.__get__(layer.mlp)
            layer.mlp.gate.forward = gate_forward.__get__(layer.mlp.gate)
            if layer_gate_dict is not None:
                layer.mlp.gate_dict = layer_gate_dict[
                    idx
                ]  # {"text": ..., "visiual": ...}
                layer.mlp.gate.gate_dict = layer.mlp.gate_dict
        layer.mlp.layer_idx = idx
        idx += 1
    return model, processor


def load_model_freq(freq_save_dir: str, model_path: str):
    # We consider both system and user prompt as the text input here
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    idx = 0
    config = model.config.text_config
    model.forward = vl_forward.__get__(model)
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    model.language_model.forward = language_model_forward.__get__(model.language_model)
    model.language_model.model.forward = model_forward.__get__(
        model.language_model.model
    )
    model.language_model.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_language.__get__(model.language_model)
    )
    special_token_id_list = processor.tokenizer.all_special_ids
    model.special_token_id_tensor = torch.tensor(special_token_id_list)
    for layer in model.language_model.model.layers:
        layer.forward = decoder_layer_forward.__get__(layer)
        if (
            config.n_routed_experts is not None
            and idx >= config.first_k_dense_replace
            and idx % config.moe_layer_freq == 0
        ):
            layer.mlp.forward = moe_forward.__get__(layer.mlp)
            layer.mlp.moe_infer = moe_infer.__get__(layer.mlp)
            layer.mlp.freq_save_dir = freq_save_dir
        layer.mlp.layer_idx = idx
        idx += 1
    return model, processor


def load_model_topk(topk_save_dir: str, model_path: str, topk: int = 6):
    # We consider both system and user prompt as the text input here
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    idx = 0
    config = model.config.text_config
    model.forward = vl_forward.__get__(model)
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    model.language_model.forward = language_model_forward.__get__(model.language_model)
    model.language_model.model.forward = model_forward.__get__(
        model.language_model.model
    )
    model.language_model.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_language.__get__(model.language_model)
    )
    special_token_id_list = processor.tokenizer.all_special_ids
    model.special_token_id_tensor = torch.tensor(special_token_id_list)
    for layer in model.language_model.model.layers:
        layer.forward = decoder_layer_forward.__get__(layer)
        if (
            config.n_routed_experts is not None
            and idx >= config.first_k_dense_replace
            and idx % config.moe_layer_freq == 0
        ):
            layer.mlp.forward = moe_forward.__get__(layer.mlp)
            layer.mlp.moe_infer = moe_infer.__get__(layer.mlp)
            layer.mlp.gate.forward = gate_forward.__get__(layer.mlp.gate)
            layer.mlp.gate.topk_save_dir = topk_save_dir
            layer.mlp.gate.mlp = layer.mlp
            layer.mlp.gate.layer_idx = idx
            layer.mlp.gate.topk = topk
        layer.mlp.layer_idx = idx
        idx += 1
    return model, processor
