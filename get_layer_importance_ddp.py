import torch
import argparse
from functools import partial
from datasets import load_dataset, concatenate_datasets
from tasks.gqa import gqa_transform
from tasks.coco import coco_transform
from tasks.video_mmmu import videommmu_transform
import torch.nn.functional as F
import pickle
import os
from utils import create_mask_after_token, create_mask_after_last_token
from loguru import logger
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from models.kimi import load_model as load_kimi_model
from models.qwen3 import load_model as load_qwen_model


@torch.no_grad()
def get_layer_importance(
    accelerator,
    model,
    model_config,
    processor,
    text_to_message,
    loss_type,
    temperature,
    dataset,
    dataloader,
    modalities,
    # topk_logits,
):
    """Compute layer-wise modality importance via KL/MSE loss when skipping each MoE layer.

    For each MoE layer and modality, skip that layer for that modality and measure
    the output distribution change. Results are gathered across DDP processes.

    Args:
        accelerator: Accelerator instance for DDP.
        model: The MoE MLLM model.
        model_config: Dict with get_lm, is_moe_layer, create_mask, eos_token.
        processor: Model processor for tokenization.
        text_to_message: Callable to format text into chat messages.
        loss_type: 'kl' or 'mse' for loss computation.
        temperature: Temperature for logits.
        dataset: Dataset name (gqa, coco, video_mmmu).
        dataloader: DataLoader over calibration samples.
        modalities: List of modality names (e.g., ['text', 'visual']).

    Returns:
        Dict mapping layer_idx -> {modality -> normalized_loss}, or None on non-main ranks.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    language_model = model_config["get_lm"](unwrapped_model)
    num_hidden_layers = language_model.config.num_hidden_layers
    is_moe_layer = model_config["is_moe_layer"]
    local_layer_loss_dict = {}
    local_total_token_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if accelerator.is_main_process:
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
        # sync all processes
        accelerator.wait_for_everyone()
        batched_messages = [
            text_to_message(text) for text in batch["model_input_org_text"]
        ]
        batched_messages = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, return_tensors="pt"
        )
        tmp = []
        if dataset == "gqa" or dataset == "coco" or dataset == "video_mmmu":
            for i, _ in enumerate(batched_messages):
                batched_messages[i] = (
                    batched_messages[i] + batch["model_input_full_answer"][i]
                )
                batched_messages[i] = batched_messages[i] + model_config["eos_token"]
                if dataset == "video_mmmu":
                    tmp.extend(batch["model_input_visual"][i])
                    frame_num = batch["model_input_frames"][i]
                    media_end_idx = batched_messages[i].find(
                        "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
                    )
                    batched_messages[i] = (
                        batched_messages[i][:media_end_idx]
                        + "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
                        * (frame_num - 1)
                        + batched_messages[i][media_end_idx:]
                    )
        else:
            raise ValueError(f"Not support {dataset}")
        if dataset == "video_mmmu":
            batch["model_input_visual"] = tmp
        inputs = processor(
            images=batch["model_input_visual"],
            text=batched_messages,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
        ).to(accelerator.device)
        # Note: `accelerator.prepare` already moved the model and dataloader to the correct device.
        # `inputs` are on CPU here, but the model forward pass will handle moving them.
        answer_masks = model_config["create_mask"](input_ids=inputs["input_ids"])
        # The model is already prepared by accelerator, so it runs on the correct GPU
        org_output = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        org_logits = org_output.logits[answer_masks, :].contiguous()
        # org_logits, org_indices = torch.topk(
        #     org_logits, topk_logits, dim=-1, sorted=False
        # )
        local_total_token_count += answer_masks.sum().item()
        for layer_idx in range(num_hidden_layers):
            if is_moe_layer(language_model.config, layer_idx):
                modality_loss_dict = {}
                for modality in modalities:
                    output = model(
                        **inputs,
                        use_cache=False,
                        return_dict=True,
                        moe_layer_skip=layer_idx,
                        skip_modality=modality,
                    )
                    logits = output.logits[answer_masks, :].contiguous()
                    # logits = logits.gather(dim=-1, index=org_indices)
                    if loss_type == "kl":
                        loss = F.relu(
                            F.kl_div(
                                F.log_softmax(logits / temperature, dim=1),
                                F.softmax(org_logits / temperature, dim=1),
                                reduction="sum",
                            )
                        )
                    elif loss_type == "mse":
                        loss = F.mse_loss(logits, org_logits, reduction="sum")
                        # if accelerator.is_main_process:
                        #     logger.info(
                        #         f"Layer {layer_idx} {modality} MSE Loss: {loss.item()}"
                        #     )
                    modality_loss_dict[modality] = loss.item()
                if layer_idx not in local_layer_loss_dict:
                    local_layer_loss_dict[layer_idx] = modality_loss_dict
                else:
                    for modality in modalities:
                        local_layer_loss_dict[layer_idx][
                            modality
                        ] += modality_loss_dict[modality]
    # --- End of per-batch processing ---
    # NEW: Aggregation step
    accelerator.wait_for_everyone()
    # Flatten the local results for gathering
    metrics_to_gather = {
        f"l{l_idx}_m{m}_loss": torch.tensor(loss, device=accelerator.device)
        for l_idx, m_losses in local_layer_loss_dict.items()
        for m, loss in m_losses.items()
    }
    metrics_to_gather["total_token_count"] = torch.tensor(
        local_total_token_count, device=accelerator.device
    )
    # Gather metrics from all processes
    gathered_metrics = accelerator.gather_for_metrics(metrics_to_gather)
    # The rest of the logic (aggregation, normalization, saving) is done only on the main process
    if accelerator.is_main_process:
        logger.info("All processes finished. Aggregating results on the main process.")
        final_layer_loss_dict = {}
        total_token_count = gathered_metrics.pop("total_token_count").sum().item()
        # Un-flatten the gathered results and sum them up
        for key, tensor_val in gathered_metrics.items():
            parts = key.split("_")
            layer_idx = int(parts[0][1:])
            modality = parts[1][1:]
            if layer_idx not in final_layer_loss_dict:
                final_layer_loss_dict[layer_idx] = {}
            # Sum the loss from all processes
            final_layer_loss_dict[layer_idx][modality] = tensor_val.sum().item()
        # --- Start of your original final normalization logic ---
        for layer_idx in final_layer_loss_dict:
            for modality in final_layer_loss_dict[layer_idx]:
                final_layer_loss_dict[layer_idx][modality] /= total_token_count
        modalities_losses = {m: 0.0 for m in modalities}
        for modality in modalities:
            for layer_idx in final_layer_loss_dict:
                modalities_losses[modality] += final_layer_loss_dict[layer_idx][
                    modality
                ]
        for modality in modalities:
            if modalities_losses[modality] > 0:
                for layer_idx in final_layer_loss_dict:
                    final_layer_loss_dict[layer_idx][modality] /= modalities_losses[
                        modality
                    ]
        # --- End of final normalization logic ---
        return final_layer_loss_dict
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute layer-wise modality importance for MoDES calibration."
    )
    parser.add_argument(
        "--name_or_path",
        type=str,
        default="storage/models/Kimi-VL-A3B-Instruct",
        help="Model path or HuggingFace model ID (e.g., Kimi-VL, Qwen3-VL-MoE).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="storage",
        help="Root directory to save layer importance results.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gqa",
        choices=["gqa", "coco", "video_mmmu"],
        help="Calibration dataset for layer importance computation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration data.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index of samples in the dataset.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Number of calibration samples to use.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl",
        choices=["mse", "kl"],
        help="Loss type for measuring output change when skipping a layer (KL or MSE).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for logits when computing loss.",
    )
    # parser.add_argument("--topk_logits", type=int, default=1000)
    args = parser.parse_args()
    accelerator = Accelerator()
    name_or_path = args.name_or_path
    save_dir = os.path.join(args.save_dir, args.dataset, name_or_path.split("/")[-1])
    loss_type = args.loss_type
    batch_size = args.batch_size
    start_idx = args.start_idx
    num_samples = args.num_samples
    temperature = args.temperature
    dataset = args.dataset
    # topk_logits = args.topk_logits
    model_name = ""

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    load_model = None
    text_to_message = None
    modalities = None
    model_config = {}
    if "Kimi-VL" in name_or_path:
        load_model = load_kimi_model
        model_name = "Kimi-VL"
        modalities = ["text", "visual"]
        model_config = {
            "get_lm": lambda m: m.language_model,
            "is_moe_layer": lambda cfg, idx: idx >= cfg.first_k_dense_replace
            and idx % cfg.moe_layer_freq == 0,
            "create_mask": partial(
                create_mask_after_token, special_token_id=163588, offset=3
            ),
            "eos_token": "<|im_end|>[EOS]",
        }
    elif "Qwen3-VL" in name_or_path:
        load_model = load_qwen_model
        model_name = "Qwen3-VL"
        special_token_id = 151644
        offset = 3
        eos_token = "<|im_end|>"
        modalities = ["text", "visual"]
        model_config = {
            "get_lm": lambda m: m.model.language_model,
            "is_moe_layer": lambda cfg, idx: (
                hasattr(cfg, "num_experts")
                and cfg.num_experts > 0
                and (idx + 1) % cfg.decoder_sparse_step == 0
                and idx not in getattr(cfg, "mlp_only_layers", [])
            ),
            "create_mask": partial(
                create_mask_after_last_token,
                special_token_id=special_token_id,
                offset=offset,
            ),
            "eos_token": eos_token,
        }
    else:
        raise ValueError(f"Not support {name_or_path}")
    text_to_message = lambda text: [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    model, processor = load_model(name_or_path, device_map="cpu")

    data = None
    if "gqa" in dataset:
        data = load_dataset(
            "storage/datasets/GQA/testdev_balanced_instructions", token=True
        )["train"]
        data.set_transform(gqa_transform)
    elif "coco" in dataset:
        data = load_dataset("storage/datasets/COCO-Caption2017/data", token=True)[
            "validation"
        ]
        data.set_transform(coco_transform)
    elif "video_mmmu" in dataset:
        assert model_name == "Kimi-VL", "VideoMMMU only support Kimi-VL"
        adaptation = load_dataset("storage/datasets/VideoMMMU/Adaptation", token=True)[
            "test"
        ]
        comprehension = load_dataset(
            "storage/datasets/VideoMMMU/Comprehension", token=True
        )["test"]
        perception = load_dataset("storage/datasets/VideoMMMU/Perception", token=True)[
            "test"
        ]
        data = concatenate_datasets([adaptation, comprehension, perception])
        data.set_transform(videommmu_transform)
    else:
        raise ValueError(f"Not support {args.dataset}")

    subset_indices = list(range(start_idx, min(start_idx + num_samples, len(data))))
    subset = Subset(data, subset_indices)

    def custom_collate_fn(batch):
        """Collate batch by stacking list of dicts into dict of lists."""
        collated_batch = {}
        keys = batch[0].keys()
        # import ipdb; ipdb.set_trace()
        for key in keys:
            collated_batch[key] = [d[key] for d in batch]
        return collated_batch

    dataloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    # model, dataloader = accelerator.prepare(model, dataloader)
    model.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)

    layer_importance = get_layer_importance(
        accelerator,
        model,
        model_config,
        processor,
        text_to_message,
        loss_type,
        temperature,
        dataset,
        dataloader,
        modalities,
        # topk_logits,
    )
    if accelerator.is_main_process:
        save_path = os.path.join(save_dir, f"{loss_type}_{start_idx}_{num_samples}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(layer_importance, f)
        logger.info(f"Results saved to {save_path}")
