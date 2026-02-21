import argparse
from models.kimi import load_model as load_kimi_model
from models.qwen3 import load_model as load_qwen3_model
from datasets import load_dataset, concatenate_datasets
from tasks.gqa import gqa_transform
from tasks.coco import coco_transform
from tasks.video_mmmu import videommmu_transform
import torch.nn.functional as F
import models.kimi as kimi_model
import models.qwen3 as qwen3_model
import pickle
import numpy as np
import os
import gc
import torch
from functools import partial
from utils import to_device
from utils import create_mask_after_token, create_mask_after_last_token
from loguru import logger
import math
from accelerate import Accelerator


@torch.no_grad()
def gen_data_loss(
    model,
    accelerator,
    loss_type,
    temperature,
    inputs_list,
    answer_masks_list,
    org_logits_list,
    tau,
    enable_load_expert_importance,
    enable_load_layer_importance,
    expert_importance_path,
    layer_importance_path,
):
    """Compute distributed loss and skip ratio for a given tau configuration.

    Each process computes local loss and token count; expert skip counters are
    read from the model modules; results are reduced across processes.

    Args:
        model: The MoE MLLM model.
        accelerator: Accelerator instance for DDP.
        loss_type: 'kl' or 'mse' for loss computation.
        temperature: Temperature for logits before loss.
        inputs_list: List of batched model inputs (on CPU).
        answer_masks_list: List of boolean masks for answer token positions.
        org_logits_list: List of original model logits at answer positions.
        tau: Dict with 'text' and 'visual' threshold values.
        enable_load_expert_importance: Whether to load expert importance.
        enable_load_layer_importance: Whether to load layer importance.
        expert_importance_path: Path to expert importance pickle.
        layer_importance_path: Path to layer importance pickle.

    Returns:
        Tuple of (average_loss, skip_ratio) scalars, same on all ranks.
    """
    # Reset expert skip counters on every process (they are module-level globals)
    kimi_model.SKIP_EXP_COUNT = 0
    kimi_model.TOTAL_EXP_COUNT = 0
    qwen3_model.SKIP_EXP_COUNT = 0
    qwen3_model.TOTAL_EXP_COUNT = 0

    local_token_count = 0
    local_loss_sum = 0.0  # accumulate summed loss over tokens

    # Compute local forward passes for the tau value
    for inputs, answer_masks, org_logits in zip(
        inputs_list, answer_masks_list, org_logits_list
    ):
        inputs = to_device(inputs, model.device)
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
            enable_tau_skip=True,
            tau=tau,
            enable_load_expert_importance=enable_load_expert_importance,
            enable_load_layer_importance=enable_load_layer_importance,
            expert_importance_path=expert_importance_path,
            layer_importance_path=layer_importance_path,
        )
        logits = outputs.logits
        org_logits = to_device(org_logits, logits.device)
        logits = logits[answer_masks, :].contiguous()
        logits = logits / temperature
        if loss_type == "kl":
            loss = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(org_logits / temperature, dim=1),
                reduction="sum",
            )
        elif loss_type == "mse":
            loss = F.mse_loss(logits, org_logits, reduction="sum")
        else:
            raise ValueError(f"Not support loss_type: {loss_type}")
        local_token_count += answer_masks.sum().item()
        # Accumulate summed loss
        local_loss_sum += loss.item()

    # Aggregate expert skip counters across all possible model backends
    local_skip = kimi_model.SKIP_EXP_COUNT + qwen3_model.SKIP_EXP_COUNT
    local_total = kimi_model.TOTAL_EXP_COUNT + qwen3_model.TOTAL_EXP_COUNT

    # Convert to tensors on the accelerator device for reduction
    t_loss_sum = torch.tensor(
        local_loss_sum, dtype=torch.float64, device=accelerator.device
    )
    t_token_cnt = torch.tensor(
        local_token_count, dtype=torch.float64, device=accelerator.device
    )
    t_skip_sum = torch.tensor(
        local_skip, dtype=torch.float64, device=accelerator.device
    )
    t_exp_total = torch.tensor(
        local_total, dtype=torch.float64, device=accelerator.device
    )

    # Reduce (sum) to get global totals
    loss_sum_all = accelerator.reduce(t_loss_sum, reduction="sum")
    token_cnt_all = accelerator.reduce(t_token_cnt, reduction="sum")
    skip_sum_all = accelerator.reduce(t_skip_sum, reduction="sum")
    exp_total_all = accelerator.reduce(t_exp_total, reduction="sum")

    # Compute globally consistent metrics (identical on all ranks)
    if token_cnt_all.item() > 0:
        loss_out = (loss_sum_all / token_cnt_all).item()
    else:
        loss_out = 0.0
    if exp_total_all.item() > 0:
        skip_ratio = (skip_sum_all / exp_total_all).item()
    else:
        skip_ratio = 0.0

    return loss_out, skip_ratio


def cache_f(f):
    """Create a per-process cache wrapper for tau evaluation.

    Caches (text_tau, visual_tau) -> (loss, skip_ratio) to avoid recomputation.

    Args:
        f: Callable f(tau) returning (loss, skip_ratio).

    Returns:
        Tuple of (wrapper_func, cache_dict). wrapper_func(i, j) calls f with tau={'text': i, 'visual': j}.
    """
    cache = {}

    def wrapper(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = f(tau={"text": i, "visual": j})
        return cache[(i, j)]

    return wrapper, cache


def minimize_g_subject_to_f(N, M, func, T, is_main_process: bool):
    """
    Args:
        N: text_tau_grid number
        M: visual_tau_grid number
        func: return loss, skip proportion (g, f)
        T: target skip proportion
        is_main_process: whether to print logs
    Returns:
        (i, jf, best_val) if best is not None else (None, None, None)
    """
    N_len = len(N)
    M_len = len(M)

    if func(N[-1], M[-1])[1] < T:
        if is_main_process:
            logger.info(f"f({N[-1]}, {M[-1]}) = {func(N[-1], M[-1])[1]} < T = {T}")
        return None, None, None

    best = None
    best_val = None
    j = M_len - 1
    for i in range(N_len):
        while j >= 0 and func(N[i], M[j])[1] >= T:
            if is_main_process:
                logger.info(
                    f"f({N[i]}, {M[j]}) = {func(N[i], M[j])[1]} >= T = {T} | g({N[i]}, {M[j]}) = {func(N[i], M[j])[0]}"
                )
            j -= 1
        jf = j + 1

        if jf < M_len:
            val = func(N[i], M[jf])[0]
            if is_main_process:
                logger.info(
                    f"f({N[i]}, {M[jf]}) = {func(N[i], M[jf])[1]} | g({N[i]}, {M[jf]}) = {val}"
                )
            if best_val is None or val < best_val:
                best = (N[i], M[jf])
                best_val = val

    return (best[0], best[1], best_val) if best is not None else (None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Frontier search for optimal tau thresholds for MoDES expert skipping."
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
        help="Root directory to save search results.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gqa",
        choices=["gqa", "coco", "video_mmmu"],
        help="Calibration dataset for frontier search.",
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
        help="Loss type for comparing original vs expert-skipping outputs (KL or MSE).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for logits when computing loss.",
    )
    parser.add_argument(
        "--expert_importance_path",
        type=str,
        default=None,
        help="Path to expert importance pickle (optional, for Kimi-VL).",
    )
    parser.add_argument(
        "--layer_importance_path",
        type=str,
        default=None,
        help="Path to layer importance pickle from get_layer_importance_ddp.py.",
    )
    parser.add_argument(
        "--target_skip_proportion",
        type=float,
        default=0.8,
        help="Target expert skipping ratio (e.g., 0.8 for 80% skip).",
    )
    parser.add_argument(
        "--grid_num",
        type=int,
        default=100,
        help="Number of grid points in tau search space.",
    )
    parser.add_argument(
        "--text_tau_min",
        type=float,
        default=0.0,
        help="Minimum value for text tau in the search grid.",
    )
    parser.add_argument(
        "--text_tau_max",
        type=float,
        default=0.71,
        help="Maximum value for text tau in the search grid.",
    )
    parser.add_argument(
        "--visual_tau_min",
        type=float,
        default=0.0,
        help="Minimum value for visual tau in the search grid.",
    )
    parser.add_argument(
        "--visual_tau_max",
        type=float,
        default=0.6,
        help="Maximum value for visual tau in the search grid.",
    )
    parser.add_argument(
        "--exp_coeff",
        type=int,
        default=10,
        help="Exponent coefficient for exponential grid mapping.",
    )
    parser.add_argument(
        "--grid_map",
        type=str,
        default="exp",
        choices=["exp", "linear"],
        help="Grid mapping: 'exp' (sigmoid) or 'linear'.",
    )
    args = parser.parse_args()

    # Initialize accelerator (DDP)
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    name_or_path = args.name_or_path
    save_dir = os.path.join(args.save_dir, args.dataset, name_or_path.split("/")[-1])
    loss_type = args.loss_type
    batch_size = args.batch_size
    start_idx = args.start_idx
    num_samples = args.num_samples
    temperature = args.temperature
    dataset = args.dataset
    expert_importance_path = args.expert_importance_path
    layer_importance_path = args.layer_importance_path
    enable_load_expert_importance = expert_importance_path is not None
    enable_load_layer_importance = layer_importance_path is not None
    target_skip_proportion = args.target_skip_proportion
    grid_num = args.grid_num
    text_tau_min = args.text_tau_min
    text_tau_max = args.text_tau_max
    visual_tau_min = args.visual_tau_min
    visual_tau_max = args.visual_tau_max
    grid_map = args.grid_map
    exp_coeff = args.exp_coeff

    model_name = ""

    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    # Choose model loader and modality specifics
    load_model = None
    text_to_message = None
    modalities = None
    if "Kimi-VL" in name_or_path:
        load_model = load_kimi_model
        text_to_message = lambda text: [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "path/to/image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        model_name = "Kimi-VL"
        modalities = ["text", "visual"]
        model_create_mask_after_token = create_mask_after_token
    elif "Qwen3-VL" in name_or_path:
        load_model = load_qwen3_model
        text_to_message = lambda text: [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "path/to/image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        model_name = "Qwen3-VL"
        modalities = ["text", "visual"]
        model_create_mask_after_token = create_mask_after_last_token
    else:
        raise ValueError(f"Not support {name_or_path}")

    # Load model and processor on each process
    model, processor = load_model(name_or_path, device_map=None)
    model.eval()
    # Prepare model with accelerator; processor remains on CPU
    # model = accelerator.prepare(model)
    model.to(accelerator.device)

    # Load dataset (every rank loads the same, will be sharded by index selection below)
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

    # Build the global index range, then shard across processes
    global_start = start_idx
    global_end = min(start_idx + num_samples, len(data))
    num_samples_per_process = (global_end - global_start) // accelerator.num_processes
    rest = (global_end - global_start) % accelerator.num_processes
    offset = 0
    if accelerator.process_index < rest:
        num_samples_per_process += 1

    if rest > 0:
        offset = min(accelerator.process_index, rest)

    my_start = (
        global_start + accelerator.process_index * num_samples_per_process + offset
    )
    my_end = my_start + num_samples_per_process

    if is_main:
        logger.info(
            f"Total samples: {global_end - global_start}, per-rank({accelerator.process_index}) assigned: [{my_start}: {my_end})"
        )

    # Build local lists for inputs, masks, and original logits
    inputs_list = []
    answer_masks_list = []
    org_logits_list = []

    # Mini-batch over my_indices
    pos = my_start
    while pos < my_end:
        batch = data[pos : min(pos + batch_size, my_end)]
        if is_main:
            logger.info(
                f"[Rank {accelerator.process_index}] Processing: [{pos}, {min(pos + batch_size, my_end)})"
            )
        pos += batch_size

        batched_messages = [
            text_to_message(text) for text in batch["model_input_org_text"]
        ]
        batched_messages = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, return_tensors="pt"
        )
        tmp = []
        if dataset == "gqa" or dataset == "coco" or dataset == "video_mmmu":
            for i, input in enumerate(batched_messages):
                batched_messages[i] = (
                    batched_messages[i] + batch["model_input_full_answer"][i]
                )
                if model_name == "Kimi-VL":
                    batched_messages[i] = batched_messages[i] + "<|im_end|>[EOS]"
                elif model_name == "Qwen3-VL":
                    batched_messages[i] = batched_messages[i] + "<|im_end|>"
                else:
                    ValueError(f"Not support {model_name}")
                if dataset == "video_mmmu":
                    tmp.extend(batch["model_input_visual"][i])
                    videos = batch["model_input_visual"][i]
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
        ).to(model.device)

        # Append CPU copies to avoid GPU memory blow-up during search
        inputs_list.append(to_device(inputs, "cpu"))

        # Determine special token for answer mask creation
        special_token_id = None
        if model_name == "Kimi-VL":
            special_token_id = 163588
            offset = 3
        elif model_name == "Qwen3-VL":
            special_token_id = 151644
            offset = 3
        else:
            raise ValueError(f"Not support {model_name}")

        answer_masks = model_create_mask_after_token(
            input_ids=inputs["input_ids"],
            special_token_id=special_token_id,
            offset=offset,  # offset 2 for <|im_end|>[EOS]
        ).to("cpu")
        answer_masks_list.append(answer_masks)

        with torch.no_grad():
            org_output = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        # Extract logits at answer positions and move to CPU
        org_logits = org_output.logits[answer_masks, :].contiguous()
        org_logits_list.append(org_logits.to("cpu"))

        # Cleanup
        del org_output, org_logits, inputs, answer_masks
        torch.cuda.empty_cache()
        gc.collect()

    # Wait until all ranks finish preparing their local lists
    accelerator.wait_for_everyone()

    # Build tau grids
    text_grids = np.linspace(text_tau_min, text_tau_max, grid_num).tolist()
    visual_grids = np.linspace(visual_tau_min, visual_tau_max, grid_num).tolist()
    if grid_map == "exp":
        text_tau_grids = [2 / (1 + math.exp(-(x - 1) * exp_coeff)) for x in text_grids]
        visual_tau_grids = [
            2 / (1 + math.exp(-(x - 1) * exp_coeff)) for x in visual_grids
        ]
    elif grid_map == "linear":
        text_tau_grids = text_grids
        visual_tau_grids = visual_grids
    else:
        raise ValueError(f"Not support {grid_map}")

    # Partially apply the distributed gen_data_loss
    gen_data_loss_fn = partial(
        gen_data_loss,
        model=model,
        accelerator=accelerator,
        loss_type=loss_type,
        temperature=temperature,
        inputs_list=inputs_list,
        answer_masks_list=answer_masks_list,
        org_logits_list=org_logits_list,
        enable_load_expert_importance=enable_load_expert_importance,
        enable_load_layer_importance=enable_load_layer_importance,
        expert_importance_path=expert_importance_path,
        layer_importance_path=layer_importance_path,
    )

    # Cache wrapper; because gen_data_loss_fn already returns globally reduced scalars,
    # each process will cache the same values for the same tau pairs.
    func, cache = cache_f(gen_data_loss_fn)

    # Search tau under the global constraints; logs only on main
    text_tau, visual_tau, best_val = minimize_g_subject_to_f(
        N=text_tau_grids,
        M=visual_tau_grids,
        func=func,
        T=target_skip_proportion,
        is_main_process=is_main,
    )

    # Synchronize before saving/printing
    accelerator.wait_for_everyone()

    if is_main:
        logger.info(
            f"text_tau = {text_tau}, visual_tau = {visual_tau}, best_val = {best_val}"
        )
        results = {
            "tau": {
                "text": text_tau,
                "visual": visual_tau,
            },
            "best_val": best_val,
            "skip_proportion": (
                cache[(text_tau, visual_tau)]
                if (text_tau, visual_tau) in cache
                else None
            ),
            "cache": cache,
        }
        if exp_coeff != 10:
            with open(
                os.path.join(
                    save_dir,
                    f"{loss_type}_{start_idx}_{num_samples}_{target_skip_proportion}_text_{text_tau_min}_{text_tau_max}_visual_{visual_tau_min}_{visual_tau_max}_grid{grid_num}_exp{exp_coeff}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(results, f)
        else:
            with open(
                os.path.join(
                    save_dir,
                    f"{loss_type}_{start_idx}_{num_samples}_{target_skip_proportion}_text_{text_tau_min}_{text_tau_max}_visual_{visual_tau_min}_{visual_tau_max}_grid{grid_num}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(results, f)
