import time
from typing import List, Optional, Tuple, Union, Dict
from models.kimi import load_model
import models.kimi
from loguru import logger as eval_logger
from tqdm import tqdm
import torch
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages
from accelerate import Accelerator, DistributedType
from lmms_eval.__main__ import cli_evaluate
import lmms_eval.models
import pickle

lmms_eval.models.AVAILABLE_CHAT_TEMPLATE_MODELS["kimi_vl"] = "eval.kimi.KimiVL"
import os
from pathlib import Path
import decord
import numpy as np
from transformers import AutoConfig
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resize_frame(frame: Image.Image, max_long_side: int) -> Image.Image:
    """Resize frame so the longer side equals max_long_side, preserving aspect ratio.

    Args:
        frame: PIL Image to resize.
        max_long_side: Maximum length of the longer side in pixels.

    Returns:
        Resized PIL Image.
    """
    w, h = frame.size
    if max(w, h) <= max_long_side:
        return frame
    if h > w:
        new_h = max_long_side
        aspect_ratio = w / h
        new_w = int(round(new_h * aspect_ratio))
    else:
        new_w = max_long_side
        aspect_ratio = h / w
        new_h = int(round(new_w * aspect_ratio))
    resized_frame = frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized_frame


def process_media(
    file_path: Union[str, Path], max_frames: int = 64, max_long_side: int = 480
) -> tuple:
    """Load video or image and return list of resized PIL frames.

    Args:
        file_path: Path to video or image file.
        max_frames: Max frames to sample from video.
        max_long_side: Max longer side length when resizing.

    Returns:
        Tuple of (list of PIL Images, frame count).

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If file type not supported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {file_path}")
    file_ext = file_path.suffix.lower()
    processed_frames = []
    if file_ext in VIDEO_EXTENSIONS:
        vr = decord.VideoReader(str(file_path), ctx=decord.cpu(0))
        num_frames = len(vr)
        if num_frames <= max_frames:
            indices = list(range(num_frames))
        else:
            indices = np.linspace(0, num_frames - 1, max_frames, dtype=int).tolist()
            if (num_frames - 1) not in indices:
                indices.pop()
                indices.append(num_frames - 1)
        frames_np = vr.get_batch(indices).asnumpy()
        for frame_np in frames_np:
            pil_frame = Image.fromarray(frame_np)
            resized_frame = _resize_frame(pil_frame, max_long_side)
            processed_frames.append(resized_frame)
    elif file_ext in IMAGE_EXTENSIONS:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            resized_image = _resize_frame(img_rgb, max_long_side)
            processed_frames.append(resized_image)
    else:
        raise ValueError(
            f"Not supported file type: '{file_ext}'. "
            f"Supported video formats: {VIDEO_EXTENSIONS}, "
            f"Supported image formats: {IMAGE_EXTENSIONS}"
        )
    return processed_frames, len(processed_frames)


def extract_thinking_and_summary(
    text: str, bot: str = "◁think▷", eot: str = "◁/think▷"
) -> tuple:
    """Split text into thinking content and summary/answer.

    Args:
        text: Full model output text.
        bot: Start marker for thinking block.
        eot: End marker for thinking block.

    Returns:
        Tuple of (thinking_text, summary_text). Empty thinking if eot absent.
    """
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return (
            text[text.index(bot) + len(bot) : text.index(eot)].strip(),
            text[text.index(eot) + len(eot) :].strip(),
        )
    return "", text


class KimiVL(lmms):
    """lmms-eval model wrapper for Kimi-VL with MoDES expert skipping support."""

    is_simple: bool = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}"
            )

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
            "layer_gate_dict": kwargs.get("layer_gate_dict", None),
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        if batch_size == 1 or kwargs.get("tau_skip_path", None) is None:
            layer_gate_dict = {}
            config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
            for i in range(config.text_config.num_hidden_layers):
                if (
                    i >= config.text_config.first_k_dense_replace
                    and i % config.text_config.moe_layer_freq == 0
                    and config.text_config.n_routed_experts is not None
                ):
                    layer_gate_dict[i] = {
                        "text": config.text_config.num_experts_per_tok,
                        "visual": config.text_config.num_experts_per_tok,
                        "text_expert_num": config.text_config.n_routed_experts,
                        "visual_expert_num": config.text_config.n_routed_experts,
                    }

            if kwargs.get("topk", None) is not None:
                topk_text = kwargs.get(
                    "topk_text", config.text_config.num_experts_per_tok
                )
                topk_visual = kwargs.get(
                    "topk_visual", config.text_config.num_experts_per_tok
                )
                start = kwargs.get("start", 0)
                end = kwargs.get("end", -1)
                for i in range(start, end + 1):
                    if i in layer_gate_dict:
                        layer_gate_dict[i]["text"] = topk_text
                        layer_gate_dict[i]["visual"] = topk_visual

            if kwargs.get("expert_num", None) is not None:
                text_expert_num = kwargs.get(
                    "text_expert_num", config.text_config.n_routed_experts
                )
                visual_expert_num = kwargs.get(
                    "visual_expert_num", config.text_config.n_routed_experts
                )
                start = kwargs.get("start", 0)
                end = kwargs.get("end", -1)
                for i in range(start, end + 1):
                    if i in layer_gate_dict:
                        layer_gate_dict[i]["text_expert_num"] = text_expert_num
                        layer_gate_dict[i]["visual_expert_num"] = visual_expert_num

            model_kwargs["layer_gate_dict"] = layer_gate_dict

        expert_importance_path = kwargs.get("expert_importance_path", None)
        enable_load_expert_importance = expert_importance_path is not None
        layer_importance_path = kwargs.get("layer_importance_path", None)
        enable_load_layer_importance = layer_importance_path is not None
        tau_skip_path = kwargs.get("tau_skip_path", None)
        enable_tau_skip = tau_skip_path is not None
        tau = None
        self.tau_skip_path = tau_skip_path if tau_skip_path else ""
        if self.tau_skip_path != "":
            with open(tau_skip_path, "rb") as f:
                eval_logger.info(f"Loading tau skip dict from {tau_skip_path}")
                tau_skip_dict = pickle.load(f)
                tau = tau_skip_dict["tau"]
        self.tau = tau
        self.enable_load_expert_importance = enable_load_expert_importance
        self.enable_load_layer_importance = enable_load_layer_importance
        self.enable_tau_skip = enable_tau_skip
        self.expert_importance_path = expert_importance_path
        self.layer_importance_path = layer_importance_path

        model, processor = load_model(model_path=pretrained, **model_kwargs)
        self._model = model
        self.processor = processor
        self._tokenizer = processor.tokenizer

        self._config = self.model.config
        # self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for KimiVL")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for KimiVL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]
            # import ipdb; ipdb.set_trace()
            media_data = []
            frame_number = []
            if visuals:
                media_data = visuals
            for video in videos:
                v, num = process_media(video)
                media_data.extend(v)
                frame_number.append(num)

            idx = 0

            def to_hf_messages(self):
                nonlocal idx
                hf_messages = []
                for message in self.messages:
                    hf_message = {"role": message.role, "content": []}
                    for content in message.content:
                        if content.type == "text":
                            hf_message["content"].append(
                                {"type": "text", "text": content.text}
                            )
                        elif content.type == "image":
                            hf_message["content"].append(
                                {"type": "image", "image": content.url}
                            )
                        elif content.type == "video":
                            for f in range(frame_number[idx]):
                                hf_message["content"].append(
                                    {"type": "image", "image": content.url}
                                )
                            idx += 1
                    hf_messages.append(hf_message)
                return hf_messages

            batched_messages = [
                to_hf_messages(chat_message) for chat_message in chat_messages
            ]
            texts = self.processor.apply_chat_template(
                batched_messages, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = self.processor(
                text=texts,
                images=media_data,
                padding=True,
                return_tensors="pt",
                padding_side="left",
                truncation=True,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }

            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
                enable_load_expert_importance=self.enable_load_expert_importance,
                enable_load_layer_importance=self.enable_load_layer_importance,
                enable_tau_skip=self.enable_tau_skip,
                tau=self.tau,
                expert_importance_path=self.expert_importance_path,
                layer_importance_path=self.layer_importance_path,
            )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            for ans, context in zip(answers, texts):
                _, clean_ans = extract_thinking_and_summary(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans
                )
                pbar.update(1)
                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        # if self.rank == 0:
        #     with open("./metric_dict.log", "a", encoding="utf-8") as f:
        #         f.write(str(metric_dict) + "\n")
        log_metrics(**metric_dict)

        pbar.close()
        return res


if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    cli_evaluate()
