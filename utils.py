import os
import torch
import functools
import gc
from loguru import logger
from collections import defaultdict
import torch
import torch.nn.functional as F


def create_mask_after_token(
    input_ids: torch.Tensor, special_token_id: int, offset: int = 2
) -> torch.Tensor:
    """Create a boolean mask for positions at/after the first occurrence of special_token_id + offset.

    Args:
        input_ids: Token ids of shape (batch_size, seq_len).
        special_token_id: Token id to locate (e.g., answer start token).
        offset: Number of positions to skip after the special token (default 2).

    Returns:
        Boolean tensor of shape (batch_size, seq_len), True where positions are part of the answer.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device
    token_indices = torch.argmax((input_ids == special_token_id).int(), dim=1)
    token_found = torch.any(input_ids == special_token_id, dim=1)
    effective_indices = torch.where(token_found, token_indices, seq_len)
    start_masking_indices = effective_indices + offset
    col_indices = torch.arange(seq_len, device=device)
    mask = col_indices >= start_masking_indices.unsqueeze(1)
    return mask


def create_mask_after_last_token(
    input_ids: torch.Tensor, special_token_id: int, offset: int = 2
) -> torch.Tensor:
    """Create a boolean mask for positions at/after the last occurrence of special_token_id + offset.

    Args:
        input_ids: Token ids of shape (batch_size, seq_len).
        special_token_id: Token id to locate (e.g., answer start token).
        offset: Number of positions to skip after the last special token (default 2).

    Returns:
        Boolean tensor of shape (batch_size, seq_len), True where positions are part of the answer.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device
    is_special_token = (input_ids == special_token_id).int()
    flipped_matches = torch.flip(is_special_token, dims=[1])
    flipped_indices = torch.argmax(flipped_matches, dim=1)
    token_indices = (seq_len - 1) - flipped_indices
    token_found = torch.any(is_special_token, dim=1)
    effective_indices = torch.where(token_found, token_indices, seq_len)
    start_masking_indices = effective_indices + offset
    col_indices = torch.arange(seq_len, device=device)
    mask = col_indices >= start_masking_indices.unsqueeze(1)
    return mask


def to_cpu_detached(x):
    """Recursively move tensors to CPU and detach from the computation graph.

    Args:
        x: A tensor, list, tuple, dict, or other object (non-tensors returned as-is).

    Returns:
        Same structure as x with tensors moved to CPU and detached.
    """
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu_detached(t) for t in x)
    if isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    return x


def to_device(x, device):
    """Recursively move tensors to the specified device.

    Args:
        x: A tensor, list, tuple, dict, or other object (non-tensors returned as-is).
        device: Target device (e.g., 'cuda:0', 'cpu').

    Returns:
        Same structure as x with tensors on the target device.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def is_tensor(x):
    """Check if x is a PyTorch tensor.

    Args:
        x: Object to check.

    Returns:
        True if x is a torch.Tensor, False otherwise.
    """
    return isinstance(x, torch.Tensor)


class StopForwardException(Exception):
    pass


class DataSaverHook:
    """Forward hook to optionally store inputs, outputs, and stop forward pass."""

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.rec = defaultdict(list)

    def __call__(self, m, args, kwargs, output):
        """Process forward pass: optionally store input/output and raise StopForwardException.

        Args:
            m: The module being hooked.
            args: Positional arguments passed to forward.
            kwargs: Keyword arguments passed to forward.
            output: Output tensor from forward.
        """
        # assert args is None, "DataSaverHook only support kwargs input"
        # logger.info(f"DataSaverHook input kwargs: {kwargs.keys()}")
        # import ipdb; ipdb.set_trace()
        if self.store_input:
            self.rec["input"].append(
                {
                    **{k: to_cpu_detached(v) for k, v in kwargs.items()},
                }
            )
        if self.store_output:
            self.rec["output"].append(to_cpu_detached(output)[0])
        if self.stop_forward:
            raise StopForwardException

    def clear(self):
        """Clear recorded data and free GPU cache."""
        gc.collect()
        self.rec = defaultdict(list)


class GetLayerInpOut:
    """Utility to capture input and output of a specific layer via forward hook."""

    def __init__(self, model) -> None:
        self.model = model
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )

    def __call__(self, layer, from_model_start=False, **kwargs):
        """Run forward and return the input and output of the given layer.

        Args:
            layer: The layer module to hook.
            from_model_start: If True, run full model forward; else run only the layer.
            **kwargs: Forward pass arguments.

        Returns:
            Tuple of (list of input dicts, list of output tensors).
        """
        self.layer = layer
        handle = self.layer.register_forward_hook(self.data_saver, with_kwargs=True)
        with torch.no_grad():
            try:
                if from_model_start:
                    _ = self.model(**kwargs)
                else:
                    _ = self.layer(**kwargs)
            except StopForwardException:
                pass

        handle.remove()
        return self.data_saver.rec["input"], self.data_saver.rec["output"]

    def clear(self):
        """Clear DataSaverHook state and free cache."""
        self.data_saver.clear()
