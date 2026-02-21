import torch
import numpy as np
from typing import List, Dict, Any
from loguru import logger
import torch.nn.functional as F


def apply_tensor_scale(A, topk_ids, topk_weight, mask):
    """Scale topk_weight by A[topk_ids] for rows where mask is True (modality-specific).

    Args:
        A: Tensor of shape [num_experts] with per-expert scale factors.
        topk_ids: Long tensor [N, k] of top-k expert indices.
        topk_weight: Float tensor [N, k] of routing weights (modified in-place).
        mask: Bool tensor [N, 1], True for rows of the target modality.

    Returns:
        topk_weight: The scaled topk_weight tensor (same object, modified in-place).
    """
    device = topk_weight.device
    A = A.to(device=device, dtype=topk_weight.dtype)
    scale = A[topk_ids]
    mask = mask.expand_as(topk_weight)
    prod = topk_weight * scale
    topk_weight[mask] = prod[mask]
    return topk_weight


def apply_scaler_scale(A, topk_weight, mask):
    """Scale topk_weight by scalar A for rows where mask is True.

    Args:
        A: Scalar or 1D tensor with layer importance for the modality.
        topk_weight: Float tensor [N, k] of routing weights (modified in-place).
        mask: Bool tensor [N, 1], True for rows of the target modality.

    Returns:
        topk_weight: The scaled topk_weight tensor (same object, modified in-place).
    """
    mask = mask.expand_as(topk_weight)
    prod = topk_weight * A
    topk_weight[mask] = prod[mask]
    return topk_weight


def threshold_and_mask(
    new_topk_weight, topk_weight, topk_ids, mask, threshold: float, max_ids: int
):
    """Zero out experts with new_topk_weight below threshold for rows where mask is True.

    Args:
        new_topk_weight: Float tensor [N, k] of importance scores for thresholding.
        topk_weight: Float tensor [N, k] of routing weights (modified in-place).
        topk_ids: Long tensor [N, k] of expert indices (modified in-place).
        mask: Bool tensor [N, 1], True for rows of the target modality.
        threshold: Minimum importance score; below this, expert is skipped.
        max_ids: Value to write to topk_ids for skipped experts (e.g., num_experts).

    Returns:
        Tuple of (topk_weight, topk_ids), same shapes as inputs, modified in-place.
    """
    device = topk_weight.device
    dtype = topk_weight.dtype
    mask = mask.expand_as(topk_weight)
    thr = torch.tensor(threshold, device=device, dtype=dtype)
    to_zero = mask & (new_topk_weight < thr)
    topk_weight.masked_fill_(to_zero, 0.0)
    topk_ids.masked_fill_(to_zero, max_ids)

    return topk_weight, topk_ids


def create_conditional_expert_mask(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Create a sparse mask indicating which experts are active per token.

    Args:
        top_k_index: Long tensor [N, k] of top-k expert indices.
        top_k_weights: Float tensor [N, k] of routing weights.
        num_experts: Total number of experts.

    Returns:
        Int tensor [M, k, N] where entry (e, k, n) is 1 if expert e is routed at (n,k).
    """
    N, k = top_k_index.shape
    M = num_experts
    values_to_scatter = (top_k_weights != 0).int().unsqueeze(-1)
    index_for_scatter = top_k_index.unsqueeze(-1)
    target_tensor = torch.zeros(N, k, M, dtype=torch.int, device=top_k_index.device)
    target_tensor.scatter_(2, index_for_scatter, values_to_scatter)
    return target_tensor.permute(2, 1, 0)
