"""Paged KV Write and linear-control kernels for contiguous FP16 K/V storage.

Input K/V shape: [num_tokens, n_features]; page cache shape:
[num_blocks, page_size, n_features]. The cache is NOT an attention kernel:
it ONLY writes token key/value vectors into physical slots.

Slot mappings MUST be a one-to-one set of valid physical slot indices.
No -1/skip sentinel, aliasing/duplicate mapping, or noncontiguous layout is supported.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KVWriteConfig:
    page_size: int = 16
    block_f: int = 128
    num_warps: int = 4


@triton.jit
def paged_kv_write_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    cache_block_stride,
    cache_slot_stride,
    slot_mapping_ptr,
    n_features,
    PAGE_SIZE: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    # Preserve the mapping logic from the Week-13 original.
    token = tl.program_id(0)
    feature_block = tl.program_id(1)
    feature_offsets = feature_block * BLOCK_F + tl.arange(0, BLOCK_F)
    feature_mask = feature_offsets < n_features

    slot = tl.load(slot_mapping_ptr + token)
    physical_block = slot // PAGE_SIZE
    physical_slot = slot % PAGE_SIZE

    src_offsets = token * n_features + feature_offsets
    dst_offsets = (physical_block * cache_block_stride
                   + physical_slot * cache_slot_stride + feature_offsets)

    k = tl.load(k_ptr + src_offsets, mask=feature_mask, other=0)
    v = tl.load(v_ptr + src_offsets, mask=feature_mask, other=0)
    tl.store(k_cache_ptr + dst_offsets, k, mask=feature_mask)
    tl.store(v_cache_ptr + dst_offsets, v, mask=feature_mask)


@triton.jit
def linear_kv_write_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    n_features,
    BLOCK_F: tl.constexpr,
):
    token = tl.program_id(0)
    feature_block = tl.program_id(1)
    feature_offsets = feature_block * BLOCK_F + tl.arange(0, BLOCK_F)
    feature_mask = feature_offsets < n_features
    offsets = token * n_features + feature_offsets
    k = tl.load(k_ptr + offsets, mask=feature_mask, other=0)
    v = tl.load(v_ptr + offsets, mask=feature_mask, other=0)
    tl.store(k_cache_ptr + offsets, k, mask=feature_mask)
    tl.store(v_cache_ptr + offsets, v, mask=feature_mask)


def _validate_source(k: torch.Tensor, v: torch.Tensor, cfg: KVWriteConfig) -> None:
    if k.ndim != 2 or v.shape != k.shape or v.ndim != 2:
        raise ValueError('Expected equal 2-D K/V tensors [num_tokens, n_features]')
    if k.shape[0] <= 0 or k.shape[1] <= 0:
        raise ValueError('Zero tokens/features not supported in this lab')
    if cfg.page_size <= 0 or cfg.block_f <= 0 or cfg.block_f & (cfg.block_f - 1):
        raise ValueError('page_size must be >0 and block_f a positive power of two')
    if cfg.num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError('num_warps must be a supported positive power-of-two configuration')
    if k.dtype != torch.float16 or v.dtype != k.dtype:
        raise TypeError('The original lab supports FP16 K/V only')
    if k.device.type != 'cuda' or v.device != k.device:
        raise ValueError('K and V must live on the same CUDA device')
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError('K and V must be contiguous [num_tokens, n_features]')


def validate_slot_mapping(slot_mapping: torch.Tensor, capacity: int, n_tokens: int) -> None:
    """Safety guard, not suitable inside benchmark timing (syncs GPU->CPU)."""
    if slot_mapping.ndim != 1 or slot_mapping.numel() != n_tokens:
        raise ValueError('slot_mapping must have one index per token')
    slots = slot_mapping.detach().cpu().tolist()  # Intentionally syncs GPU.
    if any(int(slot) < 0 or int(slot) >= capacity for slot in slots):
        raise ValueError('slot_mapping contains an out-of-range slot')
    if len(set(slots)) != len(slots):
        raise ValueError('Duplicate physical slots cause conflicting writes')


def validate_paged_inputs(
    k: torch.Tensor, v: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor, cfg: KVWriteConfig,
    *, check_slots: bool = True,
) -> None:
    _validate_source(k, v, cfg)
    if k_cache.ndim != 3 or v_cache.shape != k_cache.shape or v_cache.ndim != 3:
        raise ValueError('Expected equal 3-D caches [num_blocks, page_size, n_features]')
    if k_cache.shape[0] <= 0 or k_cache.shape[1:] != (cfg.page_size, k.shape[1]):
        raise ValueError('Cache shape does not match page_size/features')
    if k_cache.dtype != k.dtype or v_cache.dtype != k.dtype:
        raise TypeError('K/V and caches must have the same dtype')
    if k_cache.device != k.device or v_cache.device != k.device:
        raise ValueError('K/V and caches must live on the same CUDA device')
    if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError('Only contiguous 3-D cache layouts are supported')
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != k.shape[0]:
        raise ValueError('slot_mapping must have shape [num_tokens]')
    if slot_mapping.dtype != torch.int32 or slot_mapping.device != k.device or not slot_mapping.is_contiguous():
        raise ValueError('slot_mapping must be contiguous CUDA int32 on K/V device')
    if check_slots:
        validate_slot_mapping(slot_mapping, k_cache.shape[0] * cfg.page_size, k.shape[0])


def launch_paged(
    k: torch.Tensor, v: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor, cfg: KVWriteConfig = KVWriteConfig(),
    *, check_slots: bool = True,
):
    """Launch paged write; return compiled kernel handle.

    Set check_slots=False ONLY when an immutable valid slot mapping has
    already passed safety validation outside the timed region.
    """
    validate_paged_inputs(k, v, k_cache, v_cache, slot_mapping, cfg, check_slots=check_slots)
    grid = (k.shape[0], triton.cdiv(k.shape[1], cfg.block_f))
    return paged_kv_write_kernel[grid](
        k, v, k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1), slot_mapping, k.shape[1],
        PAGE_SIZE=cfg.page_size, BLOCK_F=cfg.block_f, num_warps=cfg.num_warps,
    )


def launch_linear(
    k: torch.Tensor, v: torch.Tensor,
    linear_k_cache: torch.Tensor, linear_v_cache: torch.Tensor,
    cfg: KVWriteConfig = KVWriteConfig(),
):
    """Contiguous control, used to study paging/address-generation cost."""
    _validate_source(k, v, cfg)
    if linear_k_cache.shape != k.shape or linear_v_cache.shape != k.shape:
        raise ValueError('Linear cache must match K/V shape')
    for cache in (linear_k_cache, linear_v_cache):
        if cache.device != k.device or cache.dtype != k.dtype or not cache.is_contiguous():
            raise ValueError('Linear cache must be contiguous CUDA FP16')
    grid = (k.shape[0], triton.cdiv(k.shape[1], cfg.block_f))
    return linear_kv_write_kernel[grid](
        k, v, linear_k_cache, linear_v_cache, k.shape[1],
        BLOCK_F=cfg.block_f, num_warps=cfg.num_warps,
    )
