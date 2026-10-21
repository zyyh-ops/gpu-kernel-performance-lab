from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MatmulConfig:
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3


# Keep these seven candidate configs from the original Week-13 experiment.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_pids = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_pids
    pid_n = pid % num_n_pids

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# IMPORTANT: do NOT autotune-decorate matmul_kernel itself. This wrapper alone
# chooses parameters; all fixed-config tests call the undecorated JIT kernel.
autotuned_matmul_kernel = triton.autotune(
    configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"]
)(matmul_kernel)


def validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2-D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} @ {tuple(b.shape)}")
    if min(*a.shape, *b.shape) <= 0:
        raise ValueError("Zero-sized matrices are not supported by this experiment")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("Only torch.float16 inputs are supported")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("Both matrices must be on the same CUDA device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Both matrices must be contiguous")


def launch_fixed(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, cfg: MatmulConfig):
    """Launch exact user-provided block/warp/stage config; return compiled kernel."""
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, cfg.block_m) * triton.cdiv(n, cfg.block_n),)
    return matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )


def launch_autotuned(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Select config using the autotuner and its actual tile-based launch grid."""
    m, k = a.shape
    _, n = b.shape
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    return autotuned_matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )


def matmul_triton(a: torch.Tensor, b: torch.Tensor, cfg: MatmulConfig | None = None, *, autotune: bool = False):
    validate_inputs(a, b)
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    if autotune:
        if cfg is not None:
            raise ValueError("A fixed configuration cannot be used with autotune=True")
        launch_autotuned(a, b, out)
    else:
        launch_fixed(a, b, out, cfg or MatmulConfig())
    return out
