"""Diagnose numeric disagreements in FP16/FP32-accumulate Triton GEMM.

Place under gpu-kernel-performance-lab/scripts/, run from repo root:
    python scripts/diagnose_matmul_numerics.py --M 4096 --N 4096 --K 4096
This script NEVER relaxes correctness assertions or changes benchmark data.
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton'))
from matmul import MatmulConfig, launch_fixed  # noqa: E402


def error_metrics(actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float):
    x = actual.float()
    y = reference.float()
    absolute = (x - y).abs()
    tolerated = atol + rtol * y.abs()
    fail = absolute > tolerated
    mismatch = int(fail.sum().item())
    count = absolute.numel()
    worst_flat = int(absolute.flatten().argmax().item())
    worst_row, worst_col = divmod(worst_flat, absolute.shape[1])
    rel_l2 = (torch.linalg.vector_norm(x - y) /
              torch.linalg.vector_norm(y).clamp_min(1e-12)).item()
    rms = torch.sqrt(torch.mean((x - y).square())).item()
    print(f'    mismatch = {mismatch}/{count} ({100*mismatch/count:.4f}%)'
          f'  [atol={atol:g}, rtol={rtol:g}]')
    print(f'    max_abs = {absolute.max().item():.8g}; RMSE = {rms:.8g}; rel_L2 = {rel_l2:.8g}')
    print(f'    worst_abs_index=({worst_row}, {worst_col})'
          f' actual={x[worst_row,worst_col].item():.8g}'
          f' ref={y[worst_row,worst_col].item():.8g}'
          f' allowed={tolerated[worst_row,worst_col].item():.8g}')
    if mismatch:
        examples = fail.nonzero()[:3].tolist()
        for i, j in examples:
            print(f'    failing=({i},{j}) actual={x[i,j].item():.8g}'
                  f' ref={y[i,j].item():.8g}'
                  f' error={absolute[i,j].item():.8g}'
                  f' allowed={tolerated[i,j].item():.8g}')
    return mismatch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=4096)
    p.add_argument('--N', type=int, default=4096)
    p.add_argument('--K', type=int, default=4096)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--atol', type=float, default=1e-2)
    p.add_argument('--rtol', type=float, default=1e-2)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('CUDA GPU is required')
    print('Device:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)
    print('Shape:', (args.M, args.N, args.K), 'seed:', args.seed)
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        print('cuBLAS allow_fp16_reduced_precision_reduction:',
              torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    # Avoid TF32 when computing the float32 diagnostic reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        torch.manual_seed(args.seed)
        a = torch.randn((args.M, args.K), device='cuda', dtype=torch.float16)
        b = torch.randn((args.K, args.N), device='cuda', dtype=torch.float16)
        ref16 = a @ b
        ref32 = a.float() @ b.float()
        print('\nPyTorch FP16 GEMM vs FP32 GEMM reference:')
        error_metrics(ref16, ref32, atol=args.atol, rtol=args.rtol)
        c = torch.empty((args.M, args.N), device='cuda', dtype=torch.float16)
        for warps in (4, 8):
            cfg = MatmulConfig(128, 256, 32, warps, 3)
            kernel = launch_fixed(a, b, c, cfg)
            torch.cuda.synchronize()
            print(f'\nTriton warps={warps} regs={kernel.n_regs}'
                  f' spills={kernel.n_spills} shared={kernel.metadata.shared} bytes')
            print('  Against PyTorch FP16 output:')
            error_metrics(c, ref16, atol=args.atol, rtol=args.rtol)
            print('  Against FP32 reference (before FP16 rounding):')
            error_metrics(c, ref32, atol=args.atol, rtol=args.rtol)
            print('  Against FP32 reference rounded to FP16:')
            error_metrics(c, ref32.half(), atol=args.atol, rtol=args.rtol)


if __name__ == '__main__':
    main()
