"""cuBLAS (torch.mm) 对照 benchmark：与 Triton 固定配置 GEMM 在同一口径下比较。

目的：回答面试必问的「你的 184.76 TFLOP/s 和 cuBLAS 差多少」。

公平性设计：
  1. 同一 shape、同一 dtype（FP16 输入 / FP16 输出 / FP32 累加）。
  2. 同一个计时器 triton.testing.do_bench（CUDA events），同样 warmup/rep。
  3. 两边都写进预分配的 out 张量（torch.mm(..., out=c)），避免分配器开销差异。
  4. 默认每轮之间刷 L2（do_bench fast_flush），避免 A/B 常驻 L2 抬高吞吐；
     用 --no-flush 可观察热 L2 下两者各自的变化。
  5. cuBLAS 的 FP16 归约开关（split-K 部分和用 FP16）会同时影响精度和速度，
     脚本对 on/off 两种都计时；**报告时以 off（FP32 累加）为主口径**。

关于参考值的教训（2026-11-01）：
  最初把 torch.mm 的默认输出当参考，1024³ 上出现 524 个元素失配。
  原因是 1024³ 只有 32 个 CTA（BM128/BN256），cuBLAS 为填满 SM 会切 K 做
  split-K，而 PyTorch 默认 allow_fp16_reduced_precision_reduction=True，
  split-K 部分和用 FP16 舍入，最大偏差约 2 ulp；Triton kernel 是纯 FP32 累加。
  因此本脚本在生成参考值前先关闭该开关，并用「2 ulp」量级的 atol 判正确性。

用法（在 RTX 5090 机器上，仓库根目录）：
    python benchmarks/bench_cublas.py
    python benchmarks/bench_cublas.py --sweep 1024 2048 4096 8192
    python benchmarks/bench_cublas.py --no-flush

输出：终端表格 + results/summary/cublas_vs_triton_<timestamp>.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "triton"))
from matmul import MatmulConfig, launch_fixed  # noqa: E402

# RTX 5090: FP16 Tensor Core dense, FP32 accumulate
DEFAULT_PEAK_TFLOPS = 209.5
# FP16 有 10 位显式尾数
FP16_MANTISSA_BITS = 10


def bench(fn, warmup_ms, rep_ms, repeats, fast_flush):
    timings = [
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, fast_flush=fast_flush)
        for _ in range(repeats)
    ]
    return statistics.median(timings), timings


def mismatch_count(x, y, rtol=1e-2, atol=1e-2):
    return int(((x.float() - y.float()).abs() > atol + rtol * y.float().abs()).sum().item())


def run_shape(m, n, k, cfg, args, info):
    flops = 2.0 * m * n * k
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c_blas = torch.empty((m, n), device="cuda", dtype=torch.float16)
    c_blas_fp16red = torch.empty((m, n), device="cuda", dtype=torch.float16)
    c_triton = torch.empty((m, n), device="cuda", dtype=torch.float16)

    prev_flag = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)

    # ---- 参考值与正确性：FP32 累加口径 ----
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.mm(a, b, out=c_blas)
    kernel = launch_fixed(a, b, c_triton, cfg)
    torch.cuda.synchronize()

    scale = float(c_blas.abs().max().item())
    atol = max(1e-3, 2.0 * scale * (2.0 ** -FP16_MANTISSA_BITS))  # 约 2 ulp
    max_diff = (c_triton.float() - c_blas.float()).abs().max().item()
    torch.testing.assert_close(c_triton, c_blas, rtol=1e-2, atol=atol)

    # ---- fp64 ground truth（只取前若干行，避免 fp64 太慢）----
    rows = min(64, m)
    gold = a[:rows].to(torch.float64) @ b.to(torch.float64)
    err_triton = (c_triton[:rows].to(torch.float64) - gold).abs().max().item()
    err_blas = (c_blas[:rows].to(torch.float64) - gold).abs().max().item()

    # ---- 另一个参考：cuBLAS 默认（允许 FP16 归约）----
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.mm(a, b, out=c_blas_fp16red)
    torch.cuda.synchronize()
    err_blas_fp16red = (c_blas_fp16red[:rows].to(torch.float64) - gold).abs().max().item()
    mism_fp16red = mismatch_count(c_triton, c_blas_fp16red)
    diff_between_cublas = (c_blas_fp16red.float() - c_blas.float()).abs().max().item()

    row = {
        "shape": {"M": m, "N": n, "K": k},
        "correctness": "PASS",
        "reference": "cublas fp32-accumulate (allow_fp16_reduced_precision_reduction=False)",
        "tolerance": {
            "rtol": 1e-2,
            "atol": atol,
            "atol_rule": "2 * max|ref| * 2^-10 (about 2 fp16 ulp)",
        },
        "max_abs_diff_triton_vs_reference": max_diff,
        "fp64_slice_rows": rows,
        "max_abs_err_vs_fp64": {
            "triton": err_triton,
            "cublas_fp32acc": err_blas,
            "cublas_fp16red": err_blas_fp16red,
        },
        "cublas_fp16red_vs_fp32acc_max_abs_diff": diff_between_cublas,
        "mismatch_count_vs_cublas_fp16red": mism_fp16red,
        "impls": {},
    }

    # ---- 计时：cuBLAS on / off ----
    for flag in (False, True):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
        ms, times = bench(
            lambda: torch.mm(a, b, out=c_blas),
            args.warmup_ms, args.rep_ms, args.repeats, not args.no_flush,
        )
        name = f"cublas_fp16_reduction_{'on' if flag else 'off'}"
        row["impls"][name] = {
            "time_ms": ms, "time_ms_repeats": times,
            "tflops": flops / (ms * 1e9),
            "pct_of_peak": 100.0 * flops / (ms * 1e9) / args.peak_tflops,
        }
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev_flag

    # ---- 计时：Triton ----
    ms, times = bench(
        lambda: launch_fixed(a, b, c_triton, cfg),
        args.warmup_ms, args.rep_ms, args.repeats, not args.no_flush,
    )
    row["impls"]["triton_fixed"] = {
        "time_ms": ms, "time_ms_repeats": times,
        "tflops": flops / (ms * 1e9),
        "pct_of_peak": 100.0 * flops / (ms * 1e9) / args.peak_tflops,
        "config": {
            "BLOCK_M": cfg.block_m, "BLOCK_N": cfg.block_n, "BLOCK_K": cfg.block_k,
            "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
        },
        "n_regs": kernel.n_regs,
        "n_spills": kernel.n_spills,
    }

    tri = row["impls"]["triton_fixed"]["tflops"]
    blas = row["impls"]["cublas_fp16_reduction_off"]["tflops"]
    row["triton_vs_cublas_pct"] = 100.0 * tri / blas
    info["results"].append(row)

    print(f"M={m} N={n} K={k}")
    for name, r in row["impls"].items():
        print(f"  {name:<32} {r['time_ms']:8.4f} ms  {r['tflops']:8.2f} TFLOP/s  "
              f"{r['pct_of_peak']:6.2f}% peak")
    print(f"  {'triton / cublas(fp32-acc)':<32} {'':>27} {row['triton_vs_cublas_pct']:6.2f}%")
    print(f"  correctness: max|triton-ref|={max_diff:.3e} (atol={atol:.3e}) | "
          f"err vs fp64: triton={err_triton:.3e}, cublas(fp32acc)={err_blas:.3e}, "
          f"cublas(fp16red)={err_blas_fp16red:.3e}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--sweep", type=int, nargs="+", default=None,
                    help="方形 sweep，例如 --sweep 1024 2048 4096 8192")
    ap.add_argument("--warmup-ms", type=int, default=25)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    ap.add_argument("--no-flush", action="store_true", help="不刷 L2（热缓存，仅作对照）")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        ap.error("CUDA GPU is required")

    device = torch.cuda.current_device()
    info = {
        "experiment": "cublas_vs_triton",
        "dtype": "fp16_inputs_fp32_accum_fp16_output",
        "peak_tflops_assumed": args.peak_tflops,
        "peak_note": "RTX 5090 FP16 Tensor Core dense 209.5 TFLOP/s (FP32 accumulate); "
                     "419 TFLOP/s with FP16 accumulate — do not mix the two.",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "cublas_fp16_reduction_default": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "measurement": {
            "timer": "triton.testing.do_bench (CUDA events)",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "outer_repeats": args.repeats,
            "aggregation": "median of each do_bench median",
            "l2_flushed": not args.no_flush,
            "notes": "Both impls write into preallocated out tensors; "
                     "TFLOP/s = 2MNK / latency.",
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }

    cfg = MatmulConfig(128, 256, 32, 8, 3)
    shapes = [(s, s, s) for s in args.sweep] if args.sweep else [(args.M, args.N, args.K)]
    for m, n, k in shapes:
        run_shape(m, n, k, cfg, args, info)

    out = args.output or (Path(__file__).resolve().parents[1] / "results" / "summary" /
                          f"cublas_vs_triton_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
