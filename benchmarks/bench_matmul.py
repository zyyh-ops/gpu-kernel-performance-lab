"""Controlled fixed-configuration GEMM sweeps with structured result output."""
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


def configurations(experiment):
    base = dict(block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3)
    if experiment == "spill":
        return [MatmulConfig(128, 256, 32, w, 3) for w in (4, 8)]
    if experiment == "tile":
        return [MatmulConfig(m, n, 32, w, 3) for m, n, w in
                ((128, 128, 4), (128, 128, 8), (128, 256, 4),
                 (128, 256, 8), (256, 128, 8))]
    if experiment == "stages":
        return [MatmulConfig(**(base | {"num_stages": s})) for s in (1, 2, 3, 4)]
    if experiment == "block-k":
        return [MatmulConfig(**(base | {"block_k": bk})) for bk in (16, 32, 64)]
    raise ValueError(experiment)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=["spill", "tile", "stages", "block-k"], default="spill")
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-ms", type=int, default=25)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA GPU is required")
    if min(args.M, args.N, args.K, args.repeats, args.warmup_ms, args.rep_ms) <= 0:
        ap.error("Shapes, repeat counts, and time budgets must be positive")

    torch.manual_seed(0)
    a = torch.randn((args.M, args.K), device="cuda", dtype=torch.float16)
    b = torch.randn((args.K, args.N), device="cuda", dtype=torch.float16)
    c = torch.empty((args.M, args.N), device="cuda", dtype=torch.float16)
    
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    ref_fp32 = torch.matmul(a.float(), b.float())   # Not timed. Ensure every tested configuration computes correct results.
    device = torch.cuda.current_device()
    info = {
        "experiment": args.experiment,
        "shape": {"M": args.M, "N": args.N, "K": args.K},
        "dtype": "fp16_inputs_fp32_accum_fp16_output",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "measurement": {
            "timer": "triton.testing.do_bench (CUDA events)",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "outer_repeats": args.repeats,
            "aggregation": "median of each do_bench median",
            "notes": "JIT compile and ref matmul excluded. Effective TFLOP/s = 2MNK / latency."
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }

    for cfg in configurations(args.experiment):
        def launch():
            return launch_fixed(a, b, c, cfg)

        kernel = launch()  # JIT/compile outside timing.
        torch.cuda.synchronize()
        torch.testing.assert_close(c.float(), ref_fp32, rtol=1e-2, atol=1e-2)
        timings = [triton.testing.do_bench(launch, warmup=args.warmup_ms, rep=args.rep_ms)
                   for _ in range(args.repeats)]
        ms = statistics.median(timings)
        row = {
            "config": {
                "BLOCK_M": cfg.block_m, "BLOCK_N": cfg.block_n, "BLOCK_K": cfg.block_k,
                "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
            },
            "time_ms_median": ms,
            "time_ms_outer_repeats": timings,
            "tflops": 2.0 * args.M * args.N * args.K / (ms * 1e9),
            "n_regs": kernel.n_regs,
            "n_spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
            "correctness": "PASS",
        }
        info["results"].append(row)
        print(f"{row['config']} -> {ms:.4f} ms / {row['tflops']:.2f} TFLOP/s "
              f"/ regs={row['n_regs']} spills={row['n_spills']}")

    output_path = args.output or (Path(__file__).resolve().parents[1] / "results" / "summary" /
                                 f"matmul_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
