"""Inspect compiled GEMM resources and key PTX/SASS instructions."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "triton"))
from matmul import MatmulConfig, launch_fixed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--BM", type=int, default=128)
    ap.add_argument("--BN", type=int, default=256)
    ap.add_argument("--BK", type=int, default=32)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--show", choices=["ptx", "sass", "both"], default="both")
    ap.add_argument("--save-dir", type=Path, default=None, help="Optional directory for full PTX/SASS artifacts")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA GPU is required")
    a = torch.randn((args.M, args.K), device="cuda", dtype=torch.float16)
    b = torch.randn((args.K, args.N), device="cuda", dtype=torch.float16)
    c = torch.empty((args.M, args.N), device="cuda", dtype=torch.float16)
    cfg = MatmulConfig(args.BM, args.BN, args.BK, args.warps, args.stages)
    kernel = launch_fixed(a, b, c, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)

    print(f"regs/thread={kernel.n_regs}, spills={kernel.n_spills}, "
          f"shared_bytes={kernel.metadata.shared}, warps={cfg.num_warps}")
    keywords = {
        "ptx": ("mma.sync", "ldmatrix", "cp.async", "ld.shared", "st.shared",
                "bar.sync", "ld.global", "st.global"),
        "sass": ("LDL", "STL", "HMMA", "MMA", "LDSM", "LDG", "STG", "CPASYNC"),
    }
    for name in ("ptx", "sass"):
        if args.show not in (name, "both"):
            continue
        try:
            assembly = kernel.asm.get(name)
        except Exception as exc:
            print(f"{name.upper()} unavailable: {exc}")
            continue
        if not assembly:
            print(f"{name.upper()} unavailable in this Triton/toolchain build")
            continue
        print(f"\n--- {name.upper()} filtered instructions ---")
        for line in assembly.splitlines():
            if any(word.lower() in line.lower() for word in keywords[name]):
                print(line)
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            path = args.save_dir / f"matmul_BM{cfg.block_m}_BN{cfg.block_n}_BK{cfg.block_k}_w{cfg.num_warps}_s{cfg.num_stages}.{name}"
            path.write_text(assembly, encoding="utf-8")
            print("Saved:", path)


if __name__ == "__main__":
    main()
