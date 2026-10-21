"""Run the independent autotune path and verify its selected output."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "triton"))
from matmul import autotuned_matmul_kernel, matmul_triton  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA GPU is required")
    a = torch.randn((args.M, args.K), device="cuda", dtype=torch.float16)
    b = torch.randn((args.K, args.N), device="cuda", dtype=torch.float16)
    output = matmul_triton(a, b, autotune=True)
    torch.testing.assert_close(output, a @ b, rtol=1e-2, atol=1e-2)
    print("Autotuned correctness: PASS")
    print("Best configuration:", autotuned_matmul_kernel.best_config)


if __name__ == "__main__":
    main()
