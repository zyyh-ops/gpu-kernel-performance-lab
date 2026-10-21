"""Correctness suite for the fixed-config and autotuned Tensor-Core GEMM."""
import os
from pathlib import Path
import sys
import unittest

import torch

# The directory is named `triton` for the user's file tree; it is NOT a package.
# Avoid `from triton.matmul import ...`, which would resolve to the third-party lib.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "triton"))
from matmul import MatmulConfig, matmul_triton  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class TestMatmul(unittest.TestCase):
    def test_fixed_edge_shapes(self):
        torch.manual_seed(0)
        # Includes partial M/N/K blocks; covers source's odd dimensions.
        shapes = [(32, 32, 32), (128, 128, 128), (129, 257, 65),
                  (256, 384, 512), (511, 513, 257)]
        for m, n, k in shapes:
            with self.subTest(shape=(m, n, k)):
                a = torch.randn((m, k), device="cuda", dtype=torch.float16)
                b = torch.randn((k, n), device="cuda", dtype=torch.float16)
                ref = torch.matmul(a, b)
                out = matmul_triton(a, b)
                torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    def test_large_tile_and_warp_config(self):
        a = torch.randn((137, 129), device="cuda", dtype=torch.float16)
        b = torch.randn((129, 269), device="cuda", dtype=torch.float16)
        cfg = MatmulConfig(block_m=128, block_n=256, block_k=32, num_warps=8, num_stages=3)
        torch.testing.assert_close(matmul_triton(a, b, cfg), a @ b, rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(os.getenv("RUN_LARGE") == "1", "Set RUN_LARGE=1 for 4096³ case")
    def test_large_4096(self):
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(matmul_triton(a, b), a @ b, rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(os.getenv("RUN_AUTOTUNE") == "1", "Set RUN_AUTOTUNE=1 to validate tuned path")
    def test_autotuned_path(self):
        a = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        b = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(matmul_triton(a, b, autotune=True), a @ b, rtol=1e-2, atol=1e-2)

    def test_invalid_cpu_input(self):
        a = torch.randn((16, 16), dtype=torch.float16)
        b = torch.randn((16, 16), dtype=torch.float16)
        with self.assertRaises(ValueError):
            matmul_triton(a, b)


if __name__ == "__main__":
    unittest.main()
