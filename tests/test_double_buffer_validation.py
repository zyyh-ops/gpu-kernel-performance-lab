"""Host numerical checks plus opt-in integration tests of the real CUDA binary.

GPU: DOUBLE_BUFFER_BINARY=./build/double_buffer python -m unittest discover \
    -s tests -p test_double_buffer_validation.py -v
No GPU/compiler evidence is implied by the host test.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ.get("DOUBLE_BUFFER_BINARY")
COMPILER = shutil.which("g++") or shutil.which("clang++")


class HostValidationTest(unittest.TestCase):
    @unittest.skipUnless(COMPILER, "requires g++ or clang++")
    def test_numerical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / ("validation.exe" if os.name == "nt" else "validation")
            subprocess.run([
                COMPILER, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "cuda/optimization"),
                str(ROOT / "tests/gemm_validation_host.cpp"), "-o", str(exe),
            ], check=True, capture_output=True, text=True, timeout=120)
            subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)


@unittest.skipUnless(BINARY, "set DOUBLE_BUFFER_BINARY to run actual CUDA integration tests")
class GPUValidationTest(unittest.TestCase):
    def run_case(self, path, *args):
        return subprocess.run([str(Path(BINARY).resolve()), *args, "--json", str(path)],
                              capture_output=True, text=True, timeout=180)

    def test_cpu_and_cublas_references_across_paths_and_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            for reference in ("cpu", "cublas"):
                path = Path(tmp) / f"{reference}.json"
                proc = self.run_case(path, "--mode", "test", "--test-sizes", "16,17,33",
                                     "--extra-kib", "0,32", "--reference", reference)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                data = json.loads(path.read_text())
                self.assertEqual(data["status"], "gpu_correctness_passed")
                self.assertEqual(len(data["correctness"]), 12)
                self.assertEqual(data["results"], [])
                for row in data["correctness"]:
                    self.assertEqual(row["status"], "PASS")
                    self.assertEqual(row["elements_checked"], row["N"] ** 2)
                    self.assertEqual(row["nonfinite_count"], 0)
                    self.assertLessEqual(row["max_scaled_error_finite"], 1.)
                self.assertEqual({r["extra_shared_kib"] for r in data["correctness"]}, {0, 32})

    def test_benchmark_validates_exact_config_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            args = ("--mode", "bench", "--n", "272", "--extra-kib", "32",
                    "--warmup", "1", "--repeat", "1", "--samples", "1")
            proc = self.run_case(path, *args)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            raw = path.read_bytes()
            data = json.loads(raw)
            self.assertEqual(data["status"], "gpu_correctness_and_benchmark_passed")
            self.assertEqual(len(data["correctness"]), 2)
            for row in data["correctness"]:
                self.assertEqual((row["N"], row["extra_shared_kib"]), (272, 32))
                self.assertEqual(row["reference"], "cublas_fp32_pedantic")
                self.assertEqual(row["status"], "PASS")
            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["extra_shared_kib"], 32)
            proc = self.run_case(path, *args)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(path.read_bytes(), raw)

    def test_invalid_reservation_cannot_produce_performance_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.json"
            proc = self.run_case(path, "--mode", "bench", "--n", "17", "--extra-kib", "1048576")
            self.assertNotEqual(proc.returncode, 0)
            data = json.loads(path.read_text())
            self.assertEqual(data["status"], "FAILED")
            self.assertEqual(data["results"], [])
            self.assertTrue(data["error"])


if __name__ == "__main__":
    unittest.main()
