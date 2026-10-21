# GPU Kernel Performance Lab — Matmul Phase 1 Refactor

This archive contains a **drop-in candidate** for the existing project files, not a fully GPU-validated release. Preserve the original Week-13 source and experiment logs in Git before replacing anything.

## File layout

- `triton/matmul.py`: undecorated JIT kernel, explicit launch API and separate autotune wrapper.
- `tests/test_matmul.py`: fixed-config correctness tests (with opt-in expensive tests).
- `benchmarks/bench_matmul.py`: 4096³ controlled fixed-configuration sweeps and JSON output.
- `scripts/inspect_matmul.py`: compiler resource, PTX and SASS inspection.
- `scripts/autotune_matmul.py`: isolated autotune path.
- `docs/matmul_review.md`: design rationale and validation checklist.

## Environment

Requires an NVIDIA CUDA GPU, PyTorch with CUDA support and Triton. The original experiments used RTX 4090 (sm_89) and Triton 3.8.0; preserve your existing tested environment and record versions rather than blindly upgrading.

**Import note:** `triton/` is a plain repository subfolder, not a Python package. Do not add `triton/__init__.py` or use `from triton.matmul import ...`; that collides with the installed Triton library. Scripts add this directory to `sys.path` and import `matmul` directly.

## Commands (from repository root)

```bash
python -m unittest discover -s tests -p 'test_matmul.py' -v
RUN_LARGE=1 python -m unittest discover -s tests -p 'test_matmul.py' -v
python benchmarks/bench_matmul.py --experiment spill
python benchmarks/bench_matmul.py --experiment tile
python benchmarks/bench_matmul.py --experiment stages
python benchmarks/bench_matmul.py --experiment block-k
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --warps 4 --save-dir results/raw
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --warps 8 --save-dir results/raw
python scripts/autotune_matmul.py
```

Benchmark outputs are written to `results/summary/`. In your repository `.gitignore`, you can ignore `results/raw/` until you decide which compiler artifacts to include. Keep the small summary JSON files tracked.

**Validation status:** source-level refactoring and syntax check only; requires the above GPU run to claim current results are reproduced.
