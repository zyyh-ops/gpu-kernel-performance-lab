# GPU Kernel Performance Lab — CUDA / Triton

A GPU kernel implementation and performance diagnosis laboratory built from CUDA Week 10–12 and Triton Week 13 exercises, oriented toward inference workloads. This repository prioritizes **correctness → controlled experiments → compiler evidence → causal explanation** over isolated peak numbers.

## Current scope and verification status

| Component | In this package | Status |
|---|---|---|
| Triton FP16 Matmul | Source + offline-checkable scripts | **Refactored; GPU revalidation pending** |
| Triton Paged KV Write | Source, tests, benchmark, inspection | **Refactored; GPU revalidation pending** |
| Triton Softmax / Vector Add | Listed in inventory only | **Source not yet migrated** |
| CUDA 11 kernels | Transpose implemented; 10 other originals not yet migrated | **Transpose CPU mapping checked; GPU pending** |
| Historical measurements | JSON records | **Historical experiments; not a new benchmark run** |

**IMPORTANT:** This package includes both supplied Matmul and Paged KV Write sources and their refactors. Other CUDA and Triton source files have not been supplied; do not mistake directory placeholders for implementations. Copy those original sources when ready.

## Start without a GPU

From repository root:

```bash
python scripts/offline_audit.py
python -m unittest discover -s tests -p "test_paged_kv_mapping_cpu.py" -v
```

The audit checks Python syntax and historical-result structure. The additional CPU-only unit tests verify the mathematical slot-mapping contract, **not** the GPU kernel. Neither step runs Triton or provides a GPU correctness PASS.

Read:
- [`docs/inventory.md`](docs/inventory.md) for source-file migration mapping
- [`docs/cases/matmul_spill.md`](docs/cases/matmul_spill.md) for historical Matmul performance diagnosis
- [`docs/cases/paged_kv_write.md`](docs/cases/paged_kv_write.md) for the second refactored GPU kernel case study
- [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the GPU rerun protocol
- [`docs/gpu_validation_queue.md`](docs/gpu_validation_queue.md) for pending GPU work

## Paged KV Write validation later

```bash
python -m unittest discover -s tests -p 'test_paged_kv.py' -v
python benchmarks/bench_paged_kv.py --experiment locality
python benchmarks/bench_paged_kv.py --experiment block-f
python benchmarks/bench_paged_kv.py --experiment scale
python scripts/inspect_paged_kv.py --kind both --show ptx --save-dir results/raw
```

The default launcher performs a safe one-time host-side slot-range/uniqueness check; benchmarks bypass it per launch **only after** pre-validation outside the measured region.

## Run later on an NVIDIA CUDA GPU

Install a compatible PyTorch (CUDA) + Triton environment. Original reported platform: RTX 4090, sm_89, Triton 3.8.0.

```bash
python -m unittest discover -s tests -p "test_matmul.py" -v
python benchmarks/bench_matmul.py --experiment spill
python benchmarks/bench_matmul.py --experiment stages
python benchmarks/bench_matmul.py --experiment block-k
python scripts/inspect_matmul.py --BM 128 --BN 256 --warps 4 --save-dir results/raw
python scripts/inspect_matmul.py --BM 128 --BN 256 --warps 8 --save-dir results/raw
python scripts/autotune_matmul.py
```

Store **new runs** in `results/summary/`. Never overwrite `results/historical/` with current measurements. Compare only under aligned hardware, software, workloads, and benchmark methodology.

## Limitations

No claim of GPU correctness/performance reproduction for the refactored Matmul or Paged KV Write source is made. No Nsight Compute cache/DRAM counters are available in these materials. Reported bandwidth is *effective bandwidth*, not directly measured DRAM traffic. Historical Matmul speedup concerns **one fixed-shape/config A/B**, not a speedup over PyTorch.

## Development sequence

1. Matmul: document and archive historical performance reasoning (completed offline).
2. Paged KV Write: source refactored, tests/benchmarks/docs added; GPU revalidation pending.
3. CUDA case studies: transplant source and consolidate tests/benchmarks (next).
4. Obtain GPU: run full correctness + paired benchmarks and mark reproduced cases individually.

## CUDA transpose (new Phase 4 module)

Preserved original source in `docs/archive/matrix_transpose_week10_original.cu`. Main candidate in `cuda/memory/matrix_transpose.cu`, kernel definitions in `cuda/memory/transpose_kernels.cuh`.

Offline, no CUDA toolkit required:

```bash
python -m unittest discover -s tests -p 'test_transpose_mapping_cpu.py' -v
python scripts/offline_audit.py
```

When you have a CUDA GPU and nvcc:

```bash
make build/matrix_transpose CUDA_ARCH=sm_89
make cuda-transpose-test
make cuda-transpose-bench
```

See `docs/cases/cuda_transpose.md` for the full experiment and an inconsistency in earlier written padding percentages.
