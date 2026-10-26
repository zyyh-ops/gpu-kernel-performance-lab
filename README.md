# GPU Kernel Performance Lab — CUDA / Triton

A GPU kernel implementation and performance diagnosis laboratory built from CUDA Week 10–12 and Triton Week 13 exercises, oriented toward inference workloads. This repository prioritizes **correctness → controlled experiments → compiler evidence → causal explanation** over isolated peak numbers.

## Current scope and verification status

| Component | In this package | Status |
|---|---|---|
| Triton FP16 Matmul | Source + offline-checkable scripts | **Refactored; GPU revalidation pending** |
| Triton Paged KV Write | Historical case outline only | **Source not yet migrated** |
| Triton Softmax / Vector Add | Listed in inventory only | **Source not yet migrated** |
| CUDA 11 kernels | Directory/inventory only | **Sources not yet migrated** |
| Historical measurements | JSON records | **Historical experiments; not a new benchmark run** |

**IMPORTANT:** This package contains the previously supplied Matmul Phase 1 refactor plus documentation. It does **not** invent or reconstruct CUDA/Paged-KV source files that have not been provided. Copy your original sources into the directories identified in `docs/inventory.md` when ready.

## Start without a GPU

From repository root:

```bash
python scripts/offline_audit.py
```

The audit checks Python syntax and historical-result structure using the Python standard library. It **does not** run Triton or provide a correctness PASS.

Read:
- [`docs/inventory.md`](docs/inventory.md) for source-file migration mapping
- [`docs/cases/matmul_spill.md`](docs/cases/matmul_spill.md) for historical Matmul performance diagnosis
- [`docs/cases/paged_kv_write.md`](docs/cases/paged_kv_write.md) for the next module's offline plan
- [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the GPU rerun protocol
- [`docs/gpu_validation_queue.md`](docs/gpu_validation_queue.md) for pending GPU work

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

No claim of GPU correctness/performance reproduction for the refactored Matmul source is made. No Nsight Compute cache/DRAM counters are available in these materials. Reported bandwidth is *effective bandwidth*, not directly measured DRAM traffic. Historical Matmul speedup concerns **one fixed-shape/config A/B**, not a speedup over PyTorch.

## Development sequence

1. Matmul: document and archive historical performance reasoning (completed offline).
2. Paged KV Write: review and modularize the source when provided (next).
3. CUDA case studies: transplant source and consolidate tests/benchmarks.
4. Obtain GPU: run full correctness + paired benchmarks and mark reproduced cases individually.
