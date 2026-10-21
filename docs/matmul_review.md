# Matmul Engineering Review — Phase 1

## Scope and provenance

Refactored from Week 13's original `triton/matmul.py`. This is a **source-level candidate**: Python syntax has been checked, but CUDA correctness/performance have **not** been verified on a GPU in the editing environment.

## Source issues addressed

1. **Autotuner and fixed-config parameters collided.** The undecorated JIT `matmul_kernel` handles explicit configurations, and `autotuned_matmul_kernel` separately wraps it.
2. **Grid was hard-coded using 32×32 even when actual tuned blocks were larger.** Grid is now computed from chosen fixed blocks or autotune metadata.
3. **Benchmarks were only printed.** Sweep runner now saves timestamped JSON with shape, dtype, device, software versions, per-configuration latency/TFLOP/s/resources, and correctness status.
4. **Test and inspection were embedded in one file with commented-out entry points.** Dedicated test/benchmark/inspect scripts make each operation explicit.
5. **Generated code was sometimes mislabeled as PTX when it was SASS.** Inspection labels the two separately and treats unavailable SASS as optional.

## Invariants kept

- FP16 contiguous 2-D inputs, FP32 accumulation, FP16 outputs.
- Row-major K loop with masked A/B loads, masked C stores.
- Same seven autotune candidates as the uploaded source.
- Same core experiments: tile/warp, stages, BK, autotuning, resources.

## Important methodological distinctions

- The 22.86→173.34 TFLOP/s result refers to *one fixed 4096³ workload, one BM=128, BN=256, BK=32 tile, w4 vs w8*. Do not describe it as generic PyTorch-to-Triton speedup.
- With `num_stages` fixed, `n_spills` and local-memory instructions form stronger evidence than attributing any speedup to occupancy alone.
- BF16, FP32 input, noncontiguous, arbitrary strides, and general layouts have **not** been added.
- No Nsight Compute hardware counters were collected. Do not claim a measured cache hit rate or DRAM utilization.
- Benchmarks should ideally run on the same RTX 4090, GPU clocks/power conditions, software stack, and measurement methodology before comparing to historical results.

## GPU validation checklist

- [ ] `python -m unittest discover -s tests -p test_matmul.py -v`
- [ ] `python benchmarks/bench_matmul.py --experiment spill`
- [ ] `python benchmarks/bench_matmul.py --experiment stages`
- [ ] `python benchmarks/bench_matmul.py --experiment block-k`
- [ ] `python scripts/inspect_matmul.py --BM 128 --BN 256 --warps 4 --save-dir results/raw`
- [ ] `python scripts/inspect_matmul.py --BM 128 --BN 256 --warps 8 --save-dir results/raw`
- [ ] `python scripts/autotune_matmul.py`
- [ ] Examine results against historical data before updating resume/README claims
