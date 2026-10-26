"""GPU benchmark: linear vs paged, block_f, warps, slot order, and L2 scale.

Benchmarked launch functions skip per-iteration GPU->CPU slot validation only
*after* the mapping is checked once outside the timed region.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton'))
from paged_kv_write import KVWriteConfig, launch_linear, launch_paged, validate_slot_mapping  # noqa: E402


def make_data(num_tokens, features, page_size, seed):
    torch.manual_seed(seed)
    k = torch.randn((num_tokens, features), dtype=torch.float16, device='cuda')
    v = torch.randn_like(k)
    num_blocks = triton.cdiv(num_tokens, page_size)
    kc = torch.zeros((num_blocks, page_size, features), dtype=torch.float16, device='cuda')
    vc = torch.zeros_like(kc)
    kl = torch.empty_like(k)
    vl = torch.empty_like(v)
    sequential = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
    random = torch.randperm(num_tokens, device='cuda').to(torch.int32)
    return k, v, kc, vc, kl, vl, sequential, random


def make_cases(data, page_size, block_f, warps, names):
    k, v, kc, vc, kl, vl, sequential, random = data
    cfg = KVWriteConfig(page_size=page_size, block_f=block_f, num_warps=warps)
    # Do slow O(num_tokens) validation OUTSIDE timing.
    validate_slot_mapping(sequential, kc.shape[0] * page_size, k.shape[0])
    validate_slot_mapping(random, kc.shape[0] * page_size, k.shape[0])
    cases = {
        'linear': lambda: launch_linear(k, v, kl, vl, cfg),
        'paged-sequential': lambda: launch_paged(k, v, kc, vc, sequential, cfg, check_slots=False),
        'paged-random': lambda: launch_paged(k, v, kc, vc, random, cfg, check_slots=False),
    }
    return {name: cases[name] for name in names}


def assert_correctness(data, name):
    k, v, kc, vc, kl, vl, seq, rnd = data
    if name == 'linear':
        torch.testing.assert_close(kl, k, rtol=0, atol=0)
        torch.testing.assert_close(vl, v, rtol=0, atol=0)
        return
    slots = seq if name == 'paged-sequential' else rnd
    refk = torch.zeros_like(kc)
    refv = torch.zeros_like(vc)
    refk.view(-1, k.shape[1]).index_copy_(0, slots.to(torch.int64), k)
    refv.view(-1, k.shape[1]).index_copy_(0, slots.to(torch.int64), v)
    torch.testing.assert_close(kc, refk, rtol=0, atol=0)
    torch.testing.assert_close(vc, refv, rtol=0, atol=0)


def run_group(args, num_tokens, features, block_f, warps, names):
    data = make_data(num_tokens, features, args.page_size, args.seed)
    cases = make_cases(data, args.page_size, block_f, warps, names)
    for name, fn in cases.items():
        fn()  # JIT compile and initialize output; not timed.
        torch.cuda.synchronize()
        assert_correctness(data, name)
    timings = {name: [] for name in cases}
    # Alternating A/B order reduces bias from one fixed measurement order.
    for i in range(args.repeats):
        order = list(cases)
        if i % 2:
            order.reverse()
        for name in order:
            elapsed = triton.testing.do_bench(cases[name], warmup=args.warmup_ms, rep=args.rep_ms)
            timings[name].append(elapsed)
    bytes_nominal = 4 * num_tokens * features * 2  # K+V, read+write, FP16 bytes.
    results = []
    for name in cases:
        ms = statistics.median(timings[name])
        row = {
            'variant': name, 'num_tokens': num_tokens, 'n_features': features,
            'page_size': args.page_size, 'block_f': block_f, 'num_warps': warps,
            'median_time_ms': ms, 'samples_ms': timings[name],
            'nominal_bytes': bytes_nominal,
            'effective_bw_gbps': bytes_nominal / (ms * 1e6),
            'correctness': 'PASS',
        }
        results.append(row)
        print(f'{name} T={num_tokens}, BF={block_f}, warps={warps}: '
              f'{ms:.4f} ms, {row["effective_bw_gbps"]:.2f} effective GB/s')
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', choices=['compare', 'block-f', 'warps', 'locality', 'scale'], default='locality')
    ap.add_argument('--tokens', type=int, default=32768)
    ap.add_argument('--features', type=int, default=512)
    ap.add_argument('--page-size', type=int, default=16)
    ap.add_argument('--block-f', type=int, default=512)
    ap.add_argument('--warps', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--warmup-ms', type=int, default=25)
    ap.add_argument('--rep-ms', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if min(args.tokens, args.features, args.page_size, args.block_f, args.warps,
           args.repeats, args.warmup_ms, args.rep_ms) <= 0:
        ap.error('Shapes/configs/timing budgets must be positive')
    if args.block_f & (args.block_f - 1):
        ap.error('--block-f must be a power of two')
    if not torch.cuda.is_available():
        ap.error('CUDA GPU required for benchmarking')
    if args.experiment == 'compare':
        groups = [(args.tokens, args.block_f, args.warps,
                   ['linear', 'paged-sequential', 'paged-random'])]
    elif args.experiment == 'block-f':
        groups = [(args.tokens, bf, args.warps, ['linear', 'paged-sequential'])
                  for bf in (64, 128, 256, 512)]
    elif args.experiment == 'warps':
        groups = [(args.tokens, 512, w, ['paged-sequential']) for w in (1, 2, 4, 8)]
    elif args.experiment == 'locality':
        groups = [(args.tokens, 512, 4, ['paged-sequential', 'paged-random'])]
    else:
        groups = [(t, 512, 4, ['paged-sequential', 'paged-random'])
                  for t in (4096, 8192, 16384, 32768, 65536)]
    device = torch.cuda.current_device()
    report = {
        'experiment': args.experiment,
        'validation_status': 'reproduced_on_gpu',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'platform': {
            'gpu': torch.cuda.get_device_name(device),
            'compute_capability': list(torch.cuda.get_device_capability(device)),
            'torch_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
            'triton_version': triton.__version__,
        },
        'method': {
            'tensor_dtype': 'fp16', 'timer': 'triton.testing.do_bench (CUDA events)',
            'warmup_ms': args.warmup_ms, 'rep_ms': args.rep_ms,
            'outer_repeats': args.repeats, 'paired_order': 'alternates each repeat',
            'bandwidth': 'useful K+V read+write nominal bytes / elapsed (NOT measured DRAM bytes)',
            'slot_validation': 'unique and in-range checked outside timing',
            'seed': args.seed,
        },
        'results': [],
    }
    for t, bf, w, names in groups:
        report['results'].extend(run_group(args, t, args.features, bf, w, names))
    path = args.output or (Path(__file__).resolve().parents[1] / 'results' / 'summary' /
                           f'kv_{args.experiment}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    print('Saved:', path)


if __name__ == '__main__':
    main()
