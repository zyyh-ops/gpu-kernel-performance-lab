"""Inspect PTX/SASS of linear and paged KV writers on a real GPU."""
import argparse
from pathlib import Path
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton'))
from paged_kv_write import KVWriteConfig, launch_linear, launch_paged  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=['linear', 'paged', 'both'], default='both')
    ap.add_argument('--tokens', type=int, default=32768)
    ap.add_argument('--features', type=int, default=512)
    ap.add_argument('--page-size', type=int, default=16)
    ap.add_argument('--block-f', type=int, default=512)
    ap.add_argument('--warps', type=int, default=4)
    ap.add_argument('--show', choices=['ptx', 'sass', 'both'], default='ptx')
    ap.add_argument('--save-dir', type=Path, default=None)
    args = ap.parse_args()
    if min(args.tokens, args.features, args.page_size, args.block_f, args.warps) <= 0:
        ap.error('All shapes/configuration parameters must be positive')
    if not torch.cuda.is_available():
        ap.error('CUDA GPU required')
    cfg = KVWriteConfig(args.page_size, args.block_f, args.warps)
    k = torch.randn((args.tokens, args.features), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    num_blocks = triton.cdiv(args.tokens, args.page_size)
    kc = torch.zeros((num_blocks, args.page_size, args.features), device='cuda', dtype=torch.float16)
    vc = torch.zeros_like(kc)
    kl = torch.empty_like(k)
    vl = torch.empty_like(v)
    slots = torch.arange(args.tokens, device='cuda', dtype=torch.int32)
    launches = {}
    if args.kind in ('linear', 'both'):
        launches['linear'] = lambda: launch_linear(k, v, kl, vl, cfg)
    if args.kind in ('paged', 'both'):
        launches['paged'] = lambda: launch_paged(k, v, kc, vc, slots, cfg)
    keywords = ('ld.global', 'st.global', 'shr.', 'shl.', 'and.', 'div.',
                'rem.', 'mad.', 'mul.', 'LDG', 'STG', 'SHF', 'IMAD')
    for kind, fn in launches.items():
        compiled = fn()
        torch.cuda.synchronize()
        print(f'=== {kind} BF={cfg.block_f} warps={cfg.num_warps} ===')
        print(f'n_regs={compiled.n_regs} n_spills={compiled.n_spills} '
              f'shared_bytes={compiled.metadata.shared}')
        for asm_kind in (('ptx', 'sass') if args.show == 'both' else (args.show,)):
            try:
                asm = compiled.asm.get(asm_kind)
            except Exception as exc:
                print(f'{asm_kind} unavailable: {exc}')
                continue
            if not asm:
                print(f'{asm_kind} not available for this toolchain')
                continue
            print(f'--- {asm_kind} filtered ---')
            for line in asm.splitlines():
                if any(word.lower() in line.lower() for word in keywords):
                    print(line)
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                out = args.save_dir / f'kv_{kind}_BF{cfg.block_f}_w{cfg.num_warps}.{asm_kind}'
                out.write_text(asm, encoding='utf-8')
                print('Saved:', out)


if __name__ == '__main__':
    main()
