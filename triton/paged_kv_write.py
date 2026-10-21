import torch
import triton
import triton.language as tl


@triton.jit
def paged_kv_write_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    cache_block_stride,
    cache_slot_stride,
    slot_mapping_ptr,
    n_features,
    PAGE_SIZE: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    token = tl.program_id(0)
    feature_block = tl.program_id(1)

    features_offset = feature_block * BLOCK_F + tl.arange(0, BLOCK_F)
    features_mask = features_offset < n_features

    slot = tl.load(slot_mapping_ptr + token)
    physical_block = slot // PAGE_SIZE
    physical_slot = slot % PAGE_SIZE

    src_offsets = token * n_features + features_offset
    dst_offsets = physical_block * cache_block_stride + physical_slot * cache_slot_stride + features_offset

    k = tl.load(k_ptr + src_offsets, mask=features_mask)
    v = tl.load(v_ptr + src_offsets, mask=features_mask)

    tl.store(k_cache_ptr + dst_offsets, k, mask=features_mask)
    tl.store(v_cache_ptr + dst_offsets, v, mask=features_mask)


@triton.jit
def linear_kv_write_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    n_features,
    BLOCK_F: tl.constexpr,
):
    token = tl.program_id(0)
    feature_block = tl.program_id(1)
    features = feature_block * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = features < n_features
    offset = token * n_features + features

    k = tl.load(k_ptr + offset, mask=mask)
    v = tl.load(v_ptr + offset, mask=mask)

    tl.store(k_cache_ptr + offset, k, mask=mask)
    tl.store(v_cache_ptr + offset, v, mask=mask)


def test_paged_kv_write():
    torch.manual_seed(0)

    num_tokens, num_blocks, page_size = 6, 4, 16
    num_kv_heads, head_dim = 4, 64
    n_features = num_kv_heads * head_dim

    k = torch.randn(
        (num_tokens, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    k_cache = torch.zeros(
        (num_blocks, page_size, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.zeros_like(k_cache)

    slot_mapping = torch.tensor(
        [18, 3, 47, 9, 32, 63],
        device="cuda",
        dtype=torch.int32,
    )
    BLOCK_F = 128

    grid = (
        num_tokens, triton.cdiv(n_features, BLOCK_F),
    )
    paged_kv_write_kernel[grid](
        k, v,
        k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1),
        slot_mapping,
        n_features,
        PAGE_SIZE=page_size,
        BLOCK_F=BLOCK_F,
        num_warps=4,
    )

    ref_k = torch.zeros_like(k_cache)
    ref_v = torch.zeros_like(v_cache)

    for t in range(num_tokens):
        slot = int(slot_mapping[t].item())
        block = slot // page_size
        offset = slot % page_size
        ref_k[block, offset] = k[t]
        ref_v[block, offset] = v[t]

    torch.testing.assert_close(
        k_cache, ref_k,
    )

    torch.testing.assert_close(
        v_cache, ref_v,
    )

    print("Paged KV write correctness: PASS")


import statistics
def benchmark_paged_vs_linear():
    num_tokens = 32768
    n_features = 512
    
    page_size = 16
    num_blocks = triton.cdiv(num_tokens, page_size)
    BLOCK_F = 128
    num_warps = 4

    k = torch.randn(
        (num_tokens, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    linear_k_cache = torch.empty_like(k)
    linear_v_cache = torch.empty_like(v)

    paged_k_cache = torch.empty(
        (num_blocks, page_size, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    paged_v_cache = torch.empty_like(paged_k_cache)

    sequential_slots = torch.arange(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    random_slots = torch.randperm(
        num_tokens,
        device="cuda",
    ).to(torch.int32)

    grid = (
        num_tokens,
        triton.cdiv(
            n_features, BLOCK_F,
        ),
    )

    def launch_linear():
        linear_kv_write_kernel[grid](
            k, v,
            linear_k_cache, linear_v_cache,
            n_features,
            BLOCK_F=BLOCK_F,
            num_warps=num_warps,
        )
    def launch_paged(slot_mapping):
        paged_kv_write_kernel[grid](
            k, v,
            paged_k_cache, paged_v_cache,
            paged_k_cache.stride(0),
            paged_k_cache.stride(1),
            slot_mapping,
            n_features,
            PAGE_SIZE=page_size,
            BLOCK_F=BLOCK_F,
        )

    tests = [
        ("linear", launch_linear),
        ("paged-sequential", lambda: launch_paged(sequential_slots)),
        ("paged-random", lambda: launch_paged(random_slots)),
    ]

    bytes_moved = 2 * 4 * num_tokens * n_features

    for name, fn in tests:
        fn()
        torch.cuda.synchronize()
        times = [triton.testing.do_bench(fn) for _ in range(5)]
        ms = statistics.median(times)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"{name}: ")
        print(f"time={ms:.4f}ms, BW={gbps:.2f} GB/s")


def benchmark_block_f():
    num_tokens = 32768
    n_features = 512
    
    page_size = 16
    num_blocks = triton.cdiv(num_tokens, page_size)
    num_warps = 4

    k = torch.randn(
        (num_tokens, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    linear_k_cache = torch.empty_like(k)
    linear_v_cache = torch.empty_like(v)

    paged_k_cache = torch.empty(
        (num_blocks, page_size, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    paged_v_cache = torch.empty_like(paged_k_cache)

    sequential_slots = torch.arange(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    random_slots = torch.randperm(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )

    def launch_linear(BLOCK_F):
        grid = (
            num_tokens,
            triton.cdiv(n_features, BLOCK_F),
        )
        return linear_kv_write_kernel[grid](
            k, v,
            linear_k_cache, linear_v_cache,
            n_features,
            BLOCK_F=BLOCK_F,
            num_warps=num_warps,
        )
    def launch_paged(slot_mapping, BLOCK_F):
        grid = (
            num_tokens,
            triton.cdiv(n_features, BLOCK_F),
        )
        return paged_kv_write_kernel[grid](
            k, v,
            paged_k_cache, paged_v_cache,
            paged_k_cache.stride(0),
            paged_k_cache.stride(1),
            slot_mapping,
            n_features,
            PAGE_SIZE=page_size,
            BLOCK_F=BLOCK_F,
        )

    block_sizes = [64, 128, 256, 512]

    bytes_moved = 2 * 4 * num_tokens * n_features

    for block_f in block_sizes:
        linear_kernel = launch_linear(block_f)
        paged_seq_kernel = launch_paged(sequential_slots, block_f)
        torch.cuda.synchronize()

        linear_fn = lambda: launch_linear(block_f)
        paged_fn = lambda: launch_paged(sequential_slots, block_f)
        
        linear_times = [triton.testing.do_bench(linear_fn) for _ in range(5)]
        paged_seq_times = [triton.testing.do_bench(paged_fn) for _ in range(5)]
        linear_ms = statistics.median(linear_times)
        paged_seq_ms = statistics.median(paged_seq_times)

        linear_gbps = bytes_moved / (linear_ms * 1e-3) / 1e9
        paged_gbps = bytes_moved / (paged_seq_ms * 1e-3) / 1e9

        print("=" * 40)
        print("linear: ")
        print(f"time={linear_ms:.4f}ms, BW={linear_gbps:.2f} GB/s")
        print(f"n_regs={linear_kernel.n_regs}, n_spills={linear_kernel.n_spills}")
        print("paged-sequential: ")
        print(f"time={paged_seq_ms:.4f}ms, BW={paged_gbps:.2f} GB/s")
        print(f"n_regs={paged_seq_kernel.n_regs}, n_spills={paged_seq_kernel.n_spills}")
        print(f"gap={((paged_seq_ms / linear_ms) - 1.0) * 100:.2f}%")


def inspect_kv_kernel(
    paged,
    BLOCK_F,
):
    num_tokens = 32768
    n_features = 512
    page_size = 16
    num_warps = 4

    k = torch.randn(
        (num_tokens, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    grid = (
        num_tokens,
        triton.cdiv(n_features, BLOCK_F),
    )

    if not paged:
        k_cache = torch.empty_like(k)
        v_cache = torch.empty_like(v)

        kernel = linear_kv_write_kernel.warmup(
            k, v,
            k_cache, v_cache,
            n_features,
            BLOCK_F=BLOCK_F,
            num_warps=num_warps,
            grid=grid,
        )
    else:
        num_blocks = triton.cdiv(num_tokens, page_size)
        k_cache = torch.empty(
            (num_blocks, page_size, n_features),
            device="cuda",
            dtype=torch.float16,
        )
        v_cache = torch.empty_like(k_cache)

        slots = torch.arange(
            num_tokens,
            device="cuda",
            dtype=torch.int32,
        )

        kernel = paged_kv_write_kernel.warmup(
            k, v,
            k_cache, v_cache,
            k_cache.stride(0), k_cache.stride(1),
            slots,
            n_features,
            PAGE_SIZE=page_size,
            num_warps=num_warps,
            BLOCK_F=BLOCK_F,
            grid=grid,
        )

    kernel._init_handles()
    print("=" * 70)
    print(f"{'paged' if paged else 'linear'}, BLOCK_F={BLOCK_F}")
    print(f"n_regs={kernel.n_regs}, n_spills={kernel.n_spills}")

    ptx = kernel.asm["ptx"]
    keywords = [
        "ld.global", "st.global",
        "shr.", "shl.", "and.", "div.",
        "rem.", "mad.", "mul.",
    ]
    for line in ptx.splitlines():
        if any(k in line for k in keywords):
            print(line)


def benchmark_num_warps():
    num_tokens = 32768
    n_features = 512
    
    page_size = 16
    num_blocks = triton.cdiv(num_tokens, page_size)
    BLOCK_F = 512

    k = torch.randn(
        (num_tokens, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    linear_k_cache = torch.empty_like(k)
    linear_v_cache = torch.empty_like(v)

    paged_k_cache = torch.empty(
        (num_blocks, page_size, n_features),
        device="cuda",
        dtype=torch.float16,
    )
    paged_v_cache = torch.empty_like(paged_k_cache)

    sequential_slots = torch.arange(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    random_slots = torch.randperm(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    grid = (
        num_tokens,
        triton.cdiv(n_features, BLOCK_F)
    )

    def launch_paged(slot_mapping, num_warps):
        return paged_kv_write_kernel[grid](
            k, v,
            paged_k_cache, paged_v_cache,
            paged_k_cache.stride(0),
            paged_k_cache.stride(1),
            slot_mapping,
            n_features,
            PAGE_SIZE=page_size,
            BLOCK_F=BLOCK_F,
            num_warps=num_warps,
        )

    bytes_moved = 2 * 4 * num_tokens * n_features
    num_warps_list: list[int] = [1, 2, 4, 8]

    # for num_warps in num_warps_list:
    #     paged_seq_kernel = launch_paged(sequential_slots, num_warps)
    #     torch.cuda.synchronize()

    #     paged_fn = lambda: launch_paged(sequential_slots, num_warps)
    #     paged_seq_times = [triton.testing.do_bench(paged_fn) for _ in range(5)]

    #     paged_seq_ms = statistics.median(paged_seq_times)
    #     paged_gbps = bytes_moved / (paged_seq_ms * 1e-3) / 1e9

    #     print("=" * 40)
    #     print("paged-sequential: ")
    #     print(f"num_warps={num_warps}")
    #     print(f"time={paged_seq_ms:.4f}ms, BW={paged_gbps:.2f} GB/s")
    #     print(f"n_regs={paged_seq_kernel.n_regs}, n_spills={paged_seq_kernel.n_spills}")

    paged_seq_kernel = launch_paged(sequential_slots, 4)
    paged_random_kernel = launch_paged(random_slots, 4)
    torch.cuda.synchronize()

    paged_seq_fn = lambda: launch_paged(sequential_slots, 4)
    paged_random_fn = lambda: launch_paged(random_slots, 4)

    paged_seq_times = [triton.testing.do_bench(paged_seq_fn) for _ in range(7)]
    paged_seq_ms = statistics.median(paged_seq_times)
    paged_random_times = [triton.testing.do_bench(paged_random_fn) for _ in range(7)]
    paged_random_ms = statistics.median(paged_random_times)

    paged_seq_gbps = bytes_moved / (paged_seq_ms * 1e-3) / 1e9
    paged_random_gbps = bytes_moved / (paged_random_ms * 1e-3) / 1e9

    print("=" * 40)
    print("paged-sequential: ")
    print(f"time={paged_seq_ms:.4f}ms, BW={paged_seq_gbps:.2f} GB/s")
    print(f"n_regs={paged_seq_kernel.n_regs}, n_spills={paged_seq_kernel.n_spills}")

    print("paged-random: ")
    print(f"time={paged_random_ms:.4f}ms, BW={paged_random_gbps:.2f} GB/s")
    print(f"n_regs={paged_random_kernel.n_regs}, n_spills={paged_random_kernel.n_spills}")
    print(f"{(paged_seq_gbps - paged_random_gbps) / paged_seq_gbps * 100:.2f}%")


if __name__ == "__main__":
    # test_paged_kv_write()
    # benchmark_paged_vs_linear()
    # benchmark_block_f()
    # for block_f in [128, 256, 512]:
    #     inspect_kv_kernel(False, block_f)
    #     inspect_kv_kernel(True, block_f)
    benchmark_num_warps()