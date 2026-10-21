import torch

import triton
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < N)

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(
        output_ptr + offsets,
        output,
        mask=mask
    )


def vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
):
    assert x.shape == y.shape
    assert x.is_contiguous()
    assert y.is_contiguous()

    output = torch.empty_like(x)

    n_element = x.numel()
    BLOCK_SIZE = 256
    grid = (
        triton.cdiv(n_element, BLOCK_SIZE),
    )

    vector_add_kernel[grid](
        x, y, output, n_element, BLOCK_SIZE, num_warps=4,
    )

    return output


def correctness_test():
    n_elements: list[int] = [1, 256, 257, 1024, 1025, 1 << 20]

    for n in n_elements:
        x = torch.rand(n, device=DEVICE, dtype=torch.float32)
        y = torch.rand(n, device=DEVICE, dtype=torch.float32)

        torch_out = x + y
        triton_out = vector_add(x, y)

        torch.testing.assert_close(
            torch_out, triton_out
        )

        print(f"N={n:<10} PASS")


def launch_vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
    num_warps: int,
):
    n_element = x.numel()
    grid = (
        triton.cdiv(n_element, BLOCK_SIZE),
    )
    output = torch.empty_like(x)

    vector_add_kernel[grid](
        x, y, output, n_element, BLOCK_SIZE
    )


def bench_config(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE: int,
    num_warps: int,
):
    N = x.numel()

    for _ in range(10):
        launch_vector_add(x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: launch_vector_add(x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    )
    bytes = 3 * N * x.element_size()
    gbps = bytes / (ms * 1e-3) / 1e9

    return ms, gbps


def bench_block_size():
    block_sizes: list[int] = [64, 128, 256, 512, 1024]
    N = 1 << 26

    x = torch.randn(
        N, device=DEVICE, dtype=torch.float32
    )

    y = torch.randn(
        N, device=DEVICE, dtype=torch.float32
    )

    output = torch.empty_like(x)

    for block_size in block_sizes:
        ms, gbps = bench_config(x, y, output, block_size, 4)
        print(f"block_size={block_size}, num_warps=4, time={ms:.4f}ms, bandwidth={gbps:.2f}GB/s")


def bench_num_warps():
    num_warps_list: list[int] = [1, 2, 4, 8]
    N = 1 << 26

    x = torch.randn(
        N, device=DEVICE, dtype=torch.float32
    )
    y = torch.randn(
        N, device=DEVICE, dtype=torch.float32
    )
    output = torch.empty_like(x)

    for num_warps in num_warps_list:
        ms, gbps = bench_config(x, y, output, 256, num_warps)
        print(f"block_size=256, num_warps={num_warps}, time={ms:.4f}ms, bandwidth={gbps:.2f}GB/s")


def bench():
    N = (1 << 26)
    x = torch.rand(N, device=DEVICE, dtype=torch.float32)
    y = torch.rand(N, device=DEVICE, dtype=torch.float32)

    for _ in range(10):
        vector_add(x, y)

    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: vector_add(x, y)
    )

    byte_moved = 3 * N * x.element_size()
    gbps = byte_moved / (ms * 1e-3) / 1e9

    print(f"N         = {N}")
    print(f"time      = {ms:.4f}ms")
    print(f"bandwidth = {gbps:.2f}GB/s")


def inspect_kernel(x, y, output, block_size, num_warps):
    n_elements = x.numel()
    kernel = vector_add_kernel.warmup(
        x, y, output, n_elements, 
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()

    print("=" * 70)
    print(
        f"BLOCK_SIZE={block_size}, "
        f"num_warps={num_warps}"
    )
    print(f"n_regs         = {kernel.n_regs}")
    print(f"n_spills       = {kernel.n_spills}")
    print(f"shared bytes   = {kernel.metadata.shared}")
    print(f"metadata warps = {kernel.metadata.num_warps}")
    print("available asm = ", kernel.asm.keys())


def code_evidence():
    configs = [
        (64, 4),
        (256, 1),
        (256, 4),
        (1024, 4),
    ]

    N = 1 << 26
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y = torch.randn(N, device=DEVICE, dtype=torch.float32)
    output = torch.empty_like(x)

    for block_size, num_warps in configs:
        grid = (
            triton.cdiv(N, block_size),
        )
        vector_add_kernel[grid](x, y, output, N, block_size, num_warps=num_warps)
        inspect_kernel(x, y, output, block_size, num_warps)


def print_relevant_sass(kernel):
    sass = kernel.asm["sass"]

    keywords = ["LDG", "STG", "FADD", "ISETP", "BRA", "MOV"]
    for line in sass.splitlines():
        if any(k in line for k in keywords):
            print(line)


def sass_statis():
    N = 1 << 26
    
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y = torch.randn(N, device=DEVICE, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (
        triton.cdiv(N, 256),
    )
    vector_add_kernel[grid](x, y, output, N, 256)

    configs = [(256, 1), (256, 4)]

    for block_size, num_warps in configs:
        kernel = vector_add_kernel.warmup(
            x, y, output, 
            x.numel(), BLOCK_SIZE=256,
            num_warps=num_warps, grid=(1, ),
        )

        kernel._init_handles()
        print("=" * 70)
        print(f"BLOCK_SIZE={block_size}, num_warps={num_warps}")
        print_relevant_sass(kernel)


if __name__ == "__main__":
    correctness_test()
    print()
    bench_block_size()
    print()
    bench_num_warps()
    print()
    code_evidence()
    print()
    sass_statis()