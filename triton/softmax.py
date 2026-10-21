import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    x_row_stride,
    y_row_stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    x = tl.load(
        x_ptr + row * x_row_stride + cols,
        mask=mask,
        other=-float("inf"),
    )

    maxx = tl.max(x, axis=0)
    numerator = tl.exp(x - maxx)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(
        y_ptr + y_row_stride * row + cols,
        y,
        mask=mask,
    )


def softmax_triton(x):
    assert x.ndim == 2
    assert x.shape[1] > 0
    assert x.is_cuda
    assert x.is_contiguous()

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    softmax_kernel[grid](
        x, output,
        x.stride(0),
        output.stride(0),
        N_COLS=n_cols, 
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return output


def correctness_test():
    torch.manual_seed(0)

    shapes = [
        (1, 1),
        (4, 127),
        (128, 256),
        (256, 257),
        (512, 1000),
        (1024, 1024),
        (64, 1025),
    ]

    for m, n in shapes:
        x = (
            torch.randn(
                (m, n),
                device="cuda",
                dtype=torch.float32,
            ) * 5 + 1000
        )

        ref = torch.softmax(x, dim=-1)
        out = softmax_triton(x)

        torch.testing.assert_close(
            out, ref,
            rtol=1e-4, atol=1e-5,
        )
        max_error = (out - ref).abs().max().item()

        row_sum_error = (
            out.sum(dim=-1) - 1.0
        ).abs().max().item()

        print(
            f"shape=({m}, {n}) "
            f"max_error={max_error:.3e} "
            f"row_sum_error={row_sum_error:.3e} "
            f"PASS"
        )


def inspect_softmax(n_cols, num_warps):
    n_rows = 128

    x = torch.randn(
        (n_rows, n_cols),
        device="cuda",
        dtype=torch.float32,
    )
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    kernel = softmax_kernel.warmup(
        x, y,
        x.stride(0),
        y.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        grid=(n_rows,),
    )
    kernel._init_handles()
    
    print("=" * 70)
    print(f"N_COLS       = {n_cols}")
    print(f"BLOCK_SIZE   = {block_size}")
    print(f"num_warps    = {num_warps}")
    print(f"n_regs       = {kernel.n_regs}")
    print(f"n_spills     = {kernel.n_spills}")
    print(f"shared bytes = {kernel.metadata.shared}")

    ptx = kernel.asm["ptx"]

    keywords = [
        "shfl.sync",
        "redux.sync",
        "bar.sync",
        "st.shared",
        "ld.shared",
    ]

    print("\nRelevant PTX instructions:")

    for line in ptx.splitlines():
        if any(k in line for k in keywords):
            print(line)


import statistics
import triton.testing

def benchmark_num_warps():
    M, N = 16384, 1000

    x = (
        torch.randn(
            (M, N),
            device="cuda",
            dtype=torch.float32,
        ) * 5 + 1000
    )
    output = torch.empty_like(x)
    ref = torch.softmax(x, dim=-1)

    block_size = triton.next_power_of_2(N)
    grid = (M,)

    for num_warps in [1, 2, 4, 8]:
        def launch():
            softmax_kernel[grid](
                x, output,
                x.stride(0),
                output.stride(0),
                N,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        launch()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            output, ref,
            rtol=1e-4, atol=1e-5,
        )
        times = [
            triton.testing.do_bench(launch)
            for _ in range(5)
        ]
        ms = statistics.mean(times)

        bytes_moved = 2 * M * N * 4
        gbps = bytes_moved / (ms * 1e-3) / 1e9

        print(
            f"num_warps={num_warps}, "
            f"time={ms:.4f}ms, "
            f"effective BW={gbps:.2f} GB/s"
        )


def benchmark_row_length():
    N = 1 << 24
    n_cols: list[int] = [256, 512, 1024, 2048]
    num_warps_list: list[int] = [1, 2, 4, 8]

    for n_col in n_cols:
        n_row = N // n_col
        block_size = triton.next_power_of_2(n_col)
        grid = (n_row,)

        x = torch.randn(
            (n_row, n_col),
            device="cuda",
            dtype=torch.float32,
        )
        output = torch.empty_like(x)

        print("=" * 60)
        print(f"M={n_row}, N={n_col}, BLOCK_SIZE={block_size}: ")
        for num_warps in num_warps_list:
            def launch():
                return softmax_kernel[grid](
                    x, output,
                    x.stride(0),
                    output.stride(0),
                    N_COLS=n_col,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            kernel = launch()
            torch.cuda.synchronize()

            ms = [triton.testing.do_bench(launch) for _ in range(3)]
            ms = statistics.median(ms)
            gbps = 2 * 4 * x.numel() / (ms * 1e-3) / 1e9
            print(f"num_warps={num_warps}, time={ms:.4f}ms, BW={gbps:.2f} GB/s")
            print(f"regs={kernel.n_regs}, spills={kernel.n_spills}, shared={kernel.metadata.shared}B")


def benchmark_power_of_two_cliff():
    M = 16384
    num_warps = 4
    N_list: list[int] = [1000, 1024, 1025]

    for N in N_list:
        x = torch.randn(
            (M, N),
            device="cuda",
            dtype=torch.float32,
        )
        output = torch.empty_like(x)
        block_size = triton.next_power_of_2(N)
        grid = (M,)

        def launch():
            return softmax_kernel[grid](
                x, output,
                x.stride(0),
                output.stride(0),
                N_COLS=M,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        kernel = launch()
        torch.cuda.synchronize()

        times = [triton.testing.do_bench(launch) for _ in range(3)]
        time = statistics.median(times)
        gbps = 2 * 4 * x.numel() / (time * 1e-3) / 1e9
        print(f"M={M}, N={N}, BLOCK_SIZE={block_size}: ")
        print(f"time={time:.4f}ms, BW={gbps:.2f}GB/s")
        print(f"regs={kernel.n_regs}, spills={kernel.n_spills}, shared={kernel.metadata.shared}B")


if __name__ == "__main__":
    # correctness_test()
    # inspect_softmax(n_cols=1000, num_warps=1)
    # inspect_softmax(n_cols=1000, num_warps=4)
    # benchmark_num_warps()
    # benchmark_row_length()
    benchmark_power_of_two_cliff()