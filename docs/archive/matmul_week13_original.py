import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
            },
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_pids = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_pids
    pid_n = pid % num_n_pids

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumlator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(
            a_ptrs,
            mask=(
                (offs_m[:, None] < M) & (k[None, :] < K)
            ),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(
                (k[:, None] < K) & (offs_n[None, :] < N)
            ),
            other=0.0,
        )
        accumlator = tl.dot(a, b, accumlator)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = accumlator.to(tl.float16)
    tl.store(
        c_ptrs,
        c,
        mask=(
            (offs_m[:, None] < M) & (offs_n[None, :] < N)
        ),
    )


def matmul_triton(x: torch.Tensor, y: torch.Tensor):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[0]

    assert x.dtype == torch.float16
    assert y.dtype == torch.float16
    assert x.is_contiguous()
    assert y.is_contiguous()

    BLOCK_M = BLOCK_N = BLOCK_K = 32
    M, K = x.shape
    _, N = y.shape
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    output = torch.empty(
        (M, N),
        device=x.device,
        dtype=torch.float16,
    )

    matmul_kernel[grid](
        x, y, output,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return output


def correctness_test():
    torch.manual_seed(0)
    shapes = [
        (32, 32, 32),
        (128, 128, 128),
        (129, 257, 65),
        (256, 384, 512),
        (511, 513, 257),
        (4096, 4096, 4096)
    ]

    for M, N, K in shapes:
        a = torch.randn(
            (M, K),
            device="cuda",
            dtype=torch.float16,
        )
        b = torch.randn(
            (K, N),
            device="cuda",
            dtype=torch.float16,
        )
        
        ref = torch.matmul(a, b)
        out = matmul_triton(a, b)
        triton.testing.assert_close(
            ref, out,
            atol=1e-2,
            rtol=1e-2,
        )
        max_error = (
            out.float() - ref.float()
        ).abs().max().item()

        print(
            f"M={M}, N={N}, K={K}: "
            f"max_error={max_error:.4e} "
            f"PASS"
        )


def inspect_matmul():
    M = N = K = 256
    a = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    b = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )
    c = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.float16,
    )

    BLOCK_M = BLOCK_K = BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    kernel = matmul_kernel.warmup(
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        grid=grid,
    )

    kernel._init_handles()
    print(f"n_regs       = {kernel.n_regs}")
    print(f"n_spills     = {kernel.n_spills}")
    print(f"shared bytes = {kernel.metadata.shared}")
    print(f"num_warps    = {kernel.metadata.num_warps}")

    ptx = kernel.asm["ptx"]

    keywords = [
        "mma.sync", "ldmatrix", "cp.async",
        "ld.shared", "st.shared", "bar.sync",
        "ld.global", "st.global",
    ]
    print("\nRelevant PTX:")
    for line in ptx.splitlines():
        if any(k in line for k in keywords):
            print(line)


import statistics
def benchmark_MNK():
    M = N = K = 4096
    configs = [
        (128, 128, 4),
        (128, 128, 8),
        (128, 256, 4),
        (128, 256, 8),
        (256, 128, 8),
    ]
    x = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    y = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )
    output = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.float16,
    )
    block_k = 32

    for (block_m, block_n, num_warps) in configs:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        
        def launch():
            return matmul_kernel[grid](
                x, y, output,
                M, N, K,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
        kernel = launch()
        torch.cuda.synchronize()
        times = [triton.testing.do_bench(launch) for _ in range(3)]
        ms = statistics.median(times)
        tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

        print("=" * 60)
        print(f"BLOCK_M={block_m}, BLOCK_N={block_n}, num_warps={num_warps}: ")
        print(f"time={ms:.4f}ms, TFLOP/s={tflops:.2f}")
        print(f"n_regs={kernel.n_regs}, n_spills={kernel.n_spills}, shared_bytes={kernel.metadata.shared / 1024}KB")

        if block_m == 128 and num_warps == 4:
            ptx = kernel.asm["sass"]
            keywords = [
                "LDL", "STL",
            ]
            print("\nRelevant PTX:")
            for line in ptx.splitlines():
                if any(k in line for k in keywords):
                    print(line)

    for num_stages in [1, 2, 3, 4]:
            grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
            
            def launch():
                return matmul_kernel[grid](
                    x, y, output,
                    M, N, K,
                    x.stride(0), x.stride(1),
                    y.stride(0), y.stride(1),
                    output.stride(0), output.stride(1),
                    BLOCK_M=128,
                    BLOCK_N=128,
                    BLOCK_K=block_k,
                    num_warps=8,
                    num_stages=num_stages,
                )
            kernel = launch()
            torch.cuda.synchronize()
            times = [triton.testing.do_bench(launch) for _ in range(3)]
            ms = statistics.median(times)
            tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    
            print("=" * 60)
            print(f"BLOCK_M=128, BLOCK_N=128, num_warps={num_warps}, num_stages={num_stages}: ")
            print(f"time={ms:.4f}ms, TFLOP/s={tflops:.2f}")
            print(f"n_regs={kernel.n_regs}, n_spills={kernel.n_spills}, shared_bytes={kernel.metadata.shared / 1024}KB")
    

def benchmark_K():
    M = N = K = 4096
    configs = [16, 32, 64]
    x = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    y = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )
    output = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.float16,
    )
    block_m = block_n = 128

    for block_k in configs:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        
        def launch():
            return matmul_kernel[grid](
                x, y, output,
                M, N, K,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=8,
            )
        kernel = launch()
        torch.cuda.synchronize()
        times = [triton.testing.do_bench(launch) for _ in range(3)]
        ms = statistics.median(times)
        tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

        print("=" * 60)
        print(f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, num_warps=8: ")
        print(f"time={ms:.4f}ms, TFLOP/s={tflops:.2f}")
        print(f"n_regs={kernel.n_regs}, n_spills={kernel.n_spills}, shared_bytes={kernel.metadata.shared / 1024}KB")  


def autotune():
    M = N = K = 4096
    x = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    y = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )
    output = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.float16,
    )
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    matmul_kernel[grid](
        x, y, output,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
    )
    print(matmul_kernel.best_config)


if __name__ == "__main__":
    # correctness_test()
    # inspect_matmul()
    # benchmark_MNK()
    # benchmark_K()
    autotune()