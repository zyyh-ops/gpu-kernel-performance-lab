"""CPU-only specification for TILE=16 GEMM with masked tails and ping-pong.

This models indexing and zero fill; it cannot prove CUDA async synchronization.
"""
TILE = 16


def require_supported_n(n: int) -> None:
    if n <= 0:
        raise ValueError("N must be positive")


def choose_copy_path(n: int) -> str:
    require_supported_n(n)
    return "cp_async_16_aligned" if n % TILE == 0 else "cp_async_4_masked"


def copy_thread_elements(tid: int, n: int = TILE):
    """Logical positions in a 16x16 tile assigned to one copy thread."""
    require_supported_n(n)
    if not (0 <= tid < TILE*TILE):
        raise ValueError("copy tid must be 0..255")
    if n % TILE == 0:
        if tid >= TILE*TILE//4:
            return ()
        base = tid * 4
        return tuple(range(base, base+4))
    return (tid,)


def buffer_offsets():
    t=TILE*TILE
    return {"A0": range(0,t), "A1":range(t,2*t),
            "B0":range(2*t,3*t), "B1":range(3*t,4*t)}


def schedule(num_tiles: int):
    if num_tiles<=0:
        raise ValueError("num_tiles must be positive")
    return [(t,t%2,((t+1)%2 if t+1<num_tiles else None))
            for t in range(num_tiles)]


def load_tile_pair(A,B,n,block_m,block_n,k_tile):
    """Return the exactly zero-padded A/B 16x16 shared tiles for one CTA."""
    require_supported_n(n)
    a=[[0.0]*TILE for _ in range(TILE)]
    b=[[0.0]*TILE for _ in range(TILE)]
    for r in range(TILE):
        for c in range(TILE):
            ar=block_m*TILE+r
            ac=k_tile*TILE+c
            br=k_tile*TILE+r
            bc=block_n*TILE+c
            if ar<n and ac<n:
                a[r][c]=A[ar][ac]
            if br<n and bc<n:
                b[r][c]=B[br][bc]
    return a,b


def reference_matmul(A,B,n):
    require_supported_n(n)
    return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def simulated_double_buffer(A,B,n):
    """Conceptual CPU model of every block and both ping-pong buffers."""
    require_supported_n(n)
    c=[[0.0]*n for _ in range(n)]
    num_blocks=(n-1)//TILE+1
    tiles=(n-1)//TILE+1
    for bm in range(num_blocks):
        for bn in range(num_blocks):
            pairs=[None,None]
            pairs[0]=load_tile_pair(A,B,n,bm,bn,0)
            cur=0
            for t in range(tiles):
                nxt=cur^1
                if t+1<tiles:
                    pairs[nxt]=load_tile_pair(A,B,n,bm,bn,t+1)
                aa,bb=pairs[cur]
                for r in range(TILE):
                    global_r=bm*TILE+r
                    if global_r>=n:
                        continue
                    for cc in range(TILE):
                        global_c=bn*TILE+cc
                        if global_c>=n:
                            continue
                        for kk in range(TILE):
                            c[global_r][global_c]+=aa[r][kk]*bb[kk][cc]
                cur=nxt
    return c
