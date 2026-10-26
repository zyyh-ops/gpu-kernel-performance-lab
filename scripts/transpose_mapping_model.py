"""Pure-CPU index mapping of the CUDA transpose kernels, NOT a GPU correctness test."""
TILE_DIM = 32
BLOCK_ROWS = 8


def naive_indices(n: int):
    if n < 1: raise ValueError("n must be positive")
    out = [None] * (n * n)
    for by in range((n + TILE_DIM - 1) // TILE_DIM):
        for bx in range((n + TILE_DIM - 1) // TILE_DIM):
            for ty in range(BLOCK_ROWS):
                for tx in range(TILE_DIM):
                    x, y = bx * TILE_DIM + tx, by * TILE_DIM + ty
                    for j in range(0, TILE_DIM, BLOCK_ROWS):
                        if x < n and y + j < n:
                            out[x * n + y + j] = (y + j) * n + x
    return out


def shared_indices(n: int):
    """Same index math as unpadded and padded kernels; padding changes banks, not values."""
    if n < 1: raise ValueError("n must be positive")
    out = [None] * (n * n)
    for by in range((n + TILE_DIM - 1) // TILE_DIM):
        for bx in range((n + TILE_DIM - 1) // TILE_DIM):
            tile = [[None] * TILE_DIM for _ in range(TILE_DIM)]
            for ty in range(BLOCK_ROWS):
                for tx in range(TILE_DIM):
                    x, y = bx * TILE_DIM + tx, by * TILE_DIM + ty
                    for j in range(0, TILE_DIM, BLOCK_ROWS):
                        if x < n and y + j < n:
                            tile[ty + j][tx] = (y + j) * n + x
            for ty in range(BLOCK_ROWS):
                for tx in range(TILE_DIM):
                    x, y = by * TILE_DIM + tx, bx * TILE_DIM + ty
                    for j in range(0, TILE_DIM, BLOCK_ROWS):
                        if x < n and y + j < n:
                            addr = (y + j) * n + x
                            if out[addr] is not None:
                                raise AssertionError("duplicate physical output index")
                            src = tile[tx][ty + j]
                            if src is None:
                                raise AssertionError("read from uninitialized shared tile")
                            out[addr] = src
    return out


def expected_indices(n: int):
    return [col * n + row for row in range(n) for col in range(n)]
