"""CPU-only mapping and tail zero-fill tests; not a GPU validation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from double_buffer_mapping_model import (
    TILE, require_supported_n, choose_copy_path, copy_thread_elements,
    buffer_offsets, schedule, load_tile_pair, reference_matmul,
    simulated_double_buffer,
)

class DoubleBufferMappingTest(unittest.TestCase):
    def test_fast_copy_covers_tile_once_and_aligns(self):
        flat=[x for tid in range(256) for x in copy_thread_elements(tid,16)]
        self.assertEqual(sorted(flat),list(range(TILE*TILE)))
        self.assertEqual(sum(bool(copy_thread_elements(i,16)) for i in range(256)),64)
        for tid in range(64): self.assertEqual(copy_thread_elements(tid,16)[0]%4,0)

    def test_masked_copy_covers_tile_once(self):
        self.assertEqual([x for tid in range(256) for x in copy_thread_elements(tid,17)],list(range(256)))
        self.assertEqual(choose_copy_path(17),"cp_async_4_masked")
        self.assertEqual(choose_copy_path(32),"cp_async_16_aligned")

    def test_four_buffers_are_disjoint(self):
        ranges=buffer_offsets()
        self.assertEqual(len(set().union(*(set(v) for v in ranges.values()))),4*TILE*TILE)

    def test_ping_pong_schedule(self):
        self.assertEqual(schedule(1),[(0,0,None)])
        self.assertEqual(schedule(4),[(0,0,1),(1,1,0),(2,0,1),(3,1,None)])

    def test_positive_n_only(self):
        for n in [0,-1,-16]:
            with self.assertRaises(ValueError):require_supported_n(n)
        for n in [1,7,15,16,17,31,32,33,65]:require_supported_n(n)

    def test_zero_fill_at_matrix_and_k_tail(self):
        n=17
        A=[[float(i*n+j+1) for j in range(n)] for i in range(n)]
        B=[[float(i*n+j+1) for j in range(n)] for i in range(n)]
        aa,bb=load_tile_pair(A,B,n,1,1,1)
        self.assertEqual(aa[0][0],A[16][16]);self.assertEqual(bb[0][0],B[16][16])
        self.assertEqual(aa[0][1],0.0);self.assertEqual(bb[1][0],0.0)
        self.assertEqual(aa[1][0],0.0);self.assertEqual(bb[0][1],0.0)
        self.assertEqual(sum(x!=0 for row in aa for x in row),1)
        self.assertEqual(sum(x!=0 for row in bb for x in row),1)

    def test_cpu_math_model_aligned_and_odd(self):
        for n in [1,7,15,16,17,31,32,33,47,65]:
            with self.subTest(n=n):
                A=[[float((i*17+j)%11-5)/8 for j in range(n)] for i in range(n)]
                B=[[float((i*7+j)%13-6)/8 for j in range(n)] for i in range(n)]
                ref=reference_matmul(A,B,n)
                got=simulated_double_buffer(A,B,n)
                for i in range(n):
                    for j in range(n):
                        self.assertAlmostEqual(ref[i][j],got[i][j],places=5)

if __name__=='__main__':unittest.main()
