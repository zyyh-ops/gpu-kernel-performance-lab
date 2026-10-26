"""GPU tests of refactored paged/linear KV write. Does not PASS without CUDA."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton'))
from paged_kv_write import KVWriteConfig, launch_linear, launch_paged  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU is required')
class TestPagedKVWrite(unittest.TestCase):
    def _check_case(self, num_tokens, n_features, page_size, num_blocks, slots, block_f, warps):
        torch.manual_seed(0)
        k = torch.randn(num_tokens, n_features, device='cuda', dtype=torch.float16)
        v = torch.randn_like(k)
        k_cache = torch.zeros((num_blocks, page_size, n_features), device='cuda', dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.tensor(slots, device='cuda', dtype=torch.int32)
        cfg = KVWriteConfig(page_size=page_size, block_f=block_f, num_warps=warps)
        launch_paged(k, v, k_cache, v_cache, slot_mapping, cfg)

        expected_k = torch.zeros_like(k_cache)
        expected_v = torch.zeros_like(v_cache)
        ids = slot_mapping.to(torch.int64)
        expected_k.view(-1, n_features).index_copy_(0, ids, k)
        expected_v.view(-1, n_features).index_copy_(0, ids, v)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(k_cache, expected_k))
        self.assertTrue(torch.equal(v_cache, expected_v))

    def test_original_week13_case(self):
        self._check_case(6, 256, 16, 4, [18, 3, 47, 9, 32, 63], 128, 4)

    def test_feature_tail_and_page_boundary(self):
        self._check_case(7, 257, 16, 3, [0, 15, 16, 31, 32, 42, 47], 128, 4)

    def test_more_than_one_feature_program(self):
        self._check_case(19, 513, 8, 4, list(reversed(range(19))), 256, 4)

    def test_large_block_f_and_sparse_slots(self):
        self._check_case(3, 65, 16, 2, [31, 0, 16], 512, 4)

    def test_random_unique_slots_and_warps(self):
        gen = torch.Generator().manual_seed(3)
        slots = torch.randperm(32, generator=gen)[:21].tolist()
        self._check_case(21, 512, 16, 2, slots, 512, 8)

    def test_linear_control_exact_copy(self):
        k = torch.randn((17, 257), device='cuda', dtype=torch.float16)
        v = torch.randn_like(k)
        dest_k = torch.empty_like(k)
        dest_v = torch.empty_like(v)
        launch_linear(k, v, dest_k, dest_v, KVWriteConfig(block_f=128))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dest_k, k))
        self.assertTrue(torch.equal(dest_v, v))

    def test_mapping_rejects_invalid_slots(self):
        k = torch.zeros((2, 32), device='cuda', dtype=torch.float16)
        v = torch.zeros_like(k)
        c = torch.zeros((1, 16, 32), device='cuda', dtype=torch.float16)
        d = torch.zeros_like(c)
        cfg = KVWriteConfig()
        for indices in ([-1, 1], [0, 16], [3, 3]):
            with self.subTest(indices=indices):
                slots = torch.tensor(indices, device='cuda', dtype=torch.int32)
                with self.assertRaises(ValueError):
                    launch_paged(k, v, c, d, slots, cfg)

    def test_rejects_non_contiguous_source(self):
        k = torch.empty((8, 4), device='cuda', dtype=torch.float16).T
        v = torch.empty_like(k)
        c = torch.empty((1, 16, 8), device='cuda', dtype=torch.float16)
        d = torch.empty_like(c)
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device='cuda')
        with self.assertRaises(ValueError):
            launch_paged(k, v, c, d, slots, KVWriteConfig())


if __name__ == '__main__':
    unittest.main()
