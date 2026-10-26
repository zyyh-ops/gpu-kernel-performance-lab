"""Offline mapping tests. Passing these does not mean the CUDA kernel was executed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transpose_mapping_model import expected_indices, naive_indices, shared_indices


class TransposeMappingTests(unittest.TestCase):
    def test_naive_non_multiple_tile(self):
        for n in [1, 7, 31, 32, 33, 47, 63, 65, 97, 129]:
            with self.subTest(n=n):
                self.assertEqual(naive_indices(n), expected_indices(n))

    def test_shared_non_multiple_tile(self):
        for n in [1, 7, 31, 32, 33, 47, 63, 65, 97, 129]:
            with self.subTest(n=n):
                self.assertEqual(shared_indices(n), expected_indices(n))

    def test_padding_is_bank_mapping_only(self):
        self.assertEqual(shared_indices(65), expected_indices(65))
        self.assertEqual((32 * 1) % 32, 0)   # column of [32][32] hits same bank
        self.assertEqual((33 * 1) % 32, 1)   # column of [32][33] moves bank

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            naive_indices(0)
        with self.assertRaises(ValueError):
            shared_indices(0)


if __name__ == "__main__":
    unittest.main()
