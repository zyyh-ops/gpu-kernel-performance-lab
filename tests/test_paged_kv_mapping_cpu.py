"""Pure-Python reference checks; these are NOT Triton kernel correctness tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paged_kv_mapping_model import translate_slots, write_reference


class TestPagedKVMappingCPU(unittest.TestCase):
    def test_original_six_slots(self):
        self.assertEqual(translate_slots([18, 3, 47, 9, 32, 63], 16, 4),
                         [(1, 2), (0, 3), (2, 15), (0, 9), (2, 0), (3, 15)])

    def test_page_boundary_and_untouched(self):
        cache = write_reference([[5, 6], [7, 8], [9, 10]], [15, 16, 31], 16, 2)
        self.assertEqual(cache[0][15], [5, 6])
        self.assertEqual(cache[1][0], [7, 8])
        self.assertEqual(cache[1][15], [9, 10])
        self.assertEqual(cache[0][14], [None, None])

    def test_reject_duplicate_and_invalid(self):
        for indices in ([3, 3], [-1, 2], [0, 32], [], [None]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                translate_slots(indices, 16, 2)

    def test_non_power_of_two_page_math(self):
        self.assertEqual(translate_slots([0, 6, 7, 13, 14], 7, 3),
                         [(0, 0), (0, 6), (1, 0), (1, 6), (2, 0)])


if __name__ == '__main__':
    unittest.main()
