"""按行并行 Pippenger 实现的正确性和计数测试。"""

import unittest

import numpy as np

from jones_crypto.circulant import random_circulant
from jones_crypto.jones import generate_jones_matrix, generate_public_deformations
from jones_crypto.metrics import OperationCounter
from jones_crypto.multi_exponentiation import multi_exponentiation_pippenger
from jones_crypto.parallel import multi_exponentiation_pippenger_parallel


class ParallelPippengerTests(unittest.TestCase):
    def setUp(self):
        jones = generate_jones_matrix(4, seed=2026)
        self.bases = generate_public_deformations(jones, 8)
        self.exponents = random_circulant(8, 16, seed=2027)

    def assert_matrix_vectors_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        self.assertTrue(
            all(a.equals(b) for a, b in zip(left, right, strict=True))
        )

    def test_parallel_matches_serial_for_worker_counts(self):
        expected = multi_exponentiation_pippenger(
            self.bases,
            self.exponents,
            window_size=2,
        )
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                actual = multi_exponentiation_pippenger_parallel(
                    self.bases,
                    self.exponents,
                    window_size=2,
                    max_workers=workers,
                )
                self.assert_matrix_vectors_equal(expected, actual)

    def test_parallel_counter_matches_serial_counter(self):
        serial_counter = OperationCounter()
        parallel_counter = OperationCounter()
        multi_exponentiation_pippenger(
            self.bases,
            self.exponents,
            window_size=3,
            counter=serial_counter,
        )
        multi_exponentiation_pippenger_parallel(
            self.bases,
            self.exponents,
            window_size=3,
            max_workers=4,
            counter=parallel_counter,
        )
        self.assertEqual(
            serial_counter.tropical_matrix_multiplications,
            parallel_counter.tropical_matrix_multiplications,
        )

    def test_zero_exponents(self):
        exponents = np.zeros((8, 8), dtype=np.int64)
        expected = multi_exponentiation_pippenger(self.bases, exponents)
        actual = multi_exponentiation_pippenger_parallel(
            self.bases,
            exponents,
            max_workers=4,
        )
        self.assert_matrix_vectors_equal(expected, actual)

    def test_invalid_worker_count(self):
        with self.assertRaises(ValueError):
            multi_exponentiation_pippenger_parallel(
                self.bases,
                self.exponents,
                max_workers=0,
            )


if __name__ == "__main__":
    unittest.main()
