"""Pippenger 正确性、边界情况和协议接入测试。"""

import unittest

import numpy as np

from jones_crypto.circulant import random_circulant
from jones_crypto.fixtures import paper_toy_parameters
from jones_crypto.jones import (
    generate_jones_matrix,
    generate_public_deformations,
)
from jones_crypto.multi_exponentiation import (
    multi_exponentiation_binary,
    multi_exponentiation_naive,
    multi_exponentiation_pippenger,
)
from jones_crypto.protocol import key_exchange


class PippengerTests(unittest.TestCase):
    def assert_matrix_vectors_equal(self, left, right):
        # 矩阵向量必须先长度一致，再逐个比较每个矩阵分量。
        self.assertEqual(len(left), len(right))
        self.assertTrue(
            all(a.equals(b) for a, b in zip(left, right, strict=True))
        )

    def test_paper_example(self):
        # 在论文固定参数上，不同窗口宽度都必须与朴素算法一致。
        _, bases, alice, _ = paper_toy_parameters()
        expected = multi_exponentiation_naive(bases, alice)
        for window in (1, 2, 3):
            actual = multi_exponentiation_pippenger(
                bases,
                alice,
                window_size=window,
            )
            self.assert_matrix_vectors_equal(expected, actual)

    def test_random_inputs_and_windows(self):
        # 扩大指数范围，并同时验证二进制快速幂和多个 Pippenger 窗口。
        jones = generate_jones_matrix(4, seed=91)
        bases = generate_public_deformations(jones, 8)
        for bound in (1, 2, 3, 8, 31, 256):
            exponents = random_circulant(8, bound, seed=bound)
            expected = multi_exponentiation_naive(bases, exponents)
            binary = multi_exponentiation_binary(bases, exponents)
            self.assert_matrix_vectors_equal(expected, binary)
            for window in range(1, 6):
                actual = multi_exponentiation_pippenger(
                    bases,
                    exponents,
                    window_size=window,
                )
                self.assert_matrix_vectors_equal(expected, actual)

    def test_zero_exponent_matrix(self):
        # 指数全部为 0 时，每个输出分量都应为热带单位矩阵。
        jones = generate_jones_matrix(3)
        bases = generate_public_deformations(jones, 4)
        exponents = np.zeros((4, 4), dtype=np.int64)
        expected = multi_exponentiation_naive(bases, exponents)
        actual = multi_exponentiation_pippenger(bases, exponents)
        self.assert_matrix_vectors_equal(expected, actual)

    def test_protocol_uses_optimized_engine(self):
        # 协议默认使用 Pippenger 时，Alice 和 Bob 仍应得到相同密钥。
        _, bases, alice, bob = paper_toy_parameters()
        alice_shared, bob_shared = key_exchange(bases, alice, bob)
        self.assert_matrix_vectors_equal(alice_shared, bob_shared)


if __name__ == "__main__":
    unittest.main()
