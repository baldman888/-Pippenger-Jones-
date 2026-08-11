"""max-plus 热带矩阵基础运算测试。"""

import unittest

import numpy as np

from jones_crypto.metrics import OperationCounter
from jones_crypto.tropical import TropicalMatrix


class TropicalMatrixTests(unittest.TestCase):
    def test_max_plus_product(self):
        # 验证热带矩阵乘法结果，同时检查计数器只增加一次。
        left = TropicalMatrix.from_integers([[1, 2], [3, 4]])
        right = TropicalMatrix.from_integers([[5, 6], [7, 8]])
        counter = OperationCounter()
        actual = left.multiply(right, counter)
        np.testing.assert_array_equal(actual.values, [[9, 10], [11, 12]])
        self.assertEqual(counter.tropical_matrix_multiplications, 1)

    def test_linear_and_binary_powers_agree(self):
        # 朴素线性幂与二进制平方—乘算法必须给出相同结果。
        matrix = TropicalMatrix.from_integers([[2, 1], [0, 3]])
        for exponent in range(10):
            self.assertTrue(
                matrix.power_linear(exponent).equals(
                    matrix.power_binary(exponent)
                )
            )


if __name__ == "__main__":
    unittest.main()
