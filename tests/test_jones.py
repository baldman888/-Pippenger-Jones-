"""Jones 矩阵构造与变形公式的单元测试。"""

import unittest
from fractions import Fraction

from jones_crypto.jones import deformation, generate_jones_matrix, is_jones
from jones_crypto.tropical import TropicalMatrix


class JonesTests(unittest.TestCase):
    def test_generated_matrix_is_jones(self):
        # 随机构造的矩阵必须满足 Jones 条件。
        self.assertTrue(is_jones(generate_jones_matrix(12, seed=7)))

    def test_toy_deformation(self):
        # 核对论文玩具示例中 alpha=1/2 的 Jones 变形结果。
        matrix = TropicalMatrix.from_integers(
            [[6, 5, 6], [6, 16, 12], [5, 9, 12]]
        )
        result = deformation(matrix, Fraction(1, 2), output_scale=2)
        expected = [
            [Fraction(3), Fraction(-3), Fraction(0)],
            [Fraction(-2), Fraction(8), Fraction(4)],
            [Fraction(-1), Fraction(1), Fraction(6)],
        ]
        self.assertEqual(result.as_fractions(), expected)


if __name__ == "__main__":
    unittest.main()
