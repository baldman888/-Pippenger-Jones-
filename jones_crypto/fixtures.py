"""论文第 4 节 toy example 的固定参数和预期结果。"""

from fractions import Fraction
from math import lcm

import numpy as np

from .circulant import circulant_from_first_column
from .jones import deformation
from .tropical import TropicalMatrix


def paper_toy_parameters() -> tuple[
    TropicalMatrix,
    list[TropicalMatrix],
    np.ndarray,
    np.ndarray,
]:
    """返回论文给定的 N、H、Alice 私钥 A 和 Bob 私钥 B。"""

    # 论文玩具示例中直接给出的 3×3 Jones 矩阵 N。
    n = TropicalMatrix.from_integers(
        [
            [6, 5, 6],
            [6, 16, 12],
            [5, 9, 12],
        ]
    )
    # 三个 alpha 是论文指定的 Jones 变形参数。
    alphas = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    # 统一 scale 为所有分母的最小公倍数，保证分数可以精确存储。
    scale = lcm(*(alpha.denominator for alpha in alphas))
    # 公开向量 H 的每个分量都由同一个 Jones 矩阵 N 变形得到。
    h = [deformation(n, alpha, output_scale=scale) for alpha in alphas]
    # A、B 是论文示例中的固定私钥，并非本函数随机生成。
    alice = circulant_from_first_column([2, 4, 3])
    bob = circulant_from_first_column([0, 1, 2])
    return n, h, alice, bob


def paper_expected_alice_public() -> list[list[list[Fraction]]]:
    """返回论文公布的 Alice 公钥 U，用于核对程序计算结果。"""

    # 这里保存的是论文预期值，而不是再次调用算法计算 U。
    return [
        [
            [Fraction(27), Fraction(37), Fraction(33)],
            [Fraction(38), Fraction(48), Fraction(44)],
            [Fraction(31), Fraction(41), Fraction(37)],
        ],
        [
            [Fraction(101, 3), Fraction(131, 3), Fraction(119, 3)],
            [Fraction(134, 3), Fraction(164, 3), Fraction(152, 3)],
            [Fraction(113, 3), Fraction(143, 3), Fraction(131, 3)],
        ],
        [
            [Fraction(97, 3), Fraction(127, 3), Fraction(115, 3)],
            [Fraction(130, 3), Fraction(160, 3), Fraction(148, 3)],
            [Fraction(109, 3), Fraction(139, 3), Fraction(127, 3)],
        ],
    ]
