"""Jones 矩阵、Jones 变形和可重复的公开参数构造。"""

from __future__ import annotations

from fractions import Fraction
from math import lcm

import numpy as np

from .tropical import TropicalMatrix


def is_jones(matrix: TropicalMatrix) -> bool:
    """检查所有 i、j、k 是否满足 Jones 条件。

    Jones 条件为 ``a_ij + a_jk <= a_ik + a_jj``。
    """

    a = matrix.values
    n = matrix.order
    for i in range(n):
        for j in range(n):
            if np.any(a[i, j] + a[j, :] > a[i, :] + a[j, j]):
                return False
    return True


def deformation(
    matrix: TropicalMatrix,
    alpha: Fraction,
    *,
    output_scale: int | None = None,
) -> TropicalMatrix:
    """精确计算 Jones 变形 ``N^(alpha)``。

    此处上标 ``(alpha)`` 表示论文定义的 Jones 变形，不是普通矩阵幂。
    """

    alpha = Fraction(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必须位于 [0, 1] 区间")
    # scale 是统一分母；values 中保存的整数除以 scale 才是真实矩阵元素。
    scale = output_scale or lcm(matrix.scale, alpha.denominator)
    if scale % matrix.scale or scale % alpha.denominator:
        raise ValueError("output_scale 必须能容纳所有输入分母")

    # 先把原矩阵换算到统一 scale，再计算论文定义的校正项。
    source_factor = scale // matrix.scale
    alpha_minus_one_scaled = (alpha - 1) * scale
    if alpha_minus_one_scaled.denominator != 1:
        raise ValueError("output_scale 无法精确表示 alpha")

    diagonal = np.diag(matrix.values) * source_factor
    diagonal_max = np.maximum(diagonal[:, None], diagonal[None, :])
    correction_numerator = int(alpha_minus_one_scaled)
    if np.any(diagonal_max % scale != 0):
        # 对带 scale 的一般输入，校正项可能不能直接整除；这里保留额外
        # 分母进行整数运算，最后再换算回指定的统一 scale。
        numerator = (
            matrix.values * source_factor * scale
            + correction_numerator * diagonal_max
        )
        if np.any(numerator % scale != 0):
            raise ValueError("指定的 output_scale 不足以精确表示结果")
        values = numerator // scale
    else:
        values = (
            matrix.values * source_factor
            + correction_numerator * (diagonal_max // scale)
        )
    return TropicalMatrix(values, scale)


def generate_jones_matrix(
    order: int,
    *,
    max_entry: int = 1000,
    seed: int = 2024,
) -> TropicalMatrix:
    """利用势函数和距离构造一个有限的有效 Jones 矩阵。

    构造式为 ``a_ij = r_i + c_j - |x_i-x_j|``。绝对值距离满足
    三角不等式，因此该构造满足 Jones 条件。
    """

    if order < 1:
        raise ValueError("order 必须为正整数")
    if max_entry < 8:
        raise ValueError("max_entry 必须至少为 8")
    rng = np.random.default_rng(seed)
    # quarter 用于划分安全的随机区间：x 控制距离，row/col 是行列势函数。
    # 这些范围保证结果非负，并使最大元素不超过 max_entry。
    quarter = max_entry // 4
    x = rng.integers(0, quarter + 1, size=order, dtype=np.int64)
    row = rng.integers(quarter, 2 * quarter + 1, size=order, dtype=np.int64)
    col = rng.integers(quarter, 2 * quarter + 1, size=order, dtype=np.int64)
    distance = np.abs(x[:, None] - x[None, :])
    # distance 满足三角不等式，这是该随机构造满足 Jones 条件的关键。
    values = row[:, None] + col[None, :] - distance
    matrix = TropicalMatrix(values)
    if not is_jones(matrix):
        raise AssertionError("内部 Jones 矩阵构造失败")
    return matrix


def generate_public_deformations(
    matrix: TropicalMatrix,
    count: int,
) -> list[TropicalMatrix]:
    """由一个 Jones 矩阵生成基准实验使用的公开矩阵 ``H_i``。

    论文要求各 ``H_i`` 位于同一个 Jones 矩阵生成的拟多项式半环中，
    但没有公开性能表所用随机生成器的完整细节。因此这里采用
    ``H_i = N^((i+1)/(count+1))`` 构造可重复的纯变形实验实例；
    不能把它理解成论文计时实验随机分布的精确复原。
    """

    if count < 1:
        raise ValueError("count 必须为正整数")
    # 所有 alpha 的分母都整除 count+1，因此可以共用同一个 scale。
    output_scale = lcm(matrix.scale, count + 1)
    return [
        deformation(
            matrix,
            Fraction(index + 1, count + 1),
            output_scale=output_scale,
        )
        for index in range(count)
    ]
