"""构造和检查协议私钥使用的非负整数循环矩阵。"""

from __future__ import annotations

import numpy as np


def circulant_from_first_column(first_column: np.ndarray | list[int]) -> np.ndarray:
    """根据第一列构造论文采用的循环矩阵排列。"""

    # 先统一转换成一维 int64 数组，后续才能可靠地检查维度和非负性。
    column = np.asarray(first_column, dtype=np.int64)
    if column.ndim != 1 or len(column) == 0:
        raise ValueError("first_column 必须是一维非空向量")
    if np.any(column < 0):
        raise ValueError("循环矩阵元素必须为非负整数")
    n = len(column)
    # 论文使用 A[row, col] = column[(row-col) mod n]；
    # 因而后一行可看作前一行循环右移一位。
    return np.asarray(
        [[column[(row - col) % n] for col in range(n)] for row in range(n)],
        dtype=np.int64,
    )


def random_circulant(
    order: int,
    exponent_bound: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """均匀抽取第一列，再由它构造循环矩阵。

    第一列元素取自 ``[0, exponent_bound - 1]``。矩阵的其他元素由
    循环结构确定，并不是彼此独立随机生成的。
    """

    if order < 1:
        raise ValueError("order 必须为正整数")
    if exponent_bound < 1:
        raise ValueError("exponent_bound 必须为正整数")
    rng = np.random.default_rng(seed)
    # 对私钥而言，exponent_bound 就是实验参数 s。
    first_column = rng.integers(0, exponent_bound, size=order, dtype=np.int64)
    return circulant_from_first_column(first_column)


def is_circulant(matrix: np.ndarray) -> bool:
    """判断输入是否为由第一列循环移位得到的方阵。"""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return bool(
        np.array_equal(
            matrix,
            circulant_from_first_column(matrix[:, 0]),
        )
    )
