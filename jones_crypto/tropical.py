"""精确实现 max-plus 热带矩阵运算。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np

from .metrics import OperationCounter

# 使用一个足够小的 int64 整数作为热带负无穷的内部哨兵值。
# 它不是真正的数学负无穷；取 int64 最小值的四分之一，是为了在本项目
# 的实验参数范围内执行加法时留出余量，避免整数下溢。
NEG_INF = np.int64(np.iinfo(np.int64).min // 4)


@dataclass(frozen=True)
class TropicalMatrix:
    """一个方形 max-plus 矩阵，真实元素等于 ``values / scale``。

    使用整数和统一分母 scale 可以避免浮点误差，精确保存论文中的分数。
    """

    values: np.ndarray
    scale: int = 1

    def __post_init__(self) -> None:
        """统一数据类型并检查矩阵必须为方阵、scale 必须为正数。"""

        values = np.asarray(self.values, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("热带矩阵必须是二维方阵")
        if self.scale <= 0:
            raise ValueError("scale 必须为正整数")
        object.__setattr__(self, "values", values.copy())

    @property
    def order(self) -> int:
        """返回矩阵阶数 k。"""

        return int(self.values.shape[0])

    @classmethod
    def identity(cls, order: int, scale: int = 1) -> "TropicalMatrix":
        """构造热带单位矩阵：对角线为 0，其余位置为负无穷。"""

        values = np.full((order, order), NEG_INF, dtype=np.int64)
        np.fill_diagonal(values, 0)
        return cls(values, scale)

    @classmethod
    def from_integers(cls, values: Iterable[Iterable[int]]) -> "TropicalMatrix":
        """由普通整数二维数据构造 scale=1 的热带矩阵。"""

        return cls(np.asarray(values, dtype=np.int64), 1)

    def multiply(
        self,
        other: "TropicalMatrix",
        counter: OperationCounter | None = None,
    ) -> "TropicalMatrix":
        """返回两个矩阵的 max-plus 热带乘积。"""

        self._check_compatible(other)
        if counter is not None:
            counter.tropical_matrix_multiplications += 1

        # left 形状为 (k,k,1)，right 形状为 (1,k,k)。NumPy 广播后，
        # 第 (i,j) 个元素沿中间下标 t 计算 max_t(A[i,t] + B[t,j])。
        left = self.values[:, :, None]
        right = other.values[None, :, :]
        sums = left + right
        # 负无穷与任何元素相乘（max-plus 中相加）仍为负无穷。
        invalid = (left == NEG_INF) | (right == NEG_INF)
        sums = np.where(invalid, NEG_INF, sums)
        # axis=1 对应公式中的中间下标 t。
        product = np.max(sums, axis=1)
        return TropicalMatrix(product, self.scale)

    def tropical_add(self, other: "TropicalMatrix") -> "TropicalMatrix":
        """执行热带加法，即两个矩阵对应位置逐元素取最大值。"""

        self._check_compatible(other)
        return TropicalMatrix(np.maximum(self.values, other.values), self.scale)

    def ordinary_add(self, other: "TropicalMatrix") -> "TropicalMatrix":
        """执行逐元素普通加法，用于协议中的消息掩码。"""

        self._check_finite_compatible(other)
        return TropicalMatrix(self.values + other.values, self.scale)

    def ordinary_subtract(self, other: "TropicalMatrix") -> "TropicalMatrix":
        """执行逐元素普通减法，用于解密时移除消息掩码。"""

        self._check_finite_compatible(other)
        return TropicalMatrix(self.values - other.values, self.scale)

    def power_linear(
        self,
        exponent: int,
        counter: OperationCounter | None = None,
    ) -> "TropicalMatrix":
        """用逐次乘法计算非负整数幂，作为朴素 baseline。"""

        if exponent < 0:
            raise ValueError("exponent 必须为非负整数")
        if exponent == 0:
            # 按幂的定义，任何矩阵的 0 次幂都是单位矩阵。
            return TropicalMatrix.identity(self.order, self.scale)
        result = self
        # result 已包含一个 self，因此指数 e 只需再执行 e-1 次乘法。
        for _ in range(1, exponent):
            result = result.multiply(self, counter)
        return result

    def power_binary(
        self,
        exponent: int,
        counter: OperationCounter | None = None,
    ) -> "TropicalMatrix":
        """用二进制平方—乘算法求幂，供实验对照和正确性验证。"""

        if exponent < 0:
            raise ValueError("exponent 必须为非负整数")
        result: TropicalMatrix | None = None
        base = self
        value = exponent
        while value:
            # 当前最低位为 1 时，把对应的 base 幂乘入结果。
            if value & 1:
                result = base if result is None else result.multiply(base, counter)
            value >>= 1
            if value:
                # 每右移一位，base 平方一次，依次表示 H、H^2、H^4……
                base = base.multiply(base, counter)
        return result or TropicalMatrix.identity(self.order, self.scale)

    def as_fractions(self) -> list[list[Fraction | None]]:
        """返回便于阅读的精确分数；``None`` 表示热带负无穷。"""

        return [
            [
                None if value == NEG_INF else Fraction(int(value), self.scale)
                for value in row
            ]
            for row in self.values
        ]

    def equals(self, other: "TropicalMatrix") -> bool:
        """换算到共同 scale 后比较矩阵元素是否一致。

        本项目主要用它比较相同 scale 的算法输出。由于 ``NEG_INF`` 是
        有限哨兵值，含负无穷且 scale 不同的矩阵不应直接依赖此方法比较。
        """

        if self.order != other.order:
            return False
        common_scale = np.lcm(self.scale, other.scale)
        left = self.values * (common_scale // self.scale)
        right = other.values * (common_scale // other.scale)
        return bool(np.array_equal(left, right))

    def _check_compatible(self, other: "TropicalMatrix") -> None:
        if self.order != other.order:
            raise ValueError("两个矩阵的阶数不同")
        if self.scale != other.scale:
            raise ValueError("两个矩阵的 scale 不同，请先统一 scale")

    def _check_finite_compatible(self, other: "TropicalMatrix") -> None:
        self._check_compatible(other)
        if np.any(self.values == NEG_INF) or np.any(other.values == NEG_INF):
            raise ValueError("普通加减法要求矩阵中的所有元素均为有限值")


def multiply_many(
    matrices: Iterable[TropicalMatrix],
    *,
    order: int,
    scale: int,
    counter: OperationCounter | None = None,
) -> TropicalMatrix:
    """依次连乘多个矩阵，并避免无意义的单位矩阵乘法。"""

    # 用 None 延迟初始化：第一个矩阵可直接作为结果，无需先乘单位矩阵。
    result: TropicalMatrix | None = None
    for matrix in matrices:
        result = matrix if result is None else result.multiply(matrix, counter)
    # 输入为空时，对应“空乘积”，按定义返回单位矩阵。
    return result or TropicalMatrix.identity(order, scale)
