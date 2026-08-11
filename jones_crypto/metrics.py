"""用于比较不同算法运算量的计数工具。"""

from dataclasses import dataclass


@dataclass
class OperationCounter:
    """记录算法调用的热带矩阵乘法次数。

    这里只统计完整的热带矩阵乘法，不统计矩阵内部的标量加法、max、
    内存分配或协议中的普通矩阵加减法，因此不能完全代替实际耗时。
    """

    tropical_matrix_multiplications: int = 0

    def reset(self) -> None:
        """把累计的热带矩阵乘法次数清零。"""

        self.tropical_matrix_multiplications = 0
