"""朴素算法、二进制快速幂和 Pippenger 多重指数求值器。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import OperationCounter
from .tropical import TropicalMatrix, multiply_many


def validate_action_inputs(
    bases: Sequence[TropicalMatrix],
    exponents: np.ndarray,
) -> np.ndarray:
    """检查 ``H^A`` 求值器的公共输入条件。

    ``bases`` 保存公开矩阵 ``H_i``；指数矩阵的每一列对应一个底数，
    每一行独立产生一个输出矩阵。所有底数必须具有相同阶数和 scale。
    """

    if not bases:
        raise ValueError("至少需要一个底数矩阵")
    exponent_matrix = np.asarray(exponents, dtype=np.int64)
    if exponent_matrix.ndim != 2:
        raise ValueError("指数 exponents 必须是二维矩阵")
    if exponent_matrix.shape[1] != len(bases):
        raise ValueError("指数矩阵每一行必须为每个底数提供一个指数")
    if np.any(exponent_matrix < 0):
        raise ValueError("指数必须为非负整数")
    order = bases[0].order
    scale = bases[0].scale
    if any(base.order != order or base.scale != scale for base in bases):
        raise ValueError("所有底数矩阵必须具有相同的阶数和 scale")
    return exponent_matrix


def multi_exponentiation_naive(
    bases: Sequence[TropicalMatrix],
    exponents: np.ndarray,
    *,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """使用逐次乘幂，独立计算每一个输出分量。"""

    exponent_matrix = validate_action_inputs(bases, exponents)
    order = bases[0].order
    scale = bases[0].scale
    outputs: list[TropicalMatrix] = []
    # 对指数矩阵的每一行分别计算：
    # H_1^a1 ⊗ H_2^a2 ⊗ ... ⊗ H_n^an。
    for row in exponent_matrix:
        # 指数为 0 的项等于单位矩阵，可以直接跳过。
        powered = (
            base.power_linear(int(exponent), counter)
            for base, exponent in zip(bases, row, strict=True)
            if exponent != 0
        )
        outputs.append(
            multiply_many(powered, order=order, scale=scale, counter=counter)
        )
    return outputs


def multi_exponentiation_binary(
    bases: Sequence[TropicalMatrix],
    exponents: np.ndarray,
    *,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """使用二进制平方—乘算法，独立计算每一个输出分量。"""

    exponent_matrix = validate_action_inputs(bases, exponents)
    order = bases[0].order
    scale = bases[0].scale
    outputs: list[TropicalMatrix] = []
    # 与朴素算法的输出公式相同，只替换单个 H_i^a_i 的求幂方法。
    for row in exponent_matrix:
        powered = (
            base.power_binary(int(exponent), counter)
            for base, exponent in zip(bases, row, strict=True)
            if exponent != 0
        )
        outputs.append(
            multiply_many(powered, order=order, scale=scale, counter=counter)
        )
    return outputs


def choose_window_size(number_of_bases: int, scalar_bits: int) -> int:
    """根据粗略乘法次数估算选择 Pippenger 窗口宽度。

    该结果只是在“窗口数量”和“桶数量”之间折中，不保证真实运行时间
    一定达到最优；Python 循环、内存分配和指数分布也会影响实际耗时。
    """

    if number_of_bases < 1:
        raise ValueError("number_of_bases 必须为正整数")
    if scalar_bits < 1:
        return 1
    # w 太小会产生较多窗口，w 太大则会产生 2^w 个桶。
    candidates = range(1, scalar_bits + 1)
    return min(
        candidates,
        key=lambda width: (
            (scalar_bits + width - 1) // width
        ) * (number_of_bases + (1 << (width + 1)))
        + scalar_bits,
    )


def _bucket_window(
    bases: Sequence[TropicalMatrix],
    exponent_row: np.ndarray,
    *,
    shift: int,
    window_size: int,
    counter: OperationCounter | None,
) -> TropicalMatrix | None:
    """用降序桶归约计算一个以 ``2^w`` 为基数的指数窗口。"""

    # 分桶会重新组织不同底数的乘法顺序，因此要求底数两两可交换。
    # 协议中的 H_i 均由同一个 Jones 矩阵生成，满足这一使用前提；
    # 任意热带矩阵通常不满足交换律，不能直接套用本算法。
    bucket_count = 1 << window_size
    mask = bucket_count - 1
    buckets: list[TropicalMatrix | None] = [None] * bucket_count

    for base, exponent in zip(bases, exponent_row, strict=True):
        # 右移 shift 位后用 mask 截取当前窗口的 w 个二进制位。
        digit = (int(exponent) >> shift) & mask
        if digit == 0:
            # digit=0 对当前窗口没有贡献，不放入桶中。
            continue
        previous = buckets[digit]
        # 当前窗口 digit 相同的底数先乘到同一个桶里，共享后续计算。
        buckets[digit] = (
            base if previous is None else previous.multiply(base, counter)
        )

    # running 保存“当前桶到最高桶”的累计乘积；weighted 再累计 running，
    # 使第 d 个桶最终恰好贡献 d 次，从而得到各 digit 的加权乘积。
    running: TropicalMatrix | None = None
    weighted: TropicalMatrix | None = None
    for digit in range(bucket_count - 1, 0, -1):
        bucket = buckets[digit]
        if bucket is not None:
            running = (
                bucket if running is None else running.multiply(bucket, counter)
            )
        if running is not None:
            weighted = (
                running
                if weighted is None
                else weighted.multiply(running, counter)
            )
    return weighted


def _pippenger_row(
    bases: Sequence[TropicalMatrix],
    exponent_row: np.ndarray,
    *,
    scalar_bits: int,
    window_size: int,
    counter: OperationCounter | None,
) -> TropicalMatrix:
    """使用固定窗口 Pippenger 算法计算指数矩阵中的一行。"""

    result: TropicalMatrix | None = None
    windows = (scalar_bits + window_size - 1) // window_size

    # 从最高窗口向最低窗口处理，相当于以 2^w 为基数进行 Horner 求值。
    for window in range(windows - 1, -1, -1):
        if result is not None:
            # 连续平方 w 次等价于 result^(2^w)，为低一窗口腾出位置。
            for _ in range(window_size):
                result = result.multiply(result, counter)

        # partial 是当前窗口通过分桶得到的加权乘积。
        partial = _bucket_window(
            bases,
            exponent_row,
            shift=window * window_size,
            window_size=window_size,
            counter=counter,
        )
        if partial is not None:
            result = (
                partial if result is None else result.multiply(partial, counter)
            )

    # 如果这一行指数全为 0，乘积按定义返回热带单位矩阵。
    return result or TropicalMatrix.identity(bases[0].order, bases[0].scale)


def multi_exponentiation_pippenger(
    bases: Sequence[TropicalMatrix],
    exponents: np.ndarray,
    *,
    window_size: int | None = None,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """逐行使用固定窗口 Pippenger 分桶算法计算 ``H^A``。"""

    exponent_matrix = validate_action_inputs(bases, exponents)
    # scalar_bits 是最大指数需要的二进制位数；全部指数为 0 时仍取 1。
    maximum = int(np.max(exponent_matrix, initial=0))
    scalar_bits = max(1, maximum.bit_length())
    # 所有行共用一个窗口宽度，但每一行仍独立执行 Pippenger 计算。
    width = window_size or choose_window_size(len(bases), scalar_bits)
    if width < 1:
        raise ValueError("window_size 必须为正整数")

    return [
        _pippenger_row(
            bases,
            row,
            scalar_bits=scalar_bits,
            window_size=width,
            counter=counter,
        )
        for row in exponent_matrix
    ]
