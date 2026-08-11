"""Pippenger 多重指数运算的按行并行实现。"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from .metrics import OperationCounter
from .multi_exponentiation import (
    _pippenger_row,
    choose_window_size,
    validate_action_inputs,
)
from .tropical import TropicalMatrix


def multi_exponentiation_pippenger_parallel(
    bases: Sequence[TropicalMatrix],
    exponents: np.ndarray,
    *,
    window_size: int | None = None,
    max_workers: int | None = None,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """使用线程池并行计算指数矩阵各行对应的 Pippenger 结果。

    指数矩阵的每一行都会独立产生一个输出矩阵，因此可以安全地分配给
    不同线程。函数使用 ``executor.map`` 收集结果，返回顺序仍与指数矩阵
    的行顺序一致。

    为避免多个线程同时修改同一个计数器，每个线程使用自己的局部计数器，
    所有任务结束后再由主线程汇总到 ``counter``。
    """

    exponent_matrix = validate_action_inputs(bases, exponents)
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers 必须为正整数或 None")

    if window_size is not None and window_size < 1:
        raise ValueError("window_size 必须为正整数")

    maximum = int(np.max(exponent_matrix, initial=0))
    scalar_bits = max(1, maximum.bit_length())
    width = (
        window_size
        if window_size is not None
        else choose_window_size(len(bases), scalar_bits)
    )

    # 转成元组后由所有线程只读共享；计算过程不会修改公开底数矩阵。
    shared_bases = tuple(bases)

    row_count = len(exponent_matrix)
    if row_count == 0:
        return []

    # 每个线程领取一批连续的行，避免为指数矩阵的每一行都创建独立任务。
    worker_count = min(max_workers or (os.cpu_count() or 1), row_count)
    chunk_size = (row_count + worker_count - 1) // worker_count
    row_chunks = [
        exponent_matrix[start : start + chunk_size]
        for start in range(0, row_count, chunk_size)
    ]

    def calculate_chunk(
        rows: np.ndarray,
    ) -> tuple[list[TropicalMatrix], int]:
        local_counter = OperationCounter() if counter is not None else None
        results = [
            _pippenger_row(
                shared_bases,
                row,
                scalar_bits=scalar_bits,
                window_size=width,
                counter=local_counter,
            )
            for row in rows
        ]
        operation_count = (
            local_counter.tropical_matrix_multiplications
            if local_counter is not None
            else 0
        )
        return results, operation_count

    # map 会并行执行各批次，并按照批次的输入顺序返回结果。
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_results = list(executor.map(calculate_chunk, row_chunks))

    if counter is not None:
        counter.tropical_matrix_multiplications += sum(
            operation_count for _, operation_count in chunk_results
        )

    return [
        result
        for results, _ in chunk_results
        for result in results
    ]
