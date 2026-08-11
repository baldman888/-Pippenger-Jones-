"""在完全相同的实例上比较线性、二进制快速幂与 Pippenger 算法。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

# 把项目根目录加入模块搜索路径，便于直接右击运行本脚本。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jones_crypto.circulant import random_circulant  
from jones_crypto.jones import (  
    generate_jones_matrix,
    generate_public_deformations,
)
from jones_crypto.metrics import OperationCounter  
from jones_crypto.multi_exponentiation import (  
    choose_window_size,
    multi_exponentiation_binary,
    multi_exponentiation_naive,
    multi_exponentiation_pippenger,
)

PAPER_ROWS = [
    # k、n、s，以及论文表 2 报告的一次 H^A 计算时间（秒）。
    (10, 80, 2, 1.082),
    (14, 51, 3, 1.929),
    (21, 40, 4, 2.380),
    (25, 39, 5, 5.618),
    (28, 33, 6, 5.686),
]

EXTRA_ROWS = [
    # 额外扩大指数范围，用于观察大指数下的 Pippenger 表现。
    (10, 80, 16, None),
    (10, 80, 256, None),
    (14, 128, 1024, None),
]


def timed_action(action, bases, private, *, warmups, repeats, **kwargs):
    """预热后重复测量多重指数运算，返回中位时间和乘法次数。"""

    # 预热不计入结果，用于减小首次运行产生的偶然波动。
    for _ in range(warmups):
        action(bases, private, **kwargs)

    times = []
    outputs = None
    operation_counts = []
    for _ in range(repeats):
        counter = OperationCounter()
        start = perf_counter()
        outputs = action(bases, private, counter=counter, **kwargs)
        times.append(perf_counter() - start)
        operation_counts.append(counter.tropical_matrix_multiplications)

    # 对同一输入，同一种算法每次执行的乘法次数应完全相同。
    if len(set(operation_counts)) != 1:
        raise AssertionError("同一算法多次运行的矩阵乘法次数不一致")
    return outputs, median(times), operation_counts[0]


def run_row(k, n, s, paper_seconds, requested_window, warmups, repeats):
    """在相同公开矩阵和私钥指数矩阵上运行三种算法。"""

    # 固定 seed，使三种算法得到完全相同的输入。
    jones = generate_jones_matrix(k, max_entry=1000, seed=10_000 + k)
    bases = generate_public_deformations(jones, n)
    private = random_circulant(n, s, seed=20_000 + n + s)
    scalar_bits = max(1, int(private.max()).bit_length())
    # 未指定 --window 时，根据底数数量和指数位数自动选择窗口。
    window = requested_window or choose_window_size(n, scalar_bits)

    # 三次计时只替换 H^A 求值器，其余参数和计数方式保持一致。
    naive, naive_time, naive_ops = timed_action(
        multi_exponentiation_naive,
        bases,
        private,
        warmups=warmups,
        repeats=repeats,
    )
    binary, binary_time, binary_ops = timed_action(
        multi_exponentiation_binary,
        bases,
        private,
        warmups=warmups,
        repeats=repeats,
    )
    pippenger, pippenger_time, pippenger_ops = timed_action(
        multi_exponentiation_pippenger,
        bases,
        private,
        warmups=warmups,
        repeats=repeats,
        window_size=window,
    )
    # 性能比较前先逐个检查输出，避免拿错误结果讨论加速效果。
    for name, output in (("二进制快速幂", binary), ("Pippenger", pippenger)):
        if not all(
            left.equals(right)
            for left, right in zip(naive, output, strict=True)
        ):
            raise AssertionError(f"{name} 输出与线性算法不一致")

    return {
        "k": k,
        "n": n,
        "s": s,
        "window": window,
        "paper_seconds": "" if paper_seconds is None else paper_seconds,
        "naive_seconds": naive_time,
        "binary_seconds": binary_time,
        "pippenger_seconds": pippenger_time,
        "pippenger_vs_naive_speedup": naive_time / pippenger_time,
        "pippenger_vs_binary_speedup": binary_time / pippenger_time,
        "naive_matmul": naive_ops,
        "binary_matmul": binary_ops,
        "pippenger_matmul": pippenger_ops,
        "pippenger_vs_naive_reduction_percent": (
            100.0 * (naive_ops - pippenger_ops) / naive_ops
            if naive_ops
            else 0.0
        ),
        "pippenger_vs_binary_reduction_percent": (
            100.0 * (binary_ops - pippenger_ops) / binary_ops
            if binary_ops
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="比较线性、二进制快速幂与 Pippenger 计算同一次 H^A 的性能。"
    )
    parser.add_argument("--quick", action="store_true", help="仅运行第一组参数")
    parser.add_argument("--extra", action="store_true", help="追加大指数实验参数")
    parser.add_argument("--window", type=int, help="手动指定 Pippenger 窗口宽度")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="每种算法的预热次数，默认为 2",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="每种算法的正式测量次数，默认为 7，并取中位数",
    )
    parser.add_argument("--csv", type=Path, help="可选：指定 CSV 结果保存路径")
    args = parser.parse_args()
    if args.window is not None and args.window < 1:
        parser.error("--window 必须是正整数")
    if args.warmup < 0:
        parser.error("--warmup 不能为负数")
    if args.repeat < 1:
        parser.error("--repeat 必须是正整数")

    rows = list(PAPER_ROWS)
    if args.quick:
        rows = rows[:1]
    if args.extra:
        rows.extend(EXTRA_ROWS)

    results = [
        run_row(*row, args.window, args.warmup, args.repeat)
        for row in rows
    ]

    print(f"计时方式：预热 {args.warmup} 次，正式测量 {args.repeat} 次并取中位数")
    print("三种算法使用完全相同的 H 与私钥 A，且输出已经逐项验证一致。")
    print("\n运行时间（单位：秒）")
    print(
        " k   n    s  w  线性重复乘法时间  二进制快速幂时间  Pippenger分桶法时间"
        "  P/线性加速比  P/二进制加速比"
    )
    print("-" * 116)
    for result in results:
        print(
            f"{result['k']:2d} {result['n']:3d} {result['s']:4d}"
            f" {result['window']:2d}"
            f" {result['naive_seconds']:11.6f}"
            f" {result['binary_seconds']:13.6f}"
            f" {result['pippenger_seconds']:15.6f}"
            f" {result['pippenger_vs_naive_speedup']:13.3f}x"
            f" {result['pippenger_vs_binary_speedup']:15.3f}x"
        )

    print("\n热带矩阵乘法次数")
    print(
        " k   n    s  w  线性重复乘法次数  二进制快速幂次数  Pippenger分桶法次数"
        "  相对线性减少  相对二进制减少"
    )
    print("-" * 104)
    for result in results:
        print(
            f"{result['k']:2d} {result['n']:3d} {result['s']:4d}"
            f" {result['window']:2d}"
            f" {result['naive_matmul']:11d}"
            f" {result['binary_matmul']:11d}"
            f" {result['pippenger_matmul']:14d}"
            f" {result['pippenger_vs_naive_reduction_percent']:12.2f}%"
            f" {result['pippenger_vs_binary_reduction_percent']:14.2f}%"
        )

    # 各列含义（三种算法使用完全相同的 H 与私钥 A）：
    # k：每个热带 Jones 矩阵的阶数，即单个 H_i 是 k×k 矩阵。
    # n：公开矩阵向量 H 的长度；私钥 A 同时是 n×n 循环矩阵。
    # s：私钥指数的取值上界，代码实际从 0 到 s-1 中取值。
    # w：Pippenger 的窗口宽度（位数）；每个窗口有 2^w 个桶。
    # 线性重复乘法时间：逐次乘法计算一次 H^A 的中位耗时。
    # 二进制快速幂时间：平方—乘算法计算同一次 H^A 的中位耗时。
    # Pippenger分桶法时间：Pippenger 算法计算同一次 H^A 的中位耗时。
    # P/线性加速比：线性时间 / Pippenger时间。
    # P/二进制加速比：二进制快速幂时间 / Pippenger时间。
    # 两种加速比大于 1 时，均表示 Pippenger 更快。
    # 线性重复乘法次数：逐次乘法算法执行的热带矩阵乘法次数。
    # 二进制快速幂乘法次数：平方—乘算法执行的热带矩阵乘法次数。
    # Pippenger分桶法次数：Pippenger 执行的热带矩阵乘法次数。
    # 两个减少率分别以线性算法和二进制快速幂为参照。
    # 注意：这里测量的是一次 H^A，不是完整的密钥生成、加密或解密耗时。
    print("\n说明：以上时间均对应一次 H^A，不是完整的加密或解密时间。")
    print("      加速比 > 1 表示 Pippenger 更快；减少率 > 0 表示乘法次数减少。")

    # 指定 --csv 时，把原始数值另存为 CSV，便于后续绘图和统计。
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n已保存：{args.csv.resolve()}")


if __name__ == "__main__":
    main()
