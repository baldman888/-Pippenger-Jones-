"""在同一输入上比较线性、二进制快速幂与 Pippenger 的完整 H^A 求值。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

# 允许直接运行本文件，不必先安装项目包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jones_crypto.circulant import random_circulant
from jones_crypto.jones import generate_jones_matrix, generate_public_deformations
from jones_crypto.metrics import OperationCounter
from jones_crypto.multi_exponentiation import (
    multi_exponentiation_binary,
    multi_exponentiation_naive,
    multi_exponentiation_pippenger,
)


# 每行依次为 (k, n, s, w)。
# 论文表 3：五组参数，每组使用各自的窗口宽度。
# EXPERIMENT_ROWS = [
#     (10, 80, 2, 1),
#     (14, 51, 3, 2),
#     (21, 40, 4, 2),
#     (25, 39, 5, 3),
#     (28, 33, 6, 3),
# ]

#论文表 5：同一份 (10, 80, 128) 输入，依次比较五个窗口宽度。
EXPERIMENT_ROWS = [
    (10, 80, 128, 7),
    (10, 80, 128, 4),
    (10, 80, 128, 3),
    (10, 80, 128, 2),
    (10, 80, 128, 1),
]


def same_output(reference, candidate):
    """逐项比较完整 H^A 的输出。"""

    return len(reference) == len(candidate) and all(
        left.equals(right) for left, right in zip(reference, candidate, strict=True)
    )


def run_group(k, n, s, windows, *, warmups, repeats):
    """一份输入测全部指定窗口；线性和二进制基线各只测一次。"""

    if len(set(windows)) != len(windows) or any(w < 1 for w in windows):
        raise ValueError("同组窗口宽度必须是互不重复的正整数")

    # 固定这些种子保证两种表格模式的输入生成方式相同。
    jones = generate_jones_matrix(k, max_entry=1000, seed=10_000 + k)
    bases = generate_public_deformations(jones, n)
    private = random_circulant(n, s, seed=20_000 + n + s)
    scalar_bits = max(1, int(private.max()).bit_length())

    strategies = {
        "naive": (multi_exponentiation_naive, {}),
        "binary": (multi_exponentiation_binary, {}),
    }
    for window in windows:
        strategies[f"pippenger_{window}"] = (
            multi_exponentiation_pippenger,
            {"window_size": window},
        )

    # 先验证结果，避免把错误的快速结果写入论文。
    reference = multi_exponentiation_naive(bases, private)
    for name, (action, kwargs) in strategies.items():
        if name == "naive":
            continue
        if not same_output(reference, action(bases, private, **kwargs)):
            raise AssertionError(f"{name} 的输出与线性重复乘法不一致")

    names = list(strategies)
    for round_index in range(warmups):
        for offset in range(len(names)):
            name = names[(round_index + offset) % len(names)]
            action, kwargs = strategies[name]
            action(bases, private, **kwargs)

    times = {name: [] for name in names}
    operation_counts = {name: [] for name in names}
    # 每轮改变算法执行顺序，减轻温度、缓存等时间漂移带来的偏差。
    for round_index in range(repeats):
        for offset in range(len(names)):
            name = names[(round_index + offset) % len(names)]
            action, kwargs = strategies[name]
            counter = OperationCounter()
            start = perf_counter()
            action(bases, private, counter=counter, **kwargs)
            times[name].append(perf_counter() - start)
            operation_counts[name].append(counter.tropical_matrix_multiplications)

    for name, counts in operation_counts.items():
        if len(set(counts)) != 1:
            raise AssertionError(f"{name} 的矩阵乘法次数在重复测量间不一致")

    naive_ms = 1000 * median(times["naive"])
    binary_ms = 1000 * median(times["binary"])
    naive_ops = operation_counts["naive"][0]
    binary_ops = operation_counts["binary"][0]
    results = []
    for window in windows:
        name = f"pippenger_{window}"
        pippenger_ms = 1000 * median(times[name])
        results.append(
            {
                "k": k,
                "n": n,
                "s": s,
                "window": window,
                "window_count": (scalar_bits + window - 1) // window,
                "naive_ms": naive_ms,
                "binary_ms": binary_ms,
                "pippenger_ms": pippenger_ms,
                "saved_vs_naive_ms": naive_ms - pippenger_ms,
                "saved_vs_binary_ms": binary_ms - pippenger_ms,
                "naive_matmul": naive_ops,
                "binary_matmul": binary_ops,
                "pippenger_matmul": operation_counts[name][0],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="按文件顶部启用的参数组生成论文表 3 或表 5 的实验数据。"
    )
    parser.add_argument("--quick", action="store_true", help="仅运行启用参数组的第一行")
    parser.add_argument("--warmup", type=int, default=2, help="预热轮数，默认 2")
    parser.add_argument("--repeat", type=int, default=31, help="测量轮数，默认 31，取中位数")
    parser.add_argument("--csv", type=Path, help="可选：保存汇总结果的 CSV 路径")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup 不能为负数")
    if args.repeat < 1:
        parser.error("--repeat 必须是正整数")

    rows = EXPERIMENT_ROWS[:1] if args.quick else EXPERIMENT_ROWS
    if not rows:
        parser.error("请在文件顶部启用一组实验参数")
    groups = {}
    for k, n, s, window in rows:
        groups.setdefault((k, n, s), []).append(window)

    results = []
    for (k, n, s), windows in groups.items():
        results.extend(
            run_group(k, n, s, windows, warmups=args.warmup, repeats=args.repeat)
        )

    print(f"计时方式：每种策略预热 {args.warmup} 次、测量 {args.repeat} 次，取中位数。")
    print("同一参数组共用输入；不同窗口共用线性与二进制基线。")
    print("\n运行时间（单位：ms）")
    print("(k,n,s)\tw\t线性重复乘法\t二进制快速幂\tPippenger\t相对线性节省\t相对二进制节省")
    for result in results:
        print(
            f"({result['k']},{result['n']},{result['s']})\t{result['window']}\t"
            f"{result['naive_ms']:.3f}\t{result['binary_ms']:.3f}\t"
            f"{result['pippenger_ms']:.3f}\t{result['saved_vs_naive_ms']:.3f}\t"
            f"{result['saved_vs_binary_ms']:.3f}"
        )

    print("\n热带矩阵乘法次数")
    print("(k,n,s)\tw\t窗口数t\t线性重复乘法\t二进制快速幂\tPippenger")
    for result in results:
        print(
            f"({result['k']},{result['n']},{result['s']})\t{result['window']}\t"
            f"{result['window_count']}\t{result['naive_matmul']}\t"
            f"{result['binary_matmul']}\t{result['pippenger_matmul']}"
        )

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n已保存：{args.csv.resolve()}")


if __name__ == "__main__":
    main()
