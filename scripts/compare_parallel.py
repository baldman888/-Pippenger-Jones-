"""比较串行 Pippenger 与按行多线程 Pippenger 的运行时间。"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

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
    multi_exponentiation_pippenger,
)
from jones_crypto.parallel import (  
    multi_exponentiation_pippenger_parallel,
)


PAPER_ROWS = [
    (10, 80, 2),
    (14, 51, 3),
    (21, 40, 4),
    (25, 39, 5),
    (28, 33, 6),
]


def timed_action(action, bases, private, *, warmups, repeats, **kwargs):
    """预热后重复计时，返回输出、中位时间和矩阵乘法次数。"""

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

    if len(set(operation_counts)) != 1:
        raise AssertionError("同一算法多次运行的矩阵乘法次数不一致")
    return outputs, median(times), operation_counts[0]


def outputs_equal(left, right):
    return len(left) == len(right) and all(
        a.equals(b) for a, b in zip(left, right, strict=True)
    )


def run_parameter_set(k, n, s, workers, requested_window, warmups, repeats):
    """对一组参数测试串行版本和多个线程数量。"""

    jones = generate_jones_matrix(k, max_entry=1000, seed=10_000 + k)
    bases = generate_public_deformations(jones, n)
    private = random_circulant(n, s, seed=20_000 + n + s)
    scalar_bits = max(1, int(private.max()).bit_length())
    window = requested_window or choose_window_size(n, scalar_bits)

    serial, serial_time, serial_ops = timed_action(
        multi_exponentiation_pippenger,
        bases,
        private,
        warmups=warmups,
        repeats=repeats,
        window_size=window,
    )

    rows = []
    for worker_count in workers:
        parallel, parallel_time, parallel_ops = timed_action(
            multi_exponentiation_pippenger_parallel,
            bases,
            private,
            warmups=warmups,
            repeats=repeats,
            window_size=window,
            max_workers=worker_count,
        )
        if not outputs_equal(serial, parallel):
            raise AssertionError(f"{worker_count} 线程输出与串行输出不一致")
        if serial_ops != parallel_ops:
            raise AssertionError(f"{worker_count} 线程的矩阵乘法次数与串行不一致")

        rows.append(
            {
                "k": k,
                "n": n,
                "s": s,
                "window": window,
                "workers": worker_count,
                "serial_seconds": serial_time,
                "parallel_seconds": parallel_time,
                "speedup": serial_time / parallel_time,
                "saved_milliseconds": (serial_time - parallel_time) * 1000,
                "matmul": parallel_ops,
            }
        )
    return rows


def parse_workers(value: str) -> list[int]:
    try:
        workers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("线程数必须使用逗号分隔的整数") from error
    if not workers or any(item < 1 for item in workers):
        raise argparse.ArgumentTypeError("线程数必须为正整数")
    return list(dict.fromkeys(workers))


def main():
    parser = argparse.ArgumentParser(
        description="比较串行 Pippenger 与按行多线程 Pippenger 的性能。"
    )
    parser.add_argument("--quick", action="store_true", help="只运行第一组参数")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=parse_workers("1,2,4,8"),
        help="线程数量列表，默认为 1,2,4,8",
    )
    parser.add_argument("--window", type=int, help="手动指定窗口宽度")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    parser.add_argument("--repeat", type=int, default=7, help="正式测量次数")
    parser.add_argument("--csv", type=Path, help="可选：保存 CSV 的路径")
    args = parser.parse_args()

    if args.window is not None and args.window < 1:
        parser.error("--window 必须为正整数")
    if args.warmup < 0:
        parser.error("--warmup 不能为负数")
    if args.repeat < 1:
        parser.error("--repeat 必须为正整数")

    parameter_sets = PAPER_ROWS[:1] if args.quick else PAPER_ROWS
    results = []
    for parameters in parameter_sets:
        results.extend(
            run_parameter_set(
                *parameters,
                args.workers,
                args.window,
                args.warmup,
                args.repeat,
            )
        )

    print(f"逻辑处理器数量：{os.cpu_count()}")
    print(f"计时方式：预热 {args.warmup} 次，测量 {args.repeat} 次并取中位数")
    print("串行与并行版本使用相同输入，且输出和矩阵乘法次数已经验证一致。")
    print("\n k   n    s  w  线程  串行时间(ms)  并行时间(ms)  节省时间(ms)  加速比")
    print("-" * 82)
    for result in results:
        print(
            f"{result['k']:2d} {result['n']:3d} {result['s']:4d}"
            f" {result['window']:2d} {result['workers']:5d}"
            f" {result['serial_seconds'] * 1000:13.3f}"
            f" {result['parallel_seconds'] * 1000:13.3f}"
            f" {result['saved_milliseconds']:13.3f}"
            f" {result['speedup']:8.3f}x"
        )

    print("\n说明：加速比大于 1 表示多线程更快，小于 1 表示线程开销使其变慢。")
    print("      每次计时均包含线程池的创建和关闭时间。")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"已保存：{args.csv.resolve()}")


if __name__ == "__main__":
    main()
