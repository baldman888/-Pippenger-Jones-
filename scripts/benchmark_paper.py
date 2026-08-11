"""按论文表 2 参数比较原论文与线性重复乘法的 ``H^A`` 时间。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

# 把项目根目录加入模块搜索路径，便于直接右击运行本脚本。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jones_crypto.circulant import random_circulant  # noqa: E402
from jones_crypto.jones import (  # noqa: E402
    generate_jones_matrix,
    generate_public_deformations,
)
from jones_crypto.metrics import OperationCounter  # noqa: E402
from jones_crypto.multi_exponentiation import multi_exponentiation_naive  # noqa: E402

PAPER_ROWS = [
    # k、n、s，以及论文表 2 报告的一次 H^A 计算时间（秒）。
    (10, 80, 2, 1.082),
    (14, 51, 3, 1.929),
    (21, 40, 4, 2.380),
    (25, 39, 5, 5.618),
    (28, 33, 6, 5.686),
]


def run_row(k: int, n: int, s: int, paper_seconds: float) -> dict[str, object]:
    """生成一组可重复参数，并测量朴素算法的一次 ``H^A``。"""

    # 每组使用固定 seed，保证多次运行时输入实例一致。
    jones = generate_jones_matrix(k, max_entry=1000, seed=10_000 + k)
    bases = generate_public_deformations(jones, n)
    private = random_circulant(n, s, seed=20_000 + n + s)
    counter = OperationCounter()
    # 计时范围只包含核心多重指数，不包含上面的参数生成过程。
    start = perf_counter()
    multi_exponentiation_naive(bases, private, counter=counter)
    elapsed = perf_counter() - start
    return {
        "k": k,
        "n": n,
        "s": s,
        "paper_seconds": paper_seconds,
        "local_seconds": elapsed,
        "matrix_multiplications": counter.tropical_matrix_multiplications,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按论文表 2 参数测试朴素 H^A 求值器。"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="仅运行论文中的第一组参数",
    )
    parser.add_argument("--csv", type=Path, help="可选：指定 CSV 结果保存路径")
    args = parser.parse_args()

    rows = PAPER_ROWS[:1] if args.quick else PAPER_ROWS
    results = []
    print(
        " k   n   s | 原论文时间(s) | 线性重复乘法时间(s)"
        " | 热带矩阵乘法次数"
    )
    print("说明：两列时间均对应一次 H^A 求值，不是完整协议耗时。")
    print("-" * 86)
    for row in rows:
        result = run_row(*row)
        results.append(result)
        print(
            f"{result['k']:2d}  {result['n']:2d}  {result['s']:2d} |"
            f" {result['paper_seconds']:13.3f} |"
            f" {result['local_seconds']:20.3f} |"
            f" {result['matrix_multiplications']:16d}"
        )

    # 指定 --csv 时，把相同结果另存为便于后续绘图的 CSV 文件。
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n已保存：{args.csv.resolve()}")


if __name__ == "__main__":
    main()
