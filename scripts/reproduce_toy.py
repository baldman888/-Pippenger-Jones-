"""完整复现论文第 4 节玩具示例，并使用 Pippenger 计算。"""

from __future__ import annotations

import sys
from pathlib import Path

# __file__ 是当前脚本路径，parents[1] 得到项目根目录 JonesAndPippenger/。
ROOT = Path(__file__).resolve().parents[1]
# 把项目根目录放到模块搜索路径最前面，确保可以导入 jones_crypto。
sys.path.insert(0, str(ROOT))

from jones_crypto.fixtures import (  
    paper_expected_alice_public,
    paper_toy_parameters,
)
from jones_crypto.multi_exponentiation import (  
    multi_exponentiation_naive,
    multi_exponentiation_pippenger,
)


def format_matrix(matrix) -> str:
    """把 TropicalMatrix 转成适合终端显示的多行字符串。"""

    rows = []
    for row in matrix.as_fractions():
        rows.append(
            "[" + ", ".join("-inf" if x is None else str(x) for x in row) + "]"
        )
    return "\n".join(rows)


def format_integer_matrix(matrix) -> str:
    """把 NumPy 整数矩阵转成与热带矩阵相近的终端显示形式。"""

    return "\n".join(
        "[" + ", ".join(str(int(value)) for value in row) + "]"
        for row in matrix
    )


def print_matrix_vector(
    section_number: int,
    symbol: str,
    title: str,
    matrices,
) -> None:
    """逐个打印矩阵向量的全部分量，例如 H[1]、H[2]、H[3]。"""

    print(f"\n{section_number}. {title}：")
    for index, matrix in enumerate(matrices, start=1):
        print(f"\n{symbol}[{index}]")
        print(format_matrix(matrix))


def main() -> None:
    """打印全部公开参数、私钥、中间结果、共享密钥和验证结论。"""

    # N、H、A、B 是论文玩具示例直接给出的固定数据。
    jones_matrix, bases, alice_private, bob_private = paper_toy_parameters()

    # 下面的 U、V、K_A、K_B 都由 Pippenger 算法实际计算得出。
    alice_public = multi_exponentiation_pippenger(bases, alice_private)
    bob_public = multi_exponentiation_pippenger(bases, bob_private)
    alice_shared = multi_exponentiation_pippenger(bob_public, alice_private)
    bob_shared = multi_exponentiation_pippenger(alice_public, bob_private)

    # 用原始朴素算法再算一次 U，验证优化只改变计算过程，没有改变结果。
    alice_public_naive = multi_exponentiation_naive(bases, alice_private)
    pippenger_matches_naive = all(
        optimized.equals(baseline)
        for optimized, baseline in zip(
            alice_public,
            alice_public_naive,
            strict=True,
        )
    )

    # 检查 Alice 公钥是否与论文第 4 节公布的数值完全相同。
    expected = paper_expected_alice_public()
    public_key_matches_paper = (
        [matrix.as_fractions() for matrix in alice_public] == expected
    )

    # 分量比较可以指出 K[1]、K[2]、K[3] 中是否存在不一致。
    component_matches = [
        alice_part.equals(bob_part)
        for alice_part, bob_part in zip(alice_shared, bob_shared, strict=True)
    ]
    shared_keys_match = all(component_matches)

    print("=" * 72)
    print("论文第 4 节玩具示例：Pippenger 完整复现")
    print("说明：N、H、A、B 来自论文；U、V、K_A、K_B 由程序计算。")
    print("=" * 72)

    print("\n1. Jones 矩阵 N：")
    print(format_matrix(jones_matrix))

    print_matrix_vector(2, "H", "公开矩阵向量 H", bases)

    print("\n3. Alice 私钥 A：")
    print(format_integer_matrix(alice_private))

    print("\n4. Bob 私钥 B：")
    print(format_integer_matrix(bob_private))

    print_matrix_vector(5, "U", "Alice 公钥 U = H^A（Pippenger 计算）", alice_public)
    print_matrix_vector(6, "V", "Bob 公钥 V = H^B（Pippenger 计算）", bob_public)
    print_matrix_vector(
        7,
        "K_A",
        "Alice 计算的共享密钥 K_A = V^A（Pippenger 计算）",
        alice_shared,
    )
    print_matrix_vector(
        8,
        "K_B",
        "Bob 计算的共享密钥 K_B = U^B（Pippenger 计算）",
        bob_shared,
    )

    print("\n9. 验证结果：")
    print(
        "  Pippenger 公钥 U 与原始朴素算法对比："
        + ("通过（一致）" if pippenger_matches_naive else "失败（不一致）")
    )
    print(
        "  Alice 公钥 U 与论文公布结果对比："
        + ("通过（一致）" if public_key_matches_paper else "失败（不一致）")
    )
    for index, matches in enumerate(component_matches, start=1):
        print(
            f"  共享密钥分量 K_A[{index}] 与 K_B[{index}]："
            + ("通过（一致）" if matches else "失败（不一致）")
        )
    print(
        "  Alice 与 Bob 的完整共享密钥对比："
        + ("通过（一致）" if shared_keys_match else "失败（不一致）")
    )

    overall_pass = (
        pippenger_matches_naive
        and public_key_matches_paper
        and shared_keys_match
    )
    print("\n" + "=" * 72)
    print(
        "最终结论："
        + (
            "通过，Pippenger 结果正确，论文示例复现成功"
            if overall_pass
            else "失败，复现未通过"
        )
    )
    print("=" * 72)

    # 打印完全部信息后再抛出异常，失败时仍能看到具体失败位置。
    if not pippenger_matches_naive:
        raise AssertionError("Pippenger 公钥与朴素算法结果不一致")
    if not public_key_matches_paper:
        raise AssertionError("Alice 公钥与论文公布结果不一致")
    if not shared_keys_match:
        raise AssertionError("Alice 和 Bob 得到的共享密钥不一致")


if __name__ == "__main__":
    main()
