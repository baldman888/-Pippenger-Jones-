"""实现 Protocol A 和 Protocol B，默认使用 Pippenger 求值器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import OperationCounter
from .multi_exponentiation import multi_exponentiation_pippenger
from .tropical import TropicalMatrix

# Action 统一表示 H^A 的求值算法，类似 C 语言中的函数指针类型。
# 协议公式保持不变，只需替换该函数即可切换朴素算法或 Pippenger。
Action = Callable[..., list[TropicalMatrix]]


@dataclass(frozen=True)
class Ciphertext:
    """Protocol B 的密文二元组 ``(V, Q)``。"""

    # V = H^B，是本次加密生成的一次性公钥。
    ephemeral_public: list[TropicalMatrix]
    # Q = M + U^B，是使用共享掩码处理后的消息。
    masked_message: list[TropicalMatrix]


def keygen(
    public_bases: Sequence[TropicalMatrix],
    private_key: np.ndarray,
    *,
    action: Action = multi_exponentiation_pippenger,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """计算公钥 ``U = H^A``。"""

    # action 的默认值是 Pippenger，也可以由调用者显式传入其他求值器。
    return action(public_bases, private_key, counter=counter)


def key_exchange(
    public_bases: Sequence[TropicalMatrix],
    alice_private: np.ndarray,
    bob_private: np.ndarray,
    *,
    action: Action = multi_exponentiation_pippenger,
) -> tuple[list[TropicalMatrix], list[TropicalMatrix]]:
    """执行 Protocol A，返回 Alice 和 Bob 各自得到的共享密钥。"""

    # 双方先利用各自私钥计算并交换公钥 U=H^A、V=H^B。
    alice_public = keygen(public_bases, alice_private, action=action)
    bob_public = keygen(public_bases, bob_private, action=action)
    # Alice 计算 V^A，Bob 计算 U^B；协议结构保证两者相同。
    alice_shared = action(bob_public, alice_private)
    bob_shared = action(alice_public, bob_private)
    return alice_shared, bob_shared


def encrypt(
    message: Sequence[TropicalMatrix],
    public_bases: Sequence[TropicalMatrix],
    recipient_public: Sequence[TropicalMatrix],
    ephemeral_private: np.ndarray,
    *,
    action: Action = multi_exponentiation_pippenger,
    counter: OperationCounter | None = None,
) -> Ciphertext:
    """使用掩码 ``U^B`` 对消息逐元素普通相加，生成密文。"""

    # 使用一次性私钥 B 生成密文第一部分 V=H^B。
    ephemeral_public = action(
        public_bases,
        ephemeral_private,
        counter=counter,
    )
    # 再用接收者公钥 U 和同一个 B 计算共享掩码 U^B。
    mask = action(recipient_public, ephemeral_private, counter=counter)
    if len(message) != len(mask):
        raise ValueError("消息向量与掩码向量的长度不一致")
    # 这里是逐元素普通加法，不是热带加法；消息和掩码必须都是有限矩阵。
    masked = [
        plain.ordinary_add(mask_part)
        for plain, mask_part in zip(message, mask, strict=True)
    ]
    return Ciphertext(ephemeral_public, masked)


def decrypt(
    ciphertext: Ciphertext,
    private_key: np.ndarray,
    *,
    action: Action = multi_exponentiation_pippenger,
    counter: OperationCounter | None = None,
) -> list[TropicalMatrix]:
    """重新计算 ``V^A``，再用逐元素普通减法恢复消息。"""

    # 接收者使用长期私钥 A 和密文中的 V 重新得到相同掩码。
    mask = action(
        ciphertext.ephemeral_public,
        private_key,
        counter=counter,
    )
    # 因为 Q=M+U^B 且 V^A=U^B，所以执行 Q-V^A 即可恢复 M。
    return [
        masked.ordinary_subtract(mask_part)
        for masked, mask_part in zip(ciphertext.masked_message, mask, strict=True)
    ]
