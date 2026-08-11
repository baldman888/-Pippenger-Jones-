"""Protocol A 密钥交换和 Protocol B 加解密测试。"""

import unittest

import numpy as np

from jones_crypto.fixtures import (
    paper_expected_alice_public,
    paper_toy_parameters,
)
from jones_crypto.multi_exponentiation import multi_exponentiation_naive
from jones_crypto.protocol import decrypt, encrypt, key_exchange, keygen
from jones_crypto.tropical import TropicalMatrix


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        # 每个测试都复用论文第 4 节的固定 H、A、B。
        _, self.bases, self.alice, self.bob = paper_toy_parameters()

    def test_paper_alice_public_key(self):
        # 先用朴素算法核对论文公布的 Alice 公钥 U。
        public = multi_exponentiation_naive(self.bases, self.alice)
        self.assertEqual(
            [matrix.as_fractions() for matrix in public],
            paper_expected_alice_public(),
        )

    def test_key_exchange(self):
        # Protocol A 的核心要求是双方计算出的共享密钥逐分量一致。
        alice_shared, bob_shared = key_exchange(
            self.bases,
            self.alice,
            self.bob,
        )
        self.assertTrue(
            all(
                left.equals(right)
                for left, right in zip(alice_shared, bob_shared, strict=True)
            )
        )

    def test_encrypt_decrypt(self):
        # 构造三个简单消息矩阵，验证加密后能够完整恢复原文。
        alice_public = keygen(self.bases, self.alice)
        scale = self.bases[0].scale
        messages = [
            TropicalMatrix(np.full((3, 3), index + 1), scale)
            for index in range(3)
        ]
        ciphertext = encrypt(
            messages,
            self.bases,
            alice_public,
            self.bob,
        )
        recovered = decrypt(ciphertext, self.alice)
        self.assertTrue(
            all(
                plain.equals(decoded)
                for plain, decoded in zip(messages, recovered, strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
