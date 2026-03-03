from typing import Optional
import numpy as np
from base_encoder import BaseEncoder

try:
    import cupy as cp
except Exception:
    cp = None


class PolarEncoder(BaseEncoder):
    """
    Polar 编码器（支持 NumPy / CuPy 自动切换）

    输入:
        message: (batch, K)

    输出:
        codeword: (batch, N)
    """

    # =========================================================
    # 初始化
    # =========================================================

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)

        polar_cfg = self.cfg["POLAR"]

        self.N = int(polar_cfg["N"])
        self.K = int(polar_cfg["K"])

        if (self.N & (self.N - 1)) != 0:
            raise ValueError("Polar N must be power of 2")

        frozen_np = self._build_frozen_mask_numpy()
        self.frozen_mask = self.xp.asarray(frozen_np, dtype=self.xp.bool_)

        # 预计算 bit-reversal 索引
        self.bitrev_indices = self._build_bit_reversal_indices()

    # =========================================================
    # Frozen mask
    # =========================================================

    def _build_frozen_mask_numpy(self) -> np.ndarray:
        """
        使用 3GPP 可靠性序列（适用于 N=128）
        """

        if self.N != 128:
            raise ValueError("当前示例仅支持 N=128")

        reliability_sequence = [
            0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 12, 33, 65, 20,
            24, 34, 36, 7, 66, 11, 40, 68, 72, 19, 13, 48, 14, 21, 80, 22, 96,
            35, 26, 37, 25, 38, 41, 42, 44, 69, 28, 73, 15, 49, 74, 81, 50, 82,
            84, 23, 52, 97, 27, 98, 56, 29, 30, 100, 88, 39, 45, 67, 104, 70,
            46, 76, 83, 90, 51, 58, 99, 86, 60, 101, 89, 92, 47, 105, 71, 108,
            53, 54, 77, 78, 106, 91, 57, 102, 94, 59, 110, 61, 111, 79, 87,
            112, 113, 62, 95, 103, 109, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127
        ]

        reliability_sequence = [i for i in reliability_sequence if i < self.N]
        info_positions = reliability_sequence[:self.K]

        frozen = np.ones(self.N, dtype=bool)
        frozen[info_positions] = False

        return frozen

    # =========================================================
    # 抽象方法实现
    def _encode_impl(self, message):

        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, k = msg.shape
        if k != self.K:
            raise ValueError(f"message width {k} != K ({self.K})")

        u = xp.zeros((batch, self.N), dtype=xp.uint8)

        info_positions = xp.where(~self.frozen_mask)[0]
        u[:, info_positions] = msg

        x = self._polar_transform(u)

        x = x[:, self.bitrev_indices]

        return x.astype(xp.uint8)

    def _polar_transform(self, u):

        xp = self.xp
        x = u.copy()

        n = int(np.log2(self.N))

        for stage in reversed(range(n)):
            step = 2 ** (stage + 1)
            half = step // 2

            for i in range(0, self.N, step):
                left = x[:, i:i + half].copy()
                right = x[:, i + half:i + step].copy()

                x[:, i:i + half] = left ^ right
                x[:, i + half:i + step] = right

        return x
    # =========================================================
    # Bit Reversal
    # =========================================================

    def _build_bit_reversal_indices(self):

        xp = self.xp
        n = int(np.log2(self.N))

        indices = xp.arange(self.N)
        rev = xp.zeros_like(indices)

        for i in range(n):
            rev |= ((indices >> i) & 1) << (n - 1 - i)

        return rev

    # =========================================================
    # 接口属性
    # =========================================================

    @property
    def info_length(self):
        return self.K

    @property
    def code_length(self):
        return self.N