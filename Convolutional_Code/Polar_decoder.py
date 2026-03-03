from typing import Optional
import numpy as np
from base_decoder import BaseDecoder

try:
    import cupy as cp
except Exception:
    cp = None


class PolarDecoder(BaseDecoder):
    """
    Polar SC 解码器（支持 NumPy / CuPy 后端）
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

        self.bitrev_indices = self._build_bit_reversal_indices()
    # =========================================================
    # Frozen mask
    # =========================================================
    def _build_bit_reversal_indices(self):

        xp = self.xp
        n = int(np.log2(self.N))

        indices = xp.arange(self.N)
        rev = xp.zeros_like(indices)

        for i in range(n):
            rev |= ((indices >> i) & 1) << (n - 1 - i)

        return rev
    def _build_frozen_mask_numpy(self) -> np.ndarray:
        """
        使用预定义可靠性序列（适用于 N=128）
        """

        if self.N != 128:
            raise ValueError("当前示例仅支持 N=128")

        # 3GPP 可靠性序列（前128个）
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

        # 只取 N 范围
        reliability_sequence = [i for i in reliability_sequence if i < self.N]

        # 选最可靠的 K 个作为信息位
        info_positions = reliability_sequence[:self.K]

        frozen = np.ones(self.N, dtype=bool)
        frozen[info_positions] = False

        return frozen
    # =========================================================
    # Decode
    # =========================================================
    def _decode_impl(self, llr):

        xp = self.xp
        llr = xp.asarray(llr, dtype=xp.float32)

        if llr.ndim == 1:
            llr = llr.reshape(1, -1)

        batch, N = llr.shape
        llr = llr[:, self.bitrev_indices]
        u_hat = xp.zeros((batch, N), dtype=xp.uint8)

        for b in range(batch):
            self._sc_decode(
                llr[b:b + 1],
                u_hat[b:b + 1],
                start=0
            )

        info_positions = xp.where(~self.frozen_mask)[0]

        return u_hat[:, info_positions]
    # =========================================================
    # SC 递归
    # =========================================================
    def _sc_decode(self, llr, u_hat, start):

        xp = self.xp
        length = llr.shape[1]

        if length == 1:
            if self.frozen_mask[start]:
                u_hat[:, start] = 0
            else:
                u_hat[:, start] = (llr[:, 0] < 0).astype(xp.uint8)
            return

        half = length // 2

        # 左
        llr_left = self._f(llr[:, :half], llr[:, half:])
        self._sc_decode(llr_left, u_hat, start)

        # 右
        u_left = u_hat[:, start:start + half]
        llr_right = self._g(llr[:, :half], llr[:, half:], u_left)
        self._sc_decode(llr_right, u_hat, start + half)

        u_hat[:, start:start + half] ^= \
            u_hat[:, start + half:start + length]

    def _f(self, a, b):
        xp = self.xp
        return xp.sign(a) * xp.sign(b) * xp.minimum(xp.abs(a), xp.abs(b))

    def _g(self, a, b, u):
        xp = self.xp
        return b + (1 - 2 * u) * a

    # =========================================================
    # 接口属性
    # =========================================================

    @property
    def info_length(self):
        return self.K

    @property
    def code_length(self):
        return self.N