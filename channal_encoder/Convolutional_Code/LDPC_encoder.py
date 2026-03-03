import numpy as np
from base_encoder import BaseEncoder


class LDPCEncoder(BaseEncoder):
    """
    LDPC 编码器

    参数:
        cfg: 全局 CONFIG
    """

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg, device)

        # 读取 LDPC 参数
        ldpc_code_cfg = self.cfg["LDPC"]["code"]

        self.N = ldpc_code_cfg["N"]
        self.K = ldpc_code_cfg["K"]
        self.M = self.N - self.K

        xp = self.xp

    # 构造系统型 H = [P | I]
        density = 0.1

        P = (xp.random.rand(self.M, self.K) < density).astype(xp.uint8)

        # 防止全零行
        for i in range(self.M):
            if xp.sum(P[i]) == 0:
                j = xp.random.randint(0, self.K)
                P[i, j] = 1

        I = xp.eye(self.M, dtype=xp.uint8)

        self.P = P
        self.H = xp.concatenate([P, I], axis=1)

        # 生成矩阵 G = [I | P^T]
        self.G = xp.concatenate(
            [xp.eye(self.K, dtype=xp.uint8), P.T],
            axis=1
        )


    @property
    def info_length(self):
        return self.K

    @property
    def code_length(self):
        return self.N

#编码
    def _encode_impl(self, message):
        """
        LDPC 编码

        参数:
            message: (batch, K)
        返回:
            codeword: (batch, N)
        """

        xp = self.xp

        parity = (message @ self.P.T) % 2

        codeword = xp.concatenate([message, parity], axis=1)

        return codeword.astype(xp.uint8)