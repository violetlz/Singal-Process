import numpy as np
from typing import Optional
from base_encoder import BaseEncoder

try:
    import cupy as cp
except Exception:
    cp = None


class RSCEncoder(BaseEncoder):
    """
    RSC 编码器（Recursive Systematic Convolutional）

    输出形式：
        system bits + parity bits

    构造参数:
        cfg    : 全局配置字典
        device : "cpu" 或 "gpu"
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)

        rsc_cfg = self.cfg["RSC"]

        # 生成多项式
        self.feedback = self.xp.asarray(
            rsc_cfg["feedback"], dtype=self.xp.uint8
        )
        self.parity = self.xp.asarray(
            rsc_cfg["parity"], dtype=self.xp.uint8
        )

        # 约束长度
        self.m = int(rsc_cfg["constraint_len"])

        # 是否零尾
        self.flush = bool(rsc_cfg.get("flush", True))

    # ==========================================================
    # BaseEncoder 接口
    # ==========================================================

    def _encode_impl(self, message):
        """
        BaseEncoder 统一接口

        参数:
            message : (batch, K) 0/1 比特

        返回:
            codeword : (batch, 2T)
        """

        sys_bits, parity_bits = self.encode_components(message)

        return self.xp.concatenate([sys_bits, parity_bits], axis=1)

    # ==========================================================
    # Turbo 专用接口
    # ==========================================================

    def encode_components(self, message):

        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, K = msg.shape

        reg = xp.zeros((batch, self.m), dtype=xp.uint8)

        sys_list = []
        parity_list = []

        # ======================================================
        # 主信息比特编码
        # ======================================================
        for t in range(K):
            # feedback 只使用 memory 单元
            fb = xp.sum(reg[:, 1:] * self.feedback[1:], axis=1) % 2

            u = msg[:, t]
            u_tilde = u ^ fb

            # shift
            reg[:, 1:] = reg[:, :-1]
            reg[:, 0] = u_tilde

            # parity 用完整寄存器
            p = xp.sum(reg * self.parity, axis=1) % 2

            sys_list.append(u)
            parity_list.append(p)

        # ======================================================
        # 正确 zero-tail 终止
        # ======================================================
        if self.flush:

            for _ in range(self.m - 1):
                # 关键：计算使 u_tilde=0 的输入
                fb = xp.sum(reg[:, 1:] * self.feedback[1:], axis=1) % 2
                u = fb  # 让 u_tilde = 0
                u_tilde = u ^ fb  # 必然为 0

                reg[:, 1:] = reg[:, :-1]
                reg[:, 0] = u_tilde

                p = xp.sum(reg * self.parity, axis=1) % 2

                sys_list.append(u)
                parity_list.append(p)

        sys_bits = xp.stack(sys_list, axis=1)
        parity_bits = xp.stack(parity_list, axis=1)

        return sys_bits, parity_bits