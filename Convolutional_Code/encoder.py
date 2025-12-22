from typing import Optional, List
import numpy as np
import config

try:
    import cupy as cp
except Exception:
    cp = None


class Encoder:
    """
    卷积码编码器（rate = 1/n）

    参数:
        cfg: 配置字典（优先于 config.CONFIG）
        device: 'gpu' 强制 GPU，'cpu' 强制 CPU，None 使用配置
    返回:
        无
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        # 使用传入配置或全局配置
        self.cfg = cfg if cfg is not None else config.CONFIG

        # device 决策
        if device == "cpu":
            self._use_gpu = False
        elif device == "gpu":
            self._use_gpu = (cp is not None)
        else:
            self._use_gpu = bool(self.cfg.get("GLOBAL", {}).get("use_gpu", True)) and (cp is not None)

        self.xp = cp if self._use_gpu else np


    def encode_conv(self, message):
        """
        卷积码编码（非系统码）

        参数:
            message: (batch, k) 二值数组（0/1）
        返回:
            codeword: (batch, n_out * T) 编码比特
        """

        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        # 保证 batch 维度
        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, T = msg.shape

        conv_cfg = self.cfg["CONV"]
        polynomials = conv_cfg["polynomials"]
        m = int(conv_cfg["constraint_len"])
        flush = bool(conv_cfg.get("flush", True))

        n_out = len(polynomials)

        # 强制回到 0 状态
        if flush:
            tail = xp.zeros((batch, m - 1), dtype=xp.uint8)
            msg = xp.concatenate([msg, tail], axis=1)
            T = msg.shape[1]

        # 移位寄存器（batch 并行）
        reg = xp.zeros((batch, m), dtype=xp.uint8)

        outputs = []

        for t in range(T):
            # 寄存器右移
            reg[:, 1:] = reg[:, :-1]
            reg[:, 0] = msg[:, t]

            # 多输出生成
            out_bits = []
            for g in polynomials:
                g = xp.asarray(g, dtype=xp.uint8)
                bit = xp.sum(reg * g, axis=1) % 2
                out_bits.append(bit)

            # (batch, n_out)
            outputs.append(xp.stack(out_bits, axis=1))

        # ---- 时间展开 ----
        code = xp.concatenate(outputs, axis=1)
        return code.astype(xp.uint8)


    def encode_conv_systematic(self, message):
        """
        系统卷积码编码

        参数:
            message: (batch, k) 二值数组
        返回:
            [systematic | parity] 拼接结果
        """

        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, k = msg.shape

        # 仅使用 parity 支路
        parity_full = self.encode_conv(msg)

        n_out = len(self.cfg["CONV"]["polynomials"])

        # 取第一个输出作为 parity（rate=1/2 情况）
        parity = parity_full.reshape(batch, -1, n_out)[:, :k, 0]

        return xp.concatenate([msg, parity], axis=1)
