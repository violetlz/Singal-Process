from typing import Optional
import numpy as np
import config
from base_encoder import BaseEncoder

try:
    import cupy as cp
except Exception:
    cp = None

class ConvEncoder(BaseEncoder):
    """
    卷积码编码器（rate = 1/n）

    统一入口:
        encode(message) -> codeword
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)
        self.conv_cfg = self.cfg["CONV"]

    def _encode_impl(self, message):
        """
        根据配置选择系统码 / 非系统码
        """
        if self.conv_cfg.get("systematic", False):
            return self._encode_systematic(message)
        else:
            return self._encode_nonsystematic(message)


    def _encode_nonsystematic(self, message):
        """
        非系统卷积码编码
        """
        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, T = msg.shape

        polynomials = self.conv_cfg["polynomials"]
        m = int(self.conv_cfg["constraint_len"])
        flush = bool(self.conv_cfg.get("flush", True))

        n_out = len(polynomials)

        if flush:
            tail = xp.zeros((batch, m - 1), dtype=xp.uint8)
            msg = xp.concatenate([msg, tail], axis=1)
            T = msg.shape[1]

        reg = xp.zeros((batch, m), dtype=xp.uint8)
        outputs = []

        for t in range(T):
            reg[:, 1:] = reg[:, :-1]
            reg[:, 0] = msg[:, t]

            out_bits = []
            for g in polynomials:
                g = xp.asarray(g, dtype=xp.uint8)
                bit = xp.sum(reg * g, axis=1) % 2
                out_bits.append(bit)

            outputs.append(xp.stack(out_bits, axis=1))

        code = xp.concatenate(outputs, axis=1)
        return code.astype(xp.uint8)

    def _encode_systematic(self, message):
        """
        系统卷积码编码
        """
        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, k = msg.shape

        parity_full = self._encode_nonsystematic(msg)
        n_out = len(self.conv_cfg["polynomials"])

        parity = parity_full.reshape(batch, -1, n_out)[:, :k, 0]

        return xp.concatenate([msg, parity], axis=1)

    @property
    def info_length(self):
        return self.conv_cfg.get("K", None)

    @property
    def code_length(self):
        return self.conv_cfg.get("N", None)
