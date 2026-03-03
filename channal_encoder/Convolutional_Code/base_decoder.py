from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import config

try:
    import cupy as cp
except Exception:
    cp = None


class BaseDecoder(ABC):
    """
    Decoder 基类

    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        # 读取配置
        self.cfg = cfg if cfg is not None else config.CONFIG

        # device 决策
        if device == "cpu":
            self._use_gpu = False
        elif device == "gpu":
            self._use_gpu = (cp is not None)
        else:
            self._use_gpu = bool(
                self.cfg.get("GLOBAL", {}).get("use_gpu", True)
            ) and (cp is not None)

        # numpy / cupy 统一接口
        self.xp = cp if self._use_gpu else np

    def decode(self, recv):
        """
        解码统一入口

        参数:
            recv: channel output (LLR / hard bits / soft bits)
        返回:
            decoded message bits
        """
        recv = self._check_input(recv)
        decoded = self._decode_impl(recv)
        return self._postprocess(decoded)

    @abstractmethod
    def _decode_impl(self, recv):
        """
        具体解码算法（由子类实现）
        """
        pass


# 输入检查
    def _check_input(self, recv):
        return recv


# 输出后处理
    def _postprocess(self, decoded):
        return decoded
