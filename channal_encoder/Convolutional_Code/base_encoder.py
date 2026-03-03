from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import config

try:
    import cupy as cp
except Exception:
    cp = None


class BaseEncoder(ABC):
    """
    Encoder 基类

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

    def encode(self, message):
        """
        编码统一入口

        参数:
            message: (batch, K) binary bits
        返回:
            codeword: (batch, N)
        """
        message = self._check_input(message)
        codeword = self._encode_impl(message)
        return self._postprocess(codeword)

    @abstractmethod
    def _encode_impl(self, message):
        """
        具体编码算法（由子类实现）
        """
        pass

#输入检查
    def _check_input(self, message):

        xp = self.xp
        message = xp.asarray(message, dtype=xp.uint8)

        if message.ndim == 1:
            message = message.reshape(1, -1)

        return message

#输入后处理
    def _postprocess(self, codeword):

        return codeword

    @property
#信息比特长度K
    def info_length(self):

        return None

    @property
#编码后长度N
    def code_length(self):

        return None
