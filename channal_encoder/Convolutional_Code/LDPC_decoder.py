from typing import Optional
import numpy as np
import config
from base_decoder import BaseDecoder

try:
    import cupy as cp
except Exception:
    cp = None


class LDPCDecoder(BaseDecoder):
    """
    LDPC 解码器

    功能：
        1. 读取与 Encoder 相同的 H 矩阵
        2. 构建 Tanner 图
        3. Min-Sum 迭代译码

    输入：
        LLR (batch, N) —— xp.ndarray

    输出：
        decoded bits (batch, N) —— xp.ndarray
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)

        ldpc_cfg = self.cfg["LDPC"]

        self.code_cfg = ldpc_cfg["code"]
        self.H_cfg = ldpc_cfg["H_matrix"]
        self.dec_cfg = ldpc_cfg["decoder"]

        # 基本参数
        self.N = int(self.code_cfg["N"])
        self.K = int(self.code_cfg["K"])
        self.M = self.N - self.K

        # 解码参数
        self.algorithm = self.dec_cfg.get("algorithm", "min_sum")
        self.max_iter = int(self.dec_cfg.get("max_iter", 50))
        self.early_stop = bool(self.dec_cfg.get("early_stop", True))
        self.norm_factor = float(self.dec_cfg.get("norm_factor", 1.0))
        self.llr_clip = self.dec_cfg.get("llr_clip", None)

        # 读取 H（固定使用 numpy 存储）
        self.H = self._load_H_matrix(self.H_cfg)

        if self.H.shape != (self.M, self.N):
            raise ValueError(f"H shape {self.H.shape} != ({self.M},{self.N})")

        self.H_T = self.H.T.astype(np.int8)

        self._build_graph()

# H读取
    def _load_H_matrix(self, H_cfg):

        if H_cfg.get("type") != "from_file":
            raise ValueError(
                "LDPCDecoder 必须使用 from_file 方式加载 H"
            )

        return np.load(H_cfg["path"]).astype(np.int8)

# 构建 Tanner 图
    def _build_graph(self):

        H = self.H

        self.check_nodes = []
        self.var_nodes = [[] for _ in range(self.N)]

        for c in range(self.M):
            vars_c = np.where(H[c] == 1)[0].tolist()
            self.check_nodes.append(vars_c)

            for v in vars_c:
                self.var_nodes[v].append(c)

#译码
    def _decode_impl(self, llr):

        xp = self.xp

        # 保证是 xp array
        if not isinstance(llr, xp.ndarray):
            llr = xp.asarray(llr)

        llr = llr.astype(xp.float32)

        if llr.ndim == 1:
            llr = llr.reshape(1, -1)

        batch, N = llr.shape
        if N != self.N:
            raise ValueError(f"LLR width {N} != N ({self.N})")

        if self.llr_clip is not None:
            llr = xp.clip(llr, -self.llr_clip, self.llr_clip)

        # 消息初始化
        q = {}
        r = {}

        for c in range(self.M):
            for v in self.check_nodes[c]:
                q[(c, v)] = llr[:, v].copy()
                r[(c, v)] = xp.zeros(batch, dtype=xp.float32)

        for _ in range(self.max_iter):

            # Check Node 更新
            for c in range(self.M):
                vars_c = self.check_nodes[c]

                for v in vars_c:

                    msgs = [q[(c, vp)] for vp in vars_c if vp != v]
                    msgs = xp.stack(msgs, axis=1)

                    sign = xp.prod(xp.sign(msgs), axis=1)
                    min_abs = xp.min(xp.abs(msgs), axis=1)

                    r[(c, v)] = self.norm_factor * sign * min_abs

            # Variable Node 更新
            for v in range(self.N):
                checks_v = self.var_nodes[v]

                for c in checks_v:
                    total = llr[:, v]

                    for cp_ in checks_v:
                        if cp_ != c:
                            total = total + r[(cp_, v)]

                    q[(c, v)] = total

            # 后验 LLR
            llr_post = llr.copy()

            for v in range(self.N):
                for c in self.var_nodes[v]:
                    llr_post[:, v] += r[(c, v)]

            hard = (llr_post < 0).astype(xp.int8)

            # syndrome 检查
            syndrome = (hard @ xp.asarray(self.H_T)) % 2

            if self.early_stop and xp.all(syndrome == 0):
                break

        return hard

    @property
    def info_length(self):
        return self.K

    @property
    def code_length(self):
        return self.N