import numpy as np
import config
from base_decoder import BaseDecoder
try:
    import cupy as cp
except Exception:
    cp = None


class ConvDecoder(BaseDecoder):
    """
    卷积码 Viterbi 解码器（硬判决）

    参数:
        cfg: 配置字典
        device: 'gpu' / 'cpu'
    返回:
        无
    """

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg, device)

        # 构建 trellis（CPU）
        self._build_trellis()

        # trellis 同步到 GPU
        if self._use_gpu:
            self.next_state_xp = cp.asarray(self.next_state, dtype=cp.int32)
            self.out_bits_xp = cp.asarray(self.out_bits, dtype=cp.int8)
        else:
            self.next_state_xp = self.next_state
            self.out_bits_xp = self.out_bits


    # Trellis 构建（CPU）

    def _build_trellis(self):
        conv_cfg = self.cfg["CONV"]
        self.polynomials = conv_cfg["polynomials"]
        self.m = int(conv_cfg["constraint_len"])
        self.n_out = len(self.polynomials)

        self.n_states = 2 ** (self.m - 1)

        self.next_state = np.zeros((self.n_states, 2), dtype=np.int32)
        self.out_bits = np.zeros((self.n_states, 2, self.n_out), dtype=np.int8)

        for state in range(self.n_states):
            for u in (0, 1):
                reg = [(state >> i) & 1 for i in range(self.m - 1)]
                reg = [u] + reg

                out = []
                for g in self.polynomials:
                    out.append(sum(gi * ri for gi, ri in zip(g, reg)) % 2)

                next_s = ((state << 1) | u) & (self.n_states - 1)

                self.next_state[state, u] = next_s
                self.out_bits[state, u] = out

    # Viterbi 解码

    def _decode_impl(self, recv):
        """
        Viterbi 解码（硬判决）

        参数:
            recv: (batch, T * n_out) 接收比特
        返回:
            decoded: (batch, T) 解码比特
        """

        xp = self.xp
        recv = xp.asarray(recv, dtype=xp.int32)

        if recv.ndim == 1:
            recv = recv.reshape(1, -1)

        batch, L = recv.shape
        T = L // self.n_out
        recv = recv.reshape(batch, T, self.n_out)

        # 路径度量
        INF = 1e9
        pm = xp.full((batch, self.n_states), INF, dtype=xp.float32)
        pm[:, 0] = 0.0

        # traceback 缓冲（CPU）
        prev_state = np.zeros((batch, T, self.n_states), dtype=np.int16)
        prev_input = np.zeros((batch, T, self.n_states), dtype=np.int8)

        # 前向递推
        for t in range(T):
            pm_new = xp.full_like(pm, INF)
            r = recv[:, t, :]

            for s in range(self.n_states):
                for u in (0, 1):
                    ns = self.next_state_xp[s, u]
                    expected = self.out_bits_xp[s, u]

                    # 汉明距离
                    bm = xp.sum(xp.abs(r - expected), axis=1)
                    cand = pm[:, s] + bm

                    better = cand < pm_new[:, ns]
                    pm_new[:, ns] = xp.where(better, cand, pm_new[:, ns])

                    # traceback（CPU）
                    better_cpu = better.get() if self._use_gpu else better
                    ns_cpu = int(ns.get()) if self._use_gpu else int(ns)

                    prev_state[:, t, ns_cpu] = np.where(better_cpu, s, prev_state[:, t, ns_cpu])
                    prev_input[:, t, ns_cpu] = np.where(better_cpu, u, prev_input[:, t, ns_cpu])

            pm = pm_new

        # 回溯
        pm_cpu = pm.get() if self._use_gpu else pm
        best_state = pm_cpu.argmin(axis=1)

        decoded = xp.zeros((batch, T), dtype=xp.uint8)

        for b in range(batch):
            s = best_state[b]
            for t in reversed(range(T)):
                decoded[b, t] = prev_input[b, t, s]
                s = prev_state[b, t, s]

        return decoded
