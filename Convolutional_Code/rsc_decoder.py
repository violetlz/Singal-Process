import numpy as np
from typing import Optional
from base_decoder import BaseDecoder

try:
    import cupy as cp
except Exception:
    cp = None


class RSCDecoder(BaseDecoder):
    """
    RSC 软输入软输出 (SISO) 解码器

    与 RSCEncoder 完全一致的状态更新逻辑

    构造参数:
        cfg    : 全局配置字典
        device : "cpu" 或 "gpu"
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)

        rsc_cfg = self.cfg["RSC"]

        # ===== 与 Encoder 参数完全一致 =====
        self.feedback = np.asarray(
            rsc_cfg["feedback"], dtype=np.uint8
        )
        self.parity = np.asarray(
            rsc_cfg["parity"], dtype=np.uint8
        )

        self.m = int(rsc_cfg["constraint_len"])
        self.flush = bool(rsc_cfg.get("flush", True))

        # trellis 状态数（与 m 对应）
        self.n_states = 2 ** (self.m - 1)

        self._build_trellis()

    # ==========================================================
    # Trellis 构建（严格匹配 Encoder）
    # ==========================================================

    def _build_trellis(self):
        """
        构建状态转移表

        与 RSCEncoder.encode_components 中：
            fb
            u_tilde
            shift
            parity
        逻辑完全一致
        """

        self.next_state = np.zeros(
            (self.n_states, 2), dtype=np.int32
        )
        self.out_parity = np.zeros(
            (self.n_states, 2), dtype=np.uint8
        )

        for s in range(self.n_states):

            # ============================
            # 状态 → memory 寄存器内容
            # ============================
            reg = [(s >> i) & 1 for i in range(self.m - 1)]
            reg = np.array(reg, dtype=np.uint8)

            for u in (0, 1):

                # ==========================================
                # 1️⃣ feedback (只使用 memory 单元)
                # 对应 encoder:
                # fb = sum(reg[:,1:] * feedback[1:])
                # 这里 reg 已经是 memory，不含当前输入
                # ==========================================
                fb = np.sum(reg * self.feedback[1:]) % 2

                # ==========================================
                # 2️⃣ u_tilde = u XOR fb
                # ==========================================
                u_tilde = u ^ fb

                # ==========================================
                # 3️⃣ shift
                # 对应 encoder:
                # reg[:,1:] = reg[:,:-1]
                # reg[:,0] = u_tilde
                # ==========================================
                new_reg = np.zeros(self.m - 1, dtype=np.uint8)
                new_reg[0] = u_tilde
                new_reg[1:] = reg[:-1]

                # ==========================================
                # 4️⃣ parity
                # encoder 中 parity 使用完整寄存器
                # full_reg = [u_tilde, reg]
                # ==========================================
                full_reg = np.concatenate([[u_tilde], reg])
                p = np.sum(full_reg * self.parity) % 2

                # ==========================================
                # 5️⃣ next state
                # ==========================================
                ns = 0
                for i in range(self.m - 1):
                    ns |= (new_reg[i] << i)

                self.next_state[s, u] = ns
                self.out_parity[s, u] = p

        # GPU 同步
        if self._use_gpu:
            self.next_state = cp.asarray(self.next_state)
            self.out_parity = cp.asarray(self.out_parity)

    # ==========================================================
    # BaseDecoder 接口
    # ==========================================================

    def _decode_impl(self, recv):
        raise NotImplementedError(
            "RSCDecoder 是 SISO 模块，请使用 decode_siso()"
        )

    # ==========================================================
    # Turbo 专用 SISO
    # ==========================================================

    def decode_siso(self, L_sys, L_par, L_a):
        """
        Max-Log-MAP SISO 解码

        参数:
            L_sys : (batch, T) 系统比特 LLR
            L_par : (batch, T) 校验比特 LLR
            L_a   : (batch, T) 先验 LLR

        返回:
            L_e   : (batch, T) extrinsic LLR
        """

        xp = self.xp

        L_sys = xp.asarray(L_sys, dtype=xp.float32)
        L_par = xp.asarray(L_par, dtype=xp.float32)
        L_a   = xp.asarray(L_a,   dtype=xp.float32)

        batch, T = L_sys.shape
        INF = 1e9

        alpha = xp.full(
            (batch, T + 1, self.n_states),
            -INF,
            dtype=xp.float32
        )
        beta = xp.full(
            (batch, T + 1, self.n_states),
            -INF,
            dtype=xp.float32
        )

        # =============================
        # 初始状态
        # =============================
        alpha[:, 0, 0] = 0.0
        beta[:, T, 0]  = 0.0 if self.flush else 0.0

        # =============================
        # Forward recursion
        # =============================
        for t in range(T):
            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2 * u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2 * p) * L_par[:, t]
                    )

                    alpha[:, t + 1, ns] = xp.maximum(
                        alpha[:, t + 1, ns],
                        alpha[:, t, s] + gamma
                    )

        # =============================
        # Backward recursion
        # =============================
        for t in reversed(range(T)):
            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2 * u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2 * p) * L_par[:, t]
                    )

                    beta[:, t, s] = xp.maximum(
                        beta[:, t, s],
                        beta[:, t + 1, ns] + gamma
                    )

        # =============================
        # Extrinsic LLR
        # =============================
        L_e = xp.zeros((batch, T), dtype=xp.float32)

        for t in range(T):
            num = -INF
            den = -INF

            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2 * u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2 * p) * L_par[:, t]
                    )

                    val = alpha[:, t, s] + gamma + beta[:, t + 1, ns]

                    if u == 1:
                        num = xp.maximum(num, val)
                    else:
                        den = xp.maximum(den, val)

            L_e[:, t] = num - den - L_sys[:, t] - L_a[:, t]

        return L_e