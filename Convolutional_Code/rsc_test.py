import numpy as np


# ==========================================================
# RSC 参数（必须与你 cfg["RSC"] 完全一致）
# ==========================================================

RSC_CFG = {
    "feedback":  [1, 1, 1],   # 例：1 + D
    "parity":    [1, 0, 1],   # 例：1 + D^2
    "constraint_len": 3,
    "flush": True,
}


# ==========================================================
# RSC Encoder（与你当前代码一致逻辑）
# ==========================================================

class RSCEncoder:

    def __init__(self, cfg):
        self.feedback = np.array(cfg["feedback"], dtype=np.uint8)
        self.parity   = np.array(cfg["parity"], dtype=np.uint8)
        self.m        = cfg["constraint_len"]
        self.flush    = cfg["flush"]

    def encode_components(self, message):

        msg = np.array(message, dtype=np.uint8)
        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, K = msg.shape

        reg = np.zeros((batch, self.m), dtype=np.uint8)

        sys_list = []
        parity_list = []

        for t in range(K):

            fb = np.sum(reg[:, 1:] * self.feedback[1:], axis=1) % 2

            u = msg[:, t]
            u_tilde = u ^ fb

            reg[:, 1:] = reg[:, :-1]
            reg[:, 0]  = u_tilde

            p = np.sum(reg * self.parity, axis=1) % 2

            sys_list.append(u)
            parity_list.append(p)

        if self.flush:
            for _ in range(self.m - 1):

                fb = np.sum(reg[:, 1:] * self.feedback[1:], axis=1) % 2
                u = fb
                u_tilde = u ^ fb

                reg[:, 1:] = reg[:, :-1]
                reg[:, 0]  = u_tilde

                p = np.sum(reg * self.parity, axis=1) % 2

                sys_list.append(u)
                parity_list.append(p)

        sys_bits = np.stack(sys_list, axis=1)
        parity_bits = np.stack(parity_list, axis=1)

        return sys_bits, parity_bits


# ==========================================================
# RSC Decoder（Max-Log-MAP）
# ==========================================================

class RSCDecoder:

    def __init__(self, cfg):

        self.feedback = np.array(cfg["feedback"], dtype=np.uint8)
        self.parity   = np.array(cfg["parity"], dtype=np.uint8)
        self.m        = cfg["constraint_len"]
        self.flush    = cfg["flush"]

        self.n_states = 2 ** (self.m - 1)

        self._build_trellis()

    def _build_trellis(self):

        self.next_state = np.zeros((self.n_states, 2), dtype=np.int32)
        self.out_parity = np.zeros((self.n_states, 2), dtype=np.uint8)

        for s in range(self.n_states):

            reg = [(s >> i) & 1 for i in range(self.m - 1)]
            reg = np.array(reg, dtype=np.uint8)

            for u in (0, 1):

                fb = np.sum(reg * self.feedback[1:]) % 2
                u_tilde = u ^ fb

                new_reg = np.zeros(self.m - 1, dtype=np.uint8)
                new_reg[0] = u_tilde
                new_reg[1:] = reg[:-1]

                full_reg = np.concatenate([[u_tilde],new_reg])
                p = np.sum(full_reg * self.parity) % 2

                ns = 0
                for i in range(self.m - 1):
                    ns |= (new_reg[i] << i)

                self.next_state[s, u] = ns
                self.out_parity[s, u] = p

    # =============================

    def decode_siso(self, L_sys, L_par, L_a):

        batch, T = L_sys.shape
        INF = 1e9

        alpha = np.full((batch, T + 1, self.n_states), -INF)
        beta  = np.full((batch, T + 1, self.n_states), -INF)

        alpha[:, 0, 0] = 0
        beta[:, T, 0]  = 0

        # Forward
        for t in range(T):
            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2*u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2*p) * L_par[:, t]
                    )

                    alpha[:, t+1, ns] = np.maximum(
                        alpha[:, t+1, ns],
                        alpha[:, t, s] + gamma
                    )
            alpha[:, t + 1, :] -= np.max(alpha[:, t + 1, :], axis=1, keepdims=True)

        # Backward
        for t in reversed(range(T)):
            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2*u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2*p) * L_par[:, t]
                    )

                    beta[:, t, s] = np.maximum(
                        beta[:, t, s],
                        beta[:, t+1, ns] + gamma
                    )
            beta[:, t, :] -= np.max(beta[:, t, :], axis=1, keepdims=True)

        # Extrinsic
        L_e = np.zeros((batch, T))

        for t in range(T):

            num = np.full(batch, -INF)
            den = np.full(batch, -INF)

            for s in range(self.n_states):
                for u in (0, 1):

                    ns = self.next_state[s, u]
                    p  = self.out_parity[s, u]

                    gamma = (
                        0.5 * (1 - 2*u) * (L_sys[:, t] + L_a[:, t])
                        + 0.5 * (1 - 2*p) * L_par[:, t]
                    )

                    val = alpha[:, t, s] + gamma + beta[:, t+1, ns]

                    if u == 1:
                        num = np.maximum(num, val)
                    else:
                        den = np.maximum(den, val)

            L_e[:, t] = num - den - L_sys[:, t] - L_a[:, t]

        return L_e


# ==========================================================
# 单码无噪声测试
# ==========================================================

def test_rsc():

    K = 50
    encoder = RSCEncoder(RSC_CFG)
    decoder = RSCDecoder(RSC_CFG)

    msg = np.random.randint(0, 2, (1, K))

    sys_bits, par_bits = encoder.encode_components(msg)

    # BPSK
    x_sys = 1 - 2*sys_bits
    x_par = 1 - 2*par_bits

    # 无噪声 LLR（大数）
    L_sys = 50 * x_sys
    L_par = 50 * x_par
    L_a   = np.zeros_like(L_sys)
    L_e = decoder.decode_siso(L_sys, L_par, L_a)
    L_total = L_sys + L_e

    decoded = (L_total < 0).astype(int)

    # 只比较前 K 位
    decoded_info = decoded[:, :K]
    errors = np.sum(decoded_info != msg)
    print("Errors:", errors)
    print("Errors:", np.sum(decoded[:, :K] != msg))


if __name__ == "__main__":
    test_rsc()