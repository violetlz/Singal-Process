import numpy as np
import config
from rsc_decoder import RSCDecoder
from Interleaver import Interleaver
from base_decoder import BaseDecoder

class TurboDecoder(BaseDecoder):
    """
    修正后的 TurboDecoder（与 TurboEncoder 配合，支持 NumPy/CuPy）。
    """

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg, device)
        turbo = self.cfg["TURBO"]

        self.iterations = int(turbo.get("iterations", 6))
        self.seed = int(turbo["interleaver"].get("seed", 0))

        # SISO modules
        self.rsc1 = RSCDecoder(cfg, device)
        self.rsc2 = RSCDecoder(cfg, device)

        self.interleaver = None

    def _decode_impl(self, recv_llr):
        """
        参数:
            recv_llr: (batch, 3*T) ，T = K + tail
                [L_sys | L_p1 | L_p2]
        返回:
            decoded bits: (batch, K)
        """

        if recv_llr.ndim == 1:
            recv_llr = recv_llr.reshape(1, -1)

        xp = self.xp
        batch, L = recv_llr.shape

        # T = block length per RSC (systematic or parity), tail = m-1
        T = L // 3
        tail = int(self.rsc1.m - 1)
        K = T - tail
        if K <= 0:
            raise ValueError("Derived K <= 0: check constraint_len / input length")

        # split according to T
        L_sys = recv_llr[:, :T]
        L_p1  = recv_llr[:, T:2 * T]
        L_p2  = recv_llr[:, 2 * T: 3 * T]

        # interleaver is defined on K (information bits)
        if self.interleaver is None:
            self.interleaver = Interleaver(K, self.seed)

        # prepare xp-compatible index arrays for safe advanced indexing
        if xp is np:
            perm_xp = self.interleaver.perm
            inv_perm_xp = self.interleaver.inv_perm
        else:
            # cupy: convert indices to cupy arrays
            perm_xp = xp.asarray(self.interleaver.perm)
            inv_perm_xp = xp.asarray(self.interleaver.inv_perm)

        # initialize priors (must be length T)
        L_a1 = xp.zeros((batch, T), dtype=xp.float32)
        # L_a2 not strictly needed as variable, we feed L_e1_i directly to rsc2
        # but keep for clarity
        L_a2 = xp.zeros((batch, T), dtype=xp.float32)

        # iterations
        for _ in range(self.iterations):
            # RSC1: inputs are L_sys (T), L_p1 (T), L_a1 (T)
            L_e1 = self.rsc1.decode_siso(L_sys, L_p1, L_a1)   # (batch, T)

            # --------- permute extrinsic and systematic (only first K) ----------
            # take only info part
            L_e1_info = L_e1[:, :K]
            L_sys_info = L_sys[:, :K]

            # permute info parts
            # use xp.take to be safe on cupy
            L_e1_info_i = xp.take(L_e1_info, perm_xp, axis=1)
            L_sys_info_i = xp.take(L_sys_info, perm_xp, axis=1)

            # build full-length arrays for RSC2 (info_i + tail unchanged)
            # for extrinsic tail we put zeros (no extrinsic for tails)
            if tail > 0:
                zeros_tail = xp.zeros((batch, tail), dtype=xp.float32)
                L_e1_i = xp.concatenate([L_e1_info_i, zeros_tail], axis=1)
                L_sys_i = xp.concatenate([L_sys_info_i, L_sys[:, K:]], axis=1)
            else:
                L_e1_i = L_e1_info_i
                L_sys_i = L_sys_info_i

            # --------- RSC2 decode (use permuted systematic, p2, and L_e1_i as prior) ----------
            L_e2 = self.rsc2.decode_siso(L_sys_i, L_p2, L_e1_i)   # (batch, T)

            # --------- inverse-permute extrinsic of RSC2 (only first K) ----------
            L_e2_info = L_e2[:, :K]
            L_e2_info_inv = xp.take(L_e2_info, inv_perm_xp, axis=1)

            # build new L_a1 (prior for RSC1): first K from inv, tail zeros
            if tail > 0:
                L_a1 = xp.concatenate([L_e2_info_inv, xp.zeros((batch, tail), dtype=xp.float32)], axis=1)
            else:
                L_a1 = L_e2_info_inv

            # next iteration

        # final LLR for information bits (use RSC1 systematic positions)
        # choose to form total LLR for first K bits
        L_total = L_sys[:, :K] + L_a1[:, :K]
        decoded = (L_total < 0).astype(xp.uint8)

        return decoded