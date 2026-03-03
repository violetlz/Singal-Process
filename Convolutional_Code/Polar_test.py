"""
Polar 完整测试脚本

流程：
1. 构造 Encoder
2. 构造 Decoder
3. 随机生成信息比特
4. 编码
5. BPSK 调制
6. AWGN 信道
7. 计算 LLR
8. SC 解码
9. 计算 BER
"""

import numpy as np
import config

from Polar_encoder import PolarEncoder
from Polar_decoder import PolarDecoder


# =========================================================
# 工具函数（xp 统一）
# =========================================================

def bpsk_modulation(bits, xp):
    """
    BPSK 调制
    0 -> +1
    1 -> -1
    """
    return 1 - 2 * bits.astype(xp.float32)


def awgn_channel(x, snr_db, rate, xp):
    """
    AWGN 信道

    参数：
        x: (batch, N)
        snr_db: Eb/N0 (dB)
        rate: K/N
    返回：
        y
        sigma2
    """
    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1 / (2 * rate * snr_linear)
    sigma = np.sqrt(sigma2)

    noise = sigma * xp.random.randn(*x.shape).astype(xp.float32)

    return x + noise, sigma2


def compute_llr(y, sigma2):
    """
    AWGN 下 BPSK LLR
    """
    return 2 * y / sigma2


# =========================================================
# 主测试函数
# =========================================================

def run_polar_test():

    cfg = config.CONFIG

    # -------------------------------------------------
    # 1. 构造 Encoder / Decoder
    # -------------------------------------------------
    encoder = PolarEncoder(cfg)
    decoder = PolarDecoder(cfg)

    xp = encoder.xp

    backend = "CuPy (GPU)" if encoder._use_gpu else "NumPy (CPU)"

    # -------------------------------------------------
    # 2. 基本参数
    # -------------------------------------------------
    K = encoder.info_length
    N = encoder.code_length
    rate = K / N

    batch_size = 500
    snr_db = 2.0

    print(f"Polar Test  N={N}  K={K}  Rate={rate:.3f}")
    print(f"SNR = {snr_db} dB")
    print(f"Backend: {backend}")

    # -------------------------------------------------
    # 3. 生成随机信息比特
    # -------------------------------------------------
    msg = xp.random.randint(0, 2, (batch_size, K), dtype=xp.uint8)

    # -------------------------------------------------
    # 4. 编码
    # -------------------------------------------------
    code = encoder.encode(msg)

    # -------------------------------------------------
    # 5. BPSK
    # -------------------------------------------------
    tx = bpsk_modulation(code, xp)

    # -------------------------------------------------
    # 6. AWGN
    # -------------------------------------------------
    rx, sigma2 = awgn_channel(tx, snr_db, rate, xp)

    # -------------------------------------------------
    # 7. LLR
    # -------------------------------------------------
    llr = compute_llr(rx, sigma2)

    # -------------------------------------------------
    # 8. 解码
    # -------------------------------------------------
    decoded = decoder.decode(llr)

    # -------------------------------------------------
    # 9. BER（只统计信息位）
    # -------------------------------------------------
    bit_errors = xp.sum(decoded != msg)
    total_bits = batch_size * K

    if encoder._use_gpu:
        bit_errors = int(bit_errors.get())

    ber = bit_errors / total_bits

    print(f"Bit Errors: {bit_errors}")
    print(f"BER: {ber:.6e}")


# =========================================================
# 运行
# =========================================================

if __name__ == "__main__":
    run_polar_test()