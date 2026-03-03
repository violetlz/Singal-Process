import numpy as np
import config
from factory import build_encoder
from factory import build_decoder


def turbo_test():

    cfg = config.CONFIG

    # =========================================================
    # 手动设置信息长度
    # =========================================================

    K = 128                 # 你可以改
    N = 3 * K               # Turbo 结构
    snr_db = 2.0
    batch = 100

    encoder = build_encoder("turbo", cfg=cfg)
    decoder = build_decoder("turbo", cfg=cfg)

    xp = encoder.xp

    print("=" * 60)
    print(f"Turbo Test  K={K}  N={N}  Rate={K/N:.3f}")
    print(f"SNR = {snr_db} dB")
    print(f"Backend: {'CuPy (GPU)' if xp.__name__ == 'cupy' else 'NumPy (CPU)'}")
    print("=" * 60)

    # =========================================================
    # 生成随机比特
    # =========================================================

    msg = xp.random.randint(0, 2, size=(batch, K), dtype=xp.uint8)

    # =========================================================
    # 编码
    # =========================================================

    code = encoder.encode(msg)

    # =========================================================
    # BPSK
    # =========================================================

    tx = 1 - 2 * code

    # =========================================================
    # AWGN
    # =========================================================

    snr_linear = 10 ** (snr_db / 10.0)
    sigma = xp.sqrt(1 / (2 * snr_linear))

    noise = sigma * xp.random.randn(*tx.shape)
    rx = tx + noise

    # =========================================================
    # LLR
    # =========================================================

    llr = 2 * rx / (sigma ** 2)

    # =========================================================
    # 解码
    # =========================================================

    decoded = decoder.decode(llr)
    decoded = decoded[:, :msg.shape[1]]
    # =========================================================
    # 性能统计
    # =========================================================

    bit_errors = xp.sum(decoded != msg)
    total_bits = batch * K

    if xp.__name__ == "cupy":
        bit_errors = int(bit_errors.get())

    ber = bit_errors / total_bits

    print(f"Bit Errors: {bit_errors}")
    print(f"BER: {ber:.6e}")

    # =========================================================
    # 无噪声验证
    # =========================================================

    tx_clean = 1 - 2 * code
    llr_clean = 1000 * tx_clean

    decoded_clean = decoder.decode(llr_clean)
    decoded_clean = decoded_clean[:, :msg.shape[1]]
    clean_errors = xp.sum(decoded_clean != msg)

    if xp.__name__ == "cupy":
        clean_errors = int(clean_errors.get())

    print("-" * 60)
    print("Noiseless Check:")
    print(f"Bit Errors (should be 0): {clean_errors}")
    print("=" * 60)


if __name__ == "__main__":
    turbo_test()