import numpy as np
import config

from LDPC_encoder import LDPCEncoder
from LDPC_decoder import LDPCDecoder


# 工具函数


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
    """

    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1 / (2 * rate * snr_linear)

    sigma = xp.sqrt(sigma2)

    noise = sigma * xp.random.randn(*x.shape)

    return x + noise, sigma2


def compute_llr(y, sigma2):
    """
    AWGN 下 BPSK LLR
    """
    return 2 * y / sigma2

# 主测试函数
def run_ldpc_test():

    cfg = config.CONFIG


#构造 Encoder
    encoder = LDPCEncoder(cfg)

    xp = encoder.xp
# 保存 H
    H_cpu = encoder.H.get() if xp.__name__ == "cupy" else encoder.H
    np.save("ldpc_H.npy", H_cpu)

# 修改 config 使 decoder 读取文件
    cfg["LDPC"]["H_matrix"] = {
        "type": "from_file",
        "path": "ldpc_H.npy"
    }


# 构造 Decoder
    decoder = LDPCDecoder(cfg)


# 基本参数
    K = encoder.info_length
    N = encoder.code_length
    rate = K / N

    batch_size = 200
    snr_db = 2.0

    print(f"LDPC Test  N={N}  K={K}  Rate={rate:.3f}")
    print(f"SNR = {snr_db} dB")
    print("Backend:", "CuPy (GPU)" if xp.__name__ == "cupy" else "NumPy (CPU)")


# 生成随机信息比特
    msg = xp.random.randint(0, 2, (batch_size, K), dtype=xp.uint8)

#编码

    code = encoder.encode(msg)

# BPSK 调制

    tx = bpsk_modulation(code, xp)

# AWGN 信道

    rx, sigma2 = awgn_channel(tx, snr_db, rate, xp)

# 计算 LLR

    llr = compute_llr(rx, sigma2)

# 解码

    decoded = decoder.decode(llr)

# 计算 BER
    decoded_info = decoded[:, :K]

    bit_errors = xp.sum(decoded_info != msg)
    total_bits = batch_size * K

    ber = bit_errors / total_bits

    if xp.__name__ == "cupy":
        bit_errors = bit_errors.get()
        ber = ber.get()

    print(f"Bit Errors: {bit_errors}")
    print(f"BER: {ber:.6e}")


if __name__ == "__main__":
    run_ldpc_test()