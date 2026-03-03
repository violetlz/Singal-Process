import numpy as np
import config
from factory import build_encoder, build_decoder


def test_conv_codec(
        batch_size=5,
        info_len=50,
        device=None,
        add_noise=True
):
    """
    卷积码编码/解码测试

    参数:
        batch_size: 批大小
        info_len: 信息比特长度
        device: 'cpu' / 'gpu' / None
        add_noise: 是否加入随机翻转噪声
    """

# 构建编码器 / 解码器
    encoder = build_encoder("conv", cfg=config.CONFIG, device=device)
    decoder = build_decoder("conv", cfg=config.CONFIG, device=device)

    xp = encoder.xp
# 生成随机信息比特
    xp.random.seed(0)
    message = xp.random.randint(0, 2, (batch_size, info_len)).astype(xp.uint8)

    print("原始信息比特 shape:", message.shape)

# 编码
    codeword = encoder.encode(message)

    print("编码后 shape:", codeword.shape)


# 可选：加入简单信道误码

    if add_noise:
        flip_prob = 0.01
        noise = xp.random.rand(*codeword.shape) < flip_prob
        recv = xp.bitwise_xor(codeword, noise.astype(xp.uint8))
        print("加入随机误码，误码率≈", flip_prob)
    else:
        recv = codeword.copy()

# 解码
    decoded = decoder.decode(recv)

    print("解码后 shape:", decoded.shape)


# 截断尾比特（若 flush=True）
    m = config.CONFIG["CONV"]["constraint_len"]
    flush = config.CONFIG["CONV"].get("flush", True)

    if flush:
        decoded = decoded[:, :info_len]

# 计算误码率
    bit_errors = xp.sum(message != decoded)
    total_bits = message.size
    ber = bit_errors / total_bits

    print("总比特数:", total_bits)
    print("误比特数:", bit_errors)
    print("BER:", ber)

    if bit_errors == 0:
        print("编码解码完全正确")
    else:
        print("存在误码")

    return ber


if __name__ == "__main__":
    test_conv_codec(
        batch_size=10,
        info_len=100,
        device=None,       # None / 'cpu' / 'gpu'
        add_noise=True   # 测试无噪声，调成True为有噪声，修改flip_prob
    )