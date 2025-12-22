import numpy as np
import cupy as cp
from encoder import Encoder
from decoder import Decoder
import config


def test_conv():
    # 固定随机种子（CPU + GPU）
    np.random.seed(0)
    cp.random.seed(0)

    batch = 64
    k = 100


    # random bits (CPU)

    msg = np.random.randint(0, 2, size=(batch, k)).astype(np.uint8)

    # encoder (GPU)
    enc = Encoder(config.CONFIG, device="gpu")
    code = enc.encode_conv(msg)     # cupy.ndarray


    # BSC channel (GPU)

    p = 0.05
    noise = (cp.random.rand(*code.shape) < p).astype(cp.uint8)
    recv = code ^ noise

    # decoder (GPU + CPU traceback)

    dec = Decoder(config.CONFIG, device="gpu")
    msg_hat = dec.decode_conv(recv)  # numpy.ndarray

    # BER

    bit_errors = np.sum(msg_hat[:, :k] != msg)
    ber = bit_errors / (batch * k)

    print("BER:", ber)


if __name__ == "__main__":
    test_conv()
