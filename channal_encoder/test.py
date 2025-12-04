import numpy as np
import cupy as cp
import config
from encoder import Encoder, bits_to_bytes_row
from decoder import Decoder

def to_numpy(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

def to_gpu(x):
    return cp.asarray(x) if isinstance(x, np.ndarray) else x

# ---------- Hamming 测试 ----------
def test_hamming():
    print("----- Hamming 测试 -----")
    # 明确传入配置对象以保证 encoder/decoder 使用统一配置
    enc = Encoder(cfg=config.CONFIG, device='gpu')
    dec = Decoder(cfg=config.CONFIG, device='gpu')
    k = 4; n = 7

    # 在 GPU 上生成消息并编码（仍可通过参数覆盖默认）
    msgs_gpu = (cp.random.rand(8, k) > 0.5).astype(cp.uint8)
    cw_gpu = enc.encode_hamming(msgs_gpu, n=n, k=k, secded=True)  # GPU 输出

    # 无误差译码（整体）
    corrected_all, status_all = dec.decode_hamming(cw_gpu, n=n, k=k, secded=True)
    status_all_np = to_numpy(status_all)
    assert np.all(status_all_np == 0), f"Hamming 无误差译码失败，status: {status_all_np}"
    # 比较原始消息（GPU）与 decoded info（可能在 GPU/CPU）
    info_all = corrected_all[:, :k] if isinstance(corrected_all, cp.ndarray) else corrected_all[:, :k]
    # 统一到 GPU 比较（若 info_all 是 numpy 则转 GPU）
    assert cp.array_equal(msgs_gpu, info_all if isinstance(info_all, cp.ndarray) else to_gpu(info_all)), "Hamming 无误差闭环失败"
    print("Hamming 无误差闭环通过")

    # 注入 1-bit 错误（应被纠正）
    cw_err2 = to_numpy(cw_gpu).copy()
    cw_err2[0, 2] ^= 1
    cw_err2[1, 3] ^= 1
    corrected, status = dec.decode_hamming(cw_err2, n=n, k=k, secded=True)
    status = to_numpy(status)
    assert status[0] == 0, f"Hamming 1-bit 应被纠正，但 status={status[0]}"
    # 比较信息位
    info0 = to_numpy(corrected[:, :k])[0]
    orig0 = to_numpy(msgs_gpu)[0]
    assert np.array_equal(orig0, info0), f"Hamming 1-bit 纠错后信息不匹配: orig={orig0}, got={info0}"
    print("Hamming 1-bit 纠错通过")

    # 注入 2-bit 错误（通常不可纠正）
    cw_err2 = to_numpy(cw_gpu).copy()
    cw_err2[1, 0] ^= 1
    cw_err2[1, 3] ^= 1
    corrected2, status2 = dec.decode_hamming(cw_err2, n=n, k=k, secded=True)
    status2 = to_numpy(status2)
    # 2-bit 通常无法由 (7,4) Hamming 纠正 -> status 应为非 0（2）
    assert status2[1] != 0, f"Hamming 2-bit 测试: 期望不可纠正，但 status={status2[1]}"
    print("Hamming 2-bit 不可纠正（期望）通过")


# ---------- BCH 测试 ----------
def test_bch():
    print("----- BCH 测试 -----")
    enc = Encoder(cfg=config.CONFIG, device='gpu')
    dec = Decoder(cfg=config.CONFIG, device='gpu')

    k = 11; n = 15
    batch = 6
    msgs_gpu = (cp.random.rand(batch, k) > 0.5).astype(cp.uint8)

    # 编码（内部会批量 asnumpy -> galois -> cp.asarray）
    cw_gpu = enc.encode_bch_batch(msgs_gpu, n=n, k=k, workers=None)

    # 无误差译码
    decoded, status = dec.decode_bch_batch(cw_gpu, n=n, k=k, workers=None)
    decoded_np = to_numpy(decoded)
    status_np = to_numpy(status)
    msgs_np = to_numpy(msgs_gpu)
    assert np.array_equal(decoded_np, msgs_np), f"BCH 无误差闭环失败"
    assert np.all(status_np == 0), f"BCH 无误差时 status 非零: {status_np}"
    print("BCH 无误差闭环通过")

    # 注入 1-bit 错误（BCH(15,11) 一般能纠正单比特错误）
    cw_err = to_numpy(cw_gpu).copy()
    cw_err[0, 5] ^= 1  # 单比特翻转
    decoded_err, status_err = dec.decode_bch_batch(cw_err, n=n, k=k, workers=None)
    decoded_err_np = to_numpy(decoded_err)
    status_err_np = to_numpy(status_err)
    assert status_err_np[0] == 0, f"BCH 单比特应能纠正，status={status_err_np[0]}"
    assert np.array_equal(decoded_err_np[0], msgs_np[0]), "BCH 单比特纠错后信息不匹配"
    print("BCH 单比特纠错通过")


# ---------- RS 测试 ----------
def test_rs():
    print("----- RS 测试 -----")
    enc = Encoder(cfg=config.CONFIG, device='gpu')
    dec = Decoder(cfg=config.CONFIG, device='gpu')

    batch = 4
    L_bits = 16  # 两字节消息
    msgs_bits_gpu = (cp.random.rand(batch, L_bits) > 0.5).astype(cp.uint8)

    # 在 GPU 上把 bits 打包为字节（batch）
    bitlen = L_bits
    byte_cols = (bitlen + 7) // 8
    bits = msgs_bits_gpu
    if bitlen % 8 != 0:
        pad_width = byte_cols * 8 - bitlen
        bits = cp.concatenate([bits, cp.zeros((batch, pad_width), dtype=cp.uint8)], axis=1)
    bits_reshaped = bits.reshape(batch, byte_cols, 8)
    weights = (1 << cp.arange(7, -1, -1, dtype=cp.uint8))
    packed_bytes_gpu = (bits_reshaped.astype(cp.uint8) @ weights.astype(cp.uint8)).astype(cp.uint8)  # (batch, byte_cols)

    # 批量一次性传到 CPU，生成 bytes 列表
    packed_bytes_cpu = to_numpy(packed_bytes_gpu)
    bytes_list = [bytes(row.tolist()) for row in packed_bytes_cpu]

    # 批量 RS 编码（返回 bytes 列表）
    cw_bytes_list = enc.encode_rs_batch(bytes_list, nsym=8, as_bytes=True, workers=None)

    # 无误差译码（返回 bytes 列表与 status）
    decoded_list, status = dec.decode_rs_batch(cw_bytes_list, nsym=8, as_bytes=True, workers=None)
    status_np = to_numpy(status)
    assert np.all(status_np == 0), f"RS 无误差译码失败，status：{status_np}"
    # 比较原始 bytes
    for i in range(batch):
        rec = decoded_list[i]
        if isinstance(rec, (bytes, bytearray)):
            rec_bytes = rec
        else:
            raise AssertionError("RS decode 返回非 bytes (expect bytes list)")
        # 取前 L_bits 位比较
        rec_arr = np.frombuffer(rec_bytes, dtype=np.uint8)
        rec_bits = ((rec_arr[:, None] & (1 << np.arange(7, -1, -1))) != 0).astype(np.uint8).reshape(-1)[:L_bits]
        orig_bits = to_numpy(msgs_bits_gpu[i])
        assert np.array_equal(orig_bits, rec_bits), f"RS 无误差闭环比特不匹配 i={i}"
    print("RS 无误差闭环通过")

    # 注入错误（3 个符号）
    cw_mutable = [bytearray(b) for b in cw_bytes_list]
    cw_mutable[0][0] ^= 0xFF
    cw_mutable[0][1] ^= 0x7F
    cw_mutable[0][2] ^= 0x33

    # 单条 decode 测试（捕获异常）
    try:
        decoded_list2, status2 = dec.decode_rs_batch([bytes(cw_mutable[0])], nsym=8, as_bytes=True, workers=None)
        print("decode result status2:", status2)
        print("decoded:", decoded_list2)
    except Exception as e:
        print("decode raised exception:", type(e), e)
        raise

    # 转回 bytes 列表
    cw_mutable_bytes = [bytes(b) for b in cw_mutable]

    # 批量译码（应能纠正 3 个符号错误）
    decoded_list2, status2 = dec.decode_rs_batch(cw_mutable_bytes, nsym=8, as_bytes=True, workers=None)
    status2_np = to_numpy(status2)
    assert status2_np[0] == 0, f"RS 注入 3 个符号错误应被纠正，但 status={status2_np[0]}"
    print("RS 注入 3 个符号错误纠错通过")

    # 注入超过能力的错误（例如 >4 个符号错误），这里对第 1 条注入 5 个符号错误，期待失败 status != 0
    cw_mutable2 = [bytearray(b) for b in cw_bytes_list]
    for idx in range(5):
        cw_mutable2[1][idx] ^= (0xAA + idx) & 0xFF
    cw_mutable2_bytes = [bytes(b) for b in cw_mutable2]
    _, status3 = dec.decode_rs_batch(cw_mutable2_bytes, nsym=8, as_bytes=True, workers=None)
    status3_np = to_numpy(status3)
    assert status3_np[1] != 0, f"RS 注入 5 个符号错误应无法纠正，但 status={status3_np[1]}"
    print("RS 超出纠错能力测试（期望失败）通过")


# ---------- main ----------
if __name__ == "__main__":
    test_hamming()
    test_bch()
    test_rs()
    print("所有测试通过")
