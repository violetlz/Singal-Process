# decoder.py
from typing import Optional, Union, Tuple, List
import numpy as np
import cupy as cp
import concurrent.futures
import reedsolo
from reedsolo import RSCodec
import config

ArrayLike = Union[np.ndarray, 'cp.ndarray']


def bytes_to_bits_row(b: bytes, bitlen: Optional[int] = None) -> np.ndarray:
    """
    参数:
        b: bytes
        bitlen: 可选，输出比特长度；若提供则截断到该长度
    返回:
        1D uint8 位数组
    """
    arr = np.frombuffer(b, dtype=np.uint8)
    weights = (1 << np.arange(7, -1, -1))
    bits = ((arr[:, None] & weights) != 0).astype(np.uint8).reshape(-1)
    if bitlen is not None:
        return bits[:bitlen]
    return bits


def bch_decode_row_worker(args):
    """worker: 单行 BCH 译码，返回 (decoded_row, status)"""
    row, n, k = args
    import galois
    code = galois.BCH(n, k)
    try:
        m = code.decode(row.astype(np.uint8))
        return np.asarray(m, dtype=np.uint8), 0
    except Exception:
        return np.zeros(k, dtype=np.uint8), 2


def rs_decode_bytes_worker(args):
    """
    worker: 使用 RSCodec.decode 在 worker 进程内做 robust 的纠错检测
    返回 (corrected_bytes_or_b'', status)
    status: 0=success, 2=fail/unrecoverable
    """
    b, nsym = args
    import reedsolo
    from reedsolo import RSCodec, ReedSolomonError
    try:
        rsc = RSCodec(nsym)
        dec = rsc.decode(b)
        if isinstance(dec, (tuple, list)):
            message_bytes = dec[0]
        else:
            message_bytes = dec
        try:
            cw_re = rsc.encode(message_bytes)
        except Exception:
            cw_re = reedsolo.rs_encode_msg(message_bytes, nsym)
        if cw_re == b:
            return b, 0
        else:
            return b"", 2
    except ReedSolomonError:
        return b"", 2
    except Exception:
        return b"", 2


def rs_decode_bytes_worker_rsc(args):
    """
    worker: 使用 RSCodec.decode 优先，然后 fallback 到 rs_correct_msg。
    返回 (message_bytes_or_b'', status)
    status: 0=success, 2=fail/unrecoverable
    """
    b, nsym = args
    import reedsolo
    from reedsolo import RSCodec, ReedSolomonError
    try:
        rsc = RSCodec(nsym)
        dec = rsc.decode(b)
        if isinstance(dec, (tuple, list)):
            message_bytes = dec[0]
        else:
            message_bytes = dec
        return (message_bytes, 0)
    except ReedSolomonError:
        try:
            res = reedsolo.rs_correct_msg(b, nsym)
            corrected = res[0] if isinstance(res, (tuple, list)) else res
            if corrected and len(corrected) >= nsym:
                message_bytes = corrected[:-nsym] if len(corrected) > nsym else b""
                return (message_bytes, 0) if message_bytes != b"" else (b"", 2)
            else:
                return (b"", 2)
        except Exception:
            return (b"", 2)
    except Exception:
        return (b"", 2)


class Decoder:
    """
    参数:
        cfg: 可选配置（优先于 config.CONFIG）
        device: 'gpu' 强制 GPU，'cpu' 强制 CPU，None 则使用 cfg["GLOBAL"]["use_gpu"]
    返回:
        无
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        self.cfg = cfg if cfg is not None else config.CONFIG
        if device == 'cpu':
            self._use_gpu = False
        elif device == 'gpu':
            self._use_gpu = True
        else:
            self._use_gpu = bool(self.cfg.get("GLOBAL", {}).get("use_gpu", True))

    # Hamming 译码
    def decode_hamming(self,
                       codewords: ArrayLike,
                       n: Optional[int] = None,
                       k: Optional[int] = None,
                       G: Optional[ArrayLike] = None,
                       secded: Optional[bool] = None) -> Tuple[ArrayLike, np.ndarray]:
        """
        参数:
            codewords: 1D 或 2D 码字数组（若 secded=True 输入应包含 parity bit）
            n,k: 码参数（当 secded=True 且默认 7,4 时，输入实际长度应为 8）
            G: 可选生成矩阵（系统型）
            secded: 是否使用扩展 Hamming（SECDED）
        返回:
            (纠正后的码字数组, status_array)
            status: 0 = 成功或已纠正（single），1 = 仅纠正 parity 位，2 = 检测到不可纠正错误
        """
        hcfg = self.cfg.get("HAMMING", {})
        n = int(hcfg.get("n", 7)) if n is None else int(n)
        k = int(hcfg.get("k", 4)) if k is None else int(k)
        secded = bool(hcfg.get("secded", False)) if secded is None else bool(secded)

        xp = cp if self._use_gpu else np
        cw = xp.asarray(codewords)
        if cw.ndim == 1:
            cw = cw.reshape(1, -1)

        if secded:
            expected_len = n + 1
        else:
            expected_len = n

        if cw.shape[1] != expected_len:
            raise ValueError(f"codeword length {cw.shape[1]} 不匹配期望 {expected_len}")

        if G is None:
            if (n, k) == (7, 4):
                G_np = np.array([
                    [1, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                ], dtype=np.uint8)
                G_use = G_np
            else:
                raise ValueError("默认 G 仅对 (7,4) 可用，其他情况请传入系统型 G")
        else:
            G_use = cp.asnumpy(G) if isinstance(G, cp.ndarray) else np.asarray(G)

        if not np.array_equal(G_use[:k, :k], np.eye(k, dtype=np.uint8)):
            raise ValueError("当前实现要求 G 为系统型 [I | P]")

        P = G_use[:, k:]
        H_np = np.concatenate([P.T % 2, np.eye(n - k, dtype=np.uint8)], axis=1) % 2

        if secded:
            data_bits = cw[:, :n]
            parity_bits = cw[:, n]
        else:
            data_bits = cw
            parity_bits = None

        H = cp.asarray(H_np) if self._use_gpu else H_np
        synd = (xp.dot(data_bits.astype(np.uint8), H.T.astype(np.uint8)) % 2).astype(np.uint8)

        synd_map = {}
        H_cols = H_np.T
        for j in range(n):
            key = tuple(H_cols[j].tolist())
            synd_map[key] = j

        corrected = cw.copy()
        status = np.zeros(corrected.shape[0], dtype=np.int8)

        for i in range(corrected.shape[0]):
            s = tuple(synd[i].tolist())
            if secded:
                p_prime = int(xp.sum(corrected[i, :]) % 2)
                if all(v == 0 for v in s) and p_prime == 0:
                    status[i] = 0
                elif not all(v == 0 for v in s) and p_prime == 1:
                    if s in synd_map:
                        pos = synd_map[s]
                        corrected[i, pos] ^= 1
                        status[i] = 0
                    else:
                        status[i] = 2
                elif all(v == 0 for v in s) and p_prime == 1:
                    corrected[i, n] ^= 1
                    status[i] = 1
                elif not all(v == 0 for v in s) and p_prime == 0:
                    status[i] = 2
                else:
                    status[i] = 2
            else:
                if all(v == 0 for v in s):
                    status[i] = 0
                elif s in synd_map:
                    pos = synd_map[s]
                    corrected[i, pos] ^= 1
                    status[i] = 0
                else:
                    status[i] = 2

        corrected_out = cp.asarray(corrected) if self._use_gpu else corrected
        return corrected_out, status

    # BCH 批量译码
    def decode_bch_batch(self,
                         codewords: ArrayLike,
                         n: Optional[int] = None,
                         k: Optional[int] = None,
                         workers: Optional[int] = None) -> Tuple[ArrayLike, np.ndarray]:
        """
        参数:
            codewords: 1D/2D 码字数组（cupy 或 numpy）
            n,k: BCH 参数（若 None 则从配置读取）
            workers: 并行进程数（若 None 则从配置读取）
        返回:
            (decoded_info_array, status_array)  status: 0=success,2=fail
        """
        bcfg = self.cfg.get("BCH", {})
        n = int(bcfg.get("n", 15)) if n is None else int(n)
        k = int(bcfg.get("k", 11)) if k is None else int(k)
        if workers is None:
            workers = bcfg.get("workers", None) or self.cfg.get("GLOBAL", {}).get("workers", None)

        cws_np = cp.asnumpy(codewords) if isinstance(codewords, cp.ndarray) else np.asarray(codewords)
        if cws_np.ndim == 1:
            cws_np = cws_np.reshape(1, -1)
        args = [(row, n, k) for row in cws_np.astype(np.uint8)]

        if workers is None or workers <= 1:
            results = [bch_decode_row_worker(a) for a in args]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(bch_decode_row_worker, args))

        decoded = [r[0] for r in results]
        status = [r[1] for r in results]
        decoded_np = np.stack(decoded, axis=0)
        return (cp.asarray(decoded_np) if self._use_gpu else decoded_np, np.array(status, dtype=np.int8))

    # RS 译码
    def decode_rs_batch(self,
                        codewords: Union[ArrayLike, bytes, List[bytes]],
                        nsym: Optional[int] = None,
                        as_bytes: bool = False,
                        workers: Optional[int] = None) -> Tuple[Union[ArrayLike, List[bytes]], np.ndarray]:
        """
        参数:
            codewords: ndarray 或 bytes 或 list[bytes]
            nsym: RS 校验符号数（若 None 则从配置读取）
            as_bytes: True 返回 bytes 列表
            workers: 并行进程数（若 None 则从配置读取）
        返回:
            (decoded_array_or_list, status_array)
        """
        rscfg = self.cfg.get("RS", {})
        nsym = int(rscfg.get("nsym", 8)) if nsym is None else int(nsym)
        if workers is None:
            workers = rscfg.get("workers", None) or self.cfg.get("GLOBAL", {}).get("workers", None)

        # 单条 bytes
        if isinstance(codewords, (bytes, bytearray)):
            try:
                rsc = RSCodec(nsym)
                dec = rsc.decode(codewords)
                if isinstance(dec, (tuple, list)):
                    message_bytes = dec[0]
                else:
                    message_bytes = dec
                return (message_bytes if as_bytes else np.frombuffer(message_bytes, dtype=np.uint8)), np.array([0],
                                                                                                               dtype=np.int8)
            except Exception:
                try:
                    res = reedsolo.rs_correct_msg(codewords, nsym)
                    corrected = res[0] if isinstance(res, (tuple, list)) else res
                    if corrected and len(corrected) >= nsym:
                        message_bytes = corrected[:-nsym] if len(corrected) > nsym else b""
                        if message_bytes != b"":
                            return (message_bytes if as_bytes else np.frombuffer(message_bytes, dtype=np.uint8)), np.array([0], dtype=np.int8)
                    return (b"" if as_bytes else np.array([], dtype=np.uint8)), np.array([2], dtype=np.int8)
                except Exception:
                    return (b"" if as_bytes else np.array([], dtype=np.uint8)), np.array([2], dtype=np.int8)

        # list of bytes
        if isinstance(codewords, (list, tuple)) and len(codewords) > 0 and isinstance(codewords[0], (bytes, bytearray)):
            corrected_list: List[bytes] = []
            status: List[int] = []
            if workers is None or workers <= 1:
                for b in codewords:
                    try:
                        rsc = RSCodec(nsym)
                        dec = rsc.decode(b)
                        if isinstance(dec, (tuple, list)):
                            message_bytes = dec[0]
                        else:
                            message_bytes = dec
                        corrected_list.append(message_bytes)
                        status.append(0)
                    except Exception:
                        try:
                            res = reedsolo.rs_correct_msg(b, nsym)
                            corrected = res[0] if isinstance(res, (tuple, list)) else res
                            if corrected and len(corrected) >= nsym:
                                message_bytes = corrected[:-nsym] if len(corrected) > nsym else b""
                                if message_bytes != b"":
                                    corrected_list.append(message_bytes)
                                    status.append(0)
                                    continue
                            corrected_list.append(b"")
                            status.append(2)
                        except Exception:
                            corrected_list.append(b"")
                            status.append(2)

                if as_bytes:
                    return corrected_list, np.array(status, dtype=np.int8)

                arrs = [np.frombuffer(b, dtype=np.uint8) if b != b"" else np.array([], dtype=np.uint8) for b in corrected_list]
                try:
                    stacked = np.stack(arrs, axis=0)
                    return (cp.asarray(stacked) if self._use_gpu else stacked, np.array(status, dtype=np.int8))
                except Exception:
                    return (arrs, np.array(status, dtype=np.int8))

            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    results = list(ex.map(rs_decode_bytes_worker_rsc, [(b, nsym) for b in codewords]))
                corrected_list = [r[0] for r in results]
                status = [r[1] for r in results]
                if as_bytes:
                    return corrected_list, np.array(status, dtype=np.int8)
                arrs = [np.frombuffer(b, dtype=np.uint8) if b != b"" else np.array([], dtype=np.uint8) for b in corrected_list]
                try:
                    stacked = np.stack(arrs, axis=0)
                    return (cp.asarray(stacked) if self._use_gpu else stacked, np.array(status, dtype=np.int8))
                except Exception:
                    return (arrs, np.array(status, dtype=np.int8))

        # ndarray 输入（可能是 pad 后的 array） -> 转为 bytes 列表逐条处理
        if isinstance(codewords, cp.ndarray):
            cws_np = cp.asnumpy(codewords)
        else:
            cws_np = np.asarray(codewords)

        decoded_list = []
        status = []
        for row in cws_np:
            b = bytes(np.asarray(row, dtype=np.uint8).tolist())
            try:
                rsc = RSCodec(nsym)
                dec = rsc.decode(b)
                if isinstance(dec, (tuple, list)):
                    message_bytes = dec[0]
                else:
                    message_bytes = dec
                if as_bytes:
                    decoded_list.append(message_bytes)
                else:
                    decoded_list.append(np.frombuffer(message_bytes, dtype=np.uint8))
                status.append(0)
            except Exception:
                try:
                    res = reedsolo.rs_correct_msg(b, nsym)
                    corrected = res[0] if isinstance(res, (tuple, list)) else res
                    if corrected and len(corrected) >= nsym:
                        message_bytes = corrected[:-nsym] if len(corrected) > nsym else b""
                        if as_bytes:
                            decoded_list.append(message_bytes)
                        else:
                            decoded_list.append(np.frombuffer(message_bytes, dtype=np.uint8))
                        status.append(0)
                        continue
                    if as_bytes:
                        decoded_list.append(b"")
                    else:
                        decoded_list.append(np.array([], dtype=np.uint8))
                    status.append(2)
                except Exception:
                    if as_bytes:
                        decoded_list.append(b"")
                    else:
                        decoded_list.append(np.array([], dtype=np.uint8))
                    status.append(2)

        if as_bytes:
            return decoded_list, np.array(status, dtype=np.int8)
        arrs = [np.asarray(x, dtype=np.uint8) for x in decoded_list]
        try:
            stacked = np.stack(arrs, axis=0)
            return (cp.asarray(stacked) if self._use_gpu else stacked, np.array(status, dtype=np.int8))
        except Exception:
            return (arrs, np.array(status, dtype=np.int8))
