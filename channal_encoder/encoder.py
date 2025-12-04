from typing import Optional, Union, Tuple, List
import numpy as np
import cupy as cp
import concurrent.futures
import reedsolo
from reedsolo import RSCodec
import config

ArrayLike = Union[np.ndarray, 'cp.ndarray']


def bits_to_bytes_row(bits: np.ndarray) -> bytes:
    """
    参数:
        bits: 1D uint8 位数组（0/1）
    返回:
        对应 bytes，按高位优先打包
    """
    L = bits.size
    m = (L + 7) // 8
    if L % 8 != 0:
        pad = np.zeros(m * 8 - L, dtype=np.uint8)
        bits_padded = np.concatenate([bits, pad])
    else:
        bits_padded = bits
    bits_reshaped = bits_padded.reshape(m, 8)
    weights = (1 << np.arange(7, -1, -1))
    byte_vals = bits_reshaped.dot(weights).astype(np.uint8)
    return bytes(byte_vals.tolist())


def ensure_2d_bits(arr: ArrayLike) -> np.ndarray:
    """
    参数:
        arr: 一维或二维位数组（cupy 或 numpy）
    返回:
        numpy 2D uint8 数组 (batch, L)
    """
    arr_np = cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else np.asarray(arr)
    if arr_np.ndim == 1:
        arr_np = arr_np.reshape(1, -1)
    return arr_np.astype(np.uint8)


def bch_encode_row(args):
    """worker: 单行 BCH 编码"""
    row, n, k = args
    import galois
    code = galois.BCH(n, k)
    cw = code.encode(row.astype(np.uint8))
    return np.asarray(cw, dtype=np.uint8)


def bch_decode_row(args):
    """worker: 单行 BCH 译码，返回 (decoded_row, status)"""
    row, n, k = args
    import galois
    code = galois.BCH(n, k)
    try:
        m = code.decode(row.astype(np.uint8))
        return np.asarray(m, dtype=np.uint8), 0
    except Exception:
        return np.zeros(k, dtype=np.uint8), 2


def rs_encode_bytes_worker(args):
    """worker: 单条 bytes -> encode -> return bytes"""
    b, nsym = args
    import reedsolo
    cw = reedsolo.rs_encode_msg(b, nsym)
    return cw


def rs_decode_bytes_worker(args):
    """worker: 单条 cw bytes -> correct -> return bytes or b'' and status"""
    b, nsym = args
    import reedsolo
    try:
        res = reedsolo.rs_correct_msg(b, nsym)
        corrected = res[0] if isinstance(res, (tuple, list)) else res
        return corrected, 0
    except Exception:
        return b"", 2


class Encoder:
    """
    参数:
        cfg: 可选配置字典（优先于 config.CONFIG）
        device: 'gpu' 强制 GPU，'cpu' 强制 CPU，None 则使用 cfg["GLOBAL"]["use_gpu"]
    返回:
        无
    """

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        # 使用传入配置或全局 config
        self.cfg = cfg if cfg is not None else config.CONFIG

        # device 决策：入参优先，其次使用配置
        if device == 'cpu':
            self._use_gpu = False
        elif device == 'gpu':
            self._use_gpu = True
        else:
            self._use_gpu = bool(self.cfg.get("GLOBAL", {}).get("use_gpu", True))

    # Hamming
    def encode_hamming(self,
                       message: ArrayLike,
                       n: Optional[int] = None,
                       k: Optional[int] = None,
                       G: Optional[ArrayLike] = None,
                       secded: Optional[bool] = None) -> ArrayLike:
        """
        参数:
            message: 1D 或 2D 二值数组（0/1）
            n,k,G,secded: 若为 None 则使用 self.cfg 里的对应值
        返回:
            码字数组，若运行在 GPU 返回 cupy 数组，否则 numpy 数组
        """
        # 从配置读取缺省参数
        hcfg = self.cfg.get("HAMMING", {})
        n = int(hcfg.get("n", 7)) if n is None else int(n)
        k = int(hcfg.get("k", 4)) if k is None else int(k)
        secded = bool(hcfg.get("secded", False)) if secded is None else bool(secded)

        xp = cp if self._use_gpu else np
        # 准备消息
        msg = xp.asarray(message)
        if msg.ndim == 1:
            msg = msg.reshape(1, -1)
        if msg.shape[1] != k:
            raise ValueError(f"message width {msg.shape[1]} != k ({k})")

        # 构造或使用生成矩阵
        if G is None:
            if (n, k) == (7, 4):
                G_np = np.array([
                    [1, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                ], dtype=np.uint8)
                G = cp.asarray(G_np) if self._use_gpu else G_np
            else:
                rng = np.random.RandomState(0)
                P = rng.randint(0, 2, size=(k, n - k)).astype(np.uint8)
                G_np = np.concatenate([np.eye(k, dtype=np.uint8), P], axis=1)
                G = cp.asarray(G_np) if self._use_gpu else G_np
        else:
            G = cp.asarray(G) if self._use_gpu else np.asarray(G)

        prod = xp.dot(msg.astype(np.uint8), G.astype(np.uint8)) % 2
        prod = prod.astype(np.uint8)

        if secded:
            # 计算每行的总奇偶校验位并作为最后一列追加
            parity = (xp.sum(prod, axis=1) % 2).astype(xp.uint8).reshape(-1, 1)
            prod = xp.concatenate([prod, parity], axis=1)
        return prod

    # BCH
    def encode_bch_batch(self,
                         message: ArrayLike,
                         n: Optional[int] = None,
                         k: Optional[int] = None,
                         workers: Optional[int] = None) -> ArrayLike:
        """
        参数:
            message: bits 矩阵（cupy 或 numpy）
            n,k: BCH 参数（若为 None 则从配置读取）
            workers: 并行进程数（None 表示使用配置或顺序）
        返回:
            码字数组（当 device=GPU 时返回 cupy array）
        """
        bcfg = self.cfg.get("BCH", {})
        n = int(bcfg.get("n", 15)) if n is None else int(n)
        k = int(bcfg.get("k", 11)) if k is None else int(k)
        if workers is None:
            workers = bcfg.get("workers", None) or self.cfg.get("GLOBAL", {}).get("workers", None)

        msg_np = ensure_2d_bits(message)
        if msg_np.shape[1] != k:
            raise ValueError("message width != k")

        args = [(row, n, k) for row in msg_np.astype(np.uint8)]

        if workers is None or workers <= 1:
            cws = [bch_encode_row(a) for a in args]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                cws = list(ex.map(bch_encode_row, args))

        cws_np = np.stack(cws, axis=0)
        return cp.asarray(cws_np) if self._use_gpu else cws_np

    # RS

    # 把类内 worker 保留为静态方法
    @staticmethod
    def _rs_encode_bytes_worker_rsc(args):
        """worker: 使用 RSCodec.encode 进行编码，返回 bytes"""
        b, nsym = args
        import reedsolo
        from reedsolo import RSCodec
        rsc = RSCodec(nsym)
        try:
            cw = rsc.encode(b)
        except Exception:
            # 兜底使用低层函数
            cw = reedsolo.rs_encode_msg(b, nsym)
        return cw

    def encode_rs_batch(self,
                        message: Union[ArrayLike, bytes, List[bytes]],
                        nsym: Optional[int] = None,
                        as_bytes: bool = False,
                        workers: Optional[int] = None) -> Union[ArrayLike, List[bytes]]:
        """
        参数:
            message: bits 矩阵 或 bytes 或 list[bytes]
            nsym: 校验符号数（若 None 则从配置读取）
            as_bytes: True 返回 bytes（单条或列表）
            workers: 并行 worker 数（若 None 则从配置读取）
        返回:
            码字数组或 bytes 列表（依 as_bytes）
        """
        rscfg = self.cfg.get("RS", {})
        nsym = int(rscfg.get("nsym", 8)) if nsym is None else int(nsym)
        if workers is None:
            workers = rscfg.get("workers", None) or self.cfg.get("GLOBAL", {}).get("workers", None)

        # 单条 bytes
        if isinstance(message, (bytes, bytearray)):
            try:
                rsc = RSCodec(nsym)
                cw = rsc.encode(message)
            except Exception:
                cw = reedsolo.rs_encode_msg(message, nsym)
            return cw if as_bytes else np.frombuffer(cw, dtype=np.uint8)

        # list of bytes
        if isinstance(message, (list, tuple)) and len(message) > 0 and isinstance(message[0], (bytes, bytearray)):
            if workers is None or workers <= 1:
                cws = []
                for b in message:
                    try:
                        rsc = RSCodec(nsym)
                        cw = rsc.encode(b)
                    except Exception:
                        cw = reedsolo.rs_encode_msg(b, nsym)
                    cws.append(cw)
            else:
                # 注意：为了在多进程下可 pickled，建议将 _rs_encode_bytes_worker_rsc 提升为模块顶层函数
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    cws = list(ex.map(self._rs_encode_bytes_worker_rsc, [(b, nsym) for b in message]))

            if as_bytes:
                return cws
            arrs = [np.frombuffer(b, dtype=np.uint8) for b in cws]
            cws_np = np.stack(arrs, axis=0)
            return cp.asarray(cws_np) if self._use_gpu else cws_np

        # ndarray bits 或 cupy bits
        if isinstance(message, cp.ndarray):
            msg_np = cp.asnumpy(message)
        else:
            msg_np = np.asarray(message)

        msg_np = np.asarray(msg_np, dtype=np.uint8)
        if msg_np.ndim == 1:
            if msg_np.size == 0:
                return b"" if as_bytes else (
                    cp.asarray(np.array([], dtype=np.uint8)) if self._use_gpu else np.array([], dtype=np.uint8))
            if msg_np.max() <= 1:
                b = bits_to_bytes_row(msg_np)
                try:
                    rsc = RSCodec(nsym)
                    cw = rsc.encode(b)
                except Exception:
                    cw = reedsolo.rs_encode_msg(b, nsym)
                return cw if as_bytes else (
                    cp.asarray(np.frombuffer(cw, dtype=np.uint8)) if self._use_gpu else np.frombuffer(cw,
                                                                                                      dtype=np.uint8))
            else:
                b = bytes(msg_np.tolist())
                try:
                    rsc = RSCodec(nsym)
                    cw = rsc.encode(b)
                except Exception:
                    cw = reedsolo.rs_encode_msg(b, nsym)
                return cw if as_bytes else (
                    cp.asarray(np.frombuffer(cw, dtype=np.uint8)) if self._use_gpu else np.frombuffer(cw,
                                                                                                      dtype=np.uint8))

        # 多行 bits -> 打包为 bytes 列表
        bytes_list = []
        for row in msg_np:
            row = np.asarray(row, dtype=np.uint8)
            if row.size == 0:
                bytes_list.append(b"")
            elif row.max() <= 1:
                bytes_list.append(bits_to_bytes_row(row))
            else:
                bytes_list.append(bytes(row.tolist()))

        # 并行或顺序调用
        if workers is None or workers <= 1:
            cws = []
            for b in bytes_list:
                try:
                    rsc = RSCodec(nsym)
                    cw = rsc.encode(b)
                except Exception:
                    cw = reedsolo.rs_encode_msg(b, nsym)
                cws.append(cw)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                cws = list(ex.map(self._rs_encode_bytes_worker_rsc, [(b, nsym) for b in bytes_list]))

        if as_bytes:
            return cws
        arrs = [np.frombuffer(b, dtype=np.uint8) for b in cws]
        cws_np = np.stack(arrs, axis=0)
        return cp.asarray(cws_np) if self._use_gpu else cws_np

    def encode_rs_from_packed_bytes_list(self, bytes_list: List[bytes], nsym: Optional[int] = None, pad: bool = True,
                                         workers: Optional[int] = None) -> Tuple[ArrayLike, List[int]]:
        """
        参数:
            bytes_list: CPU 上的 bytes 列表
            nsym: 校验符号数（若 None 则从配置读取）
            pad: True 则将返回的 cw pad 到相同长度并返回 ndarray；False 则返回 bytes 列表
            workers: 并行 worker 数（若 None 则使用配置）
        返回:
            (pad后的码字 ndarray 或 原始 bytes 列表, 原始每条长度 list)
        """
        rscfg = self.cfg.get("RS", {})
        nsym = int(rscfg.get("nsym", 8)) if nsym is None else int(nsym)
        if workers is None:
            workers = rscfg.get("workers", None) or self.cfg.get("GLOBAL", {}).get("workers", None)

        if workers is None or workers <= 1:
            cws = []
            for b in bytes_list:
                try:
                    rsc = RSCodec(nsym)
                    cw = rsc.encode(b)
                except Exception:
                    cw = reedsolo.rs_encode_msg(b, nsym)
                cws.append(cw)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                cws = list(ex.map(self._rs_encode_bytes_worker_rsc, [(b, nsym) for b in bytes_list]))

        if not pad:
            return cws, [len(b) for b in cws]
        arrs = [np.frombuffer(b, dtype=np.uint8) for b in cws]
        max_len = max(arr.size for arr in arrs) if arrs else 0
        padded = np.stack([np.pad(arr, (0, max_len - arr.size), constant_values=0) for arr in arrs],
                          axis=0) if arrs else np.zeros((0, 0), dtype=np.uint8)
        return (cp.asarray(padded) if self._use_gpu else padded), [arr.size for arr in arrs]
