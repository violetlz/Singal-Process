from typing import Optional
from base_encoder import BaseEncoder
from rsc_encoder import RSCEncoder
from Interleaver import Interleaver


class TurboEncoder(BaseEncoder):
    """
    Turbo 编码器

    输出格式:
        [systematic | parity1 | parity2]

    输入:
        message: (batch, K)

    输出:
        codeword: (batch, 3K)
    """

    # =========================================================
    # 初始化
    # =========================================================

    def __init__(self, cfg: Optional[dict] = None, device: Optional[str] = None):
        super().__init__(cfg, device)

        turbo = self.cfg["TURBO"]

        self.rsc1 = RSCEncoder(cfg, device)
        self.rsc2 = RSCEncoder(cfg, device)

        self.interleaver = None
        self.seed = turbo["interleaver"].get("seed", 0)

        # Turbo 不固定 K，由输入决定
        self._info_length = None
        self._code_length = None

    # =========================================================
    # BaseEncoder 抽象方法实现
    # =========================================================

    def _encode_impl(self, message):
        """
        实现 BaseEncoder 要求的抽象接口

        参数:
            message: (batch, K)

        返回:
            codeword: (batch, 3K)
        """

        xp = self.xp
        msg = xp.asarray(message, dtype=xp.uint8)

        if msg.ndim == 1:
            msg = msg.reshape(1, -1)

        batch, K = msg.shape

        # 首次根据输入长度初始化交织器
        if self.interleaver is None:
            self.interleaver = Interleaver(K, self.seed)

        # RSC1
        sys1, p1 = self.rsc1.encode_components(msg)

        # 交织后进入 RSC2
        inter_msg = self.interleaver.permute(msg)
        _, p2 = self.rsc2.encode_components(inter_msg)

        code = xp.concatenate([sys1, p1, p2], axis=1)

        # 记录长度（用于接口属性）
        self._info_length = K
        self._code_length = code.shape[1]

        return code

    # =========================================================
    # 接口属性（与其它编码器统一）
    # =========================================================

    @property
    def info_length(self):
        return self._info_length

    @property
    def code_length(self):
        return self._code_length