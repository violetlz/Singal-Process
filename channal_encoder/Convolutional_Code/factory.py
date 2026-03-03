# Encoder
from conv_encoder import ConvEncoder
from Turbo_encoder import TurboEncoder
from LDPC_encoder import LDPCEncoder
from Polar_encoder import PolarEncoder


_ENCODER_REGISTRY = {
    "conv": ConvEncoder,
    "turbo": TurboEncoder,
    "ldpc": LDPCEncoder,
    "polar": PolarEncoder,
}


def build_encoder(name: str, cfg=None, device=None):
    """
    构建编码器

    参数:
        name: 编码器名称
        cfg: 配置字典
        device: torch.device 或字符串

    返回:
        BaseEncoder
    """
    if name is None:
        raise ValueError("Encoder name must not be None")

    name = name.lower()

    if name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {name}. "
            f"Available encoders: {list(_ENCODER_REGISTRY.keys())}"
        )

    encoder_cls = _ENCODER_REGISTRY[name]
    return encoder_cls(cfg=cfg, device=device)


# Decoder
from conv_decoder import ConvDecoder
from Turbo_decoder import TurboDecoder
from LDPC_decoder import LDPCDecoder
from Polar_decoder import PolarDecoder


_DECODER_REGISTRY = {
    "conv": ConvDecoder,
    "turbo": TurboDecoder,
    "ldpc": LDPCDecoder,
    "polar": PolarDecoder,
}


def build_decoder(name: str, cfg=None, device=None):
    """
    构建解码器

    参数:
        name: 解码器名称
        cfg: 配置字典
        device: torch.device 或字符串

    返回:
        BaseDecoder
    """
    if name is None:
        raise ValueError("Decoder name must not be None")

    name = name.lower()

    if name not in _DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder type: {name}. "
            f"Available decoders: {list(_DECODER_REGISTRY.keys())}"
        )

    decoder_cls = _DECODER_REGISTRY[name]
    return decoder_cls(cfg=cfg, device=device)
