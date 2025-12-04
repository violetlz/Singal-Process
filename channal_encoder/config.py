from typing import Any, Dict, Optional
import json
import os
import copy
from datetime import datetime

DEFAULT_CONFIG: Dict[str, Any] = {
    "GLOBAL": {
        # 是否优先使用 GPU（cupy）；若无 GPU 或不想用则设为 False
        "use_gpu": True,
        # 并行 worker 默认值（None 表示由调用方/函数内部决定）
        "workers": None,
        # debug 模式，开启后会打印更多调试信息并保存失败样例
        "debug": False,
        # 随机种子
        "random_seed": None,
    },

    # I/O / 序列化 / metadata
    "IO": {
        # 是否在码字前附带 metadata header（建议 True）
        "metadata_wrapper": True,
        # header 使用的长度字段字节数（例如 4 表示前 4 字节为 header 长度）
        "packet_prefix_length": 4,
        # 比特在字节内的顺序： "msb_first"（高位在前） 或 "lsb_first"
        "bit_endianness": "msb_first",
        # 填充策略：当消息位数非 8 的倍数时如何填充
        "padding_policy": "pad_low",  # "pad_low" | "pad_high" | "no_pad"
    },

    # Hamming 参数
    "HAMMING": {
        "n": 7,
        "k": 4,
        # 是否使用扩展 Hamming（SECDED），会在码字末尾增加一个 parity bit
        "secded": True,
        # parity 类型： "even" 或 "odd"
        "parity_type": "even",
        # 默认生成矩阵 G（None 表示使用内置默认或随机生成）
        "G": None,
        # 是否对纠正结果进行重验证（开发时建议 True）
        "hamming_verify": False,
    },

    # BCH 参数（galois）
    "BCH": {
        "n": 31,
        "k": 11,
        # 可指定本原多项式（int 或 str），None 则由 galois 选择
        "poly": None,
        "workers": None,
        # 译码后是否进行重编码验证
        "bch_verify": True,
    },

    # RS 参数（reedsolo）
    "RS": {
        # 校验符号字节数（纠错能力 t = floor(nsym/2)）
        "nsym": 16,
        # 符号位宽，通常为 8 bits（一个字节）
        "symbol_size": 8,
        "workers": None,
        # 译码后是否进行重编码比对
        "verify": True,
        # decode 时是否启用低层 rs_correct_msg 作为 fallback
        "rs_fallback": True,
        # decode 成功后是否严格要求重编码完全相同
        "rs_decode_strict": False,
    },

    # 并行 / 性能调优
    "PERF": {
        # 小批量阈值：仅当 batch_size >= 该值时启用并行 worker
        "batch_size_threshold_for_workers": 64,
        # 并行 worker 上限（None 表示自动选择）
        "max_workers": None,
        # GPU <-> CPU 数据传输策略："batch" 或 "per_item"
        "gpu_transfer_strategy": "batch",
        # 是否使用 pinned memory（需额外实现）
        "use_pinned_memory": False,
    },

    # 调试 / 日志
    "DEBUG": {
        # 日志等级： "DEBUG" / "INFO" / "WARN" / "ERROR"
        "log_level": "INFO",
        # 若设置为目录路径，则在 decode 失败时把失败样例保存到该目录
        "save_failed_cases_dir": None,
        # 是否收集运行统计数据（status 分布、延迟等）
        "stats_collect": True,
    },

}

# 运行时配置对象（可被修改以影响全局行为）
CONFIG: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)


def load_from_json(path: str) -> Dict[str, Any]:
    """参数:
        path: json 文件路径
    返回:
        配置字典（以文件内容覆盖默认配置）
    """
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    merged = copy.deepcopy(DEFAULT_CONFIG)
    # 只做顶层合并：若需要深度合并请在调用方处理
    for k, v in user_cfg.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def apply_overrides(overrides: Dict[str, Any]) -> None:
    """参数:
        overrides: 要覆盖的配置字典（部分或全部）
    返回:
        None
    """
    global CONFIG
    for k, v in overrides.items():
        if k in CONFIG and isinstance(CONFIG[k], dict) and isinstance(v, dict):
            CONFIG[k].update(v)
        else:
            CONFIG[k] = v


def load_if_exists(json_path_env: str = "ENCODER_CONFIG_PATH") -> None:
    """参数:
        json_path_env: 环境变量名，若存在则从该 json 文件加载配置覆盖默认
    返回:
        None
    """
    p = os.environ.get(json_path_env)
    if p and os.path.exists(p):
        cfg = load_from_json(p)
        global CONFIG
        CONFIG = cfg



def get_config_copy() -> Dict[str, Any]:
    """参数:
        无
    返回:
        CONFIG 的深拷贝（防止上层直接修改引用）
    """
    return copy.deepcopy(CONFIG)


load_if_exists()

if __name__ == "__main__":

    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))

