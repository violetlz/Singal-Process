"""
信道编码系统全局配置文件

说明:
    - 所有编码器 / 解码器均从此文件读取默认参数
    - 通过 factory + BaseEncoder 实现按码型切换
"""

CONFIG = {

# 全局配置

    "GLOBAL": {
        # 是否启用 GPU（cupy）
        "use_gpu": True,

        # 默认数值类型
        "dtype": "uint8",

        # 随机种子
        "seed": 0
    },

# 普通卷积码
    "CONV": {
        # 生成多项式（binary 形式）
        # (7,5)_oct -> [[1,1,1],[1,0,1]]
        "polynomials": [
            [1, 1, 1],
            [1, 0, 1]
        ],

        # 约束长度 K
        "constraint_len": 3,

        # 是否 zero-tail（强制回到 0 状态）
        "flush": True,

        # 编码类型
        "systematic": False,

        #Viterbi 解码配置
        "viterbi": {
            # 分支度量
            "metric": "hamming",     # hamming / euclidean

            # traceback 深度
            # None 表示完整回溯
            "traceback": None
        }
    },

# RSC卷积码
    "RSC": {
        # 反馈多项式（recursive）
        # 通常与 parity 多项式配对
        "feedback": [1, 1, 1],     # 例如 7_oct

        # parity 多项式
        "parity": [1, 0, 1],       # 例如 5_oct

        "constraint_len": 3,

        # 是否 zero-tail
        "flush": True
    },

# Turbo
    "TURBO": {
        # 并行 RSC 分支数（经典 Turbo = 2）
        "num_encoders": 2,

        # 是否输出 systematic 比特
        "systematic": True,

        #交织器
        "interleaver": {
            # 类型
            "type": "random",       # random / block / s-random

            # 随机交织器参数
            "seed": 42,

            # s-random 交织器参数（可选）
            "s": None
        },

        #Turbo译码
        "decoder": {
            "algorithm": "log-map",     # log-map / max-log-map
            "num_iterations": 6,
            "llr_clip": 20.0
        }
    },

#LDPC
    "LDPC": {
# 基本码参数
    "code": {
        "N": 128,          # 码长
        "K": 64,           # 信息比特数
        "rate": 0.5        # 可选：K / N（用于校验）
    },

# H 矩阵构造方式
    "H_matrix": {
        "type": "random",     # random / regular / qc / standard
        "density": 0.1,       # 仅 random 有效
        "dv": None,           # regular LDPC：变量节点度
        "dc": None,           # regular LDPC：校验节点度
        "seed": 0,
        "path": None          # 若从文件加载 H
    },

# Encoder
    "encoder": {
        "systematic": True,   # 是否系统码
        "gaussian_elimination": True,
        "store_permutation": True
    },

# Decoder
    "decoder": {
        "algorithm": "min_sum",   # bp / min_sum / normalized_min_sum
        "max_iter": 50,
        "early_stop": True,
        "norm_factor": 0.75,      # 归一化 Min-Sum
        "llr_clip": 20.0
    }
},

# Polar
    "POLAR": {
        "N": 128,                 # 码长（2^n）
        "K": 64,                  # 信息位数
        "frozen_method": "simple"
        # simple / reliability
    }
}