# QPSK信号生成功能

基于CuPy的GPU加速QPSK调制信号生成器，支持5MHz带宽信号生成。

## 功能概述

本功能扩展了原有的`SignalGenerator`类，添加了完整的QPSK调制信号生成功能：

### 主要特性

1. **随机比特生成**: 生成随机比特序列
2. **QPSK调制**: 实现QPSK星座图映射
3. **升根余弦滤波**: 实现带宽限制的升根余弦滤波器
4. **带宽设计**: 精确控制信号带宽
5. **噪声添加**: 可选的AWGN噪声添加
6. **GPU加速**: 利用CuPy进行GPU并行计算

## 核心函数

### 1. `generate_random_bits(n_bits)`
生成随机比特序列
- **参数**: `n_bits` - 比特数量
- **返回**: 随机比特序列 (0或1)

### 2. `qpsk_modulation(bits)`
QPSK调制
- **参数**: `bits` - 输入比特序列
- **返回**: `(I路信号, Q路信号)`
- **星座图映射**:
  - 00 → (1, 1)
  - 01 → (-1, 1)
  - 10 → (1, -1)
  - 11 → (-1, -1)

### 3. `root_raised_cosine_filter(signal, sample_rate, symbol_rate, alpha, filter_length)`
升根余弦滤波器
- **参数**:
  - `signal` - 输入信号
  - `sample_rate` - 采样率 (Hz)
  - `symbol_rate` - 符号率 (Hz)
  - `alpha` - 滚降因子 (0-1)
  - `filter_length` - 滤波器长度
- **返回**: 滤波后的信号

### 4. `generate_qpsk_signal(bandwidth, sample_rate, duration, alpha, snr_db)`
生成QPSK调制信号
- **参数**:
  - `bandwidth` - 信号带宽 (Hz)，默认5MHz
  - `sample_rate` - 采样率 (Hz)，默认20MHz
  - `duration` - 持续时间 (s)，默认1.0s
  - `alpha` - 升根余弦滤波器滚降因子，默认0.35
  - `snr_db` - 信噪比 (dB)，None表示不添加噪声
- **返回**: 包含完整信号信息的字典

### 5. `generate_multiple_qpsk_signals(n_signals, bandwidth, sample_rate, duration, alpha, snr_db)`
生成多个QPSK信号
- **参数**: 同上，增加`n_signals`参数指定信号数量
- **返回**: 包含多个QPSK信号的字典

## 使用示例

### 基本使用

```python
from src import GPUSignalProcessor, SignalGenerator

# 初始化
gpu_processor = GPUSignalProcessor(gpu_id=0)
signal_generator = SignalGenerator(gpu_processor)

# 生成5MHz带宽的QPSK信号
qpsk_result = signal_generator.generate_qpsk_signal(
    bandwidth=5e6,      # 5MHz带宽
    sample_rate=20e6,   # 20MHz采样率
    duration=0.001,     # 1ms持续时间
    alpha=0.35,         # 升根余弦滤波器滚降因子
    snr_db=20           # 20dB信噪比
)

# 提取信号
signal = qpsk_result['signal']
bits = qpsk_result['bits']
i_symbols = qpsk_result['i_symbols']
q_symbols = qpsk_result['q_symbols']
```

### 参数说明

#### 带宽设计
- **带宽**: 5MHz (5e6 Hz)
- **符号率**: 带宽 / (1 + α) = 5MHz / 1.35 ≈ 3.7 Msymbols/s
- **比特率**: 符号率 × 2 = 7.4 Mbps (QPSK每个符号2比特)

#### 采样率选择
- **采样率**: 20MHz (20e6 Hz)
- **过采样倍数**: 20MHz / 3.7MHz ≈ 5.4倍
- **每个符号采样点数**: 5.4个采样点

#### 滚降因子影响
- **α = 0.1**: 窄带，符号间干扰小，但带宽利用率低
- **α = 0.35**: 标准值，平衡带宽利用率和符号间干扰
- **α = 0.5**: 宽带，带宽利用率高，但符号间干扰大

## 信号处理流程

1. **比特生成**: 生成随机比特序列
2. **QPSK映射**: 将比特映射到I/Q符号
3. **上采样**: 每个符号重复多个采样点
4. **升根余弦滤波**: 限制信号带宽
5. **载波调制**: 调制到指定载波频率
6. **噪声添加**: 可选添加AWGN噪声

## 输出信息

`generate_qpsk_signal()`返回的字典包含：

```python
{
    'signal': modulated_signal,        # 调制后的信号
    'i_symbols': i_symbols,           # I路符号
    'q_symbols': q_symbols,           # Q路符号
    'bits': bits,                     # 原始比特序列
    'symbol_rate': symbol_rate,       # 符号率
    'sample_rate': sample_rate,       # 采样率
    'bandwidth': bandwidth,           # 信号带宽
    'alpha': alpha,                   # 滚降因子
    'carrier_freq': carrier_freq,     # 载波频率
    'duration': duration,             # 持续时间
    'snr_db': snr_db,                # 信噪比
    'time': time                      # 时间轴
}
```

## 性能特点

### GPU加速优势
- **并行计算**: 利用GPU并行处理大量数据
- **内存优化**: 高效的内存管理和数据传输
- **计算速度**: 比CPU快数倍到数十倍

### 内存使用
- **信号长度**: 20MHz × 1ms = 20,000个采样点
- **GPU内存**: 约1MB (单精度浮点数)
- **可扩展性**: 支持更长的信号和更高的采样率

## 应用场景

1. **通信系统测试**: 生成标准QPSK测试信号
2. **信号处理算法验证**: 验证接收机算法性能
3. **频谱分析**: 分析QPSK信号的频谱特性
4. **噪声影响研究**: 研究不同信噪比下的信号质量
5. **滤波器设计**: 验证升根余弦滤波器的性能

## 运行示例

### 简单示例
```bash
python qpsk_simple_example.py
```

### 完整演示
```bash
python qpsk_signal_demo.py
```

## 注意事项

1. **GPU内存**: 确保GPU有足够的内存处理信号
2. **采样率**: 采样率应至少是带宽的2倍
3. **滚降因子**: α值影响信号带宽和符号间干扰
4. **信噪比**: 根据实际应用需求选择合适的信噪比
5. **信号长度**: 长信号需要更多GPU内存和处理时间

## 扩展功能

可以进一步扩展的功能：

1. **其他调制方式**: BPSK、8PSK、16QAM等
2. **编码功能**: 添加前向纠错编码
3. **多径信道**: 模拟多径衰落信道
4. **同步功能**: 添加载波和符号同步
5. **自适应调制**: 根据信道条件调整调制方式
