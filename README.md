# GPU信号处理库

基于CuPy的GPU加速信号处理工具包，提供高性能的信号处理功能。

## 项目结构

```
Signal_Process_On_Orin/
├── src/                          # 主要源代码包
│   ├── __init__.py              # 包初始化文件
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── gpu_processor.py     # GPU处理器核心类
│   │   └── signal_generator.py  # 信号生成器
│   ├── transforms/              # 变换模块
│   │   ├── __init__.py
│   │   ├── fft_processor.py     # FFT处理器
│   │   └── stft_processor.py    # STFT处理器
│   ├── analysis/                # 分析模块
│   │   ├── __init__.py
│   │   ├── spectral_analyzer.py # 频谱分析器
│   │   └── feature_extractor.py # 特征提取器
│   ├── filters/                 # 滤波器模块
│   │   ├── __init__.py
│   │   ├── filter_bank.py       # 滤波器组
│   │   └── adaptive_filter.py   # 自适应滤波器
│   └── utils/                   # 工具模块
│       ├── __init__.py
│       ├── visualization.py     # 可视化工具
│       └── performance_monitor.py # 性能监控工具
├── gpu_signal_processor.py      # 原始单文件版本
├── demo_visualization.py        # 可视化演示
├── example_usage.py             # 基础使用示例
├── modular_example.py           # 模块化使用示例
├── test_gpu_signal.py           # 测试文件
├── setup.py                     # 安装脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 功能特性

### 核心模块 (core)
- **GPUSignalProcessor**: GPU信号处理核心类，提供基础GPU操作
- **SignalGenerator**: 信号生成器，支持各种测试信号生成

### 变换模块 (transforms)
- **FFTProcessor**: 快速傅里叶变换处理器
  - FFT/IFFT (复数)
  - RFFT/IRFFT (实数)
  - 频率轴生成
- **STFTProcessor**: 短时傅里叶变换处理器
  - STFT/ISTFT
  - 多种窗口函数支持
  - 时频图计算

### 分析模块 (analysis)
- **SpectralAnalyzer**: 频谱分析器
  - Welch功率谱密度估计
  - 频谱峰值检测
  - 频谱中心、带宽、熵计算
- **FeatureExtractor**: 特征提取器
  - 时域统计特征
  - 频域特征
  - 频带功率比例

### 滤波器模块 (filters)
- **FilterBank**: 滤波器组
  - 巴特沃斯滤波器（低通、高通、带通、带阻）
  - 切比雪夫滤波器
  - 移动平均滤波器
  - 中值滤波器
- **AdaptiveFilter**: 自适应滤波器
  - LMS算法
  - 自适应噪声消除

### 工具模块 (utils)
- **SignalVisualizer**: 信号可视化工具
  - 时域/频域信号绘制
  - STFT频谱图
  - 滤波器频率响应
  - 特征对比热力图
- **PerformanceMonitor**: 性能监控工具
  - GPU/CPU性能对比
  - 内存使用监控
  - 性能基准测试

## 安装

### 系统要求
- Python 3.7+
- CUDA 11.0+ (用于GPU加速)
- NVIDIA GPU (支持CUDA)

### 安装步骤

1. 克隆仓库：
```bash
git clone <repository-url>
cd Signal_Process_On_Orin
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行安装脚本：
```bash
python setup.py
```

## 快速开始

### 基础使用

```python
import cupy as cp
from src import GPUSignalProcessor, FFTProcessor

# 初始化GPU处理器
gpu_processor = GPUSignalProcessor()

# 创建测试信号
signal = cp.sin(2 * cp.pi * 1000 * cp.linspace(0, 1, 44100))

# FFT处理
fft_processor = FFTProcessor(gpu_processor)
spectrum = fft_processor.fft(signal)
```

### 模块化使用

```python
from src import (
    GPUSignalProcessor, FFTProcessor, STFTProcessor,
    SpectralAnalyzer, FeatureExtractor, FilterBank,
    SignalVisualizer, PerformanceMonitor
)

# 初始化各个模块
gpu_processor = GPUSignalProcessor()
fft_processor = FFTProcessor(gpu_processor)
stft_processor = STFTProcessor(gpu_processor)
spectral_analyzer = SpectralAnalyzer(gpu_processor)
feature_extractor = FeatureExtractor(gpu_processor)
filter_bank = FilterBank(gpu_processor)
visualizer = SignalVisualizer(gpu_processor)
performance_monitor = PerformanceMonitor(gpu_processor)

# 生成信号
signal = cp.sin(2 * cp.pi * 1000 * cp.linspace(0, 1, 44100))

# 频谱分析
freq_psd, psd = spectral_analyzer.welch_psd(signal, 44100)
centroid = spectral_analyzer.estimate_spectral_centroid(psd, freq_psd)

# 特征提取
features = feature_extractor.extract_time_domain_features(signal)

# 滤波
filtered_signal = filter_bank.butterworth_lowpass(signal, 500, 44100)

# 可视化
visualizer.plot_signal_and_spectrum(signal, 44100, "信号分析")
```

## 示例

### 运行基础示例
```bash
python example_usage.py
```

### 运行模块化示例
```bash
python modular_example.py
```

### 运行可视化演示
```bash
python demo_visualization.py
```

### 运行测试
```bash
python test_gpu_signal.py
```

## 性能特性

- **GPU加速**: 利用CuPy进行GPU加速计算
- **内存优化**: 高效的内存管理和数据传输
- **模块化设计**: 支持独立使用各个功能模块
- **可扩展性**: 易于添加新的信号处理算法

## 主要优势

1. **高性能**: GPU加速，比CPU快数倍到数十倍
2. **模块化**: 清晰的模块结构，便于维护和扩展
3. **易用性**: 简单的API接口，快速上手
4. **完整性**: 涵盖信号处理的主要功能
5. **可视化**: 内置丰富的可视化工具

## 支持的信号处理功能

### 变换
- 快速傅里叶变换 (FFT/IFFT)
- 短时傅里叶变换 (STFT/ISTFT)
- 实数FFT (RFFT/IRFFT)

### 分析
- 功率谱密度估计 (Welch方法)
- 频谱峰值检测
- 频谱特征提取 (中心、带宽、熵)
- 时域统计特征
- 频带功率分析

### 滤波
- 巴特沃斯滤波器
- 切比雪夫滤波器
- 移动平均滤波器
- 中值滤波器
- LMS自适应滤波器

### 可视化
- 时域/频域信号绘制
- STFT频谱图
- 滤波器频率响应
- 特征对比图
- 性能分析图

## 故障排除

### 常见问题

1. **CuPy安装失败**
   - 确保CUDA版本兼容
   - 检查GPU驱动版本
   - 参考CuPy官方安装指南

2. **GPU内存不足**
   - 减少信号长度
   - 使用批处理模式
   - 检查GPU内存使用情况

3. **性能不理想**
   - 确保信号长度足够大
   - 检查GPU利用率
   - 使用性能监控工具分析

### 调试技巧

- 使用`PerformanceMonitor`监控性能
- 检查GPU内存使用情况
- 对比CPU和GPU结果验证正确性

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。
