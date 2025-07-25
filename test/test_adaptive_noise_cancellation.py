#!/usr/bin/env python3
"""
自适应噪声消除测试
使用5MHz带宽QPSK信号，测试AdaptiveFilter的adaptive_noise_cancellation功能
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from src import GPUSignalProcessor, SignalGenerator
from src.filters.adaptive_filter import AdaptiveFilter
from src.transforms.fft_processor import FFTProcessor


# 初始化
gpu_processor = GPUSignalProcessor(gpu_id=0)
signal_generator = SignalGenerator(gpu_processor)
adaptive_filter = AdaptiveFilter(gpu_processor)

# 生成5MHz带宽QPSK信号
qpsk_result = signal_generator.generate_qpsk_signal(
    bandwidth=5e6,
    sample_rate=20e6,
    duration=0.001,
    alpha=0.35,
    snr_db=None  # 先不加噪声
)
clean_signal = qpsk_result['signal']
sample_rate = qpsk_result['sample_rate']
time = qpsk_result['time']

# 生成相关噪声
noise_base = cp.random.normal(0, 0.2, len(clean_signal))
primary_signal = clean_signal + noise_base
reference_signal = noise_base  # 相关噪声作为参考

# 自适应噪声消除
filtered_signal, filtered_reference = adaptive_filter.adaptive_noise_cancellation(
    primary_signal=cp.asarray(primary_signal),
    reference_signal=cp.asarray(reference_signal),
    filter_length=64,
    mu=0.01
)

# 转为numpy便于绘图
primary_signal_np = cp.asnumpy(primary_signal)
reference_signal_np = cp.asnumpy(reference_signal)
filtered_signal_np = cp.asnumpy(filtered_signal)
clean_signal_np = cp.asnumpy(clean_signal)
time_np = cp.asnumpy(time)

# 绘图
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(time_np, clean_signal_np, label='原始信号')
plt.title('原始QPSK信号')
plt.ylabel('幅度')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time_np, primary_signal_np, label='带噪主信号', color='orange')
plt.title('带噪主信号（信号+噪声）')
plt.ylabel('幅度')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time_np, reference_signal_np, label='参考噪声', color='green')
plt.title('参考噪声信号')
plt.ylabel('幅度')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time_np, filtered_signal_np, label='自适应滤波后信号', color='red')
plt.title('自适应噪声消除后信号')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

plt.tight_layout()
plt.savefig('test_adaptive_noise_cancellation.png', dpi=300, bbox_inches='tight')
plt.show()

# 初始化FFT处理器
fft_processor = FFTProcessor(gpu_processor)

# 计算正频率部分的频谱
spectrum_clean = fft_processor.rfft(clean_signal)
spectrum_primary = fft_processor.rfft(primary_signal)
spectrum_filtered = fft_processor.rfft(filtered_signal)

freq_axis = cp.fft.rfftfreq(len(clean_signal), d=1/sample_rate)
freq_axis_np = cp.asnumpy(freq_axis)

# 绘制频域对比图
plt.figure(figsize=(12, 6))
plt.semilogy(freq_axis_np, cp.asnumpy(cp.abs(spectrum_primary)), label='带噪主信号')
plt.semilogy(freq_axis_np, cp.asnumpy(cp.abs(spectrum_filtered)), label='自适应滤波后信号')
plt.semilogy(freq_axis_np, cp.asnumpy(cp.abs(spectrum_clean)), label='原始信号')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度（对数）')
plt.title('信号频谱对比（正频率部分）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_adaptive_noise_cancellation_freq.png', dpi=300, bbox_inches='tight')
plt.show()

# --- LMS参数网格测试 ---
lms_filter_lengths = [64, 128, 256]
lms_mus = [0.01, 0.005, 0.001]

for flen in lms_filter_lengths:
    for mu in lms_mus:
        name = f'LMS-{flen}-{mu}'
        filtered_signal, _ = adaptive_filter.lms_filter(
            input_signal=cp.asarray(reference_signal),
            desired_signal=cp.asarray(primary_signal),
            filter_length=flen,
            mu=mu
        )
        spectrum = np.abs(cp.asnumpy(fft_processor.rfft(filtered_signal)))
        plt.figure(figsize=(12, 6))
        plt.semilogy(freq_axis_np, cp.asnumpy(cp.abs(spectrum_primary)), label='带噪主信号')
        plt.semilogy(freq_axis_np, spectrum, label=f'{name} 滤波后')
        plt.semilogy(freq_axis_np, cp.asnumpy(cp.abs(spectrum_clean)), label='原始信号')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度（对数）')
        plt.title(f'{name} 自适应滤波频谱对比（正频率部分）')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'test_adaptive_noise_cancellation_freq_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()

print('测试完成，图片保存为 test_adaptive_noise_cancellation.png') 