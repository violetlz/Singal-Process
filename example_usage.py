#!/usr/bin/env python3
"""
GPU信号处理使用示例
展示如何使用GPUSignalProcessor进行基本的FFT和STFT操作
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_processor import GPUSignalProcessor

def example_basic_fft():
    """基本FFT示例"""
    print("=== 基本FFT示例 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成简单的正弦波信号
    sample_rate = 44100
    duration = 0.1
    t = cp.linspace(0, duration, int(sample_rate * duration))
    frequency = 1000  # 1kHz
    signal = cp.sin(2 * cp.pi * frequency * t)

    # 执行FFT
    spectrum = processor.fft(signal)

    # 获取频率轴
    freq_axis = processor.get_frequency_axis(sample_rate, len(spectrum))

    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 时域信号
    ax1.plot(cp.asnumpy(t), cp.asnumpy(signal))
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title('时域信号')
    ax1.grid(True)

    # 频域信号
    ax2.plot(cp.asnumpy(freq_axis), cp.asnumpy(cp.abs(spectrum)))
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度')
    ax2.set_title('频域信号')
    ax2.grid(True)
    ax2.set_xlim(0, 5000)

    plt.tight_layout()
    plt.savefig('example_fft.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("FFT示例完成，图片保存为 example_fft.png")

def example_stft_analysis():
    """STFT分析示例"""
    print("\n=== STFT分析示例 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成调频信号
    sample_rate = 44100
    duration = 0.5
    t = cp.linspace(0, duration, int(sample_rate * duration))

    # 频率从500Hz变化到2000Hz
    frequency = 500 + 1500 * t / duration
    signal = cp.sin(2 * cp.pi * frequency * t)

    # 执行STFT
    window_size = 1024
    hop_size = 512
    stft_result = processor.stft(signal, window_size=window_size, hop_size=hop_size)

    # 获取时间和频率轴
    time_axis = processor.get_time_axis(len(signal), sample_rate, hop_size, window_size)
    freq_axis = processor.get_frequency_axis(sample_rate, stft_result.shape[1])

    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 时域信号
    ax1.plot(cp.asnumpy(t), cp.asnumpy(signal))
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title('调频信号 - 时域')
    ax1.grid(True)

    # STFT频谱图
    im = ax2.pcolormesh(cp.asnumpy(time_axis), cp.asnumpy(freq_axis),
                       cp.asnumpy(cp.abs(stft_result.T)),
                       shading='gouraud', cmap='viridis')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_title('STFT频谱图')
    plt.colorbar(im, ax=ax2, label='幅度')

    plt.tight_layout()
    plt.savefig('example_stft.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("STFT示例完成，图片保存为 example_stft.png")

def example_signal_reconstruction():
    """信号重构示例"""
    print("\n=== 信号重构示例 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成复合信号
    sample_rate = 44100
    duration = 0.1
    t = cp.linspace(0, duration, int(sample_rate * duration))

    # 包含多个频率成分的信号
    signal = (cp.sin(2 * cp.pi * 1000 * t) +  # 1kHz
              0.5 * cp.sin(2 * cp.pi * 3000 * t) +  # 3kHz
              0.3 * cp.sin(2 * cp.pi * 5000 * t))   # 5kHz

    # 添加一些噪声
    signal += 0.1 * cp.random.normal(0, 1, len(signal))

    # FFT重构
    spectrum = processor.fft(signal)
    reconstructed_fft = processor.ifft(spectrum)

    # STFT重构
    stft_result = processor.stft(signal, window_size=512, hop_size=256)
    reconstructed_stft = processor.istft(stft_result, hop_size=256, window_size=512)

    # 计算重构误差
    error_fft = cp.mean(cp.abs(signal - cp.real(reconstructed_fft)))
    error_stft = cp.mean(cp.abs(signal[:len(reconstructed_stft)] - reconstructed_stft))

    print(f"FFT重构误差: {error_fft:.2e}")
    print(f"STFT重构误差: {error_stft:.2e}")

    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # 原始信号
    axes[0, 0].plot(cp.asnumpy(t), cp.asnumpy(signal))
    axes[0, 0].set_title('原始信号')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].grid(True)

    # FFT重构信号
    axes[0, 1].plot(cp.asnumpy(t), cp.asnumpy(cp.real(reconstructed_fft)))
    axes[0, 1].set_title('FFT重构信号')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].grid(True)

    # STFT重构信号
    t_stft = cp.linspace(0, len(reconstructed_stft) / sample_rate, len(reconstructed_stft))
    axes[1, 0].plot(cp.asnumpy(t_stft), cp.asnumpy(reconstructed_stft))
    axes[1, 0].set_title('STFT重构信号')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].grid(True)

    # 重构误差
    axes[1, 1].plot(cp.asnumpy(t), cp.asnumpy(signal - cp.real(reconstructed_fft)),
                   label='FFT误差', alpha=0.7)
    axes[1, 1].plot(cp.asnumpy(t_stft),
                   cp.asnumpy(signal[:len(reconstructed_stft)] - reconstructed_stft),
                   label='STFT误差', alpha=0.7)
    axes[1, 1].set_title('重构误差')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('example_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("信号重构示例完成，图片保存为 example_reconstruction.png")

def example_window_comparison():
    """窗口函数对比示例"""
    print("\n=== 窗口函数对比示例 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成测试信号
    sample_rate = 44100
    duration = 0.2
    t = cp.linspace(0, duration, int(sample_rate * duration))

    # 包含多个频率成分的信号
    signal = (cp.sin(2 * cp.pi * 1000 * t) +
              0.5 * cp.sin(2 * cp.pi * 3000 * t) +
              0.3 * cp.sin(2 * cp.pi * 5000 * t))

    # 不同窗口函数
    windows = ['hann', 'hamming', 'blackman', 'rect']
    window_names = ['汉宁窗', '海明窗', '布莱克曼窗', '矩形窗']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, (window, name) in enumerate(zip(windows, window_names)):
        row, col = i // 2, i % 2

        # 计算STFT
        stft_result = processor.stft(signal, window_size=1024,
                                   hop_size=512, window=window)

        # 获取时间和频率轴
        time_axis = processor.get_time_axis(len(signal), sample_rate, 512, 1024)
        freq_axis = processor.get_frequency_axis(sample_rate, stft_result.shape[1])

        # 绘制频谱图
        im = axes[row, col].pcolormesh(cp.asnumpy(time_axis), cp.asnumpy(freq_axis),
                                      cp.asnumpy(cp.abs(stft_result.T)),
                                      shading='gouraud', cmap='viridis')
        axes[row, col].set_title(f'{name} STFT')
        axes[row, col].set_xlabel('时间 (s)')
        axes[row, col].set_ylabel('频率 (Hz)')
        plt.colorbar(im, ax=axes[row, col], label='幅度')

    plt.tight_layout()
    plt.savefig('example_windows.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("窗口函数对比示例完成，图片保存为 example_windows.png")

def main():
    """主函数"""
    print("GPU信号处理使用示例")
    print("=" * 40)

    try:
        # 检查GPU可用性
        print(f"检测到 {cp.cuda.runtime.getDeviceCount()} 个GPU设备")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"GPU {i}: {props['name'].decode()}")

        # 运行示例
        example_basic_fft()
        example_stft_analysis()
        example_signal_reconstruction()
        example_window_comparison()

        print("\n所有示例完成！生成的图片文件：")
        print("- example_fft.png")
        print("- example_stft.png")
        print("- example_reconstruction.png")
        print("- example_windows.png")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保已正确安装CuPy和CUDA")

if __name__ == "__main__":
    main()
