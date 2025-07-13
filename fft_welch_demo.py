#!/usr/bin/env python3
"""
FFT和功率谱密度估计（Welch方法）测试和效果展示
使用模块化的GPU信号处理库
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from src import (
    GPUSignalProcessor, FFTProcessor, SpectralAnalyzer,
    SignalVisualizer, PerformanceMonitor
)

def generate_test_signals(sample_rate=44100, duration=2.0):
    """
    生成各种测试信号

    Args:
        sample_rate: 采样率
        duration: 信号持续时间

    Returns:
        测试信号字典
    """
    signal_length = int(sample_rate * duration)
    t = cp.linspace(0, duration, signal_length)

    signals = {}

    # 1. 单频正弦信号
    signals['single_tone'] = cp.sin(2 * cp.pi * 1000 * t)

    # 2. 多频复合信号
    signals['multi_tone'] = (
        cp.sin(2 * cp.pi * 500 * t) +    # 500 Hz
        cp.sin(2 * cp.pi * 1500 * t) +   # 1500 Hz
        cp.sin(2 * cp.pi * 2500 * t)     # 2500 Hz
    )

    # 3. 调频信号
    fm_freq = 1000 + 500 * cp.sin(2 * cp.pi * 2 * t)  # 1000±500 Hz
    signals['fm_signal'] = cp.sin(2 * cp.pi * fm_freq * t)

    # 4. 带噪声的信号
    noise = cp.random.normal(0, 0.3, signal_length)
    signals['noisy_signal'] = signals['multi_tone'] + noise

    # 5. 脉冲信号
    pulse_signal = cp.zeros(signal_length)
    pulse_indices = cp.arange(0, signal_length, sample_rate // 10)  # 每0.1秒一个脉冲
    pulse_signal[pulse_indices] = 1.0
    signals['pulse_signal'] = pulse_signal

    # 6. 随机信号
    signals['random_signal'] = cp.random.normal(0, 1, signal_length)

    return signals, t, sample_rate

def test_fft_performance():
    """测试FFT性能"""
    print("=" * 60)
    print("FFT性能测试")
    print("=" * 60)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)
    performance_monitor = PerformanceMonitor(gpu_processor)

    # 测试不同信号长度
    signal_lengths = [1024, 4096, 16384, 65536, 262144, 1048576]
    sample_rate = 44100

    print(f"{'信号长度':<12} {'GPU时间(ms)':<15} {'CPU时间(ms)':<15} {'加速比':<10}")
    print("-" * 60)

    gpu_times = []
    cpu_times = []
    speedups = []

    for length in signal_lengths:
        # 生成测试信号
        signal = cp.sin(2 * cp.pi * 1000 * cp.linspace(0, 1, length))

        # GPU FFT
        def gpu_fft(sig):
            return fft_processor.fft(sig)

        # CPU FFT
        def cpu_fft(sig):
            return np.fft.fft(cp.asnumpy(sig))

        # 性能测试
        gpu_stats = performance_monitor.benchmark_function(gpu_fft, signal, num_runs=10)
        cpu_stats = performance_monitor.benchmark_function(cpu_fft, signal, num_runs=10)

        gpu_time_ms = gpu_stats['mean_time'] * 1000
        cpu_time_ms = cpu_stats['mean_time'] * 1000
        if gpu_time_ms == 0:
            speedup = float('inf')
            speedup_str = "N/A"
        else:
            speedup = cpu_time_ms / gpu_time_ms
            speedup_str = f"{speedup:<10.2f}x"

        gpu_times.append(gpu_time_ms)
        cpu_times.append(cpu_time_ms)
        speedups.append(speedup)

        print(f"{length:<12} {gpu_time_ms:<15.3f} {cpu_time_ms:<15.3f} {speedup_str}")

    # 绘制性能对比图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.semilogx(signal_lengths, gpu_times, 'b-o', label='GPU', linewidth=2, markersize=8)
    plt.semilogx(signal_lengths, cpu_times, 'r-s', label='CPU', linewidth=2, markersize=8)
    plt.xlabel('信号长度')
    plt.ylabel('执行时间 (ms)')
    plt.title('FFT性能对比')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.semilogx(signal_lengths, speedups, 'g-^', linewidth=2, markersize=8)
    plt.xlabel('信号长度')
    plt.ylabel('加速比')
    plt.title('GPU加速比')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fft_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_fft_analysis():
    """演示FFT分析效果"""
    print("\n" + "=" * 60)
    print("FFT分析效果演示")
    print("=" * 60)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)
    visualizer = SignalVisualizer(gpu_processor)

    # 生成测试信号
    signals, t, sample_rate = generate_test_signals()

    # 分析每种信号
    for signal_name, signal in signals.items():
        print(f"\n分析信号: {signal_name}")

        # 执行FFT
        start_time = time.time()
        spectrum = fft_processor.fft(signal)
        fft_time = time.time() - start_time

        # 获取频率轴
        freq_axis = fft_processor.get_frequency_axis(sample_rate, len(spectrum))

        # 计算功率谱
        power_spectrum = cp.abs(spectrum) ** 2

        # 找到主要频率成分
        peak_indices = cp.argsort(power_spectrum)[-5:]  # 前5个峰值
        peak_frequencies = freq_axis[peak_indices]
        peak_powers = power_spectrum[peak_indices]

        print(f"  FFT执行时间: {fft_time*1000:.3f} ms")
        print(f"  主要频率成分:")
        for i, (freq, power) in enumerate(zip(peak_frequencies, peak_powers)):
            print(f"    {i+1}. {freq:.1f} Hz (功率: {power:.2f})")

        # 可视化
        visualizer.plot_signal_and_spectrum(
            signal, sample_rate, f"{signal_name} - FFT分析",
            f"fft_analysis_{signal_name}.png"
        )

def demonstrate_welch_psd():
    """演示Welch功率谱密度估计"""
    print("\n" + "=" * 60)
    print("Welch功率谱密度估计演示")
    print("=" * 60)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    spectral_analyzer = SpectralAnalyzer(gpu_processor)
    visualizer = SignalVisualizer(gpu_processor)

    # 生成测试信号
    signals, t, sample_rate = generate_test_signals()

    # 测试不同窗口大小
    window_sizes = [256, 512, 1024, 2048]
    hop_sizes = [w//2 for w in window_sizes]

    # 分析带噪声的信号
    signal = signals['noisy_signal']

    print(f"\n分析信号: noisy_signal")
    print(f"信号长度: {len(signal)}")
    print(f"采样率: {sample_rate} Hz")

    # 比较不同窗口大小的效果
    plt.figure(figsize=(15, 10))

    for i, (window_size, hop_size) in enumerate(zip(window_sizes, hop_sizes)):
        print(f"\n窗口大小: {window_size}, 跳跃大小: {hop_size}")

        # 计算Welch PSD
        start_time = time.time()
        freq_psd, psd = spectral_analyzer.welch_psd(
            signal, sample_rate, window_size, hop_size, 'hann'
        )
        welch_time = time.time() - start_time

        # 频谱特征
        centroid = spectral_analyzer.estimate_spectral_centroid(psd, freq_psd)
        bandwidth = spectral_analyzer.estimate_spectral_bandwidth(psd, freq_psd)
        entropy = spectral_analyzer.compute_spectral_entropy(psd)

        print(f"  Welch PSD执行时间: {welch_time*1000:.3f} ms")
        print(f"  频谱中心: {centroid:.2f} Hz")
        print(f"  频谱带宽: {bandwidth:.2f} Hz")
        print(f"  频谱熵: {entropy:.2f}")

        # 峰值检测
        peaks, peak_values = spectral_analyzer.find_peaks(psd, threshold=0.1)
        print(f"  检测到 {len(peaks)} 个峰值")

        # 绘制子图
        plt.subplot(2, 2, i+1)
        plt.semilogy(cp.asnumpy(freq_psd), cp.asnumpy(psd))
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title(f'Welch PSD (窗口={window_size}, 跳跃={hop_size})')
        plt.grid(True)

        # 标记峰值
        if len(peaks) > 0:
            peak_freqs = cp.asnumpy(freq_psd[peaks])
            peak_powers = cp.asnumpy(psd[peaks])
            plt.plot(peak_freqs, peak_powers, 'ro', markersize=8)

    plt.tight_layout()
    plt.savefig('welch_psd_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_fft_vs_welch():
    """比较FFT和Welch方法"""
    print("\n" + "=" * 60)
    print("FFT vs Welch方法比较")
    print("=" * 60)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # 生成测试信号
    signals, t, sample_rate = generate_test_signals()
    signal = signals['noisy_signal']

    # FFT方法
    print("\n1. FFT方法")
    start_time = time.time()
    spectrum = fft_processor.fft(signal)
    freq_axis = fft_processor.get_frequency_axis(sample_rate, len(spectrum))
    power_spectrum = cp.abs(spectrum) ** 2
    fft_time = time.time() - start_time

    print(f"  执行时间: {fft_time*1000:.3f} ms")
    print(f"  频率分辨率: {sample_rate/len(signal):.2f} Hz")

    # Welch方法
    print("\n2. Welch方法")
    start_time = time.time()
    freq_psd, psd = spectral_analyzer.welch_psd(signal, sample_rate, 1024, 512)
    welch_time = time.time() - start_time

    print(f"  执行时间: {welch_time*1000:.3f} ms")
    print(f"  频率分辨率: {sample_rate/1024:.2f} Hz")

    # 可视化比较
    plt.figure(figsize=(15, 6))

    # FFT结果
    plt.subplot(1, 2, 1)
    plt.semilogy(cp.asnumpy(freq_axis), cp.asnumpy(power_spectrum))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱')
    plt.title('FFT功率谱')
    plt.grid(True)

    # Welch结果
    plt.subplot(1, 2, 2)
    plt.semilogy(cp.asnumpy(freq_psd), cp.asnumpy(psd))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.title('Welch功率谱密度')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fft_vs_welch_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 性能对比
    print(f"\n性能对比:")
    print(f"  FFT方法: {fft_time*1000:.3f} ms")
    print(f"  Welch方法: {welch_time*1000:.3f} ms")
    if fft_time == 0:
        time_ratio_str = "N/A"
    else:
        time_ratio_str = f"{welch_time/fft_time:.2f}x"
    print(f"  时间比: {time_ratio_str}")

def test_different_window_functions():
    """测试不同窗口函数对Welch方法的影响"""
    print("\n" + "=" * 60)
    print("不同窗口函数对Welch方法的影响")
    print("=" * 60)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # 生成测试信号
    signals, t, sample_rate = generate_test_signals()
    signal = signals['multi_tone']  # 使用多频信号

    # 测试不同窗口函数
    windows = ['hann', 'hamming', 'blackman', 'rect']
    window_names = ['汉宁窗', '海明窗', '布莱克曼窗', '矩形窗']

    plt.figure(figsize=(15, 10))

    for i, (window, window_name) in enumerate(zip(windows, window_names)):
        print(f"\n窗口函数: {window_name} ({window})")

        # 计算Welch PSD
        freq_psd, psd = spectral_analyzer.welch_psd(
            signal, sample_rate, 1024, 512, window
        )

        # 频谱特征
        centroid = spectral_analyzer.estimate_spectral_centroid(psd, freq_psd)
        bandwidth = spectral_analyzer.estimate_spectral_bandwidth(psd, freq_psd)
        entropy = spectral_analyzer.compute_spectral_entropy(psd)

        print(f"  频谱中心: {centroid:.2f} Hz")
        print(f"  频谱带宽: {bandwidth:.2f} Hz")
        print(f"  频谱熵: {entropy:.2f}")

        # 绘制结果
        plt.subplot(2, 2, i+1)
        plt.semilogy(cp.asnumpy(freq_psd), cp.asnumpy(psd))
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title(f'{window_name} ({window})')
        plt.grid(True)

        # 标记期望的频率成分
        expected_freqs = [500, 1500, 2500]
        for freq in expected_freqs:
            plt.axvline(x=freq, color='r', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('window_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("FFT和功率谱密度估计（Welch方法）测试和效果展示")
    print("=" * 80)

    try:
        # 1. FFT性能测试
        test_fft_performance()

        # 2. FFT分析效果演示
        demonstrate_fft_analysis()

        # 3. Welch功率谱密度估计演示
        demonstrate_welch_psd()

        # 4. FFT vs Welch方法比较
        compare_fft_vs_welch()

        # 5. 不同窗口函数测试
        test_different_window_functions()

        print("\n" + "=" * 80)
        print("所有测试和演示完成！")
        print("生成的图片文件:")
        print("- fft_performance_comparison.png")
        print("- fft_analysis_*.png (各种信号的FFT分析)")
        print("- welch_psd_comparison.png")
        print("- fft_vs_welch_comparison.png")
        print("- window_function_comparison.png")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
