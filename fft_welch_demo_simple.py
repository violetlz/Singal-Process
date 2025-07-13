#!/usr/bin/env python3
"""
FFT和功率谱密度估计（Welch方法）测试和效果展示 - 简化版
使用模块化的GPU信号处理库
"""

try:
    import cupy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from src import (
        GPUSignalProcessor, FFTProcessor, SpectralAnalyzer,
        SignalVisualizer, PerformanceMonitor
    )
    print("所有依赖包导入成功！")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖包：pip install -r requirements.txt")
    exit(1)

def generate_test_signals(sample_rate=44100, duration=1.0):
    """生成测试信号"""
    signal_length = int(sample_rate * duration)
    t = cp.linspace(0, duration, signal_length)

    signals = {}

    # 单频正弦信号
    signals['single_tone'] = cp.sin(2 * cp.pi * 1000 * t)

    # 多频复合信号
    signals['multi_tone'] = (
        cp.sin(2 * cp.pi * 500 * t) +    # 500 Hz
        cp.sin(2 * cp.pi * 1500 * t) +   # 1500 Hz
        cp.sin(2 * cp.pi * 2500 * t)     # 2500 Hz
    )

    # 带噪声的信号
    noise = cp.random.normal(0, 0.3, signal_length)
    signals['noisy_signal'] = signals['multi_tone'] + noise

    return signals, t, sample_rate

def test_fft_basic():
    """基础FFT测试"""
    print("=" * 50)
    print("基础FFT测试")
    print("=" * 50)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)

    # 生成测试信号
    sample_rate = 44100
    duration = 0.1  # 0.1秒
    signal_length = int(sample_rate * duration)
    t = cp.linspace(0, duration, signal_length)

    # 创建1kHz正弦信号
    signal = cp.sin(2 * cp.pi * 1000 * t)

    print(f"信号长度: {signal_length}")
    print(f"采样率: {sample_rate} Hz")
    print(f"信号频率: 1000 Hz")

    # 执行FFT
    start_time = time.time()
    spectrum = fft_processor.fft(signal)
    fft_time = time.time() - start_time

    # 获取频率轴
    freq_axis = fft_processor.get_frequency_axis(sample_rate, len(spectrum))

    # 计算功率谱
    power_spectrum = cp.abs(spectrum) ** 2

    # 找到峰值
    peak_index = cp.argmax(power_spectrum)
    peak_frequency = freq_axis[peak_index]
    peak_power = power_spectrum[peak_index]

    print(f"FFT执行时间: {fft_time*1000:.3f} ms")
    print(f"检测到的峰值频率: {peak_frequency:.1f} Hz")
    print(f"峰值功率: {peak_power:.2f}")

    # 简单可视化
    plt.figure(figsize=(12, 4))

    # 时域信号
    plt.subplot(1, 2, 1)
    plt.plot(cp.asnumpy(t[:1000]), cp.asnumpy(signal[:1000]))  # 只显示前1000个点
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.title('时域信号 (1kHz正弦波)')
    plt.grid(True)

    # 频域信号
    plt.subplot(1, 2, 2)
    plt.plot(cp.asnumpy(freq_axis), cp.asnumpy(power_spectrum))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱')
    plt.title('频域信号')
    plt.grid(True)
    plt.xlim(0, 5000)  # 限制显示范围

    plt.tight_layout()
    plt.savefig('fft_basic_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    return signal, spectrum, freq_axis, power_spectrum

def test_welch_psd_basic():
    """基础Welch PSD测试"""
    print("\n" + "=" * 50)
    print("基础Welch PSD测试")
    print("=" * 50)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # 生成带噪声的多频信号
    signals, t, sample_rate = generate_test_signals()
    signal = signals['noisy_signal']

    print(f"信号长度: {len(signal)}")
    print(f"采样率: {sample_rate} Hz")
    print(f"信号包含频率: 500 Hz, 1500 Hz, 2500 Hz + 噪声")

    # 计算Welch PSD
    start_time = time.time()
    freq_psd, psd = spectral_analyzer.welch_psd(signal, sample_rate, 1024, 512)
    welch_time = time.time() - start_time

    # 频谱特征
    centroid = spectral_analyzer.estimate_spectral_centroid(psd, freq_psd)
    bandwidth = spectral_analyzer.estimate_spectral_bandwidth(psd, freq_psd)
    entropy = spectral_analyzer.compute_spectral_entropy(psd)

    print(f"Welch PSD执行时间: {welch_time*1000:.3f} ms")
    print(f"频谱中心: {centroid:.2f} Hz")
    print(f"频谱带宽: {bandwidth:.2f} Hz")
    print(f"频谱熵: {entropy:.2f}")

    # 峰值检测
    peaks, peak_values = spectral_analyzer.find_peaks(psd, threshold=0.1)
    print(f"检测到 {len(peaks)} 个峰值")

    for i, (peak_idx, peak_val) in enumerate(zip(peaks, peak_values)):
        peak_freq = freq_psd[peak_idx]
        print(f"  峰值 {i+1}: {peak_freq:.1f} Hz (功率: {peak_val:.4f})")

    # 可视化
    plt.figure(figsize=(12, 4))

    # 时域信号
    plt.subplot(1, 2, 1)
    plt.plot(cp.asnumpy(t[:2000]), cp.asnumpy(signal[:2000]))  # 只显示前2000个点
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.title('时域信号 (多频+噪声)')
    plt.grid(True)

    # Welch PSD
    plt.subplot(1, 2, 2)
    plt.semilogy(cp.asnumpy(freq_psd), cp.asnumpy(psd))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.title('Welch功率谱密度')
    plt.grid(True)

    # 标记峰值
    if len(peaks) > 0:
        peak_freqs = cp.asnumpy(freq_psd[peaks])
        peak_powers = cp.asnumpy(psd[peaks])
        plt.plot(peak_freqs, peak_powers, 'ro', markersize=8, label='检测到的峰值')
        plt.legend()

    # 标记期望的频率
    expected_freqs = [500, 1500, 2500]
    for freq in expected_freqs:
        plt.axvline(x=freq, color='g', linestyle='--', alpha=0.7, label=f'{freq} Hz')

    plt.tight_layout()
    plt.savefig('welch_psd_basic_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    return signal, freq_psd, psd, peaks

def compare_fft_vs_welch():
    """比较FFT和Welch方法"""
    print("\n" + "=" * 50)
    print("FFT vs Welch方法比较")
    print("=" * 50)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # 生成测试信号
    signals, t, sample_rate = generate_test_signals()
    signal = signals['noisy_signal']

    print(f"测试信号: 多频信号 + 噪声")
    print(f"信号长度: {len(signal)}")
    print(f"采样率: {sample_rate} Hz")

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
    plt.xlim(0, 5000)

    # Welch结果
    plt.subplot(1, 2, 2)
    plt.semilogy(cp.asnumpy(freq_psd), cp.asnumpy(psd))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.title('Welch功率谱密度')
    plt.grid(True)
    plt.xlim(0, 5000)

    plt.tight_layout()
    plt.savefig('fft_vs_welch_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 性能对比
    print(f"\n性能对比:")
    print(f"  FFT方法: {fft_time*1000:.3f} ms")
    print(f"  Welch方法: {welch_time*1000:.3f} ms")
    print(f"  时间比: {welch_time/fft_time:.2f}x")

    # 质量对比
    print(f"\n质量对比:")
    print(f"  FFT频率分辨率: {sample_rate/len(signal):.2f} Hz")
    print(f"  Welch频率分辨率: {sample_rate/1024:.2f} Hz")
    print(f"  分辨率比: {(sample_rate/len(signal))/(sample_rate/1024):.2f}x")

def test_performance_scaling():
    """测试性能随信号长度的变化"""
    print("\n" + "=" * 50)
    print("性能随信号长度的变化")
    print("=" * 50)

    # 初始化
    gpu_processor = GPUSignalProcessor()
    fft_processor = FFTProcessor(gpu_processor)
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # 测试不同信号长度
    signal_lengths = [1024, 4096, 16384, 65536, 262144]
    sample_rate = 44100

    print(f"{'信号长度':<12} {'FFT时间(ms)':<15} {'Welch时间(ms)':<15}")
    print("-" * 50)

    fft_times = []
    welch_times = []

    for length in signal_lengths:
        # 生成测试信号
        t = cp.linspace(0, length/sample_rate, length)
        signal = cp.sin(2 * cp.pi * 1000 * t) + cp.random.normal(0, 0.1, length)

        # FFT测试
        start_time = time.time()
        spectrum = fft_processor.fft(signal)
        fft_time = time.time() - start_time

        # Welch测试
        start_time = time.time()
        freq_psd, psd = spectral_analyzer.welch_psd(signal, sample_rate, 1024, 512)
        welch_time = time.time() - start_time

        fft_times.append(fft_time * 1000)
        welch_times.append(welch_time * 1000)

        print(f"{length:<12} {fft_time*1000:<15.3f} {welch_time*1000:<15.3f}")

    # 绘制性能图
    plt.figure(figsize=(10, 6))

    plt.semilogx(signal_lengths, fft_times, 'b-o', label='FFT', linewidth=2, markersize=8)
    plt.semilogx(signal_lengths, welch_times, 'r-s', label='Welch PSD', linewidth=2, markersize=8)
    plt.xlabel('信号长度')
    plt.ylabel('执行时间 (ms)')
    plt.title('FFT vs Welch PSD 性能对比')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("FFT和功率谱密度估计（Welch方法）测试和效果展示")
    print("=" * 80)

    try:
        # 1. 基础FFT测试
        test_fft_basic()

        # 2. 基础Welch PSD测试
        test_welch_psd_basic()

        # 3. FFT vs Welch方法比较
        compare_fft_vs_welch()

        # 4. 性能随信号长度的变化
        test_performance_scaling()

        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("生成的图片文件:")
        print("- fft_basic_test.png")
        print("- welch_psd_basic_test.png")
        print("- fft_vs_welch_comparison.png")
        print("- performance_scaling.png")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
