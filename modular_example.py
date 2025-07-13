#!/usr/bin/env python3
"""
模块化GPU信号处理库使用示例
展示如何使用src包中的各个模块
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from src import (
    GPUSignalProcessor, SignalGenerator, FFTProcessor, STFTProcessor,
    SpectralAnalyzer, FeatureExtractor, FilterBank, AdaptiveFilter,
    SignalVisualizer, PerformanceMonitor
)

def main():
    """主函数"""
    print("GPU信号处理模块化库示例")
    print("=" * 50)

    # 初始化GPU处理器
    gpu_processor = GPUSignalProcessor()
    signal_generator = SignalGenerator(gpu_processor)

    # 生成测试信号
    sample_rate = 44100
    duration = 1.0  # 1秒
    signal_length = int(sample_rate * duration)

    # 生成复合信号：正弦波 + 噪声
    t = cp.linspace(0, duration, signal_length)
    clean_signal = (cp.sin(2 * cp.pi * 1000 * t) +  # 1kHz
                   cp.sin(2 * cp.pi * 2000 * t) +  # 2kHz
                   cp.sin(2 * cp.pi * 3000 * t))   # 3kHz

    # 添加噪声
    noise = cp.random.normal(0, 0.1, signal_length)
    noisy_signal = clean_signal + noise

    print(f"信号长度: {signal_length}")
    print(f"采样率: {sample_rate} Hz")
    print(f"信号持续时间: {duration} 秒")

    # 1. FFT处理示例
    print("\n1. FFT处理示例")
    fft_processor = FFTProcessor(gpu_processor)

    # 执行FFT
    spectrum = fft_processor.fft(noisy_signal)
    freq_axis = fft_processor.get_frequency_axis(sample_rate, len(spectrum))

    print(f"FFT完成，频谱长度: {len(spectrum)}")

    # 2. STFT处理示例
    print("\n2. STFT处理示例")
    stft_processor = STFTProcessor(gpu_processor)

    # 执行STFT
    window_size = 1024
    hop_size = 512
    stft_result = stft_processor.stft(noisy_signal, window_size, hop_size)

    print(f"STFT完成，时频图形状: {stft_result.shape}")

    # 3. 频谱分析示例
    print("\n3. 频谱分析示例")
    spectral_analyzer = SpectralAnalyzer(gpu_processor)

    # Welch功率谱密度估计
    freq_psd, psd = spectral_analyzer.welch_psd(noisy_signal, sample_rate)

    # 频谱峰值检测
    peaks, peak_values = spectral_analyzer.find_peaks(psd, threshold=0.1)

    # 频谱特征
    centroid = spectral_analyzer.estimate_spectral_centroid(psd, freq_psd)
    bandwidth = spectral_analyzer.estimate_spectral_bandwidth(psd, freq_psd)
    entropy = spectral_analyzer.compute_spectral_entropy(psd)

    print(f"频谱中心: {centroid:.2f} Hz")
    print(f"频谱带宽: {bandwidth:.2f} Hz")
    print(f"频谱熵: {entropy:.2f}")
    print(f"检测到 {len(peaks)} 个峰值")

    # 4. 特征提取示例
    print("\n4. 特征提取示例")
    feature_extractor = FeatureExtractor(gpu_processor)

    # 时域特征
    time_features = feature_extractor.extract_time_domain_features(noisy_signal)

    # 频域特征
    freq_features = feature_extractor.extract_frequency_domain_features(noisy_signal, sample_rate)

    # 频带功率比例
    bands = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    band_features = feature_extractor.extract_frequency_band_power_ratios(
        noisy_signal, sample_rate, bands)

    print("时域特征:")
    for key, value in time_features.items():
        print(f"  {key}: {value:.4f}")

    print("频域特征:")
    for key, value in freq_features.items():
        print(f"  {key}: {value:.4f}")

    # 5. 滤波示例
    print("\n5. 滤波示例")
    filter_bank = FilterBank(gpu_processor)

    # 低通滤波
    cutoff_freq = 1500  # 1.5kHz
    lowpass_filtered = filter_bank.butterworth_lowpass(
        noisy_signal, cutoff_freq, sample_rate)

    # 高通滤波
    highpass_filtered = filter_bank.butterworth_highpass(
        noisy_signal, cutoff_freq, sample_rate)

    # 带通滤波
    bandpass_filtered = filter_bank.butterworth_bandpass(
        noisy_signal, 800, 2200, sample_rate)

    print(f"低通滤波完成 (截止频率: {cutoff_freq} Hz)")
    print(f"高通滤波完成 (截止频率: {cutoff_freq} Hz)")
    print(f"带通滤波完成 (800-2200 Hz)")

    # 6. 自适应滤波示例
    print("\n6. 自适应滤波示例")
    adaptive_filter = AdaptiveFilter(gpu_processor)

    # 创建参考噪声信号
    reference_noise = cp.random.normal(0, 0.1, signal_length)

    # LMS自适应滤波
    output_signal, error_signal = adaptive_filter.lms_filter(
        reference_noise, noisy_signal, filter_length=32, mu=0.01)

    print("LMS自适应滤波完成")

    # 7. 可视化示例
    print("\n7. 可视化示例")
    visualizer = SignalVisualizer(gpu_processor)

    # 绘制原始信号和频谱
    visualizer.plot_signal_and_spectrum(
        noisy_signal, sample_rate, "原始信号分析")

    # 绘制STFT频谱图
    visualizer.plot_stft_spectrogram(
        noisy_signal, sample_rate, window_size, hop_size, "STFT频谱图")

    # 绘制特征对比
    features_dict = {
        '原始信号': {**time_features, **freq_features},
        '低通滤波': feature_extractor.extract_time_domain_features(lowpass_filtered),
        '高通滤波': feature_extractor.extract_time_domain_features(highpass_filtered),
        '带通滤波': feature_extractor.extract_time_domain_features(bandpass_filtered)
    }
    visualizer.plot_feature_comparison(features_dict, "信号特征对比")

    # 8. 性能监控示例
    print("\n8. 性能监控示例")
    performance_monitor = PerformanceMonitor(gpu_processor)

    # 基准测试FFT性能
    def gpu_fft(signal):
        return cp.fft.fft(signal)

    def cpu_fft(signal):
        return np.fft.fft(cp.asnumpy(signal))

    # 比较GPU和CPU FFT性能
    comparison = performance_monitor.compare_gpu_cpu_performance(
        gpu_fft, cpu_fft, noisy_signal, num_runs=5)

    print(f"GPU FFT平均时间: {comparison['gpu']['mean_time']:.6f} 秒")
    print(f"CPU FFT平均时间: {comparison['cpu']['mean_time']:.6f} 秒")
    print(f"GPU加速比: {comparison['speedup']:.2f}x")

    # 生成性能报告
    report = performance_monitor.get_performance_report()
    print("\n性能报告:")
    print(report)

    print("\n示例完成！")

if __name__ == "__main__":
    main()
