#!/usr/bin/env python3
"""
频谱检测与估计功能测试
使用5MHz带宽QPSK信号测试频谱分析器和特征提取器
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from src import (
    GPUSignalProcessor, SignalGenerator, FFTProcessor, 
    SpectralAnalyzer, FeatureExtractor, SignalVisualizer
)

def test_spectral_detection_and_estimation():
    """测试频谱检测与估计功能"""
    print("=== 频谱检测与估计功能测试 ===")
    
    # 初始化各个模块
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)
    fft_processor = FFTProcessor(gpu_processor)
    spectral_analyzer = SpectralAnalyzer(gpu_processor)
    feature_extractor = FeatureExtractor(gpu_processor)
    visualizer = SignalVisualizer(gpu_processor)
    
    # 生成5MHz带宽的QPSK信号
    print("生成5MHz带宽的QPSK信号...")
    qpsk_result = signal_generator.generate_qpsk_signal(
        bandwidth=5e6,      # 5MHz带宽
        sample_rate=20e6,   # 20MHz采样率
        duration=0.001,     # 1ms持续时间
        alpha=0.35,         # 升根余弦滤波器滚降因子
        snr_db=20           # 20dB信噪比
    )
    
    signal = qpsk_result['signal']
    sample_rate = qpsk_result['sample_rate']
    bandwidth = qpsk_result['bandwidth']
    carrier_freq = qpsk_result['carrier_freq']
    
    print(f"信号参数:")
    print(f"  带宽: {bandwidth/1e6:.1f} MHz")
    print(f"  载波频率: {carrier_freq/1e6:.1f} MHz")
    print(f"  采样率: {sample_rate/1e6:.1f} MHz")
    print(f"  信号长度: {len(signal)} 采样点")
    
    # 1. 频谱检测与估计
    print("\n--- 1. 频谱检测与估计 ---")
    
    # 计算正频率FFT功率谱
    spectrum_rfft = cp.fft.rfft(signal)
    psd_rfft = cp.abs(spectrum_rfft) ** 2
    freq_axis_rfft = cp.fft.rfftfreq(len(signal), d=1/sample_rate)
    
    # Welch功率谱密度估计
    print("执行Welch功率谱密度估计...")
    welch_freq, welch_psd = spectral_analyzer.welch_psd(
        signal, sample_rate, window_size=1024, hop_size=512, window='hann'
    )
    
    # 频谱峰值检测（用正频率部分）
    print("执行频谱峰值检测...")
    peak_indices, peak_values = spectral_analyzer.find_peaks(
        cp.asnumpy(psd_rfft), threshold=0.1, min_distance=50
    )
    freq_axis_np = cp.asnumpy(freq_axis_rfft)
    peak_indices_np = peak_indices.get()
    peak_frequencies = freq_axis_np[peak_indices_np]
    
    print(f"检测到的峰值数量: {len(peak_indices)}")
    for i, (freq, value) in enumerate(zip(peak_frequencies, peak_values)):
        print(f"  峰值 {i+1}: 频率 = {freq/1e6:.2f} MHz, 幅度 = {value:.2e}")
    
    # 频谱特征估计（用正频率部分）
    print("\n执行频谱特征估计...")
    spectral_centroid = spectral_analyzer.estimate_spectral_centroid(psd_rfft, freq_axis_rfft)
    print(f"频谱中心: {spectral_centroid/1e6:.2f} MHz")
    spectral_bandwidth = spectral_analyzer.estimate_spectral_bandwidth(psd_rfft, freq_axis_rfft)
    print(f"频谱带宽: {spectral_bandwidth/1e6:.2f} MHz")
    spectral_entropy = spectral_analyzer.compute_spectral_entropy(psd_rfft)
    print(f"频谱熵: {spectral_entropy:.4f}")
    
    # 2. 信号能量谱提取
    print("\n--- 2. 信号能量谱提取 ---")
    
    # 时域特征提取
    print("提取时域特征...")
    time_features = feature_extractor.extract_time_domain_features(signal)
    
    print("时域特征:")
    for key, value in time_features.items():
        if 'energy' in key or 'power' in key:
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 频域特征提取
    print("\n提取频域特征...")
    freq_features = feature_extractor.extract_frequency_domain_features(signal, sample_rate)
    
    print("频域特征:")
    for key, value in freq_features.items():
        if 'spectral_centroid' in key or 'spectral_bandwidth' in key or 'spectral_rolloff' in key:
            print(f"  {key}: {value/1e6:.2f} MHz")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 频带功率比例提取
    print("\n提取频带功率比例...")
    
    # 定义频带（针对5MHz带宽信号）
    bands = [
        (0, 1e6),      # 0-1 MHz
        (1e6, 3e6),    # 1-3 MHz
        (3e6, 5e6),    # 3-5 MHz
        (5e6, 7e6),    # 5-7 MHz
        (7e6, 10e6),   # 7-10 MHz
    ]
    
    band_features = feature_extractor.extract_frequency_band_power_ratios(
        signal, sample_rate, bands
    )
    
    print("频带功率比例:")
    for i, (low_freq, high_freq) in enumerate(bands):
        key = f'band_{i+1}_power_ratio'
        print(f"  {low_freq/1e6:.0f}-{high_freq/1e6:.0f} MHz: {band_features[key]:.4f}")
    
    # 3. 可视化结果
    print("\n--- 3. 可视化结果 ---")
    
    # 创建综合可视化，增加能量谱图
    fig, axs = plt.subplots(3, 2, figsize=(16, 16))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]
    ax5 = axs[2, 0]
    # ax6 预留

    # 时域信号
    time = qpsk_result['time']
    ax1.plot(cp.asnumpy(time[:2000]), cp.asnumpy(signal[:2000]))
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title('QPSK信号 - 时域')
    ax1.grid(True)

    # 功率谱密度
    ax2.semilogy(freq_axis_np, cp.asnumpy(psd_rfft), label='FFT功率谱')
    ax2.semilogy(cp.asnumpy(welch_freq), cp.asnumpy(welch_psd), label='Welch功率谱', linestyle='--')
    if len(peak_frequencies) > 0:
        peak_psd_values = cp.asnumpy(psd_rfft)[peak_indices_np]
        ax2.plot(peak_frequencies, peak_psd_values, 'ro', markersize=8, label='检测到的峰值')
    # 标记载波频率和带宽
    ax2.axvline(x=carrier_freq, color='green', linestyle='-', alpha=0.7, label=f'载波: {carrier_freq/1e6:.1f} MHz')
    ax2.axvline(x=carrier_freq + bandwidth/2, color='red', linestyle='--', alpha=0.7, label=f'带宽边界: {carrier_freq-bandwidth/2:.1f}~{carrier_freq+bandwidth/2:.1f} MHz')
    ax2.axvline(x=carrier_freq - bandwidth/2, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('功率谱密度')
    ax2.set_title('功率谱密度分析')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, sample_rate/2)

    # 频带功率比例柱状图
    band_names = [f'{low/1e6:.0f}-{high/1e6:.0f} MHz' for low, high in bands]
    band_values = [band_features[f'band_{i+1}_power_ratio'] for i in range(len(bands))]
    ax3.bar(band_names, band_values, alpha=0.7, color='skyblue')
    ax3.set_xlabel('频带')
    ax3.set_ylabel('功率比例')
    ax3.set_title('频带功率比例分布')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 特征对比
    feature_names = ['频谱中心\n(MHz)', '频谱带宽\n(MHz)', '频谱熵', '过零率']
    feature_values = [
        freq_features['spectral_centroid']/1e6,
        freq_features['spectral_bandwidth']/1e6,
        freq_features['spectral_entropy'],
        time_features['zero_crossing_rate']
    ]
    ax4.bar(feature_names, feature_values, alpha=0.7, color='lightcoral')
    ax4.set_ylabel('特征值')
    ax4.set_title('信号特征对比')
    ax4.grid(True, alpha=0.3)

    # 能量谱图也用正频率
    energy_spectrum = np.abs(cp.asnumpy(spectrum_rfft)) ** 2
    ax5.plot(freq_axis_np, energy_spectrum)
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('能量')
    ax5.set_title('信号能量谱')
    ax5.grid(True)

    # 隐藏ax6
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig('spectral_detection_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 单独绘制能量谱 ---
    plt.figure(figsize=(10, 4))
    plt.plot(freq_axis_np, energy_spectrum)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('能量')
    plt.title('信号能量谱')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n测试完成，图片保存为 spectral_detection_test.png")
    
    # 4. 性能分析
    print("\n--- 4. 性能分析 ---")
    
    # 计算检测精度
    expected_carrier_freq = carrier_freq
    detected_peaks = peak_frequencies
    
    # 找到最接近载波频率的峰值
    if len(detected_peaks) > 0:
        carrier_detection_error = np.min(np.abs(detected_peaks - expected_carrier_freq))
        print(f"载波频率检测误差: {carrier_detection_error/1e6:.3f} MHz")
        
        # 检查是否检测到负频率的镜像
        negative_carrier_error = np.min(np.abs(detected_peaks + expected_carrier_freq))
        print(f"负载波频率检测误差: {negative_carrier_error/1e6:.3f} MHz")
    
    # 带宽估计精度
    estimated_bandwidth = spectral_bandwidth
    actual_bandwidth = bandwidth
    bandwidth_error = abs(estimated_bandwidth - actual_bandwidth)
    print(f"带宽估计误差: {bandwidth_error/1e6:.3f} MHz")
    
    # 频谱中心精度
    estimated_centroid = spectral_centroid
    centroid_error = abs(estimated_centroid - expected_carrier_freq)
    print(f"频谱中心估计误差: {centroid_error/1e6:.3f} MHz")
    
    return {
        'signal_params': qpsk_result,
        'peak_detection': {
            'frequencies': peak_frequencies,
            'values': peak_values,
            'count': len(peak_indices)
        },
        'spectral_features': {
            'centroid': spectral_centroid,
            'bandwidth': spectral_bandwidth,
            'entropy': spectral_entropy
        },
        'time_features': time_features,
        'freq_features': freq_features,
        'band_features': band_features
    }

def test_multiple_signals():
    """测试多个不同参数的信号"""
    print("\n=== 多信号测试 ===")
    
    # 初始化
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)
    spectral_analyzer = SpectralAnalyzer(gpu_processor)
    feature_extractor = FeatureExtractor(gpu_processor)
    
    # 测试不同信噪比的信号
    snr_levels = [None, 30, 20, 10, 5]
    snr_labels = ['无噪声', '30dB', '20dB', '10dB', '5dB']
    
    results = {}
    
    for snr_db, snr_label in zip(snr_levels, snr_labels):
        print(f"\n测试信噪比 {snr_label} 的信号...")
        
        # 生成信号
        qpsk_result = signal_generator.generate_qpsk_signal(
            bandwidth=5e6,
            sample_rate=20e6,
            duration=0.001,
            alpha=0.35,
            snr_db=snr_db
        )
        
        signal = qpsk_result['signal']
        sample_rate = qpsk_result['sample_rate']
        
        # 频谱分析
        spectrum = cp.fft.fft(signal)
        freq_axis = cp.linspace(-sample_rate/2, sample_rate/2, len(spectrum))
        spectrum_shifted = cp.fft.fftshift(spectrum)
        psd = cp.abs(spectrum_shifted) ** 2
        
        # 峰值检测
        peak_indices, peak_values = spectral_analyzer.find_peaks(
            cp.asnumpy(psd), threshold=0.1, min_distance=50
        )
        
        # 特征提取
        time_features = feature_extractor.extract_time_domain_features(signal)
        freq_features = feature_extractor.extract_frequency_domain_features(signal, sample_rate)
        
        results[snr_label] = {
            'peak_count': len(peak_indices),
            'spectral_entropy': freq_features['spectral_entropy'],
            'signal_power': time_features['power'],
            'zero_crossing_rate': time_features['zero_crossing_rate']
        }
    
    # 显示结果对比
    print("\n多信号测试结果对比:")
    print(f"{'信噪比':<10} {'峰值数':<8} {'频谱熵':<10} {'信号功率':<12} {'过零率':<10}")
    print("-" * 60)
    
    for snr_label, result in results.items():
        print(f"{snr_label:<10} {result['peak_count']:<8} {result['spectral_entropy']:<10.4f} "
              f"{result['signal_power']:<12.2e} {result['zero_crossing_rate']:<10.4f}")

if __name__ == "__main__":
    # 运行主测试
    main_results = test_spectral_detection_and_estimation()
    
    # 运行多信号测试
    test_multiple_signals()
    
    print("\n所有测试完成！") 