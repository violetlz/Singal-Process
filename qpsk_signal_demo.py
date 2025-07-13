#!/usr/bin/env python3
"""
QPSK信号生成演示
展示如何使用SignalGenerator生成5MHz带宽的QPSK调制信号
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from src import GPUSignalProcessor, SignalGenerator, FFTProcessor, SignalVisualizer

def demo_qpsk_signal_generation():
    """QPSK信号生成演示"""
    print("=== QPSK信号生成演示 ===")

    # 初始化GPU处理器和信号生成器
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)
    fft_processor = FFTProcessor(gpu_processor)
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

    # 提取信号和参数
    signal = qpsk_result['signal']
    symbol_rate = qpsk_result['symbol_rate']
    carrier_freq = qpsk_result['carrier_freq']
    bandwidth = qpsk_result['bandwidth']
    sample_rate = qpsk_result['sample_rate']
    time = qpsk_result['time']

    print(f"信号参数:")
    print(f"  带宽: {bandwidth/1e6:.1f} MHz")
    print(f"  符号率: {symbol_rate/1e6:.2f} Msymbols/s")
    print(f"  载波频率: {carrier_freq/1e6:.1f} MHz")
    print(f"  采样率: {sample_rate/1e6:.1f} MHz")
    print(f"  信号长度: {len(signal)} 采样点")

    # 计算正频率部分的频谱
    spectrum = cp.fft.rfft(signal)
    freq_axis = cp.fft.rfftfreq(len(signal), d=1/sample_rate)

    # 绘制时域和频域信号
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 时域信号
    ax1.plot(cp.asnumpy(time[:1000]), cp.asnumpy(signal[:1000]))
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title('QPSK信号 - 时域 (前1000个采样点)')
    ax1.grid(True)

    # 频域信号（正频率）
    freq_axis_np = cp.asnumpy(freq_axis)
    spectrum_np = cp.asnumpy(cp.abs(spectrum))
    ax2.plot(freq_axis_np, spectrum_np)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度')
    ax2.set_title('QPSK信号 - 频域')
    ax2.grid(True)
    ax2.set_xlim(0, sample_rate/2)

    # 功率谱密度（正频率）
    psd = cp.abs(spectrum) ** 2
    psd_np = cp.asnumpy(psd)
    ax3.semilogy(freq_axis_np, psd_np)
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('功率谱密度')
    ax3.set_title('QPSK信号 - 功率谱密度')
    ax3.grid(True)
    ax3.set_xlim(0, sample_rate/2)

    # 星座图
    i_symbols = qpsk_result['i_symbols']
    q_symbols = qpsk_result['q_symbols']
    ax4.scatter(cp.asnumpy(i_symbols), cp.asnumpy(q_symbols), alpha=0.6)
    ax4.set_xlabel('I路')
    ax4.set_ylabel('Q路')
    ax4.set_title('QPSK星座图')
    ax4.grid(True)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('qpsk_signal_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("QPSK信号演示完成，图片保存为 qpsk_signal_demo.png")

def demo_qpsk_bandwidth_analysis():
    """QPSK信号带宽分析演示"""
    print("\n=== QPSK信号带宽分析演示 ===")

    # 初始化
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)
    fft_processor = FFTProcessor(gpu_processor)

    # 生成不同滚降因子的QPSK信号
    alphas = [0.1, 0.35, 0.5, 0.8]
    bandwidth = 5e6
    sample_rate = 20e6
    duration = 0.001

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, alpha in enumerate(alphas):
        print(f"生成滚降因子 α={alpha} 的QPSK信号...")

        qpsk_result = signal_generator.generate_qpsk_signal(
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            duration=duration,
            alpha=alpha,
            snr_db=25
        )

        signal = qpsk_result['signal']
        symbol_rate = qpsk_result['symbol_rate']

        # 计算正频率部分的频谱
        spectrum = cp.fft.rfft(signal)
        freq_axis = cp.fft.rfftfreq(len(signal), d=1/sample_rate)

        # 计算功率谱密度
        psd = cp.abs(spectrum) ** 2

        # 绘制功率谱密度
        freq_axis_np = cp.asnumpy(freq_axis)
        psd_np = cp.asnumpy(psd)

        axes[i].semilogy(freq_axis_np, psd_np)
        axes[i].set_xlabel('频率 (Hz)')
        axes[i].set_ylabel('功率谱密度')
        axes[i].set_title(f'QPSK信号 - α={alpha}\n符号率: {symbol_rate/1e6:.2f} Msymbols/s')
        axes[i].grid(True)
        axes[i].set_xlim(0, sample_rate/2)

        # 标记带宽（只在正频率部分）
        axes[i].axvline(x=bandwidth, color='red', linestyle='--', alpha=0.7, label=f'带宽: {bandwidth/1e6:.1f} MHz')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('qpsk_bandwidth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("QPSK带宽分析演示完成，图片保存为 qpsk_bandwidth_analysis.png")

def demo_qpsk_noise_analysis():
    """QPSK信号噪声分析演示"""
    print("\n=== QPSK信号噪声分析演示 ===")

    # 初始化
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)
    fft_processor = FFTProcessor(gpu_processor)

    # 生成不同信噪比的QPSK信号
    snr_levels = [None, 30, 20, 10, 5]  # None表示无噪声
    snr_labels = ['无噪声', '30dB', '20dB', '10dB', '5dB']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (snr_db, snr_label) in enumerate(zip(snr_levels, snr_labels)):
        print(f"生成信噪比 {snr_label} 的QPSK信号...")

        qpsk_result = signal_generator.generate_qpsk_signal(
            bandwidth=5e6,
            sample_rate=20e6,
            duration=0.001,
            alpha=0.35,
            snr_db=snr_db
        )

        signal = qpsk_result['signal']
        time = qpsk_result['time']

        # 计算频谱
        spectrum = fft_processor.fft(signal)
        freq_axis = fft_processor.get_frequency_axis(20e6, len(spectrum))

        # 绘制时域信号
        axes[i].plot(cp.asnumpy(time[:500]), cp.asnumpy(signal[:500]))
        axes[i].set_xlabel('时间 (s)')
        axes[i].set_ylabel('幅度')
        axes[i].set_title(f'QPSK信号 - {snr_label}')
        axes[i].grid(True)

    # 最后一个子图显示频谱对比
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('qpsk_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("QPSK噪声分析演示完成，图片保存为 qpsk_noise_analysis.png")

def demo_multiple_qpsk_signals():
    """多个QPSK信号生成演示"""
    print("\n=== 多个QPSK信号生成演示 ===")

    # 初始化
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)

    # 生成多个QPSK信号
    print("生成5个QPSK信号...")
    qpsk_signals = signal_generator.generate_multiple_qpsk_signals(
        n_signals=5,
        bandwidth=5e6,
        sample_rate=20e6,
        duration=0.001,
        alpha=0.35,
        snr_db=20
    )

    # 绘制所有信号
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (signal_name, signal_data) in enumerate(qpsk_signals.items()):
        if i >= 5:  # 只显示前5个信号
            break

        signal = signal_data['signal']
        time = signal_data['time']

        axes[i].plot(cp.asnumpy(time[:1000]), cp.asnumpy(signal[:1000]))
        axes[i].set_xlabel('时间 (s)')
        axes[i].set_ylabel('幅度')
        axes[i].set_title(f'{signal_name} - 时域')
        axes[i].grid(True)

    # 隐藏最后一个子图
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('multiple_qpsk_signals.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("多个QPSK信号演示完成，图片保存为 multiple_qpsk_signals.png")

def main():
    """主函数"""
    print("QPSK信号生成演示程序")
    print("=" * 50)

    try:
        # 运行各种演示
        demo_qpsk_signal_generation()
        demo_qpsk_bandwidth_analysis()
        #demo_qpsk_noise_analysis()
        #demo_multiple_qpsk_signals()

        print("\n所有演示完成！")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
