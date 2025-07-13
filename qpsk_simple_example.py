#!/usr/bin/env python3
"""
简单的QPSK信号生成示例
展示如何生成5MHz带宽的QPSK调制信号
"""

import cupy as cp
from src import GPUSignalProcessor, SignalGenerator

def main():
    """主函数"""
    print("QPSK信号生成示例")
    print("=" * 30)

    # 初始化GPU处理器和信号生成器
    gpu_processor = GPUSignalProcessor(gpu_id=0)
    signal_generator = SignalGenerator(gpu_processor)

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
    bits = qpsk_result['bits']
    i_symbols = qpsk_result['i_symbols']
    q_symbols = qpsk_result['q_symbols']

    # 打印信号参数
    print(f"\n信号参数:")
    print(f"  带宽: {bandwidth/1e6:.1f} MHz")
    print(f"  符号率: {symbol_rate/1e6:.2f} Msymbols/s")
    print(f"  载波频率: {carrier_freq/1e6:.1f} MHz")
    print(f"  采样率: {sample_rate/1e6:.1f} MHz")
    print(f"  信号长度: {len(signal)} 采样点")
    print(f"  比特数量: {len(bits)}")
    print(f"  符号数量: {len(i_symbols)}")

    # 计算信号统计信息
    signal_mean = cp.mean(signal)
    signal_std = cp.std(signal)
    signal_power = cp.mean(signal ** 2)

    print(f"\n信号统计信息:")
    print(f"  均值: {signal_mean:.6f}")
    print(f"  标准差: {signal_std:.6f}")
    print(f"  功率: {signal_power:.6f}")

    # 显示前几个比特和符号
    print(f"\n前10个比特: {cp.asnumpy(bits[:10])}")
    print(f"前5个I符号: {cp.asnumpy(i_symbols[:5])}")
    print(f"前5个Q符号: {cp.asnumpy(q_symbols[:5])}")

    # 计算频谱
    spectrum = cp.fft.fft(signal)
    freq_axis = cp.fft.fftfreq(len(signal), 1/sample_rate)

    # 找到频谱峰值
    magnitude_spectrum = cp.abs(spectrum)
    peak_idx = cp.argmax(magnitude_spectrum)
    peak_freq = freq_axis[peak_idx]

    print(f"\n频谱分析:")
    print(f"  峰值频率: {peak_freq/1e6:.2f} MHz")
    print(f"  峰值幅度: {magnitude_spectrum[peak_idx]:.2f}")

    print("\nQPSK信号生成完成！")

    return qpsk_result

if __name__ == "__main__":
    main()
