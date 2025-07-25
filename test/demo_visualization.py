import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gpu_signal_processor import GPUSignalProcessor
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_signal_and_spectrum(processor, signal, title, sample_rate, ax1, ax2):
    """绘制信号和频谱"""
    # 计算FFT
    spectrum = processor.fft(signal)
    freq_axis = processor.get_frequency_axis(sample_rate, len(spectrum))

    # 转换为numpy数组用于绘图
    signal_np = cp.asnumpy(signal)
    spectrum_np = cp.asnumpy(cp.abs(spectrum))
    freq_axis_np = cp.asnumpy(freq_axis)

    # 绘制时域信号
    time_axis = np.arange(len(signal_np)) / sample_rate
    ax1.plot(time_axis, signal_np)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title(f'{title} - 时域')
    ax1.grid(True)

    # 绘制频域信号
    ax2.plot(freq_axis_np, spectrum_np)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度')
    ax2.set_title(f'{title} - 频域')
    ax2.grid(True)
    ax2.set_xlim(0, sample_rate / 2)

def plot_stft_spectrogram(processor, signal, title, sample_rate, ax,
                         window_size=1024, hop_size=512):
    """绘制STFT频谱图"""
    # 计算STFT
    stft_result = processor.stft(signal, window_size, hop_size)

    # 获取时间和频率轴
    time_axis = processor.get_time_axis(len(signal), sample_rate, hop_size, window_size)
    freq_axis = processor.get_frequency_axis(sample_rate, stft_result.shape[1])

    # 转换为numpy数组用于绘图
    stft_magnitude = cp.asnumpy(cp.abs(stft_result))
    time_axis_np = cp.asnumpy(time_axis)
    freq_axis_np = cp.asnumpy(freq_axis)

    # 绘制频谱图
    im = ax.pcolormesh(time_axis_np, freq_axis_np, stft_magnitude.T,
                       shading='gouraud', cmap='viridis')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('频率 (Hz)')
    ax.set_title(f'{title} - STFT频谱图')
    plt.colorbar(im, ax=ax, label='幅度')

def demo_fft():
    """FFT演示"""
    print("=== FFT演示 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成测试信号
    signals = processor.generate_test_signals(sample_rate=44100, duration=0.1)

    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('GPU加速FFT演示', fontsize=16)

    # 演示不同信号的FFT
    plot_signal_and_spectrum(processor, signals['sine_1khz'],
                            '1kHz正弦波', signals['sample_rate'],
                            axes[0, 0], axes[0, 1])

    plot_signal_and_spectrum(processor, signals['fm_signal'],
                            '调频信号', signals['sample_rate'],
                            axes[1, 0], axes[1, 1])

    plot_signal_and_spectrum(processor, signals['composite'],
                            '复合信号', signals['sample_rate'],
                            axes[2, 0], axes[2, 1])

    plt.tight_layout()
    plt.savefig('fft_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_stft():
    """STFT演示"""
    print("=== STFT演示 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成更长的测试信号用于STFT
    signals = processor.generate_test_signals(sample_rate=44100, duration=1.0)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GPU加速STFT演示', fontsize=16)

    # 演示不同信号的STFT
    plot_stft_spectrogram(processor, signals['fm_signal'],
                         '调频信号', signals['sample_rate'], axes[0, 0])

    plot_stft_spectrogram(processor, signals['am_signal'],
                         '调幅信号', signals['sample_rate'], axes[0, 1])

    plot_stft_spectrogram(processor, signals['composite'],
                         '复合信号', signals['sample_rate'], axes[1, 0])

    # 添加时域信号对比
    time_axis = cp.asnumpy(signals['time'])
    composite_np = cp.asnumpy(signals['composite'])
    axes[1, 1].plot(time_axis, composite_np)
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('幅度')
    axes[1, 1].set_title('复合信号 - 时域')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('stft_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_performance_comparison():
    """性能对比演示"""
    print("=== 性能对比演示 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成不同长度的测试信号
    signal_lengths = [2**10, 2**12, 2**14, 2**16, 2**18]
    gpu_times = []
    cpu_times = []

    for length in signal_lengths:
        print(f"测试信号长度: {length}")

        # 生成测试信号
        t = cp.linspace(0, 1, length)
        signal = cp.sin(2 * cp.pi * 1000 * t) + cp.random.normal(0, 0.1, length)

        # GPU FFT计时
        start_time = time.time()
        gpu_spectrum = processor.fft(signal)
        cp.cuda.Stream.null.synchronize()  # 确保GPU计算完成
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)

        # CPU FFT计时
        signal_cpu = cp.asnumpy(signal)
        start_time = time.time()
        cpu_spectrum = np.fft.fft(signal_cpu)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)

        print(f"  GPU时间: {gpu_time:.6f}s")
        print(f"  CPU时间: {cpu_time:.6f}s")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    plt.plot(signal_lengths, gpu_times, 'o-', label='GPU (CuPy)', linewidth=2, markersize=8)
    plt.plot(signal_lengths, cpu_times, 's-', label='CPU (NumPy)', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('信号长度')
    plt.ylabel('计算时间 (s)')
    plt.title('GPU vs CPU FFT性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_window_comparison():
    """不同窗口函数对比"""
    print("=== 窗口函数对比演示 ===")

    # 初始化GPU处理器
    processor = GPUSignalProcessor(gpu_id=0)

    # 生成测试信号
    signals = processor.generate_test_signals(sample_rate=44100, duration=0.5)

    # 不同窗口类型
    windows = ['hann', 'hamming', 'blackman', 'rect']
    window_names = ['汉宁窗', '海明窗', '布莱克曼窗', '矩形窗']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同窗口函数的STFT效果对比', fontsize=16)

    for i, (window, name) in enumerate(zip(windows, window_names)):
        row, col = i // 2, i % 2

        # 计算STFT
        stft_result = processor.stft(signals['composite'], window_size=1024,
                                   hop_size=512, window=window)

        # 获取时间和频率轴
        time_axis = processor.get_time_axis(len(signals['composite']),
                                          signals['sample_rate'], 512, 1024)
        freq_axis = processor.get_frequency_axis(signals['sample_rate'],
                                               stft_result.shape[1])

        # 转换为numpy数组用于绘图
        stft_magnitude = cp.asnumpy(cp.abs(stft_result))
        time_axis_np = cp.asnumpy(time_axis)
        freq_axis_np = cp.asnumpy(freq_axis)

        # 绘制频谱图
        im = axes[row, col].pcolormesh(time_axis_np, freq_axis_np, stft_magnitude.T,
                                      shading='gouraud', cmap='viridis')
        axes[row, col].set_xlabel('时间 (s)')
        axes[row, col].set_ylabel('频率 (Hz)')
        axes[row, col].set_title(f'{name} STFT')
        plt.colorbar(im, ax=axes[row, col], label='幅度')

    plt.tight_layout()
    plt.savefig('window_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("GPU信号处理演示程序")
    print("=" * 50)

    try:
        # 检查GPU可用性
        print(f"检测到 {cp.cuda.runtime.getDeviceCount()} 个GPU设备")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"GPU {i}: {props['name'].decode()}")

        # 运行演示
        demo_fft()
        demo_stft()
        demo_performance_comparison()
        demo_window_comparison()

        print("\n所有演示完成！生成的图片文件：")
        print("- fft_demo.png")
        print("- stft_demo.png")
        print("- performance_comparison.png")
        print("- window_comparison.png")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保已正确安装CuPy和CUDA")

if __name__ == "__main__":
    main()
