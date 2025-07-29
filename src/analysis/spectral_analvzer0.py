import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)
# 循环向上查找包含'src'文件夹的目录作为项目根目录
project_root = None
temp_dir = current_dir
while temp_dir != os.path.dirname(temp_dir):  # 直到到达系统根目录
    if os.path.isdir(os.path.join(temp_dir, 'src')):
        project_root = temp_dir
        break
    temp_dir = os.path.dirname(temp_dir)

if project_root:
    sys.path.append(project_root)
    from src.core.gpu_processor import GPUSignalProcessor
    from src.core.signal_generator import SignalGenerator
else:
    raise FileNotFoundError("未找到包含'src'文件夹的项目根目录，请检查项目结构")

class SpectrumAnalyzer:
    """
    频谱分析器

    """

    def __init__(self, gpu_id=0, output_dir='../../result/spectrum_analysis',
                 default_signal_params=None, default_spectrum_params=None):
        """
        初始化频谱分析器，指定GPU设备并设置默认参数

        参数:
            output_dir: 结果输出目录
            default_signal_params: 信号生成默认参数
            default_spectrum_params: 频谱分析默认参数
        """
        # 初始化GPU处理器和信号生成器
        self.gpu_processor = GPUSignalProcessor(gpu_id=gpu_id)
        self.signal_generator = SignalGenerator(self.gpu_processor)

        # 设置默认参数
        self.signal_params = default_signal_params if default_signal_params is not None else {
            "bandwidth": 5e6,
            "sample_rate": 20e6,
            "duration": 0.001,
            "alpha": 0.35,
            "snr_db": 20
        }

        self.spectrum_params = default_spectrum_params if default_spectrum_params is not None else {
            "fft_size": 4096,  # FFT点数
            "peak_threshold": 0.5,  # 峰值检测阈值（相对最大值的比例）
            "welch_window": "hann",  # Welch方法窗口类型
            "welch_nfft": 2048,  # Welch方法FFT点数
            "welch_noverlap": 1024,  # Welch方法重叠点数
            "welch_segment_size": 4096,  # Welch方法每段信号长度
            "visualize": True  # 是否可视化结果
        }

        # 信号存储
        self.signal = None  # 原始信号（GPU）
        self.sample_rate = None  # 采样率
        self.time = None  # 时间轴（GPU）

        # 频谱分析结果存储
        self.energy_spectrum = None  # 能量谱
        self.freq_axis = None  # 频率轴
        self.peaks = None  # 检测到的峰值（CPU）
        self.power_spectrum_density = None  # 功率谱密度
        self.welch_freq = None  # Welch方法频率轴

        # 输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_signal(self, custom_signal_params=None):
        """

        参数:
            custom_signal_params: 自定义信号生成参数，覆盖默认值

        return:
            包含生成信号信息的字典
        """
        # 合并默认参数和自定义参数
        used_params = self.signal_params.copy()
        if custom_signal_params is not None:
            used_params.update(custom_signal_params)

        # 生成信号
        signal_result = self.signal_generator.generate_qpsk_signal(**used_params)

        # 存储信号及相关参数
        self.signal = signal_result['signal']  # 已在GPU
        self.sample_rate = used_params['sample_rate']
        self.time = signal_result['time']  # 时间轴（GPU）

        return {
            "signal": self.signal,
            "sample_rate": self.sample_rate,
            "time": self.time
        }

    def load_external_signal(self, signal, sample_rate):
        """
        加载外部信号进行分析，自动转为GPU数据

        参数:
            signal: 外部信号
            sample_rate: 信号采样率
        """
        if isinstance(signal, np.ndarray):
            self.signal = cp.asarray(signal, dtype=cp.complex64)  # 转GPU
        else:
            self.signal = signal.astype(cp.complex64)  # 确保是GPU数组

        self.sample_rate = sample_rate
        self.time = cp.arange(len(self.signal), dtype=cp.float32) / self.sample_rate  # GPU时间轴

    def compute_energy_spectrum(self):
        """
        计算信号的能量谱
        """
        if self.signal is None:
            raise ValueError("请先生成或加载信号（调用generate_signal或load_external_signal）")

        # 截取或补零到指定的FFT点数
        fft_size = self.spectrum_params['fft_size']
        signal_length = len(self.signal)

        # 处理信号补零或截断
        if signal_length < fft_size:
            signal_padded = cp.zeros(fft_size, dtype=cp.complex64)
            signal_padded[:signal_length] = self.signal
        else:
            signal_padded = self.signal[:fft_size]

        # 计算FFT并获取能量谱（|X(f)|²）
        fft_result = cp.fft.fft(signal_padded)
        self.energy_spectrum = cp.abs(fft_result) ** 2

        # 计算频率轴（GPU上），仅保留正频率
        freq = cp.fft.fftfreq(fft_size, 1 / self.sample_rate)
        positive_mask = freq >= 0
        self.freq_axis = freq[positive_mask]
        self.energy_spectrum = self.energy_spectrum[positive_mask]

    def detect_spectrum_peaks(self):
        """
        频谱峰值检测

        return:
            包含峰值频率和对应幅度的字典列表
        """
        if self.energy_spectrum is None:
            self.compute_energy_spectrum()  # 确保能量谱已计算

        # 提取峰值检测参数
        threshold = self.spectrum_params['peak_threshold']
        max_amplitude = cp.max(self.energy_spectrum)
        peak_amplitude_threshold = max_amplitude * threshold

        # GPU上生成局部最大值掩码（中间点大于左右邻居）
        left = self.energy_spectrum[:-2]
        mid = self.energy_spectrum[1:-1]
        right = self.energy_spectrum[2:]
        local_max_mask = (mid > left) & (mid > right)

        # 结合阈值筛选峰值
        threshold_mask = mid >= peak_amplitude_threshold
        peak_mask = local_max_mask & threshold_mask

        # 找到峰值位置并提取频率和幅度（转为CPU输出）
        peak_indices = cp.where(peak_mask)[0] + 1  # 补偿切片偏移
        peak_freqs = self.freq_axis[peak_indices].get()  # 转CPU
        peak_amps = self.energy_spectrum[peak_indices].get()  # 转CPU

        # 整理为字典列表
        self.peaks = [
            {"frequency": freq, "amplitude": amp}
            for freq, amp in zip(peak_freqs, peak_amps)
        ]
        return self.peaks

    def estimate_power_spectral_density(self):
        """
        Welch方法估计功率谱密度
        """
        if self.signal is None:
            raise ValueError("请先生成或加载信号（调用generate_signal或load_external_signal）")

        # 提取Welch方法参数
        window_type = self.spectrum_params['welch_window']
        nfft = self.spectrum_params['welch_nfft']
        noverlap = self.spectrum_params['welch_noverlap']
        segment_size = self.spectrum_params['welch_segment_size']

        # 计算每段信号的步长
        step = segment_size - noverlap
        if step <= 0:
            raise ValueError("Welch方法：段长度必须大于重叠长度")

        # 生成窗口函数
        if window_type == "hann":
            window = cp.hanning(segment_size).astype(cp.float32)
        elif window_type == "hamming":
            window = cp.hamming(segment_size).astype(cp.float32)
        elif window_type == "blackman":
            window = cp.blackman(segment_size).astype(cp.float32)
        else:
            raise ValueError(f"不支持的窗口类型: {window_type}")

        # 计算窗口能量校正因子
        window_power = cp.sum(window ** 2)
        if window_power == 0:
            raise ValueError("窗口函数能量为零，无法用于Welch方法")

        # 将信号分块
        num_segments = (len(self.signal) - segment_size) // step + 1
        if num_segments < 1:
            raise ValueError("信号长度不足，无法分块用于Welch方法")

        # 初始化功率谱密度存储
        psd = cp.zeros(nfft // 2 + 1, dtype=cp.float32)

        # 循环处理每段信号
        for i in range(num_segments):
            # 提取当前段
            start = i * step
            end = start + segment_size
            segment = self.signal[start:end]

            # 加窗
            segment_windowed = segment * window

            # FFT计算
            fft_result = cp.fft.fft(segment_windowed, n=nfft)

            # 计算功率谱并累加（仅保留正频率）
            power = cp.abs(fft_result[:nfft // 2 + 1]) ** 2 / (self.sample_rate * window_power)
            psd += power

        # 平均所有段的结果
        self.power_spectrum_density = 10 * cp.log10(psd / num_segments + 1e-10)

        # 计算Welch频率轴
        self.welch_freq = cp.fft.fftfreq(nfft, 1 / self.sample_rate)[:nfft // 2 + 1]

        return {
            "frequency": self.welch_freq,
            "power_spectral_density": self.power_spectrum_density
        }

    def visualize_results(self):
        """
        可视化频谱分析结果
        """
        if not self.spectrum_params['visualize']:
            return

        # 确保所有分析已执行
        if self.energy_spectrum is None:
            self.compute_energy_spectrum()
        if self.peaks is None:
            self.detect_spectrum_peaks()
        if self.power_spectrum_density is None:
            self.estimate_power_spectral_density()

        # 将GPU数据转为CPU用于绘图
        signal_cpu = self.signal.get()
        time_cpu = self.time.get()
        energy_spectrum_cpu = self.energy_spectrum.get()
        freq_axis_cpu = self.freq_axis.get()
        welch_freq_cpu = self.welch_freq.get()
        psd_cpu = self.power_spectrum_density.get()

        # 1. 绘制原始信号时域波形
        plt.figure(figsize=(12, 6))
        plt.plot(time_cpu[:1000], cp.real(signal_cpu[:1000]).get(), label='I')
        plt.plot(time_cpu[:1000], cp.imag(signal_cpu[:1000]).get(), label='Q', alpha=0.7)
        plt.title('Original Signal Time Domain Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'time_domain_waveform.png'), dpi=300)
        plt.show()

        # 2. 绘制能量谱并标记峰值
        plt.figure(figsize=(12, 6))
        plt.plot(freq_axis_cpu / 1e6, 10 * np.log10(energy_spectrum_cpu + 1e-10), label='Energy Spectrum')

        # 标记峰值
        for peak in self.peaks:
            plt.scatter(peak['frequency'] / 1e6, 10 * np.log10(peak['amplitude'] + 1e-10),
                        color='red', marker='x', label='Detected Peaks' if peak == self.peaks[0] else "")

        plt.title('Signal Energy Spectrum with Peak Detection (Full GPU Computation)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Energy (dB)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'energy_spectrum_with_peaks.png'), dpi=300)
        plt.show()

        # 3. 绘制功率谱密度（Welch方法，GPU计算）
        plt.figure(figsize=(12, 6))
        plt.plot(welch_freq_cpu / 1e6, psd_cpu, label='Power Spectral Density (GPU Welch Method)')
        plt.title('Power Spectral Density Estimation (Full GPU Acceleration)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'power_spectral_density.png'), dpi=300)
        plt.show()

    def print_analysis_results(self):
        """
        打印频谱分析结果
        """
        print("频谱分析结果：")
        print(f"采样率: {self.sample_rate / 1e6:.2f} MHz")
        print(f"检测到的峰值数量: {len(self.peaks)}")
        for i, peak in enumerate(self.peaks, 1):
            print(f"  峰值 {i}: 频率 = {peak['frequency'] / 1e6:.4f} MHz, 幅度 = {peak['amplitude']:.2f}")

        # 计算功率谱密度峰值
        psd_peak = cp.max(self.power_spectrum_density).get()
        print(f"功率谱密度峰值: {psd_peak:.2f} dB/Hz")

    def run(self, custom_signal_params=None, custom_spectrum_params=None):
        """
        完整的频谱分析流程

        参数:
            custom_signal_params: 自定义信号生成参数
            custom_spectrum_params: 自定义频谱分析参数

        return:
            包含所有分析结果的字典
        """
        # 更新参数
        if custom_signal_params is not None:
            self.signal_params.update(custom_signal_params)
        if custom_spectrum_params is not None:
            self.spectrum_params.update(custom_spectrum_params)

        # 生成信号
        self.generate_signal()

        # 执行频谱分析
        self.compute_energy_spectrum()
        self.detect_spectrum_peaks()
        self.estimate_power_spectral_density()

        # 可视化结果
        if self.spectrum_params['visualize']:
            self.visualize_results()

        # 打印分析结果
        self.print_analysis_results()

        # 返回结果
        return {
            "energy_spectrum": self.energy_spectrum.get(),
            "freq_axis": self.freq_axis.get(),
            "peaks": self.peaks,
            "power_spectral_density": self.power_spectrum_density.get(),
            "welch_freq": self.welch_freq.get()
        }


if __name__ == "__main__":
    # 初始化频谱分析器并运行默认分析（全GPU）
    analyzer = SpectrumAnalyzer(gpu_id=0)
    analyzer.run()

    # # 自定义参数运行示例
    # custom_analyzer = SpectrumAnalyzer(
    #     gpu_id=0,
    #     output_dir='./result/gpu_spectrum_analysis',
    #     default_spectrum_params={
    #         "fft_size": 8192,
    #         "peak_threshold": 0.3,
    #         "welch_nfft": 8192,
    #         "welch_segment_size": 8192
    #     }
    # )
    # custom_analyzer.run(
    #     custom_signal_params={"snr_db": 10, "duration": 0.005},
    #     custom_spectrum_params={"visualize": True}
    # )