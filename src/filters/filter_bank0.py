import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin
import os
import sys

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

class FIR_Filter:
    def __init__(self, gpu_id=0, output_dir='../../result/test4',default_qpsk_params=None, default_filter_params=None):
        """
        初始化QPSK信号处理与滤波类
        """
        # 初始化GPU处理器和信号生成器
        self.gpu_processor = GPUSignalProcessor(gpu_id=gpu_id)
        self.signal_generator = SignalGenerator(self.gpu_processor)
        self.qpsk_params = default_qpsk_params if default_qpsk_params is not None else {
            "bandwidth": 5e6,
            "sample_rate": 20e6,
            "duration": 0.001,
            "alpha": 0.35,
            "snr_db": 20
        }
        self.filter_params = default_filter_params if default_filter_params is not None else {
            "order": 101,
            "window": "hamming"
        }
        # 信号与滤波器参数
        self.signal = None
        self.bits = None
        self.i_symbols = None
        self.q_symbols = None
        self.sample_rate = None
        self.filters = {}
        self.filtered_signals = {}

        # 输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_signal(self, custom_qpsk_params=None):
        """
        生成QPSK信号
        """

        qpsk_signal = self.signal_generator.generate_qpsk_signal(**self.qpsk_params)
        used_qpsk_params = self.qpsk_params.copy()
        if custom_qpsk_params is not None:
            used_qpsk_params.update(custom_qpsk_params)

        qpsk_signal = self.signal_generator.generate_qpsk_signal(**used_qpsk_params)

        # 提取信号参数
        self.signal = qpsk_signal['signal']
        self.bits = qpsk_signal['bits']
        self.i_symbols = qpsk_signal['i_symbols']
        self.q_symbols = qpsk_signal['q_symbols']
        self.sample_rate = qpsk_signal["sample_rate"]

        return qpsk_signal

    def design_fir_filters(self, filter_type, cutoff_freq,custom_filter_params=None):
        """
        设计FIR滤波器（低通、高通、带通）
        """
        used_filter_params = self.filter_params.copy()
        if custom_filter_params is not None:
            used_filter_params.update(custom_filter_params)
        nyquist = 0.5 * self.sample_rate
        filter_order = used_filter_params["order"]
        window = used_filter_params["window"]

        if filter_type == 'lpf':
            normalized_cutoff = cutoff_freq / nyquist
            h = firwin(filter_order, normalized_cutoff, window=window)
        elif filter_type == 'hpf':
            normalized_cutoff = cutoff_freq / nyquist
            h = firwin(filter_order, normalized_cutoff, window=window, pass_zero=False)
        elif filter_type == 'bpf':
            low_freq, high_freq = cutoff_freq
            normalized_low = low_freq / nyquist
            normalized_high = high_freq / nyquist
            h = firwin(filter_order, [normalized_low, normalized_high], window=window, pass_zero=False)
        else:
            raise ValueError("Unsupported filter type. Use 'lpf', 'hpf', or 'bpf'.")

        self.filters[filter_type] = cp.array(h, dtype=cp.float32)
        return self.filters[filter_type]

    def apply_filter(self, filter_type, input_signal=None):
        """
        应用滤波器处理信号
        """
        if filter_type not in self.filters:
            raise ValueError(f"Filter {filter_type} not designed yet.")

        # 默认处理原始QPSK信号
        if input_signal is None:
            input_signal = self.signal

        # 执行滤波
        filtered_signal = cp.convolve(input_signal, self.filters[filter_type], mode='same')
        self.filtered_signals[filter_type] = filtered_signal

        return filtered_signal

    def visualize_results(self):
        """
        可视化原始信号和滤波结果
        """
        # 确保有滤波结果可显示
        if not self.filtered_signals:
            print("No filtered signals to visualize. Apply filters first.")
            return

        # 转换到CPU进行绘图
        signal_cpu = cp.asnumpy(self.signal)

        # 绘制原始信号星座图
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.scatter(signal_cpu.real, signal_cpu.imag, s=3, alpha=0.5)
        plt.title('Original QPSK Signal Constellation')
        plt.xlabel('In-phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True)
        plt.axis('equal')

        # 绘制各滤波器处理后的星座图
        filter_names = list(self.filtered_signals.keys())
        for i, filter_type in enumerate(filter_names, 2):
            filtered_cpu = cp.asnumpy(self.filtered_signals[filter_type])
            plt.subplot(2, 2, i)
            plt.scatter(filtered_cpu.real, filtered_cpu.imag, s=3, alpha=0.5)
            plt.title(f'{filter_type.upper()} Filtered QPSK Constellation')
            plt.xlabel('In-phase (I)')
            plt.ylabel('Quadrature (Q)')
            plt.grid(True)
            plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qpsk_constellations.png'), dpi=300)
        plt.show()

        # 绘制滤波器频率响应
        plt.figure(figsize=(12, 8))
        freq = cp.fft.fftshift(cp.fft.fftfreq(2048, 1 / self.sample_rate))

        for filter_type, h in self.filters.items():
            H = cp.fft.fft(h, 2048)
            H = cp.fft.fftshift(H)
            mag = cp.abs(H)
            mag_db = 20 * cp.log10(mag + 1e-10)
            plt.plot(cp.asnumpy(freq), cp.asnumpy(mag_db), label=f'{filter_type}')

        plt.xlim(0, self.sample_rate / 2)
        plt.ylim(-100, 5)
        plt.title('Filter Frequency Responses')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'filter_responses.png'), dpi=300)
        plt.show()

    def run(self, custom_qpsk_params=None, filter_configs=None):
        """
        生成信号→滤波器设计和滤波处理→可视化
        """
        self.generate_signal(custom_qpsk_params)

        if filter_configs is None:
            self.design_fir_filters('lpf', cutoff_freq=7.5e6)
            self.design_fir_filters('hpf', cutoff_freq=2.5e6)
            self.design_fir_filters('bpf', cutoff_freq=[2.5e6, 7.5e6])
        else:
            for filter_conf in filter_configs:
                filter_type = filter_conf["type"]
                cutoff_freq = filter_conf["cutoff"]
                custom_filter_params = filter_conf.get("custom_params", None)
                self.design_fir_filters(filter_type, cutoff_freq, custom_filter_params)

        # 应用滤波器
        for filter_type in self.filters.keys():
            self.apply_filter(filter_type)

        self.visualize_results()

if __name__ == "__main__":
    main = FIR_Filter(gpu_id=0)
    main.run()

    # custom_qpsk_params = {
    #     "bandwidth": 6e6,
    #     "sample_rate": 25e6,
    #     "duration": 0.002,
    #     "alpha": 0.4,
    #     "snr_db": 15
    # }
    # filter_configs = [
    #     {
    #         "type": "lpf",
    #         "cutoff": 8e6,
    #         "custom_params": {
    #             "order": 120,
    #             "window": "blackman"
    #         }
    #     },
    #     {
    #         "type": "hpf",
    #         "cutoff": 3e6,
    #         "custom_params": {
    #             "order": 110,
    #             "window": "hann"
    #         }
    #     },
    #     {
    #         "type": "bpf",
    #         "cutoff": [3.5e6, 8.5e6],
    #         "custom_params": {
    #             "order": 130,
    #             "window": "kaiser"
    #         }
    #     }
    # ]
    # custom_run = FIR_Filter(gpu_id=0, output_dir='./result/custom_test')
    # custom_run.run(custom_qpsk_params, filter_configs)