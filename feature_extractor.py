import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from src import GPUSignalProcessor, SignalGenerator
import os


class Feature_Extractor:
    """
    信号特征提取器
    用于提取时间域统计特征和频率分量功率特征，
    """

    def __init__(self, gpu_id=0, output_dir='./result/features',
                 default_qpsk_params=None, default_feature_params=None):
        """
        初始化特征提取器

        参数:
            output_dir: 输出目录
            default_qpsk_params: 信号默认参数
            default_feature_params: 特征提取的默认参数
        """
        # 初始化GPU处理器和信号生成器
        self.gpu_processor = GPUSignalProcessor(gpu_id=gpu_id)
        self.signal_generator = SignalGenerator(self.gpu_processor)

        # 设置默认参数
        self.qpsk_params = default_qpsk_params if default_qpsk_params is not None else {
            "bandwidth": 5e6,
            "sample_rate": 20e6,
            "duration": 0.001,
            "alpha": 0.35,
            "snr_db": 20
        }

        self.feature_params = default_feature_params if default_feature_params is not None else {
            "freq_bins": 10,  # 频率分箱数量
            "visualize": True  # 是否可视化特征
        }

        # 信号与特征存储
        self.signal = None
        self.bits = None
        self.i_symbols = None
        self.q_symbols = None
        self.sample_rate = None
        self.fft_result = None
        self.features = None

        # 输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_signal(self, custom_qpsk_params=None):
        """
        生成QPSK信号

        参数:
            custom_qpsk_params: 自定义信号生成参数，可覆盖默认值

        return:
            qpsk_signal:  生成的信号字典
        """
        # 合并默认参数和自定义参数
        used_qpsk_params = self.qpsk_params.copy()
        if custom_qpsk_params is not None:
            used_qpsk_params.update(custom_qpsk_params)

        # 生成信号
        qpsk_signal = self.signal_generator.generate_qpsk_signal(**used_qpsk_params)

        # 提取信号分量
        self.signal = qpsk_signal['signal']
        self.bits = qpsk_signal['bits']
        self.i_symbols = qpsk_signal['i_symbols']
        self.q_symbols = qpsk_signal['q_symbols']
        self.sample_rate = used_qpsk_params["sample_rate"]

        # 重置缓存的FFT结果
        self.fft_result = None

        return qpsk_signal

    def load_external_signal(self, signal, sample_rate, bits=None, i_symbols=None, q_symbols=None):
        """
        加载外部生成的信号

        参数:
            signal: 外部生成的信号
            sample_rate: 采样率
            bits: 信号对应的比
            i_symbols: I路符号
            q_symbols: Q路符号
        """
        if not isinstance(signal, cp.ndarray):
            self.signal = cp.asarray(signal, dtype=cp.complex64)
        else:
            self.signal = signal.astype(cp.complex64)

        self.sample_rate = sample_rate
        self.bits = bits
        self.i_symbols = i_symbols
        self.q_symbols = q_symbols
        self.fft_result = None

    def _compute_fft(self):
        """计算信号的快速傅里叶变换，结果缓存"""
        if self.fft_result is None and self.signal is not None:
            self.fft_result = cp.fft.fft(self.signal)
        return self.fft_result

    def extract_time_domain_features(self):
        """
        提取时间域统计特征

        return:
            dict: 包含各种时间域特征的字典
        """
        if self.signal is None:
            raise ValueError("请先生成或加载信号（调用generate_signal或load_external_signal方法）")

        # 计算I/Q分量
        i_component = cp.real(self.signal)
        q_component = cp.imag(self.signal)

        # 基本统计特征
        features = {
            # 幅度特征
            'amplitude_mean': cp.mean(cp.abs(self.signal)),
            'amplitude_std': cp.std(cp.abs(self.signal)),
            'amplitude_max': cp.max(cp.abs(self.signal)),
            'amplitude_min': cp.min(cp.abs(self.signal)),
            'amplitude_rms': cp.sqrt(cp.mean(cp.abs(self.signal) ** 2)),

            # I分量特征
            'i_mean': cp.mean(i_component),
            'i_std': cp.std(i_component),
            'i_max': cp.max(i_component),
            'i_min': cp.min(i_component),
            'i_rms': cp.sqrt(cp.mean(i_component ** 2)),

            # Q分量特征
            'q_mean': cp.mean(q_component),
            'q_std': cp.std(q_component),
            'q_max': cp.max(q_component),
            'q_min': cp.min(q_component),
            'q_rms': cp.sqrt(cp.mean(q_component ** 2)),

            # 相位特征（弧度）
            'phase_mean': cp.mean(cp.angle(self.signal)),
            'phase_std': cp.std(cp.angle(self.signal)),

            # 高阶统计量
            'amplitude_skewness': cp.mean(
                ((cp.abs(self.signal) - cp.mean(cp.abs(self.signal))) /
                 cp.std(cp.abs(self.signal))) ** 3
            ),
            'amplitude_kurtosis': cp.mean(
                ((cp.abs(self.signal) - cp.mean(cp.abs(self.signal))) /
                 cp.std(cp.abs(self.signal))) ** 4
            ) - 3
        }

        return {k: float(v.get()) for k, v in features.items()}

    def extract_frequency_features(self, num_bins=None):
        """
        提取频率分量功率特征

        参数:
            num_bins: 频率分箱的数量，若为None则使用默认参数

        return:
            dict: 包含各种频率域特征的字典
        """
        if self.signal is None:
            raise ValueError("请先生成或加载信号（调用generate_signal或load_external_signal方法）")

        # 使用指定的分箱数或默认值
        freq_bins = num_bins if num_bins is not None else self.feature_params["freq_bins"]

        # 获取FFT结果和频率轴
        fft_vals = self._compute_fft()
        n = len(fft_vals)
        freq_axis = cp.fft.fftfreq(n, 1 / self.sample_rate)

        # 计算功率谱
        power_spectrum = cp.abs(fft_vals) ** 2 / n

        # 频率分箱特征
        freq_bins_edges = cp.linspace(
            cp.min(freq_axis),
            cp.max(freq_axis),
            freq_bins + 1
        )

        bin_powers = []
        for i in range(freq_bins):
            mask = (freq_axis >= freq_bins_edges[i]) & (freq_axis < freq_bins_edges[i + 1])
            bin_power = cp.sum(power_spectrum[mask])
            bin_powers.append(bin_power)

        # 总功率
        total_power = cp.sum(power_spectrum)

        # 最大功率及其频率
        max_power_idx = cp.argmax(power_spectrum)
        max_power = power_spectrum[max_power_idx]
        max_power_freq = freq_axis[max_power_idx]

        # 频率特征字典
        features = {
            'total_power': total_power,
            'max_power': max_power,
            'max_power_frequency': max_power_freq,
            'normalized_max_power': max_power / total_power if total_power != 0 else 0
        }

        # 添加分箱功率及其归一化值
        for i, power in enumerate(bin_powers):
            features[f'bin_{i}_power'] = power
            if total_power != 0:
                features[f'bin_{i}_normalized_power'] = power / total_power
            else:
                features[f'bin_{i}_normalized_power'] = 0

        return {k: float(v.get()) for k, v in features.items()}

    def extract_all_features(self, freq_bins=None):
        """
        提取所有特征（时间域+频率域）

        参数:
            freq_bins: 频率分箱的数量，若为None则使用默认参数

        return:
            dict: 包含所有特征的字典
        """
        time_features = self.extract_time_domain_features()
        freq_features = self.extract_frequency_features(freq_bins)

        # 合并特征字典并保存
        self.features = {**time_features, **freq_features}
        return self.features

    def save_features(self, filename='features.npz'):
        """
        保存提取的特征到文件

        参数:
            filename: 保存的文件名
        """
        if self.features is None:
            raise ValueError("请先提取特征（调用extract_all_features方法）")

        save_path = os.path.join(self.output_dir, filename)
        np.savez(save_path, **self.features)
        print(f"特征已保存到: {save_path}")
        return save_path

    def visualize_features(self):
        """可视化信号和特征"""
        if self.signal is None:
            raise ValueError("请先生成或加载信号（调用generate_signal或load_external_signal方法）")

        # 转换到CPU进行绘图
        signal_cpu = cp.asnumpy(self.signal)
        i_component = np.real(signal_cpu)
        q_component = np.imag(signal_cpu)
        amplitude = np.abs(signal_cpu)
        phase = np.angle(signal_cpu)

        # 创建时间轴
        n_samples = len(signal_cpu)
        time = np.arange(n_samples) / self.sample_rate * 1e6  # 转换为微秒

        # 绘制时间域特征
        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 1)
        plt.plot(time, i_component)
        plt.title('I Component')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (μs)')
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(time, q_component)
        plt.title('Q Component')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (μs)')
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(time, amplitude)
        plt.title('Signal Amplitude')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (μs)')
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(time, phase)
        plt.title('Signal Phase')
        plt.ylabel('Phase (rad)')
        plt.xlabel('Time (μs)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_domain_features.png'), dpi=300)
        plt.show()

        # 绘制频率域特征
        plt.figure(figsize=(12, 6))

        fft_vals = cp.asnumpy(self._compute_fft())
        freq_axis = cp.asnumpy(cp.fft.fftfreq(n_samples, 1 / self.sample_rate))
        power_spectrum = np.abs(fft_vals) ** 2 / n_samples

        # 只显示正频率部分
        positive_mask = freq_axis >= 0
        freq_positive = freq_axis[positive_mask] / 1e6  # 转换为MHz
        power_positive = power_spectrum[positive_mask]

        plt.plot(freq_positive, 10 * np.log10(power_positive + 1e-10))
        plt.title('Power Spectrum')
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (MHz)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'frequency_spectrum.png'), dpi=300)
        plt.show()

        # 绘制星座图
        plt.figure(figsize=(8, 8))
        plt.scatter(i_component, q_component, s=3, alpha=0.5)
        plt.title('QPSK Signal Constellation')
        plt.xlabel('In-phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_dir, 'qpsk_constellation.png'), dpi=300)
        plt.show()

    def run(self, custom_qpsk_params=None, custom_feature_params=None):
        """
        完整的特征提取流程：生成信号→提取特征→可视化（可选）→保存特征

        参数:
            custom_qpsk_params: 自定义QPSK信号生成参数
            custom_feature_params: 自定义特征提取参数

        return:
            提取的特征字典
        """
        # 更新特征参数（如果提供）
        if custom_feature_params is not None:
            self.feature_params.update(custom_feature_params)

        # 生成信号
        self.generate_signal(custom_qpsk_params)

        # 提取特征
        features = self.extract_all_features()

        # 可视化
        if self.feature_params.get('visualize', True):
            self.visualize_features()

        # 保存特征
        self.save_features()

        return features


if __name__ == "__main__":
    # 默认参数运行
    default_extractor = Feature_Extractor(gpu_id=0)
    default_features = default_extractor.run()



    # # 自定义参数运行示例
    # custom_qpsk_params = {
    #     "bandwidth": 6e6,
    #     "sample_rate": 25e6,
    #     "duration": 0.002,
    #     "alpha": 0.4,
    #     "snr_db": 15
    # }
    #
    # custom_feature_params = {
    #     "freq_bins": 15,
    #     "visualize": True
    # }
    #
    # custom_extractor = QPSKFeatureExtractor(
    #     gpu_id=0,
    #     output_dir='./result/custom_features',
    #     default_qpsk_params=custom_qpsk_params,
    #     default_feature_params=custom_feature_params
    # )
    #
    # custom_features = custom_extractor.run()
    # print(f"使用自定义参数提取了 {len(custom_features)} 个特征")
    #
    # # 展示部分特征
    # print("\n部分时间域特征:")
    # for name in ['amplitude_mean', 'amplitude_std', 'i_mean', 'q_mean']:
    #     print(f"  {name}: {custom_features[name]:.6f}")
    #
    # print("\n部分频率域特征:")
    # for name in ['total_power', 'max_power', 'max_power_frequency']:
    #     print(f"  {name}: {custom_features[name]:.6f}")
