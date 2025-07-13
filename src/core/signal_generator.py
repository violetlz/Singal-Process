"""
信号生成器
提供各种测试信号的生成功能
"""

import cupy as cp
import numpy as np
from typing import Dict, Optional, Union, Tuple
import warnings

class SignalGenerator:
    """
    信号生成器类
    提供各种测试信号的生成功能
    """

    def __init__(self, gpu_processor=None):
        """
        初始化信号生成器

        Args:
            gpu_processor: GPU处理器实例
        """
        self.gpu_processor = gpu_processor

    def generate_sine_wave(self, frequency: float, sample_rate: float,
                          duration: float, amplitude: float = 1.0,
                          phase: float = 0.0) -> cp.ndarray:
        """
        生成正弦波信号

        Args:
            frequency: 频率 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            amplitude: 幅度
            phase: 相位 (rad)

        Returns:
            正弦波信号
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = amplitude * cp.sin(2 * cp.pi * frequency * t + phase)
        return signal

    def generate_chirp_signal(self, f0: float, f1: float, sample_rate: float,
                             duration: float, method: str = 'linear') -> cp.ndarray:
        """
        生成调频信号

        Args:
            f0: 起始频率 (Hz)
            f1: 结束频率 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            method: 调频方法 ('linear', 'quadratic', 'logarithmic')

        Returns:
            调频信号
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        if method == 'linear':
            frequency = f0 + (f1 - f0) * t / duration
        elif method == 'quadratic':
            frequency = f0 + (f1 - f0) * (t / duration) ** 2
        elif method == 'logarithmic':
            frequency = f0 * (f1 / f0) ** (t / duration)
        else:
            raise ValueError(f"不支持的调频方法: {method}")

        # 计算瞬时相位
        phase = 2 * cp.pi * cp.cumsum(frequency) / sample_rate
        signal = cp.sin(phase)

        return signal

    def generate_am_signal(self, carrier_freq: float, mod_freq: float,
                          sample_rate: float, duration: float,
                          mod_depth: float = 0.5) -> cp.ndarray:
        """
        生成调幅信号

        Args:
            carrier_freq: 载波频率 (Hz)
            mod_freq: 调制频率 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            mod_depth: 调制深度 (0-1)

        Returns:
            调幅信号
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # 载波信号
        carrier = cp.sin(2 * cp.pi * carrier_freq * t)

        # 调制信号
        modulation = 1 + mod_depth * cp.sin(2 * cp.pi * mod_freq * t)

        # 调幅信号
        am_signal = modulation * carrier

        return am_signal

    def generate_noise(self, sample_rate: float, duration: float,
                      noise_type: str = 'white', std: float = 1.0) -> cp.ndarray:
        """
        生成噪声信号

        Args:
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            noise_type: 噪声类型 ('white', 'pink', 'brown')
            std: 标准差

        Returns:
            噪声信号
        """
        n_samples = int(sample_rate * duration)

        if noise_type == 'white':
            noise = cp.random.normal(0, std, n_samples)
        elif noise_type == 'pink':
            # 简单的粉红噪声实现
            white_noise = cp.random.normal(0, std, n_samples)
            # 使用FFT实现1/f频谱
            spectrum = cp.fft.fft(white_noise)
            freq = cp.fft.fftfreq(n_samples, 1/sample_rate)
            # 避免除零
            freq[0] = 1
            pink_spectrum = spectrum / cp.sqrt(cp.abs(freq))
            pink_spectrum[0] = 0  # DC分量设为0
            noise = cp.real(cp.fft.ifft(pink_spectrum))
        elif noise_type == 'brown':
            # 布朗噪声 (随机游走)
            white_noise = cp.random.normal(0, std, n_samples)
            noise = cp.cumsum(white_noise)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")

        return noise

    def generate_composite_signal(self, sample_rate: float, duration: float,
                                components: Dict[str, Dict]) -> cp.ndarray:
        """
        生成复合信号

        Args:
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            components: 信号分量字典
                {
                    'sine_1': {'type': 'sine', 'freq': 1000, 'amp': 1.0},
                    'chirp_1': {'type': 'chirp', 'f0': 500, 'f1': 2000, 'amp': 0.5},
                    'noise_1': {'type': 'noise', 'std': 0.1}
                }

        Returns:
            复合信号
        """
        n_samples = int(sample_rate * duration)
        composite = cp.zeros(n_samples)

        for name, params in components.items():
            signal_type = params.get('type', 'sine')
            amplitude = params.get('amp', 1.0)

            if signal_type == 'sine':
                freq = params.get('freq', 1000)
                signal = self.generate_sine_wave(freq, sample_rate, duration, amplitude)
            elif signal_type == 'chirp':
                f0 = params.get('f0', 500)
                f1 = params.get('f1', 2000)
                signal = self.generate_chirp_signal(f0, f1, sample_rate, duration)
                signal *= amplitude
            elif signal_type == 'noise':
                std = params.get('std', 0.1)
                signal = self.generate_noise(sample_rate, duration, 'white', std)
                signal *= amplitude
            else:
                raise ValueError(f"不支持的信号类型: {signal_type}")

            composite += signal

        return composite

    def generate_test_signals(self, sample_rate: float = 44100,
                            duration: float = 2.0) -> Dict:
        """
        生成标准测试信号集

        Args:
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)

        Returns:
            测试信号字典
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # 生成各种测试信号
        signals = {
            'time': t,
            'sample_rate': sample_rate,
            'sine_1khz': self.generate_sine_wave(1000, sample_rate, duration),
            'sine_5khz': self.generate_sine_wave(5000, sample_rate, duration, 0.5),
            'chirp_linear': self.generate_chirp_signal(500, 2000, sample_rate, duration),
            'chirp_log': self.generate_chirp_signal(500, 2000, sample_rate, duration, 'logarithmic'),
            'am_signal': self.generate_am_signal(2000, 10, sample_rate, duration),
            'white_noise': self.generate_noise(sample_rate, duration, 'white', 0.1),
            'pink_noise': self.generate_noise(sample_rate, duration, 'pink', 0.1),
        }

        # 生成复合信号
        components = {
            'sine_1': {'type': 'sine', 'freq': 1000, 'amp': 1.0},
            'sine_2': {'type': 'sine', 'freq': 3000, 'amp': 0.5},
            'chirp_1': {'type': 'chirp', 'f0': 500, 'f1': 1500, 'amp': 0.3},
            'noise_1': {'type': 'noise', 'std': 0.1}
        }
        signals['composite'] = self.generate_composite_signal(sample_rate, duration, components)

        return signals

    def add_noise(self, signal: cp.ndarray, snr_db: float,
                  noise_type: str = 'white') -> cp.ndarray:
        """
        向信号添加指定信噪比的噪声

        Args:
            signal: 原始信号
            snr_db: 信噪比 (dB)
            noise_type: 噪声类型

        Returns:
            添加噪声后的信号
        """
        # 计算信号功率
        signal_power = cp.mean(signal ** 2)

        # 计算目标噪声功率
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear

        # 生成噪声
        noise = self.generate_noise(1, len(signal), noise_type, 1.0)

        # 调整噪声功率
        current_noise_power = cp.mean(noise ** 2)
        noise_scale = cp.sqrt(target_noise_power / current_noise_power)
        noise *= noise_scale

        return signal + noise

    def generate_random_bits(self, n_bits: int) -> cp.ndarray:
        """
        生成随机比特序列

        Args:
            n_bits: 比特数量

        Returns:
            随机比特序列 (0或1)
        """
        return cp.random.randint(0, 2, n_bits, dtype=cp.int32)

    def qpsk_modulation(self, bits: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        QPSK调制

        Args:
            bits: 输入比特序列

        Returns:
            (I路信号, Q路信号)
        """
        # 确保比特数量为偶数
        if len(bits) % 2 != 0:
            bits = cp.concatenate([bits, cp.array([0])])

        # 将比特分成I路和Q路
        i_bits = bits[::2]
        q_bits = bits[1::2]

        # QPSK星座图映射
        # 00 -> (1, 1), 01 -> (-1, 1), 10 -> (1, -1), 11 -> (-1, -1)
        i_symbols = 1 - 2 * i_bits  # 0->1, 1->-1
        q_symbols = 1 - 2 * q_bits  # 0->1, 1->-1

        return i_symbols, q_symbols

    def root_raised_cosine_filter(self, signal: cp.ndarray, sample_rate: float,
                                 symbol_rate: float, alpha: float = 0.35,
                                 filter_length: int = None) -> cp.ndarray:
        """
        升根余弦滤波器

        Args:
            signal: 输入信号
            sample_rate: 采样率 (Hz)
            symbol_rate: 符号率 (Hz)
            alpha: 滚降因子 (0-1)
            filter_length: 滤波器长度

        Returns:
            滤波后的信号
        """
        if filter_length is None:
            filter_length = int(8 * sample_rate / symbol_rate)

        # 计算滤波器参数
        t = cp.linspace(-filter_length/2, filter_length/2, filter_length) / (sample_rate / symbol_rate)

        # 避免除零
        t = cp.where(t == 0, 1e-10, t)

        # 升根余弦滤波器响应
        h = cp.zeros_like(t)

        # 主瓣
        main_condition = cp.abs(t) == 1 / (4 * alpha)
        h = cp.where(main_condition,
                     (1 + alpha) * cp.pi / 4 * cp.sin(cp.pi * (1 - alpha) / (4 * alpha)),
                     h)

        # 其他位置
        other_condition = ~main_condition
        h = cp.where(other_condition,
                     (cp.sin(cp.pi * t * (1 - alpha)) +
                      4 * alpha * t * cp.cos(cp.pi * t * (1 + alpha))) /
                     (cp.pi * t * (1 - (4 * alpha * t) ** 2)),
                     h)

        # 归一化
        h = h / cp.sum(h)

        # 应用滤波器
        filtered_signal = cp.convolve(signal, h, mode='same')

        return filtered_signal

    def generate_qpsk_signal(self, bandwidth: float = 5e6, sample_rate: float = 20e6,
                           duration: float = 1.0, alpha: float = 0.35,
                           snr_db: Optional[float] = None) -> Dict:
        """
        生成5MHz带宽的QPSK调制信号

        Args:
            bandwidth: 信号带宽 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            alpha: 升根余弦滤波器滚降因子
            snr_db: 信噪比 (dB)，None表示不添加噪声

        Returns:
            包含QPSK信号信息的字典
        """
        # 计算符号率 (带宽 = 符号率 * (1 + alpha))
        symbol_rate = bandwidth / (1 + alpha)

        # 计算符号数量
        n_symbols = int(symbol_rate * duration)

        # 生成随机比特 (QPSK每个符号2比特)
        n_bits = n_symbols * 2
        bits = self.generate_random_bits(n_bits)

        # QPSK调制
        i_symbols, q_symbols = self.qpsk_modulation(bits)

        # 上采样 (每个符号多个采样点)
        samples_per_symbol = int(sample_rate / symbol_rate)
        i_upsampled = cp.repeat(i_symbols, samples_per_symbol)
        q_upsampled = cp.repeat(q_symbols, samples_per_symbol)

        # 应用升根余弦滤波器
        i_filtered = self.root_raised_cosine_filter(i_upsampled, sample_rate, symbol_rate, alpha)
        q_filtered = self.root_raised_cosine_filter(q_upsampled, sample_rate, symbol_rate, alpha)

        # 生成载波
        t = cp.linspace(0, duration, len(i_filtered), endpoint=False)
        carrier_freq = bandwidth / 2  # 载波频率设为带宽的一半
        carrier_i = cp.cos(2 * cp.pi * carrier_freq * t)
        carrier_q = cp.sin(2 * cp.pi * carrier_freq * t)

        # 调制到载波
        modulated_signal = i_filtered * carrier_i + q_filtered * carrier_q

        # 添加噪声（如果指定）
        if snr_db is not None:
            modulated_signal = self.add_noise(modulated_signal, snr_db)

        # 返回结果
        result = {
            'signal': modulated_signal,
            'i_symbols': i_symbols,
            'q_symbols': q_symbols,
            'bits': bits,
            'symbol_rate': symbol_rate,
            'sample_rate': sample_rate,
            'bandwidth': bandwidth,
            'alpha': alpha,
            'carrier_freq': carrier_freq,
            'duration': duration,
            'snr_db': snr_db,
            'time': t
        }

        return result

    def generate_multiple_qpsk_signals(self, n_signals: int = 5,
                                     bandwidth: float = 5e6,
                                     sample_rate: float = 20e6,
                                     duration: float = 1.0,
                                     alpha: float = 0.35,
                                     snr_db: Optional[float] = None) -> Dict:
        """
        生成多个QPSK信号用于测试

        Args:
            n_signals: 信号数量
            bandwidth: 信号带宽 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            alpha: 升根余弦滤波器滚降因子
            snr_db: 信噪比 (dB)

        Returns:
            包含多个QPSK信号的字典
        """
        signals = {}

        for i in range(n_signals):
            signal_name = f'qpsk_signal_{i+1}'
            signals[signal_name] = self.generate_qpsk_signal(
                bandwidth=bandwidth,
                sample_rate=sample_rate,
                duration=duration,
                alpha=alpha,
                snr_db=snr_db
            )

        return signals

    def generate_single_tone_signal(self, frequency: float = 1000, sample_rate: float = 44100, duration: float = 1.0, amplitude: float = 1.0, phase: float = 0.0) -> cp.ndarray:
        """
        生成单音信号（单一频率正弦波）

        Args:
            frequency: 频率 (Hz)
            sample_rate: 采样率 (Hz)
            duration: 持续时间 (s)
            amplitude: 幅度
            phase: 初始相位 (rad)

        Returns:
            单音信号
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = amplitude * cp.sin(2 * cp.pi * frequency * t + phase)
        return signal
