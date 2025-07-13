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
