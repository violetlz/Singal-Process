"""
FFT处理器
提供快速傅里叶变换和逆变换功能
"""

import cupy as cp
import numpy as np
from typing import Optional, Union, Tuple
import warnings

class FFTProcessor:
    """
    FFT处理器类
    提供快速傅里叶变换和逆变换功能
    """

    def __init__(self, gpu_processor=None):
        """
        初始化FFT处理器

        Args:
            gpu_processor: GPU处理器实例
        """
        self.gpu_processor = gpu_processor

    def fft(self, signal: cp.ndarray, n: Optional[int] = None) -> cp.ndarray:
        """
        执行快速傅里叶变换

        Args:
            signal: 输入信号 (GPU数组)
            n: FFT长度，如果为None则使用信号长度

        Returns:
            复数频谱
        """
        if n is None:
            n = len(signal)

        # 确保信号在GPU上
        if not isinstance(signal, cp.ndarray):
            signal = cp.asarray(signal)

        # 执行FFT
        spectrum = cp.fft.fft(signal, n=n)
        return spectrum

    def ifft(self, spectrum: cp.ndarray, n: Optional[int] = None) -> cp.ndarray:
        """
        执行逆快速傅里叶变换

        Args:
            spectrum: 复数频谱 (GPU数组)
            n: IFFT长度，如果为None则使用频谱长度

        Returns:
            重构的信号
        """
        if n is None:
            n = len(spectrum)

        # 确保频谱在GPU上
        if not isinstance(spectrum, cp.ndarray):
            spectrum = cp.asarray(spectrum)

        # 执行IFFT
        signal = cp.fft.ifft(spectrum, n=n)
        return signal

    def rfft(self, signal: cp.ndarray, n: Optional[int] = None) -> cp.ndarray:
        """
        执行实数FFT（只返回正频率部分）

        Args:
            signal: 输入信号 (GPU数组)
            n: FFT长度，如果为None则使用信号长度

        Returns:
            实数频谱（正频率部分）
        """
        if n is None:
            n = len(signal)

        # 确保信号在GPU上
        if not isinstance(signal, cp.ndarray):
            signal = cp.asarray(signal)

        # 执行实数FFT
        spectrum = cp.fft.rfft(signal, n=n)
        return spectrum

    def irfft(self, spectrum: cp.ndarray, n: Optional[int] = None) -> cp.ndarray:
        """
        执行逆实数FFT

        Args:
            spectrum: 实数频谱 (GPU数组)
            n: IFFT长度，如果为None则从频谱长度推断

        Returns:
            重构的实数信号
        """
        if n is None:
            n = (len(spectrum) - 1) * 2

        # 确保频谱在GPU上
        if not isinstance(spectrum, cp.ndarray):
            spectrum = cp.asarray(spectrum)

        # 执行逆实数FFT
        signal = cp.fft.irfft(spectrum, n=n)
        return signal

    def fftshift(self, spectrum: cp.ndarray) -> cp.ndarray:
        """
        将频谱的零频率分量移到中心

        Args:
            spectrum: 输入频谱

        Returns:
            移位后的频谱
        """
        return cp.fft.fftshift(spectrum)

    def ifftshift(self, spectrum: cp.ndarray) -> cp.ndarray:
        """
        将频谱的零频率分量移回原位置

        Args:
            spectrum: 输入频谱

        Returns:
            移位后的频谱
        """
        return cp.fft.ifftshift(spectrum)

    def get_frequency_axis(self, sample_rate: float, n_points: int,
                          shift: bool = False) -> cp.ndarray:
        """
        获取频率轴

        Args:
            sample_rate: 采样率
            n_points: 频率点数
            shift: 是否进行fftshift

        Returns:
            频率轴
        """
        if shift:
            freq_axis = cp.fft.fftfreq(n_points, 1/sample_rate)
        else:
            freq_axis = cp.linspace(0, sample_rate/2, n_points)

        return freq_axis

    def get_power_spectrum(self, signal: cp.ndarray,
                          normalize: bool = True) -> cp.ndarray:
        """
        计算功率谱

        Args:
            signal: 输入信号
            normalize: 是否归一化

        Returns:
            功率谱
        """
        spectrum = self.fft(signal)
        power_spectrum = cp.abs(spectrum) ** 2

        if normalize:
            power_spectrum /= len(signal)

        return power_spectrum

    def get_magnitude_spectrum(self, signal: cp.ndarray,
                             normalize: bool = True) -> cp.ndarray:
        """
        计算幅度谱

        Args:
            signal: 输入信号
            normalize: 是否归一化

        Returns:
            幅度谱
        """
        spectrum = self.fft(signal)
        magnitude_spectrum = cp.abs(spectrum)

        if normalize:
            magnitude_spectrum /= len(signal)

        return magnitude_spectrum

    def get_phase_spectrum(self, signal: cp.ndarray) -> cp.ndarray:
        """
        计算相位谱

        Args:
            signal: 输入信号

        Returns:
            相位谱
        """
        spectrum = self.fft(signal)
        phase_spectrum = cp.angle(spectrum)
        return phase_spectrum

    def zero_padding(self, signal: cp.ndarray, target_length: int) -> cp.ndarray:
        """
        对信号进行零填充

        Args:
            signal: 输入信号
            target_length: 目标长度

        Returns:
            零填充后的信号
        """
        if len(signal) >= target_length:
            return signal

        padded_signal = cp.zeros(target_length, dtype=signal.dtype)
        padded_signal[:len(signal)] = signal

        return padded_signal

    def window_fft(self, signal: cp.ndarray, window: str = 'hann',
                  n: Optional[int] = None) -> cp.ndarray:
        """
        应用窗函数后进行FFT

        Args:
            signal: 输入信号
            window: 窗函数类型 ('hann', 'hamming', 'blackman', 'rect')
            n: FFT长度

        Returns:
            窗函数FFT结果
        """
        if n is None:
            n = len(signal)

        # 创建窗函数
        if window == 'hann':
            window_func = cp.hanning(len(signal))
        elif window == 'hamming':
            window_func = cp.hamming(len(signal))
        elif window == 'blackman':
            window_func = cp.blackman(len(signal))
        elif window == 'rect':
            window_func = cp.ones(len(signal))
        else:
            raise ValueError(f"不支持的窗函数类型: {window}")

        # 应用窗函数
        windowed_signal = signal * window_func

        # 执行FFT
        spectrum = self.fft(windowed_signal, n=n)

        return spectrum
