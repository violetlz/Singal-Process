"""
频谱分析器
提供功率谱密度估计、频谱峰值检测等功能
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

class SpectralAnalyzer:
    """
    频谱分析器
    提供功率谱密度估计、频谱峰值检测等功能
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor

    def welch_psd(self, signal: cp.ndarray, sample_rate: float,
                  window_size: int = 1024, hop_size: int = 512,
                  window: str = 'hann') -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Welch方法功率谱密度估计

        Args:
            signal: 输入信号
            sample_rate: 采样率
            window_size: 窗口大小
            hop_size: 跳跃大小
            window: 窗口类型

        Returns:
            (频率轴, 功率谱密度)
        """
        # 计算STFT
        stft_result = self._compute_stft(signal, window_size, hop_size, window)

        # 计算功率谱
        power_spectrum = cp.abs(stft_result) ** 2

        # 平均所有帧
        psd = cp.mean(power_spectrum, axis=0)

        # 归一化
        psd /= (sample_rate * window_size)

        # 频率轴
        freq_axis = cp.linspace(0, sample_rate / 2, len(psd))

        return freq_axis, psd

    def find_peaks(self, spectrum: cp.ndarray, threshold: float = 0.1,
                   min_distance: int = 10) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        频谱峰值检测

        Args:
            spectrum: 频谱
            threshold: 阈值
            min_distance: 最小峰值距离

        Returns:
            (峰值位置, 峰值幅度)
        """
        # 找到局部最大值
        peaks = []
        peak_values = []

        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold):

                # 检查最小距离
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
                    peak_values.append(spectrum[i])

        return cp.array(peaks), cp.array(peak_values)

    def estimate_spectral_centroid(self, spectrum: cp.ndarray,
                                 freq_axis: cp.ndarray) -> float:
        """
        估计频谱中心

        Args:
            spectrum: 功率谱
            freq_axis: 频率轴

        Returns:
            频谱中心频率
        """
        weighted_sum = cp.sum(spectrum * freq_axis)
        total_power = cp.sum(spectrum)

        if total_power == 0:
            return 0.0

        return float(weighted_sum / total_power)

    def estimate_spectral_bandwidth(self, spectrum: cp.ndarray,
                                  freq_axis: cp.ndarray) -> float:
        """
        估计频谱带宽

        Args:
            spectrum: 功率谱
            freq_axis: 频率轴

        Returns:
            频谱带宽
        """
        centroid = self.estimate_spectral_centroid(spectrum, freq_axis)

        # 计算二阶矩
        variance = cp.sum(spectrum * (freq_axis - centroid) ** 2)
        total_power = cp.sum(spectrum)

        if total_power == 0:
            return 0.0

        bandwidth = cp.sqrt(variance / total_power)
        return float(bandwidth)

    def compute_spectral_entropy(self, spectrum: cp.ndarray) -> float:
        """
        计算频谱熵

        Args:
            spectrum: 功率谱

        Returns:
            频谱熵
        """
        # 归一化
        normalized_spectrum = spectrum / cp.sum(spectrum)

        # 避免log(0)
        normalized_spectrum = cp.where(normalized_spectrum > 0,
                                     normalized_spectrum, 1e-10)

        # 计算熵
        entropy = -cp.sum(normalized_spectrum * cp.log2(normalized_spectrum))
        return float(entropy)

    def _compute_stft(self, signal: cp.ndarray, window_size: int,
                     hop_size: int, window: str) -> cp.ndarray:
        """计算STFT"""
        # 创建窗口函数
        if window == 'hann':
            window_func = cp.hanning(window_size)
        elif window == 'hamming':
            window_func = cp.hamming(window_size)
        elif window == 'blackman':
            window_func = cp.blackman(window_size)
        elif window == 'rect':
            window_func = cp.ones(window_size)
        else:
            raise ValueError(f"不支持的窗口类型: {window}")

        # 计算帧数
        num_frames = 1 + (len(signal) - window_size) // hop_size

        # 初始化STFT结果
        stft_result = cp.zeros((num_frames, window_size // 2 + 1), dtype=cp.complex128)

        # 逐帧处理
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + window_size

            if end_idx > len(signal):
                frame = cp.concatenate([signal[start_idx:], cp.zeros(end_idx - len(signal))])
            else:
                frame = signal[start_idx:end_idx]

            # 应用窗口并执行FFT
            windowed_frame = frame * window_func
            spectrum = cp.fft.rfft(windowed_frame)
            stft_result[i, :] = spectrum

        return stft_result
