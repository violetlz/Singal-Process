"""
STFT处理器
提供短时傅里叶变换和逆变换功能
"""

import cupy as cp
import numpy as np
from typing import Optional, Union, Tuple
import warnings

class STFTProcessor:
    """
    STFT处理器类
    提供短时傅里叶变换和逆变换功能
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor

    def stft(self, signal: cp.ndarray, window_size: int = 1024,
             hop_size: int = 512, window: str = 'hann') -> cp.ndarray:
        """
        执行短时傅里叶变换

        Args:
            signal: 输入信号 (GPU数组)
            window_size: 窗口大小
            hop_size: 跳跃大小
            window: 窗口类型 ('hann', 'hamming', 'blackman', 'rect')

        Returns:
            复数时频图 [时间帧, 频率]
        """
        if not isinstance(signal, cp.ndarray):
            signal = cp.asarray(signal)

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
                # 处理最后一帧，用零填充
                frame = cp.concatenate([signal[start_idx:], cp.zeros(end_idx - len(signal))])
            else:
                frame = signal[start_idx:end_idx]

            # 应用窗口并执行FFT
            windowed_frame = frame * window_func
            spectrum = cp.fft.rfft(windowed_frame)
            stft_result[i, :] = spectrum

        return stft_result

    def istft(self, stft_result: cp.ndarray, hop_size: int = 512,
              window_size: Optional[int] = None) -> cp.ndarray:
        """
        执行逆短时傅里叶变换

        Args:
            stft_result: STFT结果 [时间帧, 频率]
            hop_size: 跳跃大小
            window_size: 窗口大小，如果为None则从STFT结果推断

        Returns:
            重构的信号
        """
        if window_size is None:
            window_size = (stft_result.shape[1] - 1) * 2

        # 计算信号长度
        signal_length = (stft_result.shape[0] - 1) * hop_size + window_size

        # 初始化输出信号
        output_signal = cp.zeros(signal_length, dtype=cp.float64)
        window_sum = cp.zeros(signal_length, dtype=cp.float64)

        # 创建窗口函数（使用与STFT相同的窗口）
        window_func = cp.hanning(window_size)

        # 逐帧重构
        for i in range(stft_result.shape[0]):
            start_idx = i * hop_size
            end_idx = start_idx + window_size

            # 执行IFFT
            frame = cp.fft.irfft(stft_result[i, :], n=window_size)

            # 应用窗口并累加
            windowed_frame = frame * window_func
            output_signal[start_idx:end_idx] += windowed_frame
            window_sum[start_idx:end_idx] += window_func

        # 归一化
        window_sum[window_sum == 0] = 1  # 避免除零
        output_signal /= window_sum

        return output_signal

    def get_time_axis(self, signal_length: int, sample_rate: float,
                     hop_size: int, window_size: int) -> cp.ndarray:
        """
        获取时间轴

        Args:
            signal_length: 信号长度
            sample_rate: 采样率
            hop_size: 跳跃大小
            window_size: 窗口大小

        Returns:
            时间轴
        """
        num_frames = 1 + (signal_length - window_size) // hop_size
        frame_times = cp.arange(num_frames) * hop_size / sample_rate
        return frame_times

    def get_frequency_axis(self, sample_rate: float, n_points: int) -> cp.ndarray:
        """
        获取频率轴

        Args:
            sample_rate: 采样率
            n_points: 频率点数

        Returns:
            频率轴
        """
        return cp.linspace(0, sample_rate / 2, n_points)

    def compute_magnitude_spectrogram(self, stft_result: cp.ndarray) -> cp.ndarray:
        """
        计算幅度谱图

        Args:
            stft_result: STFT结果

        Returns:
            幅度谱图
        """
        return cp.abs(stft_result)

    def compute_power_spectrogram(self, stft_result: cp.ndarray) -> cp.ndarray:
        """
        计算功率谱图

        Args:
            stft_result: STFT结果

        Returns:
            功率谱图
        """
        return cp.abs(stft_result) ** 2

    def compute_phase_spectrogram(self, stft_result: cp.ndarray) -> cp.ndarray:
        """
        计算相位谱图

        Args:
            stft_result: STFT结果

        Returns:
            相位谱图
        """
        return cp.angle(stft_result)

    def apply_spectral_mask(self, stft_result: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
        """
        应用频谱掩码

        Args:
            stft_result: STFT结果
            mask: 频谱掩码

        Returns:
            应用掩码后的STFT结果
        """
        return stft_result * mask
