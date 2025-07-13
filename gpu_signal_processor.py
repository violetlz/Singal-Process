import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

class GPUSignalProcessor:
    """
    GPU加速的信号处理器，使用CuPy实现FFT和STFT
    """

    def __init__(self, gpu_id: int = 0):
        """
        初始化GPU信号处理器

        Args:
            gpu_id: 指定使用的GPU设备ID
        """
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        self.device.use()
        print(f"使用GPU设备 {gpu_id}: {cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()}")

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
        # 确保信号在GPU上
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

    def generate_test_signals(self, sample_rate: float = 44100, duration: float = 2.0) -> dict:
        """
        生成测试信号

        Args:
            sample_rate: 采样率
            duration: 信号持续时间

        Returns:
            包含各种测试信号的字典
        """
        t = cp.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # 正弦波
        sine_1khz = cp.sin(2 * cp.pi * 1000 * t)
        sine_5khz = cp.sin(2 * cp.pi * 5000 * t)

        # 调频信号
        fm_signal = cp.sin(2 * cp.pi * (1000 * t + 500 * t**2))

        # 调幅信号
        am_signal = (1 + 0.5 * cp.sin(2 * cp.pi * 10 * t)) * cp.sin(2 * cp.pi * 2000 * t)

        # 白噪声
        noise = cp.random.normal(0, 0.1, len(t))

        # 复合信号
        composite = sine_1khz + 0.5 * sine_5khz + 0.3 * fm_signal + 0.1 * noise

        return {
            'time': t,
            'sine_1khz': sine_1khz,
            'sine_5khz': sine_5khz,
            'fm_signal': fm_signal,
            'am_signal': am_signal,
            'noise': noise,
            'composite': composite,
            'sample_rate': sample_rate
        }
