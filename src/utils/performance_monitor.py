"""
性能监控工具
提供GPU和CPU性能监控功能
"""

import cupy as cp
import numpy as np
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
import warnings

class PerformanceMonitor:
    """
    性能监控工具类
    提供GPU和CPU性能监控功能
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor
        self.performance_history = []

    def benchmark_function(self, func: Callable, *args,
                          num_runs: int = 10, warmup_runs: int = 3,
                          **kwargs) -> Dict[str, float]:
        """
        基准测试函数性能

        Args:
            func: 要测试的函数
            *args: 函数参数
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数
            **kwargs: 函数关键字参数

        Returns:
            性能统计字典
        """
        # 预热运行
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # 同步GPU
        if self.gpu_processor:
            self.gpu_processor.synchronize()

        # 性能测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = func(*args, **kwargs)

            # 同步GPU
            if self.gpu_processor:
                self.gpu_processor.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

        # 计算统计信息
        times = np.array(times)
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'total_time': float(np.sum(times))
        }

        return stats

    def compare_gpu_cpu_performance(self, gpu_func: Callable, cpu_func: Callable,
                                  *args, num_runs: int = 10, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        比较GPU和CPU性能

        Args:
            gpu_func: GPU函数
            cpu_func: CPU函数
            *args: 函数参数
            num_runs: 测试运行次数
            **kwargs: 函数关键字参数

        Returns:
            性能比较结果
        """
        # GPU性能测试
        gpu_stats = self.benchmark_function(gpu_func, *args, num_runs=num_runs, **kwargs)

        # CPU性能测试
        cpu_stats = self.benchmark_function(cpu_func, *args, num_runs=num_runs, **kwargs)

        # 计算加速比
        speedup = cpu_stats['mean_time'] / gpu_stats['mean_time']

        results = {
            'gpu': gpu_stats,
            'cpu': cpu_stats,
            'speedup': speedup
        }

        return results

    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        监控GPU内存使用情况

        Returns:
            内存使用信息
        """
        if not self.gpu_processor:
            return {}

        free, total = self.gpu_processor.get_memory_info()
        used = total - free

        memory_info = {
            'total_memory_gb': total / (1024**3),
            'used_memory_gb': used / (1024**3),
            'free_memory_gb': free / (1024**3),
            'memory_usage_percent': (used / total) * 100
        }

        return memory_info

    def profile_fft_performance(self, signal_lengths: List[int],
                              sample_rate: float = 44100) -> Dict[str, List[float]]:
        """
        分析FFT性能

        Args:
            signal_lengths: 信号长度列表
            sample_rate: 采样率

        Returns:
            FFT性能分析结果
        """
        gpu_times = []
        cpu_times = []

        for length in signal_lengths:
            # 生成测试信号
            t = cp.linspace(0, 1, length)
            signal = cp.sin(2 * cp.pi * 1000 * t) + cp.random.normal(0, 0.1, length)

            # GPU FFT
            gpu_stats = self.benchmark_function(cp.fft.fft, signal, num_runs=5)
            gpu_times.append(gpu_stats['mean_time'])

            # CPU FFT
            signal_cpu = cp.asnumpy(signal)
            cpu_stats = self.benchmark_function(np.fft.fft, signal_cpu, num_runs=5)
            cpu_times.append(cpu_stats['mean_time'])

        return {
            'signal_lengths': signal_lengths,
            'gpu_times': gpu_times,
            'cpu_times': cpu_times,
            'speedups': [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
        }

    def profile_stft_performance(self, signal_lengths: List[int],
                               window_size: int = 1024, hop_size: int = 512,
                               sample_rate: float = 44100) -> Dict[str, List[float]]:
        """
        分析STFT性能

        Args:
            signal_lengths: 信号长度列表
            window_size: 窗口大小
            hop_size: 跳跃大小
            sample_rate: 采样率

        Returns:
            STFT性能分析结果
        """
        gpu_times = []
        cpu_times = []

        for length in signal_lengths:
            # 生成测试信号
            t = cp.linspace(0, 1, length)
            signal = cp.sin(2 * cp.pi * 1000 * t) + cp.random.normal(0, 0.1, length)

            # GPU STFT
            def gpu_stft(sig):
                return self._compute_gpu_stft(sig, window_size, hop_size)

            gpu_stats = self.benchmark_function(gpu_stft, signal, num_runs=3)
            gpu_times.append(gpu_stats['mean_time'])

            # CPU STFT
            signal_cpu = cp.asnumpy(signal)
            def cpu_stft(sig):
                return self._compute_cpu_stft(sig, window_size, hop_size)

            cpu_stats = self.benchmark_function(cpu_stft, signal_cpu, num_runs=3)
            cpu_times.append(cpu_stats['mean_time'])

        return {
            'signal_lengths': signal_lengths,
            'gpu_times': gpu_times,
            'cpu_times': cpu_times,
            'speedups': [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
        }

    def get_performance_report(self) -> str:
        """
        生成性能报告

        Returns:
            性能报告字符串
        """
        report = "GPU信号处理性能报告\n"
        report += "=" * 50 + "\n\n"

        # GPU信息
        if self.gpu_processor:
            device_info = self.gpu_processor.get_device_info()
            report += f"GPU设备: {device_info['name']}\n"
            report += f"计算能力: {device_info['compute_capability']}\n"
            report += f"总内存: {device_info['memory_total'] / (1024**3):.2f} GB\n"
            report += f"多处理器数量: {device_info['multiprocessor_count']}\n\n"

        # 内存使用情况
        memory_info = self.monitor_memory_usage()
        if memory_info:
            report += "内存使用情况:\n"
            report += f"  总内存: {memory_info['total_memory_gb']:.2f} GB\n"
            report += f"  已用内存: {memory_info['used_memory_gb']:.2f} GB\n"
            report += f"  可用内存: {memory_info['free_memory_gb']:.2f} GB\n"
            report += f"  使用率: {memory_info['memory_usage_percent']:.1f}%\n\n"

        # 性能历史
        if self.performance_history:
            report += "性能历史记录:\n"
            for i, record in enumerate(self.performance_history[-5:]):  # 显示最近5条记录
                report += f"  记录 {i+1}: {record}\n"

        return report

    def _compute_gpu_stft(self, signal: cp.ndarray, window_size: int, hop_size: int) -> cp.ndarray:
        """GPU STFT计算"""
        window_func = cp.hanning(window_size)
        num_frames = 1 + (len(signal) - window_size) // hop_size
        stft_result = cp.zeros((num_frames, window_size // 2 + 1), dtype=cp.complex128)

        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + window_size

            if end_idx > len(signal):
                frame = cp.concatenate([signal[start_idx:], cp.zeros(end_idx - len(signal))])
            else:
                frame = signal[start_idx:end_idx]

            windowed_frame = frame * window_func
            spectrum = cp.fft.rfft(windowed_frame)
            stft_result[i, :] = spectrum

        return stft_result

    def _compute_cpu_stft(self, signal: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
        """CPU STFT计算"""
        window_func = np.hanning(window_size)
        num_frames = 1 + (len(signal) - window_size) // hop_size
        stft_result = np.zeros((num_frames, window_size // 2 + 1), dtype=np.complex128)

        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + window_size

            if end_idx > len(signal):
                frame = np.concatenate([signal[start_idx:], np.zeros(end_idx - len(signal))])
            else:
                frame = signal[start_idx:end_idx]

            windowed_frame = frame * window_func
            spectrum = np.fft.rfft(windowed_frame)
            stft_result[i, :] = spectrum

        return stft_result
