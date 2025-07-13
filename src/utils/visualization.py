"""
信号可视化工具
提供各种信号和频谱的可视化功能
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import warnings

class SignalVisualizer:
    """
    信号可视化工具类
    提供各种信号和频谱的可视化功能
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_signal_and_spectrum(self, signal: cp.ndarray, sample_rate: float,
                                title: str = "信号分析", save_path: Optional[str] = None):
        """
        绘制信号和频谱

        Args:
            signal: 输入信号
            sample_rate: 采样率
            title: 图表标题
            save_path: 保存路径
        """
        # 计算FFT
        spectrum = cp.fft.fft(signal)
        freq_axis = cp.fft.fftfreq(len(signal), 1/sample_rate)

        # 转换为numpy数组用于绘图
        signal_np = cp.asnumpy(signal)
        spectrum_np = cp.asnumpy(cp.abs(spectrum))
        freq_axis_np = cp.asnumpy(freq_axis)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 绘制时域信号
        time_axis = np.arange(len(signal_np)) / sample_rate
        ax1.plot(time_axis, signal_np)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('幅度')
        ax1.set_title(f'{title} - 时域')
        ax1.grid(True)

        # 绘制频域信号
        ax2.plot(freq_axis_np, spectrum_np)
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('幅度')
        ax2.set_title(f'{title} - 频域')
        ax2.grid(True)
        ax2.set_xlim(-sample_rate/2, sample_rate/2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_stft_spectrogram(self, signal: cp.ndarray, sample_rate: float,
                             window_size: int = 1024, hop_size: int = 512,
                             title: str = "STFT频谱图", save_path: Optional[str] = None):
        """
        绘制STFT频谱图

        Args:
            signal: 输入信号
            sample_rate: 采样率
            window_size: 窗口大小
            hop_size: 跳跃大小
            title: 图表标题
            save_path: 保存路径
        """
        # 计算STFT
        stft_result = self._compute_stft(signal, window_size, hop_size)

        # 获取时间和频率轴
        time_axis = self._get_time_axis(len(signal), sample_rate, hop_size, window_size)
        freq_axis = cp.linspace(0, sample_rate / 2, stft_result.shape[1])

        # 转换为numpy数组用于绘图
        stft_magnitude = cp.asnumpy(cp.abs(stft_result))
        time_axis_np = cp.asnumpy(time_axis)
        freq_axis_np = cp.asnumpy(freq_axis)

        # 绘制频谱图
        plt.figure(figsize=(12, 6))
        im = plt.pcolormesh(time_axis_np, freq_axis_np, stft_magnitude.T,
                           shading='gouraud', cmap='viridis')
        plt.xlabel('时间 (s)')
        plt.ylabel('频率 (Hz)')
        plt.title(title)
        plt.colorbar(im, label='幅度')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_filter_response(self, b: cp.ndarray, a: cp.ndarray, sample_rate: float,
                           title: str = "滤波器频率响应", save_path: Optional[str] = None):
        """
        绘制滤波器频率响应

        Args:
            b: 分子系数
            a: 分母系数
            sample_rate: 采样率
            title: 图表标题
            save_path: 保存路径
        """
        # 计算频率响应
        w, h = self._compute_frequency_response(b, a, sample_rate)

        # 转换为numpy数组
        w_np = cp.asnumpy(w)
        h_np = cp.asnumpy(h)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 幅度响应
        ax1.semilogx(w_np, 20 * np.log10(np.abs(h_np)))
        ax1.set_xlabel('频率 (Hz)')
        ax1.set_ylabel('幅度 (dB)')
        ax1.set_title(f'{title} - 幅度响应')
        ax1.grid(True)
        ax1.set_xlim(w_np[1], w_np[-1])

        # 相位响应
        ax2.semilogx(w_np, np.angle(h_np, deg=True))
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('相位 (度)')
        ax2.set_title(f'{title} - 相位响应')
        ax2.grid(True)
        ax2.set_xlim(w_np[1], w_np[-1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_adaptive_filter_convergence(self, error_signal: cp.ndarray,
                                       filter_history: cp.ndarray,
                                       title: str = "自适应滤波器收敛",
                                       save_path: Optional[str] = None):
        """
        绘制自适应滤波器收敛过程

        Args:
            error_signal: 误差信号
            filter_history: 滤波器系数历史
            title: 图表标题
            save_path: 保存路径
        """
        # 转换为numpy数组
        error_np = cp.asnumpy(error_signal)
        history_np = cp.asnumpy(filter_history)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 误差信号
        ax1.plot(error_np)
        ax1.set_xlabel('样本')
        ax1.set_ylabel('误差')
        ax1.set_title(f'{title} - 误差信号')
        ax1.grid(True)

        # 滤波器系数收敛
        for i in range(min(5, history_np.shape[1])):  # 只显示前5个系数
            ax2.plot(history_np[:, i], label=f'系数 {i+1}')
        ax2.set_xlabel('样本')
        ax2.set_ylabel('系数值')
        ax2.set_title(f'{title} - 滤波器系数收敛')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_feature_comparison(self, features_dict: Dict[str, Dict[str, float]],
                              title: str = "特征对比", save_path: Optional[str] = None):
        """
        绘制特征对比图

        Args:
            features_dict: 特征字典 {信号名: {特征名: 值}}
            title: 图表标题
            save_path: 保存路径
        """
        # 提取特征名和信号名
        signal_names = list(features_dict.keys())
        feature_names = list(features_dict[signal_names[0]].keys())

        # 创建数据矩阵
        data = []
        for signal_name in signal_names:
            row = [features_dict[signal_name][feature_name] for feature_name in feature_names]
            data.append(row)

        data = np.array(data)

        # 创建热力图
        plt.figure(figsize=(max(8, len(feature_names)*0.8), max(6, len(signal_names)*0.4)))
        sns.heatmap(data, annot=True, fmt='.3f',
                   xticklabels=feature_names, yticklabels=signal_names,
                   cmap='viridis', cbar_kws={'label': '特征值'})
        plt.title(title)
        plt.xlabel('特征')
        plt.ylabel('信号')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _compute_stft(self, signal: cp.ndarray, window_size: int, hop_size: int) -> cp.ndarray:
        """计算STFT"""
        # 创建窗口函数
        window_func = cp.hanning(window_size)

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

    def _get_time_axis(self, signal_length: int, sample_rate: float,
                      hop_size: int, window_size: int) -> cp.ndarray:
        """获取时间轴"""
        num_frames = 1 + (signal_length - window_size) // hop_size
        frame_times = cp.arange(num_frames) * hop_size / sample_rate
        return frame_times

    def _compute_frequency_response(self, b: cp.ndarray, a: cp.ndarray,
                                  sample_rate: float) -> Tuple[cp.ndarray, cp.ndarray]:
        """计算频率响应"""
        # 计算频率点
        w = cp.logspace(-3, 0, 1000) * sample_rate / 2

        # 计算频率响应
        h = cp.zeros(len(w), dtype=cp.complex128)
        for i, freq in enumerate(w):
            z = cp.exp(1j * 2 * cp.pi * freq / sample_rate)
            h[i] = cp.polyval(b, z) / cp.polyval(a, z)

        return w, h
