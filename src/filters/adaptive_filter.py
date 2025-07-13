"""
自适应滤波器
提供LMS自适应滤波算法
"""

import cupy as cp
import numpy as np
from typing import Tuple

class AdaptiveFilter:
    """
    自适应滤波器类
    提供LMS自适应滤波算法
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor

    def lms_filter(self, input_signal: cp.ndarray, desired_signal: cp.ndarray,
                   filter_length: int = 64, mu: float = 0.01) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        LMS（最小均方）自适应滤波器

        Args:
            input_signal: 输入信号
            desired_signal: 期望信号
            filter_length: 滤波器长度
            mu: 步长参数

        Returns:
            (输出信号, 误差信号)
        """
        # 初始化
        n_samples = len(input_signal)
        output_signal = cp.zeros(n_samples)
        error_signal = cp.zeros(n_samples)
        filter_coeffs = cp.zeros(filter_length)

        # LMS算法
        for n in range(n_samples):
            # 构建输入向量
            if n < filter_length - 1:
                x = cp.concatenate([cp.zeros(filter_length - n - 1), input_signal[:n+1]])
            else:
                x = input_signal[n-filter_length+1:n+1]

            # 计算输出
            output_signal[n] = cp.dot(filter_coeffs, x)

            # 计算误差
            error_signal[n] = desired_signal[n] - output_signal[n]

            # 更新滤波器系数
            filter_coeffs += mu * error_signal[n] * x

        return output_signal, error_signal

    def nlms_filter(self, input_signal: cp.ndarray, desired_signal: cp.ndarray,
                    filter_length: int = 64, mu: float = 0.1,
                    epsilon: float = 1e-6) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        归一化LMS自适应滤波器

        Args:
            input_signal: 输入信号
            desired_signal: 期望信号
            filter_length: 滤波器长度
            mu: 步长参数
            epsilon: 正则化参数

        Returns:
            (输出信号, 误差信号, 滤波器系数历史)
        """
        # 初始化
        n_samples = len(input_signal)
        output_signal = cp.zeros(n_samples)
        error_signal = cp.zeros(n_samples)
        filter_coeffs = cp.zeros(filter_length)
        filter_history = cp.zeros((n_samples, filter_length))

        # NLMS算法
        for n in range(n_samples):
            # 构建输入向量
            if n < filter_length - 1:
                x = cp.concatenate([cp.zeros(filter_length - n - 1), input_signal[:n+1]])
            else:
                x = input_signal[n-filter_length+1:n+1]

            # 计算输出
            output_signal[n] = cp.dot(filter_coeffs, x)

            # 计算误差
            error_signal[n] = desired_signal[n] - output_signal[n]

            # 计算归一化步长
            x_norm = cp.dot(x, x) + epsilon
            normalized_mu = mu / x_norm

            # 更新滤波器系数
            filter_coeffs += normalized_mu * error_signal[n] * x

            # 保存滤波器系数历史
            filter_history[n] = filter_coeffs.copy()

        return output_signal, error_signal, filter_history

    def rls_filter(self, input_signal: cp.ndarray, desired_signal: cp.ndarray,
                   filter_length: int = 64, lambda_: float = 0.99,
                   delta: float = 1.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        RLS（递归最小二乘）自适应滤波器

        Args:
            input_signal: 输入信号
            desired_signal: 期望信号
            filter_length: 滤波器长度
            lambda_: 遗忘因子
            delta: 初始化参数

        Returns:
            (输出信号, 误差信号, 滤波器系数历史)
        """
        # 初始化
        n_samples = len(input_signal)
        output_signal = cp.zeros(n_samples)
        error_signal = cp.zeros(n_samples)
        filter_coeffs = cp.zeros(filter_length)
        filter_history = cp.zeros((n_samples, filter_length))

        # 初始化P矩阵
        P = cp.eye(filter_length) / delta

        # RLS算法
        for n in range(n_samples):
            # 构建输入向量
            if n < filter_length - 1:
                x = cp.concatenate([cp.zeros(filter_length - n - 1), input_signal[:n+1]])
            else:
                x = input_signal[n-filter_length+1:n+1]

            # 计算增益向量
            k = cp.dot(P, x) / (lambda_ + cp.dot(cp.dot(x, P), x))

            # 计算输出
            output_signal[n] = cp.dot(filter_coeffs, x)

            # 计算误差
            error_signal[n] = desired_signal[n] - output_signal[n]

            # 更新滤波器系数
            filter_coeffs += k * error_signal[n]

            # 更新P矩阵
            P = (P - cp.outer(k, cp.dot(x, P))) / lambda_

            # 保存滤波器系数历史
            filter_history[n] = filter_coeffs.copy()

        return output_signal, error_signal, filter_history

    def kalman_filter(self, input_signal: cp.ndarray, measurement_noise: float = 0.1,
                     process_noise: float = 0.01) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        卡尔曼滤波器

        Args:
            input_signal: 输入信号
            measurement_noise: 测量噪声方差
            process_noise: 过程噪声方差

        Returns:
            (滤波后的信号, 状态估计)
        """
        n_samples = len(input_signal)
        filtered_signal = cp.zeros(n_samples)
        state_estimate = cp.zeros(n_samples)

        # 初始化
        x_hat = input_signal[0]  # 初始状态估计
        P = 1.0  # 初始协方差

        for n in range(n_samples):
            # 预测步骤
            x_hat_minus = x_hat
            P_minus = P + process_noise

            # 更新步骤
            K = P_minus / (P_minus + measurement_noise)  # 卡尔曼增益
            x_hat = x_hat_minus + K * (input_signal[n] - x_hat_minus)
            P = (1 - K) * P_minus

            # 保存结果
            filtered_signal[n] = x_hat
            state_estimate[n] = x_hat

        return filtered_signal, state_estimate

    def wiener_filter(self, input_signal: cp.ndarray, noise_psd: cp.ndarray,
                     signal_psd: cp.ndarray) -> cp.ndarray:
        """
        维纳滤波器

        Args:
            input_signal: 输入信号
            noise_psd: 噪声功率谱密度
            signal_psd: 信号功率谱密度

        Returns:
            滤波后的信号
        """
        # 计算维纳滤波器频率响应
        H = signal_psd / (signal_psd + noise_psd)

        # 应用滤波器
        spectrum = cp.fft.fft(input_signal)
        filtered_spectrum = spectrum * H
        filtered_signal = cp.real(cp.fft.ifft(filtered_spectrum))

        return filtered_signal

    def adaptive_noise_cancellation(self, primary_signal: cp.ndarray,
                                  reference_signal: cp.ndarray,
                                  filter_length: int = 64, mu: float = 0.01) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        自适应噪声消除

        Args:
            primary_signal: 主信号（包含噪声）
            reference_signal: 参考信号（噪声参考）
            filter_length: 滤波器长度
            mu: 步长参数

        Returns:
            (消除噪声后的信号, 滤波器输出)
        """
        # 使用LMS算法训练滤波器
        output_signal, error_signal = self.lms_filter(reference_signal, primary_signal,
                                             filter_length, mu)

        # 使用最终的滤波器系数进行噪声消除
        final_coeffs = error_signal[-1]
        filtered_reference = cp.convolve(reference_signal, final_coeffs, mode='same')

        # 从主信号中减去滤波后的参考信号
        noise_cancelled_signal = primary_signal - filtered_reference

        return noise_cancelled_signal, filtered_reference
