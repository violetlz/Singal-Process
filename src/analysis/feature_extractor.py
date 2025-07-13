"""
特征提取器
提供时域和频域特征提取功能
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

class FeatureExtractor:
    """
    特征提取器
    提供时域和频域特征提取功能
    """

    def __init__(self, gpu_processor=None):
        self.gpu_processor = gpu_processor

    def extract_time_domain_features(self, signal: cp.ndarray) -> Dict[str, float]:
        """
        提取时域统计特征

        Args:
            signal: 输入信号

        Returns:
            时域特征字典
        """
        features = {}

        # 基本统计特征
        features['mean'] = float(cp.mean(signal))
        features['std'] = float(cp.std(signal))
        features['variance'] = float(cp.var(signal))
        features['rms'] = float(cp.sqrt(cp.mean(signal ** 2)))
        features['peak'] = float(cp.max(cp.abs(signal)))
        features['peak_to_peak'] = float(cp.max(signal) - cp.min(signal))

        # 形状特征
        features['skewness'] = float(self._compute_skewness(signal))
        features['kurtosis'] = float(self._compute_kurtosis(signal))

        # 能量特征
        features['energy'] = float(cp.sum(signal ** 2))
        features['power'] = float(cp.mean(signal ** 2))

        # 过零率
        features['zero_crossing_rate'] = float(self._compute_zero_crossing_rate(signal))

        return features

    def extract_frequency_domain_features(self, signal: cp.ndarray,
                                        sample_rate: float) -> Dict[str, float]:
        """
        提取频域特征

        Args:
            signal: 输入信号
            sample_rate: 采样率

        Returns:
            频域特征字典
        """
        features = {}

        # 计算功率谱
        spectrum = cp.fft.rfft(signal)
        power_spectrum = cp.abs(spectrum) ** 2
        freq_axis = cp.linspace(0, sample_rate / 2, len(power_spectrum))

        # 频谱中心
        features['spectral_centroid'] = float(self._compute_spectral_centroid(power_spectrum, freq_axis))

        # 频谱带宽
        features['spectral_bandwidth'] = float(self._compute_spectral_bandwidth(power_spectrum, freq_axis))

        # 频谱熵
        features['spectral_entropy'] = float(self._compute_spectral_entropy(power_spectrum))

        # 频谱滚降点
        features['spectral_rolloff'] = float(self._compute_spectral_rolloff(power_spectrum, freq_axis))

        # 频谱通量
        features['spectral_flux'] = float(self._compute_spectral_flux(power_spectrum))

        return features

    def extract_frequency_band_power_ratios(self, signal: cp.ndarray,
                                          sample_rate: float,
                                          bands: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        提取频带功率比例特征

        Args:
            signal: 输入信号
            sample_rate: 采样率
            bands: 频带列表 [(low_freq, high_freq), ...]

        Returns:
            频带功率比例字典
        """
        # 计算功率谱
        spectrum = cp.fft.rfft(signal)
        power_spectrum = cp.abs(spectrum) ** 2
        freq_axis = cp.linspace(0, sample_rate / 2, len(power_spectrum))

        # 总功率
        total_power = cp.sum(power_spectrum)

        features = {}
        for i, (low_freq, high_freq) in enumerate(bands):
            # 找到频带内的索引
            mask = (freq_axis >= low_freq) & (freq_axis <= high_freq)
            band_power = cp.sum(power_spectrum[mask])

            # 计算功率比例
            power_ratio = float(band_power / total_power) if total_power > 0 else 0.0
            features[f'band_{i+1}_power_ratio'] = power_ratio

        return features

    def _compute_skewness(self, signal: cp.ndarray) -> float:
        """计算偏度"""
        mean = cp.mean(signal)
        std = cp.std(signal)
        if std == 0:
            return 0.0
        skewness = cp.mean(((signal - mean) / std) ** 3)
        return float(skewness)

    def _compute_kurtosis(self, signal: cp.ndarray) -> float:
        """计算峰度"""
        mean = cp.mean(signal)
        std = cp.std(signal)
        if std == 0:
            return 0.0
        kurtosis = cp.mean(((signal - mean) / std) ** 4) - 3
        return float(kurtosis)

    def _compute_zero_crossing_rate(self, signal: cp.ndarray) -> float:
        """计算过零率"""
        zero_crossings = cp.sum(cp.diff(cp.sign(signal)) != 0)
        return float(zero_crossings / (len(signal) - 1))

    def _compute_spectral_centroid(self, power_spectrum: cp.ndarray,
                                 freq_axis: cp.ndarray) -> float:
        """计算频谱中心"""
        weighted_sum = cp.sum(power_spectrum * freq_axis)
        total_power = cp.sum(power_spectrum)
        return float(weighted_sum / total_power) if total_power > 0 else 0.0

    def _compute_spectral_bandwidth(self, power_spectrum: cp.ndarray,
                                  freq_axis: cp.ndarray) -> float:
        """计算频谱带宽"""
        centroid = self._compute_spectral_centroid(power_spectrum, freq_axis)
        variance = cp.sum(power_spectrum * (freq_axis - centroid) ** 2)
        total_power = cp.sum(power_spectrum)
        bandwidth = cp.sqrt(variance / total_power) if total_power > 0 else 0.0
        return float(bandwidth)

    def _compute_spectral_entropy(self, power_spectrum: cp.ndarray) -> float:
        """计算频谱熵"""
        normalized_spectrum = power_spectrum / cp.sum(power_spectrum)
        normalized_spectrum = cp.where(normalized_spectrum > 0, normalized_spectrum, 1e-10)
        entropy = -cp.sum(normalized_spectrum * cp.log2(normalized_spectrum))
        return float(entropy)

    def _compute_spectral_rolloff(self, power_spectrum: cp.ndarray,
                                freq_axis: cp.ndarray, percentile: float = 0.85) -> float:
        """计算频谱滚降点"""
        total_power = cp.sum(power_spectrum)
        target_power = total_power * percentile

        cumulative_power = cp.cumsum(power_spectrum)
        rolloff_idx = cp.searchsorted(cumulative_power, target_power)

        if rolloff_idx >= len(freq_axis):
            return float(freq_axis[-1])

        return float(freq_axis[rolloff_idx])

    def _compute_spectral_flux(self, power_spectrum: cp.ndarray) -> float:
        """计算频谱通量"""
        # 简单的频谱通量计算（与前一帧的差异）
        if len(power_spectrum) < 2:
            return 0.0

        flux = cp.sum(cp.abs(cp.diff(power_spectrum)))
        return float(flux)
