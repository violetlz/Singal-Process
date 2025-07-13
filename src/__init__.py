"""
GPU信号处理库
基于CuPy的GPU加速信号处理工具包
"""

__version__ = "1.0.0"
__author__ = "Signal Processing Team"

from .core.gpu_processor import GPUSignalProcessor
from .core.signal_generator import SignalGenerator
from .transforms.fft_processor import FFTProcessor
from .transforms.stft_processor import STFTProcessor
from .analysis.spectral_analyzer import SpectralAnalyzer
from .analysis.feature_extractor import FeatureExtractor
from .filters.filter_bank import FilterBank
from .filters.adaptive_filter import AdaptiveFilter
from .utils.visualization import SignalVisualizer
from .utils.performance_monitor import PerformanceMonitor

__all__ = [
    'GPUSignalProcessor',
    'SignalGenerator',
    'FFTProcessor',
    'STFTProcessor',
    'SpectralAnalyzer',
    'FeatureExtractor',
    'FilterBank',
    'AdaptiveFilter',
    'SignalVisualizer',
    'PerformanceMonitor'
]
