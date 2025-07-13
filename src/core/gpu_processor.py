"""
GPU处理器核心类
提供GPU设备管理和基础功能
"""

import cupy as cp
import numpy as np
from typing import Optional, Union, Tuple
import warnings

class GPUSignalProcessor:
    """
    GPU信号处理器核心类
    提供GPU设备管理和基础功能
    """

    def __init__(self, gpu_id: int = 0):
        """
        初始化GPU处理器

        Args:
            gpu_id: 指定使用的GPU设备ID
        """
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        self.device.use()

        # 获取GPU信息
        self.gpu_props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        self.gpu_name = self.gpu_props['name'].decode()
        self.memory_total = self.gpu_props['totalGlobalMem']
        self.memory_free = cp.cuda.runtime.memGetInfo()[0]

        print(f"使用GPU设备 {gpu_id}: {self.gpu_name}")
        print(f"GPU内存: {self.memory_total / 1024**3:.2f} GB")

    def get_memory_info(self) -> Tuple[int, int]:
        """
        获取GPU内存信息

        Returns:
            (free_memory, total_memory) in bytes
        """
        free, total = cp.cuda.runtime.memGetInfo()
        return free, total

    def clear_memory(self):
        """清理GPU内存"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()

    def to_gpu(self, data: Union[np.ndarray, cp.ndarray]) -> cp.ndarray:
        """
        将数据转移到GPU

        Args:
            data: 输入数据

        Returns:
            GPU数组
        """
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif isinstance(data, cp.ndarray):
            return data
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def to_cpu(self, data: cp.ndarray) -> np.ndarray:
        """
        将数据从GPU转移到CPU

        Args:
            data: GPU数组

        Returns:
            CPU数组
        """
        return cp.asnumpy(data)

    def check_memory(self, required_bytes: int) -> bool:
        """
        检查是否有足够的GPU内存

        Args:
            required_bytes: 需要的字节数

        Returns:
            是否有足够内存
        """
        free, _ = self.get_memory_info()
        return free >= required_bytes

    def get_device_info(self) -> dict:
        """
        获取GPU设备信息

        Returns:
            GPU设备信息字典
        """
        return {
            'id': self.gpu_id,
            'name': self.gpu_name,
            'memory_total': self.memory_total,
            'memory_free': self.get_memory_info()[0],
            'compute_capability': f"{self.gpu_props['major']}.{self.gpu_props['minor']}",
            'multiprocessor_count': self.gpu_props['multiProcessorCount'],
            'max_threads_per_block': self.gpu_props['maxThreadsPerBlock']
        }

    def synchronize(self):
        """同步GPU操作"""
        cp.cuda.Stream.null.synchronize()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.clear_memory()
