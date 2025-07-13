#!/usr/bin/env python3
"""
GPU信号处理项目安装脚本
自动检测CUDA版本并安装相应的CuPy版本
"""

import subprocess
import sys
import os
import platform

def run_command(command, check=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, check=check,
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_cuda_version():
    """检查CUDA版本"""
    print("检查CUDA版本...")

    # 尝试使用nvcc命令
    success, stdout, stderr = run_command("nvcc --version", check=False)
    if success:
        # 解析版本号
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                version_str = line.split('release')[1].split(',')[0].strip()
                version_parts = version_str.split('.')
                if len(version_parts) >= 2:
                    major = int(version_parts[0])
                    minor = int(version_parts[1])
                    print(f"检测到CUDA版本: {major}.{minor}")
                    return major, minor

    # 尝试检查环境变量
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"CUDA_HOME: {cuda_home}")
        # 检查版本文件
        version_file = os.path.join(cuda_home, 'version.txt')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                content = f.read()
                if 'CUDA Version' in content:
                    version_str = content.split('CUDA Version')[1].split('\n')[0].strip()
                    version_parts = version_str.split('.')
                    if len(version_parts) >= 2:
                        major = int(version_parts[0])
                        minor = int(version_parts[1])
                        print(f"检测到CUDA版本: {major}.{minor}")
                        return major, minor

    print("无法自动检测CUDA版本")
    return None, None

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("错误: 需要Python 3.7或更高版本")
        return False

    return True

def install_cupy(cuda_major, cuda_minor):
    """安装CuPy"""
    print("安装CuPy...")

    # 根据CUDA版本选择CuPy包
    if cuda_major == 12:
        cupy_package = "cupy-cuda12x"
    elif cuda_major == 11:
        cupy_package = "cupy-cuda11x"
    elif cuda_major == 10:
        cupy_package = "cupy-cuda10x"
    else:
        print(f"不支持的CUDA版本: {cuda_major}.{cuda_minor}")
        print("请手动安装适合的CuPy版本")
        return False

    print(f"安装 {cupy_package}...")
    success, stdout, stderr = run_command(f"pip install {cupy_package}")

    if success:
        print("CuPy安装成功")
        return True
    else:
        print(f"CuPy安装失败: {stderr}")
        return False

def install_requirements():
    """安装其他依赖"""
    print("安装其他依赖...")

    if os.path.exists('requirements.txt'):
        success, stdout, stderr = run_command("pip install -r requirements.txt")
        if success:
            print("依赖安装成功")
            return True
        else:
            print(f"依赖安装失败: {stderr}")
            return False
    else:
        print("requirements.txt文件不存在，跳过依赖安装")
        return True

def test_installation():
    """测试安装"""
    print("测试安装...")

    test_code = """
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# 测试CuPy
print(f"CuPy版本: {cp.__version__}")
print(f"CUDA设备数量: {cp.cuda.runtime.getDeviceCount()}")

# 测试基本功能
x = cp.array([1, 2, 3, 4])
y = cp.fft.fft(x)
print("CuPy FFT测试成功")

# 测试NumPy
z = np.array([1, 2, 3, 4])
w = np.fft.fft(z)
print("NumPy FFT测试成功")

print("所有测试通过！")
"""

    success, stdout, stderr = run_command(f"{sys.executable} -c '{test_code}'")

    if success:
        print("安装测试成功！")
        return True
    else:
        print(f"安装测试失败: {stderr}")
        return False

def main():
    """主函数"""
    print("GPU信号处理项目安装脚本")
    print("=" * 40)

    # 检查Python版本
    if not check_python_version():
        return

    # 检查CUDA版本
    cuda_major, cuda_minor = check_cuda_version()

    if cuda_major is None:
        print("\n请手动安装CUDA Toolkit，然后重新运行此脚本")
        print("CUDA下载地址: https://developer.nvidia.com/cuda-downloads")
        return

    # 安装CuPy
    if not install_cupy(cuda_major, cuda_minor):
        return

    # 安装其他依赖
    if not install_requirements():
        return

    # 测试安装
    if not test_installation():
        return

    print("\n🎉 安装完成！")
    print("\n下一步:")
    print("1. 运行测试: python test_gpu_signal.py")
    print("2. 运行演示: python demo_visualization.py")
    print("3. 查看文档: README.md")

if __name__ == "__main__":
    main()
