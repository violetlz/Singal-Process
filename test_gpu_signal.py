#!/usr/bin/env python3
"""
GPU信号处理测试脚本
用于验证FFT和STFT功能是否正常工作
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_processor import GPUSignalProcessor

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本GPU信号处理功能...")

    try:
        # 初始化处理器
        processor = GPUSignalProcessor(gpu_id=0)
        print("✓ GPU处理器初始化成功")

        # 生成简单测试信号
        t = cp.linspace(0, 1, 1024)
        signal = cp.sin(2 * cp.pi * 100 * t)  # 100Hz正弦波

        # 测试FFT
        spectrum = processor.fft(signal)
        reconstructed = processor.ifft(spectrum)

        # 验证重构误差
        error = cp.mean(cp.abs(signal - cp.real(reconstructed)))
        print(f"✓ FFT/IFFT重构误差: {error:.2e}")

        # 测试STFT
        stft_result = processor.stft(signal, window_size=256, hop_size=128)
        reconstructed_stft = processor.istft(stft_result, hop_size=128, window_size=256)

        # 验证STFT重构误差
        error_stft = cp.mean(cp.abs(signal[:len(reconstructed_stft)] - reconstructed_stft))
        print(f"✓ STFT/ISTFT重构误差: {error_stft:.2e}")

        print("✓ 所有基本功能测试通过")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_signal_generation():
    """测试信号生成功能"""
    print("\n测试信号生成功能...")

    try:
        processor = GPUSignalProcessor(gpu_id=0)
        signals = processor.generate_test_signals(sample_rate=44100, duration=0.1)

        # 检查生成的信号
        expected_keys = ['time', 'sine_1khz', 'sine_5khz', 'fm_signal',
                        'am_signal', 'noise', 'composite', 'sample_rate']

        for key in expected_keys:
            if key in signals:
                print(f"✓ 生成信号: {key}")
            else:
                print(f"✗ 缺少信号: {key}")
                return False

        print("✓ 信号生成功能正常")
        return True

    except Exception as e:
        print(f"✗ 信号生成测试失败: {e}")
        return False

def test_performance():
    """测试性能"""
    print("\n测试性能...")

    try:
        processor = GPUSignalProcessor(gpu_id=0)

        # 生成不同长度的信号进行测试
        lengths = [1024, 4096, 16384]

        for length in lengths:
            print(f"测试信号长度: {length}")

            # 生成测试信号
            t = cp.linspace(0, 1, length)
            signal = cp.sin(2 * cp.pi * 1000 * t) + cp.random.normal(0, 0.1, length)

            # GPU FFT
            import time
            start_time = time.time()
            spectrum = processor.fft(signal)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time

            # CPU FFT对比
            signal_cpu = cp.asnumpy(signal)
            start_time = time.time()
            spectrum_cpu = np.fft.fft(signal_cpu)
            cpu_time = time.time() - start_time

            print(f"  GPU时间: {gpu_time:.6f}s")
            print(f"  CPU时间: {cpu_time:.6f}s")
            print(f"  加速比: {cpu_time/gpu_time:.2f}x")

        print("✓ 性能测试完成")
        return True

    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def create_simple_demo():
    """创建简单的演示图"""
    print("\n创建简单演示图...")

    try:
        processor = GPUSignalProcessor(gpu_id=0)
        signals = processor.generate_test_signals(sample_rate=44100, duration=0.05)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('GPU信号处理简单演示', fontsize=14)

        # 1kHz正弦波
        time_axis = cp.asnumpy(signals['time'])
        sine_1khz = cp.asnumpy(signals['sine_1khz'])

        axes[0, 0].plot(time_axis, sine_1khz)
        axes[0, 0].set_title('1kHz正弦波 - 时域')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].grid(True)

        # FFT频谱
        spectrum = processor.fft(signals['sine_1khz'])
        freq_axis = processor.get_frequency_axis(signals['sample_rate'], len(spectrum))

        axes[0, 1].plot(cp.asnumpy(freq_axis), cp.asnumpy(cp.abs(spectrum)))
        axes[0, 1].set_title('1kHz正弦波 - 频域')
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, 5000)

        # 调频信号STFT
        stft_result = processor.stft(signals['fm_signal'], window_size=512, hop_size=256)
        time_axis_stft = processor.get_time_axis(len(signals['fm_signal']),
                                               signals['sample_rate'], 256, 512)
        freq_axis_stft = processor.get_frequency_axis(signals['sample_rate'],
                                                    stft_result.shape[1])

        im = axes[1, 0].pcolormesh(cp.asnumpy(time_axis_stft),
                                  cp.asnumpy(freq_axis_stft),
                                  cp.asnumpy(cp.abs(stft_result.T)),
                                  shading='gouraud', cmap='viridis')
        axes[1, 0].set_title('调频信号 - STFT')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('频率 (Hz)')
        plt.colorbar(im, ax=axes[1, 0])

        # 复合信号
        composite = cp.asnumpy(signals['composite'])
        axes[1, 1].plot(time_axis, composite)
        axes[1, 1].set_title('复合信号 - 时域')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('simple_demo.png', dpi=300, bbox_inches='tight')
        print("✓ 演示图保存为 simple_demo.png")

        return True

    except Exception as e:
        print(f"✗ 创建演示图失败: {e}")
        return False

def main():
    """主测试函数"""
    print("GPU信号处理测试程序")
    print("=" * 40)

    # 检查CuPy是否可用
    try:
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA设备数量: {cp.cuda.runtime.getDeviceCount()}")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"GPU {i}: {props['name'].decode()}")
    except Exception as e:
        print(f"✗ CuPy初始化失败: {e}")
        print("请确保已正确安装CuPy和CUDA")
        return

    # 运行测试
    tests = [
        test_basic_functionality,
        test_signal_generation,
        test_performance,
        create_simple_demo
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！GPU信号处理功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查安装和配置。")

if __name__ == "__main__":
    main()
