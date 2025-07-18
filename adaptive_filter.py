import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from src import GPUSignalProcessor, SignalGenerator

class AdaptiveFilter:
    def __init__(self, gpu_id=0):
        """
        初始化相关处理器和生成器
        """
        self.gpu_processor = GPUSignalProcessor(gpu_id=gpu_id)
        self.signal_generator = SignalGenerator(self.gpu_processor)
        self.order = 32
        self.mu = 0.005
        self.lambda_ = 0.99
        self.qpsk_params = {
            "bandwidth": 5e6,
            "sample_rate": 20e6,
            "duration": 0.001,
            "alpha": 0.35,
            "snr_db": 20
        }
        self.original_qpsk_params = {**self.qpsk_params, "snr_db": None}
        self.noisy_signal = None
        self.original_signal = None
        self.sample_rate = None
        self.t = None

        self.y_lms = None
        self.e_lms = None
        self.w_lms = None
        self.lms_time = None

        self.y_rls = None
        self.e_rls = None
        self.w_rls = None
        self.rls_time = None

        self.mse_lms = None
        self.mse_rls = None

    def generate_signals(self):
        """
        生成带噪声的QPSK信号和原始无噪声QPSK信号
        """
        qpsk_result = self.signal_generator.generate_qpsk_signal(**self.qpsk_params)
        self.noisy_signal = qpsk_result['signal']
        original_result = self.signal_generator.generate_qpsk_signal(**self.original_qpsk_params)
        self.original_signal = original_result['signal']
        self.sample_rate = qpsk_result['sample_rate']
        self.t = qpsk_result['time']

    def lms_filter(self, x, d, n_iters=None):
        """
        LMS自适应滤波器实现
        参数：
        x: 输入信号
        d: 期望信号
        n_iters: 迭代次数 (默认全信号长度)
        return:
        y输出信号, e误差信号, w最终权重
        """
        n_iters = n_iters or len(x)
        w = cp.zeros(self.order, dtype=cp.complex64)
        y = cp.zeros(n_iters, dtype=cp.complex64)
        e = cp.zeros(n_iters, dtype=cp.complex64)
        x_padded = cp.concatenate([cp.zeros(self.order - 1, dtype=cp.complex64), x])

        for n in range(self.order - 1, n_iters):
            start_idx = n - self.order + 1
            x_n = x_padded[start_idx:start_idx + self.order][::-1]
            y[n] = cp.dot(w.conj(), x_n)
            e[n] = d[n] - y[n]
            w += self.mu * e[n] * x_n.conj()

        return y, e, w

    def rls_filter(self, x, d, delta=0.01):
        """
        RLS自适应滤波器实现
        参数：
        x: 输入信号
        d: 期望信号
        delta: 初始化参数
        return:
        y输出信号, e误差信号, w最终权重
        """
        n_samples = len(x)
        w = cp.zeros(self.order, dtype=cp.complex64)
        P = cp.eye(self.order, dtype=cp.complex64) / delta
        y = cp.zeros(n_samples, dtype=cp.complex64)
        e = cp.zeros(n_samples, dtype=cp.complex64)
        x_padded = cp.concatenate([cp.zeros(self.order - 1, dtype=cp.complex64), x])

        for n in range(self.order - 1, n_samples):
            start_idx = n - self.order + 1
            x_n = x_padded[start_idx:start_idx + self.order][::-1]
            pi = P @ x_n
            k = pi / (self.lambda_ + cp.dot(x_n.conj(), pi))
            y[n] = cp.dot(w.conj(), x_n)
            e[n] = d[n] - y[n]
            w += k * e[n]
            P = (P - cp.outer(k, x_n.conj()) @ P) / self.lambda_

        return y, e, w

    def process_signals(self):
        """
        调用滤波器对信号进行处理，并计算性能指标
        """
        # LMS滤波器处理
        start_time = time.time()
        self.y_lms, self.e_lms, self.w_lms = self.lms_filter(self.noisy_signal, self.original_signal)
        self.lms_time = time.time() - start_time

        # RLS滤波器处理
        start_time = time.time()
        self.y_rls, self.e_rls, self.w_rls = self.rls_filter(self.noisy_signal, self.original_signal)
        self.rls_time = time.time() - start_time

        # 计算性能指标
        self.mse_lms = 10 * cp.log10(cp.mean(cp.abs(self.e_lms) ** 2))
        self.mse_rls = 10 * cp.log10(cp.mean(cp.abs(self.e_rls) ** 2))

    def visualize_results(self):
        """
        可视化星座图和误差收敛曲线
        """
        t_cpu = cp.asnumpy(self.t)
        original_cpu = cp.asnumpy(self.original_signal)
        noisy_cpu = cp.asnumpy(self.noisy_signal)
        y_lms_cpu = cp.asnumpy(self.y_lms)
        y_rls_cpu = cp.asnumpy(self.y_rls)

        # 绘制星座图
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(original_cpu.real, original_cpu.imag, s=3, label='Original')
        plt.xlabel('I ')
        plt.ylabel('Q ')
        plt.title('Original QPSK Constellation')
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(2, 2, 2)
        plt.scatter(noisy_cpu.real, noisy_cpu.imag, s=3, label='Noisy', color='orange')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title('Noisy QPSK Constellation')
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(2, 2, 3)
        plt.scatter(y_lms_cpu.real, y_lms_cpu.imag, s=3, label='LMS', color='green')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title(f'LMS Filtered (MSE: {self.mse_lms:.2f} dB)')
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(2, 2, 4)
        plt.scatter(y_rls_cpu.real, y_rls_cpu.imag, s=1, label='RLS', color='red')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title(f'RLS Filtered (MSE: {self.mse_rls:.2f} dB)')
        plt.grid(True)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

        # 绘制误差收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(cp.asnumpy(10 * cp.log10(cp.abs(self.e_lms) ** 2 + 1e-10)), label='LMS')
        plt.plot(cp.asnumpy(10 * cp.log10(cp.abs(self.e_rls) ** 2 + 1e-10)), label='RLS')
        plt.title('Error Convergence (QPSK Signal)')
        plt.xlabel('Sample Index')
        plt.ylabel('Error Power (dB)')
        plt.legend()
        plt.grid(True)
        plt.ylim(-40, 20)
        plt.show()

    def print_performance(self):
        """
        打印滤波器性能指标
        """
        print(f"LMS Execution Time: {self.lms_time:.4f} s, Final MSE: {self.mse_lms:.2f} dB")
        print(f"RLS Execution Time: {self.rls_time:.4f} s, Final MSE: {self.mse_rls:.2f} dB")
        print(f"LMS:MSE {self.mse_lms:.2f}dB")
        print(f"RLS:MSE {self.mse_rls:.2f}dB")

    def run(self):
        """
        运行整个流程：生成信号→滤波处理→可视化→打印性能
        """
        self.generate_signals()
        self.process_signals()
        self.visualize_results()
        self.print_performance()


if __name__ == "__main__":
    main = AdaptiveFilter(gpu_id=0)
    main.run()