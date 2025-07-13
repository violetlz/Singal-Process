# FFT和功率谱密度估计（Welch方法）演示

本演示展示了如何使用模块化的GPU信号处理库进行FFT和功率谱密度估计（Welch方法）的测试和效果展示。

## 文件说明

### 主要演示文件

1. **`fft_welch_demo.py`** - 完整版演示文件
   - 包含详细的性能测试和效果展示
   - 生成多种测试信号
   - 比较不同窗口函数的效果
   - 生成丰富的可视化图表

2. **`fft_welch_demo_simple.py`** - 简化版演示文件
   - 基础功能演示
   - 适合快速测试和验证
   - 包含错误处理和导入检查

## 功能特性

### 1. FFT测试和演示
- **基础FFT测试**: 对单频正弦信号进行FFT分析
- **性能测试**: 比较GPU和CPU的FFT性能
- **多信号分析**: 分析不同类型的信号（单频、多频、调频、脉冲、随机等）
- **可视化**: 生成时域和频域的对比图

### 2. Welch功率谱密度估计
- **基础Welch PSD**: 对带噪声的多频信号进行功率谱密度估计
- **参数影响**: 测试不同窗口大小对结果的影响
- **窗口函数**: 比较不同窗口函数（汉宁窗、海明窗、布莱克曼窗、矩形窗）的效果
- **频谱特征**: 计算频谱中心、带宽、熵等特征

### 3. FFT vs Welch方法比较
- **性能对比**: 比较两种方法的执行时间
- **质量对比**: 比较频率分辨率和估计质量
- **适用场景**: 分析两种方法的优缺点

### 4. 性能分析
- **性能缩放**: 测试性能随信号长度的变化
- **GPU加速**: 展示GPU相对于CPU的加速效果
- **内存使用**: 监控GPU内存使用情况

## 生成的测试信号

演示中生成以下类型的测试信号：

1. **单频正弦信号** (1000 Hz)
   - 用于基础FFT测试
   - 验证频率检测的准确性

2. **多频复合信号** (500 Hz, 1500 Hz, 2500 Hz)
   - 用于测试多频率成分的检测
   - 验证频谱分析的能力

3. **调频信号** (1000±500 Hz)
   - 用于测试时变频率的检测
   - 展示STFT的优势

4. **带噪声的信号**
   - 多频信号 + 高斯白噪声
   - 用于测试噪声环境下的频谱估计

5. **脉冲信号**
   - 周期性脉冲
   - 用于测试瞬态信号的检测

6. **随机信号**
   - 高斯白噪声
   - 用于测试随机信号的频谱特性

## 生成的图片文件

运行演示后会生成以下图片文件：

### 基础测试图片
- `fft_basic_test.png` - 基础FFT测试结果
- `welch_psd_basic_test.png` - 基础Welch PSD测试结果

### 比较分析图片
- `fft_vs_welch_comparison.png` - FFT和Welch方法对比
- `performance_scaling.png` - 性能随信号长度的变化

### 详细分析图片（完整版）
- `fft_performance_comparison.png` - FFT性能对比
- `fft_analysis_*.png` - 各种信号的FFT分析
- `welch_psd_comparison.png` - 不同窗口大小的Welch PSD
- `window_function_comparison.png` - 不同窗口函数的效果

## 使用方法

### 1. 运行简化版演示
```bash
python fft_welch_demo_simple.py
```

### 2. 运行完整版演示
```bash
python fft_welch_demo.py
```

### 3. 自定义测试
可以修改演示文件中的参数来测试不同的场景：

```python
# 修改信号参数
sample_rate = 44100  # 采样率
duration = 1.0       # 信号持续时间

# 修改Welch PSD参数
window_size = 1024   # 窗口大小
hop_size = 512       # 跳跃大小
window = 'hann'      # 窗口函数

# 修改性能测试参数
signal_lengths = [1024, 4096, 16384, 65536, 262144]  # 测试的信号长度
```

## 预期结果

### 1. FFT测试结果
- 单频信号：应能准确检测到1000 Hz的峰值
- 多频信号：应能检测到500 Hz、1500 Hz、2500 Hz的峰值
- 性能：GPU应比CPU快数倍到数十倍

### 2. Welch PSD测试结果
- 噪声抑制：应能有效抑制噪声，突出主要频率成分
- 峰值检测：应能准确检测到主要频率成分
- 频谱特征：应能计算出合理的频谱中心、带宽和熵

### 3. 性能对比结果
- FFT方法：速度快，但频率分辨率固定
- Welch方法：速度较慢，但频率分辨率可调，噪声抑制效果好

## 技术要点

### 1. FFT方法
- **优点**: 计算速度快，实现简单
- **缺点**: 频率分辨率固定，对噪声敏感
- **适用**: 信号长度固定，噪声较少的场景

### 2. Welch方法
- **优点**: 频率分辨率可调，噪声抑制效果好
- **缺点**: 计算复杂度较高
- **适用**: 长信号，噪声较多的场景

### 3. GPU加速
- **FFT**: GPU加速效果显著，特别是大信号长度
- **Welch**: GPU加速效果相对较小，主要受限于串行处理部分

## 故障排除

### 1. 导入错误
如果出现导入错误，请检查：
```bash
pip install -r requirements.txt
```

### 2. GPU内存不足
如果出现GPU内存不足错误：
- 减少信号长度
- 减少窗口大小
- 使用批处理模式

### 3. 性能不理想
如果GPU加速效果不理想：
- 确保信号长度足够大（>1024）
- 检查GPU利用率
- 使用性能监控工具分析瓶颈

## 扩展功能

基于这个演示，可以进一步扩展：

1. **实时处理**: 实现实时FFT和Welch PSD
2. **自适应参数**: 根据信号特性自动调整参数
3. **多GPU并行**: 利用多个GPU进行并行处理
4. **其他变换**: 添加小波变换、希尔伯特变换等
5. **应用场景**: 音频处理、雷达信号处理、生物信号分析等

## 参考文献

1. Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms." IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.

2. Oppenheim, A. V., & Schafer, R. W. (2010). "Discrete-time signal processing." Pearson Higher Education.

3. Smith, S. W. (1997). "The scientist and engineer's guide to digital signal processing." California Technical Publishing.
