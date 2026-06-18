# scene_fmcw_corner_reflector 仿真流程总结

## 1. 目标与方法概述
该示例使用 `RadarSimPy` 搭建一个 **77 GHz FMCW 雷达 + 角反射器目标** 场景，完成以下流程：

- 雷达与目标建模
- 时域回波仿真（含噪声）
- 距离-多普勒（Range-Doppler）处理
- 2D OS-CFAR 自适应门限检测
- 结果可视化与验证

---

## 2. 雷达系统建模

### 2.1 发射机（Transmitter）
- 频率扫频：`[77e9 - 50e6, 77e9 + 50e6]`（带宽 100 MHz）
- Chirp 时长：`80 us`
- PRP：`100 us`
- Chirp 数：`256`
- 发射功率：`25 dBm`
- 天线位置：`(0, 0, 0)`

### 2.2 接收机（Receiver）
- 采样率：`2 MHz`
- 噪声系数：`8 dB`
- RF 增益：`20 dB`
- 基带增益：`30 dB`
- 负载电阻：`500 ohm`
- 天线位置：`(0, 0, 0)`（与发射同址，单站体制）

### 2.3 雷达对象
将 `tx` 与 `rx` 组合为完整 `Radar` 系统。

---

## 3. 场景与目标建模

- 目标类型：三面角反射器（STL 模型）
- 模型路径：`../models/cr.stl`
- 目标位置：`(50, 0, 0)` m
- 目标速度：`(-5, 0, 0)` m/s（朝雷达接近）
- 目标列表：`targets = [target_1]`

在仿真前对角反射器网格进行可视化，用于检查模型几何正确性。

---

## 4. 回波仿真

调用：
- `data = sim_radar(radar, targets, density=0.2)`

说明：
- `density=0.2`：较高射线密度，用于更准确刻画角反射器多次反射路径。
- 从仿真结果提取并构造基带信号：
  - `baseband = data["baseband"] + data["noise"]`
- 数据维度语义：
  - 输入处理前：`[channels, pulses, samples]`

---

## 5. 距离-多普勒处理（Range-Doppler FFT）

### 5.1 加窗
- 距离向（快时间）Chebyshev 窗，旁瓣抑制 60 dB
- 多普勒向（慢时间）Chebyshev 窗，旁瓣抑制 60 dB

### 5.2 二维 FFT
调用：
- `range_doppler = proc.range_doppler_fft(baseband, rwin=range_window, dwin=doppler_window)`

输出数据语义：
- `range_doppler` 维度约为 `[channels, Doppler_bins, range_bins]`

---

## 6. 坐标轴构建与 RD 图可视化

根据雷达参数计算：
- 最大距离 `max_range`
- 无模糊速度 `unambiguous_speed`
- `range_axis`
- `doppler_axis`

随后绘制 3D Range-Doppler 图，预期主峰位于：
- 距离约 `50 m`
- 速度约 `-5 m/s`

---

## 7. 2D OS-CFAR 检测

### 7.1 预处理
- 对多通道幅值图做平均：
  - `rdop_avg = np.mean(np.abs(range_doppler), axis=0)`

### 7.2 OS-CFAR 参数
调用：
- `cfar = proc.cfar_os_2d(...)`

参数：
- `guard=2`
- `trailing=20`
- `pfa=1e-4`
- `k=1500`
- `detector="linear"`

输出：
- 与 RD 图同尺寸的自适应门限面 `cfar`

检测判据：
- 信号幅值 > CFAR 门限 即判定目标存在

---

## 8. 结果叠加展示

将以下两者在同一图中叠加：
- 雷达回波表面（Radar Returns）
- CFAR 门限表面（CFAR Threshold）

观察重点：
- 目标峰值是否稳定高于门限
- 门限是否随局部噪声/杂波自适应变化

---

## 9. 结论与可扩展方向

### 结论
- 示例验证了角反射器目标在 FMCW 体制下的距离-速度可检测性。
- 通过 OS-CFAR，可在设定虚警率下实现自适应检测。

### 可扩展实验
- 与 `CA-CFAR` 对比鲁棒性
- 增加第二个目标测试掩蔽效应
- 增大带宽提高距离分辨率
- 增加 chirp 数提高速度分辨率
