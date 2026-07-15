# simulator_ofdm.grc 运行逻辑

基于 Sionna GPU 的 OFDM 单站感知仿真 flowgraph，在 GNU Radio 中完成「发端波形 → 信道/目标回波 → 接收与 DD 谱 → 谱图显示」的闭环。

## 总体数据流

```
SionnaBootstrap（无连线，启动预热）
        │
Sionna OFDM Tx ──► Sionna Static Target ──► Sionna DD Rx
  (PRI burst)        (单目标回波 + 自耦合)         │
                                                   ├─ out(0) complex[2048] ──► Null Sink
                                                   └─ out(1) float[2048]    ──► QT GUI Spectrogram
```

GUI 顶部提供 **target_range** / **target_velocity** 滑块（界面标签仍为 range / Velocity），实时调整仿真目标参数。

## Burst 发送（PRI 模式）

默认 `burst_pri_sec = 0.1`（100 ms PRI）。每个 PRI 周期：

```
┌──────── CPI ────────┬──── 静默零填充 ────┐
│ 1_310_720 样点      │ PRI - CPI 样点      │
│ packet_len tag @0   │ 无 tag              │
└─────────────────────┴─────────────────────┘
         ◄──────── PRI = burst_pri_sec × samp_rate ────────►
```

默认 `burst_pri_sec = 0.1`（100 ms PRI）。`burst_pri_sec = 0` 时为**连续发送**。

| 变量              | 默认      | 说明                                                         |
| ----------------- | --------- | ------------------------------------------------------------ |
| `burst_pri_sec` | `0.1` s   | PRI（秒）；`0` 为连续发送                          |
| `burst_mode`    | 自动 True | `burst_pri_sec > 0`；收端/信道块 tag 门控         |
| `samp_rate`     | 30.72 MHz | burst 模式下换算 PRI 样点数                                  |

约束：`int(burst_pri_sec × samp_rate) > 1_310_720`，否则发端构造报错。

收端 `burst_mode = (burst_pri_sec > 0)`：仅在收到发端 `packet_len` tag 后才开始累积 CPI，静默段样点丢弃，避免误触发 DD 处理。

## 配置优先级（GRC > TOML）

- **GRC 变量是 flowgraph 的主入口**：`fft_len`、`ofdm_symbols`、`cp_len`、`subcarrier_spacing`、`center_freq`、`seed` 等块参数优先于 `config/simulation/sensing/sensing_monostatic.toml` 同名字段。
- **Sionna 块内部**通过 `gr_config.merge_config()` 合并：`EffectiveConfig = merge(TOML, GrcOverrides)`，GPU 波形与 DD 处理以 **effective** 为准。
- **Bootstrap** 在 Python 中通过 `merge_config` + `SensingPerformance` 计算并**打印**感知性能（分辨率、R_max、v_max 等）。
- GRC 中**不再**维护 `bandwidth`、`R_max`、`v_max` 等派生 Variable；谱图轴使用与默认配置接近的占位值，滑块范围由 `run_simulator_ofdm.py` 启动时刷新。
- **仅 TOML**（GRC 不覆盖）：ZC 源、窗函数、CFAR、`rt_simulator` 场景、QAM/SNR 等。
- **仅 GRC**：`burst_pri_sec`、`tx_device`、谱图 `interval`、滑块、`sionna_static_target` 参数等。
- **Python 脚本**（如 `run_sensing_monostatic.py`）不涉及 GRC，仍纯 TOML。

## 启动顺序

1. **Import 块**：将 `gnuradio/` 与 `src/` 加入 `sys.path`，供 Sionna 自定义块导入 ISAC 模块。
2. **SionnaBootstrap**（无 I/O 连线）：在 TX/RX 块实例化前执行一次 GPU 预热：
   - 在 GPU 上生成 ZC 发端包（时域 + 频域参考）；
   - 初始化 Sionna 接收链并跑一遍 dummy DD 谱；
   - 打印感知性能参数（距离/速度分辨率、最大量程等）。
3. **块实例化与连线**：GNU Radio 调度器启动后，各块按 buffer 需求开始流转样点。

## 各块职责

| 块                                                                 | 状态 | 作用                                                                                    |
| ------------------------------------------------------------------ | ---- | --------------------------------------------------------------------------------------- |
| `sionna_ofdm_tx_0`                                               | 启用 | 缓存 CPI 按 PRI burst 输出：CPI + tag，然后零填充静默；`burst_pri_sec=0` 时为连续循环 |
| `sionna_static_target_0`                                         | 启用 | Torch 点目标仿真（替代 gr-radar）：滑块 range/velocity + RCS + 自耦合（-10 dB）       |
| `sionna_dd_rx_0`                                                 | 启用 | tag 门控收包 + GPU DD 谱；双路输出                                                      |
| `blocks_null_sink_0`                                             | 启用 | 消费 out(0) 复数 IQ 矩阵行（暂代 OS-CFAR）                                              |
| `radar_qtgui_spectrogram_plot_0`                                 | 启用 | 接收 out(1) log10 谱，按 `packet_len` tag 组帧刷新                                    |
| `analog_noise_source` + `blocks_add_xx`                        | 禁用 | 可选 AWGN 通路                                                                          |
| `radar_os_cfar_2d_vc` → `estimator_ofdm` → `print_results` | 禁用 | CFAR 检测链                                                                             |

## Sionna DD Rx 双输出

| 端口   | 类型    | vlen               | 内容                         |
| ------ | ------- | ------------------ | ---------------------------- |
| out(0) | complex | `fft_len` (2048) | DD 谱复数 IQ，供 CFAR        |
| out(1) | float   | `fft_len` (2048) | log10(\|·\|² + ε)，供谱图 |

每输出 **512 行**（= `ofdm_symbols`）构成一帧 DD 矩阵；帧首打 `packet_len=512` tag。多普勒维在 RX 内 `flipud`，以匹配谱图 Y 轴 `[-v_max, v_max]`。

## 感知性能（仅 Python）

| 环节 | 说明 |
|------|------|
| `SionnaBootstrap` | 启动时 `merge_config` → 控制台打印 EffectiveConfig 性能 |
| `flowgraph_perf.apply_sensing_perf_ui` | `run_simulator_ofdm.py` 在 flowgraph 构造后刷新 `target_range` / `target_velocity` 滑块上下限 |
| 谱图 `axis_x/y` | GRC 中为占位 `[0,10000]`、`[-150,150]`；大幅改 OFDM 参数后请对照 Bootstrap 打印值调整 |

## 关键 GRC 变量（默认 15 kHz 子载波间隔）

| 变量                   | 默认          | 说明                              |
| ---------------------- | ------------- | --------------------------------- |
| `fft_len`            | 2048          | 子载波数                          |
| `ofdm_symbols`       | 512           | 慢时间 FFT 长度                   |
| `cp_len`             | 512           | 循环前缀                          |
| `subcarrier_spacing` | 15 kHz        | 子载波间隔                        |
| `center_freq`        | 6 GHz         | 载频                              |
| `samp_rate`          | 30.72 MHz     | `fft_len × subcarrier_spacing`（信道仿真用） |
| `burst_pri_sec`      | 0.1 s         | PRI；设为 0 恢复连续发送          |
| `config_file`        | `config/simulation/sensing/sensing_monostatic.toml` | Sionna 配置          |
| `tx_device`          | `cuda:0`      | GPU 设备                          |
| `target_range`       | 100 m（滑块） | 仿真目标距离                      |
| `target_velocity`    | 5 m/s（滑块） | 仿真目标速度                      |

一帧 CPI 时域长度：`512 × (2048 + 512) = 1 310 720` 样点（≈ 42.7 ms）。

## Tag 与矩阵维度

- **Tag 键**：`packet_len`
- **发端 tag 值**：时域样点数（1 310 720），仅 CPI 起点
- **收端 out tag 值**：频域行数（512）
- **谱图 vlen**：2048；矩阵 512 × 2048

## 谱图刷新

- 变量 `spectrogram_interval`（毫秒）：连续模式 `5000`；**burst 模式自动 ≥ 60000**。
- burst 下 DD 输出频率 ≈ 1/PRI（受 GPU 耗时限制可能更低）。
- gr-radar 谱图在缓冲区为空时仍会定时 `refresh()`，会抛 `data buffer has size zero` 并崩溃；interval 须大于首帧 GPU DD 延迟。

## 故障：启动后谱图崩溃 `data buffer has size zero`

**根因**：QTimer 在首帧 DD 到达前触发。burst + 全 CPI GPU 处理时首帧常需 10–60 s，原 `interval=5000` 过短。

**处理**：增大 `spectrogram_interval`（默认 burst 下已为 60000 ms）；仍崩溃时可试 `120000`，然后重新 `grcc` 并运行。

## 运行方式

```bash
bash src/isac_imp/install_grc_blocks.sh   # 首次或修改 sionna_*.block.yml 后必跑
cd gnuradio
grcc simulator_ofdm.grc
python -u run_simulator_ofdm.py    # 含感知性能 UI 刷新；勿直接跑 simulator_ofdm.py
```

若 **Sionna Bootstrap** 属性里仍出现 `Bandwidth`、`R_max` 等红色空项，说明 GRC 缓存了旧块定义：先运行 `install_grc_blocks.sh`，**完全退出并重启 GRC**，再打开 `simulator_ofdm.grc`。正确块应只有 `config_file`、`seed`、`device` 及 OFDM 五元组（`fft_len` 等），感知性能仅在 Python Bootstrap 控制台打印。

连续发送：将 `burst_pri_sec` 设为 `0`。

## 故障：调整 range/velocity 滑块谱图不更新

**处理**：确认 `burst_pri_sec=0` 或已增大 `spectrogram_interval`；重新 `grcc` 后运行。burst 下首帧完成前谱图可能空白。

## 相关文件

| 文件                               | 说明                          |
| ---------------------------------- | ----------------------------- |
| `simulator_ofdm.grc`             | Flowgraph 定义                |
| `simulator_ofdm.py`              | GRC 生成的 flowgraph 类         |
| `run_simulator_ofdm.py`        | 推荐入口（刷新滑块感知范围）    |
| `flowgraph_perf.py`              | merge_config → Qt 滑块范围      |
| `sionna_tx.py`                   | 发端 burst 状态机 + Bootstrap |
| `sionna_rx.py`                   | 收端 tag 门控 + DD 谱         |
| `sionna_*.block.yml`             | GRC 自定义块                  |
| `gr_config.py`                   | GRC/TOML merge 与 EffectiveConfig |
| `config/simulation/sensing/sensing_monostatic.toml` | Sionna 配置                       |
| `sionna_static_target.py`        | Torch 点目标信道 GRC 块           |
| `sionna_static_target.block.yml` | GRC 块定义                        |
| `verify_dd_axis.py`              | DD 轴标定验证                 |
