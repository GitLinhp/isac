# OFDM Burst 信号处理流程

本文档描述测试流图 `usrp_ofdm_sink_source_dd` 的端到端信号处理：Style1 定时突发发射、同 epoch 定点收样、Torch CUDA 距离谱，以及内嵌 Range Profile / MUSIC。

对应文件：

- [`usrp_ofdm_sink_source_dd.grc`](usrp_ofdm_sink_source_dd.grc) / [`usrp_ofdm_sink_source_dd.py`](usrp_ofdm_sink_source_dd.py)
- TX：[`src/isac_imp/ofdm_burst_tx_source.py`](../../../../src/isac_imp/ofdm_burst_tx_source.py)
- RX：[`src/isac_imp/ofdm_burst_rx.py`](../../../../src/isac_imp/ofdm_burst_rx.py)

---

## 1. 概述

本流图实现**零多普勒 OFDM 单站雷达**（环回 / 近距目标）：

1. **启动时**在 CUDA 上一次性预计算一整 CPI 的频域网格与时域 IQ，拷回 Host。
2. **运行期** Host 仅 memcpy 重放时域突发，打 Style1 时间标签，并向 USRP Source **异步**下发 `NUM_SAMPS_AND_DONE`（后台线程，避免 MPM RPC 卡住 TX pacing）。
3. **同 `tx_epoch`** 时刻 USRP 发射与接收各 `8704` 个样点。
4. RX 在 Host 做延迟对齐后，将 CPI 送入 CUDA 做去 CP / FFT / 信道估计 / 距离 FFT，再在 Qt 上显示 ROI 距离谱（可选异步 MUSIC）。

核心块仅两端：`OfdmBurstTxSource` + `OfdmBurstRx`（无流输出 sink；plot / MUSIC 内嵌）。

---

## 2. 流图拓扑

```mermaid
flowchart LR
  subgraph txHost [Host_TX]
    TxSrc["OfdmBurstTxSource"]
  end
  subgraph usrp [USRP_X4xx]
    Sink["uhd_usrp_sink\nTX/RX ant"]
    Source["uhd_usrp_source\nRX1 ant"]
  end
  subgraph rxHost [Host_RX]
    Rx["OfdmBurstRx\nCUDA DSP + plot + MUSIC"]
  end

  TxSrc -->|"out0 complex64 IQ"| Sink
  TxSrc -->|"msg tx_freq_cpi\n首 CPI 一次"| Rx
  TxSrc -->|"msg tx_schedule"| Rx
  Source -->|"fc32 IQ + rx_time"| Rx
  Sink -.->|"RF air / loopback"| Source
```

连接关系（与生成代码一致）：

| 源                                | 目的                        | 类型                                      |
| --------------------------------- | --------------------------- | ----------------------------------------- |
| `OfdmBurstTxSource.out0`        | `uhd_usrp_sink_0`         | 时域 IQ`complex64`（唯一流输出）        |
| `OfdmBurstTxSource.tx_freq_cpi` | `OfdmBurstRx.tx_freq_cpi` | PMT 消息（首 CPI 一次；RX 粘性复用）      |
| `OfdmBurstTxSource.tx_schedule` | `OfdmBurstRx.tx_schedule` | PMT 消息（每 CPI）                        |
| `uhd_usrp_source_0`             | `OfdmBurstRx.in0`         | 接收 IQ`fc32`                           |

**反压 / 吞吐说明**：

1. TX/RX 均为单流口；无 out1/in1、无 null_sink/source。
2. 频域参考在 `start()` 预缓存 PMT，仅首 CPI `pub` 一次；RX 用 `_tx_cpi_latest` 粘性复用，避免每 CPI `init_c32vector`/`c32vector_elements` 拖住 TX。
3. 后续 CPI command lead=`20ms`；`host_gap_ms` 与 `epoch_dt_ms` 用于验收。
4. qtgui time sink 默认 disabled，不接 out0。
---

## 3. 关键参数

摘自当前 flowgraph 默认值（GUI 可调项已注明）。

| 参数                                | 默认值                                    | 含义                                            |
| ----------------------------------- | ----------------------------------------- | ----------------------------------------------- |
| `fft_len`                         | 2048                                      | OFDM FFT 长度                                   |
| `cp_len`                          | **128**                                 | 循环前缀长度（约可罩 ~78 m 往返）               |
| `num_symbols` | 4                                         | 每 CPI OFDM 符号数                              |
| `subcarrier_spacing`              | 120 kHz                                   | 子载波间隔                                      |
| `samp_rate`                       | `fft_len × scs` = **245.76 MHz** | IQ 采样率                                       |
| `burst_len_samples`               | 4×(2048+128) = **8704**            | 一 CPI 时域样点数（约 35.4 µs）                |
| `idle_ms`                         | 10                                        | 突发间隙（主机侧 idle）                         |
| `time_lead_s` / `wait_to_start` | 0.5                                       | **首 CPI** 发射提前量（后续 CPI ≤20 ms） |
| `freq`                            | 6 GHz                                     | RF 中心频率                                     |
| `factor`                          | 0.008                                     | TX 幅度缩放（GUI）                              |
| `TX_gain` / `RX_gain`           | 20                                        | USRP 增益（GUI）                                |
| `num_delay_samp`                  | 277                                       | RX 环回延迟补偿样点（GUI）                      |
| `zeropadding_fac`                 | 4                                         | 距离 FFT 零填充 →`vlen_out=8192`             |
| `range_bin_step`                  | `c/(2·fs·zp)` ≈ 0.153 m              | 距离 bin                                        |
| `range_roi`                       | (0.0, 3.5) m                              | 显示 / MUSIC ROI                                |
| `music_enable`                    | False                                     | 异步 1D MUSIC（GUI 复选框）                     |
| `device`                          | `'cuda'`                                | TX 预计算与 RX 每 CPI DSP                       |
| `length_tag_key`                  | `packet_len`                            | 频域参考 CPI 边界 tag                           |

理论 CPI 率（忽略调度抖动）：

$$
\frac{1}{\mathtt{idle\_ms}/1000 + \mathtt{burst\_len}/\mathtt{samp\_rate}} \approx 99.6\,\mathrm{Hz}
$$

---

## 4. 启动与绑定

| 时机   | 函数 / Snippet                     | 作用                                                                     | 设备         |
| ------ | ---------------------------------- | ------------------------------------------------------------------------ | ------------ |
| 建块前 | `patch_usrp_source_factory()`    | USRP Source 以`issue_stream_cmd_on_start=False` 创建，禁止启动即连续收 | Host         |
| 建块后 | `set_time_now`（Sink + Source）  | 同机 TX/RX 绝对时间对齐                                                  | Host → USRP |
| 建块后 | `set_max_noutput_items(max(16384, burst_len))` + `min_output_buffer≥2·burst` | 允许一次 `work` 拉满一 CPI（8704），避免默认 ~8191 分块卡死 | Host / GR |
| 建块后 | `bind_scheduled_rx(usrp_source)` | TX 持有 Source 句柄；每突发异步 `issue_stream_cmd` | Host         |

实现位置：流图 snippet；调度补丁见 [`scheduled_usrp_source.py`](../../../../src/isac_imp/scheduled_usrp_source.py)。

---

## 5. TX 路径

### 5.1 `start()`：一次性 CUDA 预计算

入口：`OfdmBurstTxSourceBlock.start()` → `_precompute_cpi_torch()`。

| 步骤 | 模块 / 函数                                                  | 做什么                                                   | 设备           | 数据                                        |
| ---- | ------------------------------------------------------------ | -------------------------------------------------------- | -------------- | ------------------------------------------- |
| A    | `_build_freq_grid_torch`（`sionna_resource_grid_tx.py`） | BinarySource → QAM → ResourceGridMapper →`fftshift` | **CUDA** | `(4, 2048) complex64` 频域 CPI            |
| B    | `_modulate_cpi_torch`                                      | 批量`ifftshift` + `ifft(norm=forward)` + 拼 CP       | **CUDA** | `(4, 2176)` → reshape `(8704,)`       |
| C    | `× factor`                                                | 幅度缩放                                                 | CUDA           | `td_cpi`                                  |
| D    | `cuda.synchronize` + `.cpu().numpy()`                    | **一次 D2H**                                       | Host           | `freq_cpi`、`td_cpi` 驻留 **CPU** |

运行期**不再上 GPU**：只 Host memcpy 重放，避免 USRP underflow。

调制语义对齐 GNU Radio `fft_vcc`(IFFT, shift) + `ofdm_cyclic_prefixer`(rolloff=0)。独立 GR 块 `SionnaResourceGridTx` / `SionnaOfdmModulator` 逻辑已内联，本图不连接。

### 5.2 `general_work()`：CPI 重放 + Style1 调度

每个 CPI：

1. **首 CPI**：`message_port_pub(tx_freq_cpi, 缓存 PMT)`（仅一次）；后续 CPI 跳过。
2. **`_begin_burst`**：
   - 首 CPI：`epoch = max(now + time_lead_s, next)`（约 +0.5 s 给硬件上臂）。
   - 后续 CPI：command lead **20 ms**；落后则按 `burst_period_s` 整周期追赶。
   - `add_style1_sob_time` → tags `tx_sob` + `tx_time`。
   - `message_port_pub(tx_schedule, make_tx_schedule_msg(epoch))`。
   - `_issue_scheduled_recv(epoch)`：将 `NUM_SAMPS_AND_DONE` **入异步队列**（单 worker），不阻塞 TX work / pacing。
   - `_next_tx_epoch += burst_period_s`。
3. **out0**：拷贝 `td_cpi`；末样点 `add_style1_eob`。
4. 若 `idle_ms > 0`：突发结束后 idle，期间 `work` 返回 0。

| 输出              | 形状 / 类型                         | 下游                  |
| ----------------- | ----------------------------------- | --------------------- |
| out0              | `complex64` 标量流，CPI=8704      | USRP Sink（唯一消费者） |
| `tx_freq_cpi`   | PMT `(dict meta, c32vector)`          | RX（首 CPI 一次）     |
| `tx_schedule`   | PMT`(uint64 sec, double frac)`    | RX 每 CPI             |

诊断：`epoch_dt_ms` / `host_gap_ms` / `write_ms` / `cmd_q` / `cmd_drop` 节流打印；稳态 `host_gap≈period`，`write_ms` 应远小于 period。

Style1（[`burst_pack.py`](../../../../src/isac_imp/burst_pack.py)）：Sink 的 `len_tag_name=''`，靠 `tx_sob` / `tx_time` / `tx_eob` 界定突发，**不是** length-tag 突发。

### 5.3 `uhd_usrp_sink_0`

| 项   | 值                                               |
| ---- | ------------------------------------------------ |
| 设备 | USRP X4xx FPGA / RF                              |
| 格式 | `cpu_format=fc32`                              |
| 天线 | `TX/RX`                                        |
| 缓冲 | `num_send_frames=512`, `send_buff_size=25e6` |
| 行为 | 按`tx_time` 在计划绝对时刻发射                 |

---

## 6. 空口 / 环回

TX RF →（环回电缆或近距目标散射）→ RX RF。

同机调度下 RX 与 TX **共享同一 `tx_epoch`**：Source 仅在该时刻收 **恰好 8704** 样点，突发起点带 UHD `rx_time` tag。

---

## 7. RX 路径

### 7.1 `uhd_usrp_source_0`

| 项   | 值                                                 |
| ---- | -------------------------------------------------- |
| 设备 | USRP FPGA/RF → Host                               |
| 格式 | `fc32`                                           |
| 天线 | `RX1`                                            |
| 启动 | 已 patch：不连续流；由 TX **异步** `issue_stream_cmd` 驱动 |
| 其它 | `max_noutput ≥ max(16384, burst_len)`；`min_output_buffer ≥ 2·burst` |

### 7.2 `OfdmBurstRxBlock`（统一 sink）

`out_sig=None`：无流输出。内嵌 `_RangeProfileDisplay` 与可选 `RangeMusicEstimator`。

| 步骤 | 方法                     | 做什么                                                                                 | 设备            | 数据                   |
| ---- | ------------------------ | -------------------------------------------------------------------------------------- | --------------- | ---------------------- |
| 1    | `_on_tx_schedule`      | 收到计划 epoch（轻量 ack）                                                            | Host            | msg                    |
| 2    | `_on_tx_freq_cpi`      | 解析一次 → `_tx_cpi_latest` + 入队（粘性复用）                                        | Host            | TX 参考 CPI            |
| 3    | `_ingest_iq`           | 找`rx_time` → 从该偏移 bulk memcpy 满 `burst_len`                                 | Host            | `(8704,) complex64` |
| 4    | `_apply_delay`         | 左移`num_delay_samp`（默认 277），尾部填 0                                            | Host            | 原地                   |
| 5    | 异步 `process_rx_cpi_torch` | 入队后 H2D → CUDA → D2H（不阻塞 GR `work`） | **CUDA**（后台线程） | 见下 |
| 6    | `_emit_plot_and_music` | ROI 裁剪刷新图；可选异步 MUSIC | Host / Qt + CPU | — |

**`forecast`**：仅对 in0 索取 IQ（整 CPI / 剩余样点）；无第二流口。

### 7.3 CUDA 距离谱（`process_rx_cpi_torch`）

1. reshape `(4, 2176)`，去 CP → `(4, 2048)`。
2. `fft` + `fftshift` → RX 频域（对齐 GR CP remover + `fft_vcc` fwd+shift）。
3. 按符号 `H = tx_freq / rx_freq`。
4. 零填充到 `8192`，× Blackman–Harris 窗。
5. 距离维 FFT；功率跨符号求和 → `10·log10` → **dB 谱** `(8192,) float32`。
6. 复数谱跨符号求和 → **`(8192,) complex64`**（供 MUSIC）。

### 7.4 显示与 MUSIC（内嵌，非独立 GR 块）

- **Plot**：`_RangeProfileDisplay`（`range_profile_plot.py`）；`compute_range_roi` 裁 `(0, 3.5) m`；节流约 0.10 s。
- **MUSIC**：`RangeMusicEstimator`（`isac/sensing/detection/range_music_estimator.py`）；`ThreadPoolExecutor` 异步；默认 `num_sources=1`，`subarray_size=16`；结果画竖线。输入为全谱复数 `cx`，估计器内部再按 ROI 切片。

---

## 8. 单 CPI 时序

```text
t ≈ epoch − 0.5 s（首帧）/ epoch − 几十 ms（后续）
  TX work: tx_freq_cpi(once) + Style1 + tx_schedule + enqueue stream_cmd
  Host memcpy out0 → USRP Sink DMA

t = epoch
  USRP TX 发 8704 samp（~35.4 µs @ 245.76 MHz）
  USRP RX 同刻收 8704 samp（NUM_SAMPS_AND_DONE）

t > epoch
  RX: delay(277) → CUDA 距离谱 → plot（± MUSIC）
  Host idle ≈ idle_ms（10 ms）后下一 CPI
```

```mermaid
sequenceDiagram
  participant TX as OfdmBurstTxSource
  participant Sink as USRP_Sink
  participant Src as USRP_Source
  participant RX as OfdmBurstRx
  participant GPU as CUDA

  TX->>TX: start precompute once
  Note over TX,GPU: CUDA RG plus IFFT CP then D2H
  TX->>RX: tx_freq_cpi msg plus tx_schedule
  TX->>Src: async NUM_SAMPS_AND_DONE at epoch
  TX->>Sink: Style1 IQ at epoch
  Sink-->>Src: RF loopback
  Src->>RX: IQ plus rx_time
  RX->>RX: delay align Host
  RX->>GPU: process_rx_cpi_torch
  GPU-->>RX: db profile plus complex
  RX->>RX: plot ROI plus optional MUSIC
```

---

## 9. 设备与数据类型总表

| 阶段                       | 设备            | 典型类型 / 形状                       |
| -------------------------- | --------------- | ------------------------------------- |
| Sionna RG + IFFT/CP 预计算 | CUDA → Host    | `(4,2048) c64`，`(8704,) c64`    |
| GR TX 重放 / Style1        | Host CPU        | 同上 memcpy                           |
| USRP Sink / Source         | FPGA/RF ↔ Host | `fc32` 标量 IQ                      |
| RX 对齐 / delay            | Host            | `(8704,) c64`                      |
| 去 CP / FFT / 距离谱       | CUDA → Host    | dB`(8192,) f32`；cx `(8192,) c64` |
| Plot                       | Qt 主线程       | ROI 子集                              |
| MUSIC                      | Host 线程池     | `peak_ranges_m` 列表                |

**结论**：DSP 与可视化已收进 `OfdmBurstRx`；TX 侧 CUDA 仅用于 `start()` 预计算，运行期全 Host 重放以保证实时。

---

## 10. 相关源码索引

### 本流图直接使用

| 模块              | 路径                                                    | 角色                                                 |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| Flowgraph         | `usrp_ofdm_sink_source_dd.py` / `.grc`              | 拓扑与参数                                           |
| TX 统一源         | `src/isac_imp/ofdm_burst_tx_source.py`                | 预计算 + Style1 + 定点收样命令                       |
| RX 统一 sink      | `src/isac_imp/ofdm_burst_rx.py`                       | IQ/参考对齐 + CUDA 距离谱 + plot/MUSIC               |
| 频域网格          | `src/isac_imp/sionna_resource_grid_tx.py`             | `_build_freq_grid_torch`（被 TX `start()` 调用） |
| Style1 / 调度消息 | `src/isac_imp/burst_pack.py`                          | `tx_sob`/`tx_time`/`tx_eob`、`tx_schedule`/`tx_freq_cpi` |
| Source 工厂补丁   | `src/isac_imp/scheduled_usrp_source.py`               | 禁止启动连续收                                       |
| ROI               | `src/isac_imp/range_profile_roi_slice.py`             | `compute_range_roi`                                |
| 显示类            | `src/isac_imp/range_profile_plot.py`                  | `_RangeProfileDisplay`（被 RX 实例化）             |
| MUSIC 估计器      | `src/isac/sensing/detection/range_music_estimator.py` | 异步测距                                             |

### 本流图未接入（遗留 / 旧多块链）

| 模块                         | 说明                                 |
| ---------------------------- | ------------------------------------ |
| `sionna_ofdm_modulator.py` | 逻辑已并入 TX Torch IFFT+CP          |
| `ofdm_range_profile.py`    | 旧多块距离谱；算法由 RX Torch 复现   |
| `RangeProfilePlotBlock`    | 独立 GR 块未连；显示类仍被 RX 使用   |
| `range_music_block.py`     | 独立 GR 块未连；估计器由 RX 直接调用 |
