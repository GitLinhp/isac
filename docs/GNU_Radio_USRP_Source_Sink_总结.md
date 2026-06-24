# GNU Radio USRP Source / Sink 中文总结

> 本文档基于 [GNU Radio Wiki - USRP Source](https://wiki.gnuradio.org/index.php?title=USRP_Source) 与 [USRP Sink](https://wiki.gnuradio.org/index.php?title=USRP_Sink)，并结合本地 GNU Radio 3.10 GRC 块定义与 [官方 UHD 接口文档](https://gnuradio.readthedocs.io/en/latest/page_uhd.html) 整理而成。

---

## 目录

1. [概述](#1-概述)
2. [架构与数据流](#2-架构与数据流)
3. [公共参数（General / Advanced）](#3-公共参数general--advanced)
4. [RF Options（每通道 ChX）](#4-rf-options每通道-chx)
5. [USRP Source 详解](#5-usrp-source-详解)
6. [USRP Sink 详解](#6-usrp-sink-详解)
7. [消息端口与运行时控制](#7-消息端口与运行时控制)
8. [Source 与 Sink 对比速查](#8-source-与-sink-对比速查)
9. [最佳实践与常见问题](#9-最佳实践与常见问题)
10. [参考链接](#10-参考链接)

---

## 1. 概述

`USRP Source` 与 `USRP Sink` 是 GNU Radio 中通过 **UHD（USRP Hardware Driver）** 与 Ettus Research USRP 系列软件无线电硬件交互的核心块，属于 `gnuradio.uhd` 包：

```python
from gnuradio import uhd
```

| 块 | 角色 | 功能 |
|---|---|---|
| **USRP Source** | 接收机（Receiver） | 从 USRP 硬件读取 IQ 样本，输出到 GNU Radio 流图 |
| **USRP Sink** | 发射机（Transmitter） | 从 GNU Radio 流图读取 IQ 样本，发送到 USRP 硬件发射 |

两者均继承自 `usrp_block`，共享采样率、中心频率、增益、天线、带宽、时钟/时间源等大量 API。在 GRC 中分别显示为 **UHD: USRP Source** 与 **UHD: USRP Sink**。

---

## 2. 架构与数据流

```
接收流图：
  USRP 硬件 ──► USRP Source ──► [信号处理块] ──► ...

发射流图：
  [信号源/处理块] ──► USRP Sink ──► USRP 硬件
```

**消息端口**（两者均有）：

- **输入 `command`**：运行时动态修改 RF 参数（频率、增益等）
- **输出 `async_msgs`**：异步状态/事件通知（如溢出、欠载）

---

## 3. 公共参数（General / Advanced）

以下参数在 Source 与 Sink 的 GRC 属性面板中基本一致。

### 3.1 设备与流配置

| 参数 | 说明 |
|------|------|
| **Device Address** | 用于定位 UHD 设备的分隔字符串。留空则使用系统中发现的第一个设备。示例：`serial=12345678`（USRP1）、`addr=192.168.10.2`（USRP2/N 系列）、`addr0=192.168.10.2, addr1=192.168.10.3`（多设备） |
| **Device Arguments** | 附加设备参数字符串 |
| **Input/Output Type** | GNU Radio 流数据类型：`fc32`（Complex float32，默认）、`sc16`（Complex int16）、`item32`（VITA word32） |
| **Wire Format** | 总线/网络传输格式：`Automatic`、`sc16`、`sc12`、`sc8`。Complex bytes 可在精度与带宽间权衡，并非所有设备支持全部格式 |
| **Stream Args** | 传给 UHD streamer 的可选键值对，用法由实现决定（如 sc8 格式下的 `peak=0.003906` 影响 16/8 位整数缩放） |
| **Stream Channels** | 指定使用的通道列表，如 `[0, 1]`；留空则按 Num Channels 自动分配 |

### 3.2 同步与时钟

| 参数 | 说明 |
|------|------|
| **Sync** | 初始时间对齐方式：`Unknown PPS`（尝试同步到 PPS 边沿）、`PC Clock`（设为 PC 系统时间）、`PC Clock on Next PPS`、`GPS Time on Next PPS`、`No Sync`（不同步） |
| **Start Time (seconds)** | 输出/输入样本的起始绝对时间；-1 表示默认 |
| **Clock Rate (Hz)** | 主板主时钟频率，与采样率相关但不同。B2X0、E31X 等使用灵活时钟，通常等于请求采样率或其倍数；一般保持默认即可 |
| **Num Mboards** | 本配置中的 USRP 主板（物理设备）数量，最多 8 |
| **MbX: Clock Source** | 时钟参考源：`Internal`（默认）、`External`（10 MHz SMA 输入）、`MIMO Cable`、`O/B GPSDO`（板载 GPSDO） |
| **MbX: Time Source** | 时间参考源：`External`（PPS SMA 输入）、`MIMO Cable`、`O/B GPSDO` |
| **MbX: Subdev Spec** | 子设备选择字符串，格式为 `dboard_slot:subdev_name` 列表。留空则 UHD 自动选择第一个子设备。单通道示例：`:AB`；双通道示例：`:A :B` |

### 3.3 通道与采样率

| 参数 | 说明 |
|------|------|
| **Num Channels** | 多 USRP 配置中的总通道数。例：4 块主板 × 每板 2 通道 = 8 通道 |
| **Sample Rate (Sps)** | 每秒采样数，在接收时等于可观测带宽（Hz）。UHD 驱动尽力匹配请求值；若无法实现，运行时会打印错误 |

> **注意**：若 Source 与 Sink 引用同一设备，时钟/时间参考源只需在其中一侧设置即可。

---

## 4. RF Options（每通道 ChX）

每个通道（Ch0、Ch1…）在 **RF Options** 分类下独立配置。

| 参数 | 说明 |
|------|------|
| **Center Frequency (Hz)** | RF 链整体中心频率。可传入浮点数，也可传入 `uhd.tune_request` 对象以精细控制 LO/DSP 调谐 |
| **Gain Value** | 增益值。Absolute 模式下通常为 0 至 USRP 最大增益（约 70–90 dB）；Normalized 模式下为 0.0–1.0（1.0 映射到最大增益） |
| **Gain Type** | `Absolute (dB)` / `Normalized (0–1)` / `Absolute Power (dBm)`（部分设备支持，需校准数据） |
| **Antenna** | 天线端口选择，取决于 daughterboard。Source 常见 `RX2`、`RX1`、`TX/RX`；Sink 常见 `TX/RX` |
| **Bandwidth (Hz)** | USRP 抗混叠滤波带宽；设为 0 使用默认。仅部分子设备支持可配置带宽 |
| **LO Source / LO Export**（高级） | 本振来源（internal/external/companion）及是否导出 LO |

### 4.1 调谐示例（tune_request）

```python
# 带 LO 偏移调谐
uhd.tune_request(f_target, f_offset)

# 不使用 DSP（手动策略）
uhd.tune_request(target_freq, dsp_freq=0, dsp_freq_policy=uhd.tune_request.POLICY_MANUAL)
```

### 4.2 Source 独有：AGC 与前端校正

| 参数 | 说明 |
|------|------|
| **ChX: AGC** | 自动增益控制：`Default` / `Disabled` / `Enabled`。启用 AGC 后，手动 Gain 设置将被忽略 |
| **ChX: Enable DC Offset Correction** | 去除 DC 偏移（频域中表现为零频尖峰）：`Default` / `Automatic` / `Disabled` / `Manual` |
| **ChX: Enable IQ Imbalance Correction** | 校正 I/Q 路径不匹配（会导致星座图拉伸）：同上四种模式 |

Sink 块在 GRC 中无 AGC/FE Corrections 面板，但 Python API 仍提供 `set_dc_offset()`、`set_iq_balance()` 等方法。

---

## 5. USRP Source 详解

### 5.1 基本功能

USRP Source 从 USRP 设备持续接收 IQ 样本，并通过输出流端口提供给下游块。默认行为为 **连续接收**（continuous streaming）。

**Python 创建示例：**

```python
src = uhd.usrp_source(
    "",  # device_addr，空字符串表示第一个设备
    uhd.stream_args(cpu_format="fc32", channels=[0])
)
src.set_samp_rate(1e6)
src.set_center_freq(915e6, 0)
src.set_gain(40, 0)
src.set_antenna("RX2", 0)
```

### 5.2 消息端口

| 端口 | 方向 | 说明 |
|------|------|------|
| `command` | 输入 | 接收 PMT 字典命令，动态修改 freq/gain/antenna 等 |
| `async_msgs` | 输出 | GNU Radio ≥ 3.10.5 起，发布溢出（overflow）等异步事件 |

**溢出（Overflow）**：终端显示 `O` 表示主机消费数据速度跟不上 USRP 产生数据的速度，可能由 USB/网络瓶颈、CPU 负载过高或流图处理过重引起。

### 5.3 高级 API

| 方法 | 说明 |
|------|------|
| `issue_stream_cmd(cmd)` | 高级用法：向所有通道下发 `stream_cmd_t`，覆盖默认连续接收。启动流图后需先 `stop()` 再下发命令 |
| `finite_acquisition(nsamps)` | Python 便捷有限采集，返回 `complex float` 向量；**不用于调度器** |
| `finite_acquisition_v(nsamps)` | 多通道版有限采集 |
| `set_rx_agc(enable, chan)` | 启用/禁用 AGC |
| `set_recv_timeout(timeout, one_packet)` | 设置接收超时 |
| `set_start_time(time_spec_t)` | 设置输出样本的起始绝对时间 |
| `set_auto_dc_offset()` / `set_auto_iq_balance()` | 自动 DC/IQ 校正 |

**stream_cmd 模式**（UHD `stream_cmd_t`）：

- `STREAM_MODE_START_CONTINUOUS`：连续流
- `STREAM_MODE_STOP_CONTINUOUS`：停止连续流
- `STREAM_MODE_NUM_SAMPS_AND_DONE`：采集指定样本数后结束
- `STREAM_MODE_NUM_SAMPS_AND_MORE`：采集指定样本数，期待后续连续命令

---

## 6. USRP Sink 详解

### 6.1 基本功能

USRP Sink 读取上游 IQ 样本流并发送到 USRP 硬件进行 RF 发射。

**Python 创建示例：**

```python
snk = uhd.usrp_sink(
    "",
    uhd.stream_args(cpu_format="fc32", channels=[0]),
    ""  # tsb_tag_name，留空表示不使用 Tagged Stream 模式
)
snk.set_samp_rate(1e6)
snk.set_center_freq(915e6, 0)
snk.set_gain(30, 0)
snk.set_antenna("TX/RX", 0)
```

### 6.2 Sink 独有参数：TSB Tag Name

| 参数 | 说明 |
|------|------|
| **TSB tag name**（`len_tag_name`） | 非空时，Sink 进入 **Tagged Stream** 模式，监听该键名的长度标签以确定发射突发长度。留空则使用 SOB/EOB 标签模式 |

### 6.3 突发发射（Bursty Transmission）

USRP Sink 支持两种互斥的突发发射方式，用于分组/定时发射，避免欠载（Underrun，`U`）。

#### Style 1：SOB/EOB 标签（`tsb_tag_name` 留空）

使用流标签（Stream Tags）控制突发时序：

| 标签 | 位置 | 值 | 说明 |
|------|------|-----|------|
| `tx_sob` | 突发**首**样本 | `pmt.PMT_T` | Start of Burst |
| `tx_time` | 突发**首**样本 | `(uint64秒, double小数秒)` | 突发开始的绝对时间 |
| `tx_eob` | 突发**末**样本 | `pmt.PMT_T` | End of Burst，通知 USRP 可 idle 发射硬件 |

**时间戳格式示例：** 时间 `1416299676.3453495` 表示为 `(1416299676, 0.3453495)`。

**注意事项：**

- `Burst Tagger` 块**不能**用于此模式，因为它不注入 `tx_time` 标签
- `tx_eob` 仅禁用欠载警告，**不会**自动切换 TX/RX 路径；`tx_eob` 之后的样本仍会被当作突发内容发送
- 相邻突发的 `tx_sob` 偏移通常紧接上一个 `tx_eob` 之后

#### Style 2：Tagged Stream（设置 `tsb_tag_name`，如 `tx_pkt_len`）

适用于分组/包处理场景：

- 在突发**首**样本上放置长度标签（键名 = `tsb_tag_name`）和 `tx_time` 标签
- 长度值为 PMT 编码的**整数**，表示 PDU 的**样本数**（非字节数）
- 突发之间**不可重叠、不可有间隙**
- 若首标签 offset=0、包长 1000，则下一组标签应出现在 offset=1000
- 设置 `tsb_tag_name` 后，所有 `tx_sob`/`tx_eob` 标签将被**忽略**

### 6.4 Sink 消费的其他流标签

| 标签 | 说明 |
|------|------|
| `tx_freq` | double 或 `(channel, frequency)` 对，触发 USRP 调谐 |
| `tx_command` | 携带 PMT 命令，在样本时刻执行 |
| `tsb_tag_name`（用户指定） | Tagged Stream 长度标签 |

当使用 SOB/EOB 或长度标签时，Sink 识别数据为突发模式，并配置 USRP 在突发最后一采样后避免欠载。

### 6.5 欠载（Underflow）

终端显示 `U` 表示主机向 USRP 供数速度不够快，常见于突发间隙或 CPU/流图处理瓶颈。

---

## 7. 消息端口与运行时控制

Source 与 Sink 均支持通过 **`command` 消息端口**在流图运行期间异步修改参数。命令为 PMT 字典（key/value 对）。

### 7.1 Python 示例

```python
import pmt

cmd = pmt.make_dict()
cmd = pmt.dict_add(cmd, pmt.intern("freq"), pmt.from_double(352e6))
cmd = pmt.dict_add(cmd, pmt.intern("gain"), pmt.from_double(50))
cmd = pmt.dict_add(cmd, pmt.intern("chan"), pmt.from_long(0))

# 在流图中：msg_connect((msg_source, 'out'), (usrp_block, 'command'))
```

### 7.2 常用命令键

| 命令键 | 值类型 | 说明 |
|--------|--------|------|
| `freq` | double | 设置 Tx/Rx 频率（默认所有通道） |
| `lo_offset` | double | LO 偏移（不影响有效中心频率） |
| `tune` | tune_request | 完整调谐请求（中心频率 + DSP 偏移） |
| `mtune` | dict | 手动完整调谐（支持 rf_freq_policy、dsp_freq_policy 等） |
| `gain` | double | 增益（dB） |
| `power_dbm` | double | 功率参考电平（dBm，部分设备） |
| `rate` | double | 采样率（影响所有通道） |
| `bandwidth` | double | 带宽 |
| `antenna` | string | 天线选择 |
| `chan` | int | 指定通道；`-1` 表示所有通道 |
| `time` | timestamp | 命令生效时间 `(full_secs, frac_secs)`；`PMT_NIL` 清除 |
| `mboard` | int | 主板索引 |
| `gpio` | dict | GPIO 控制（bank/attr/value/mask） |

> 推荐使用**字典格式**一次性发送多个参数，而非多条单键消息：同一 timestamp 对所有键生效，且所有设置同时应用。

---

## 8. Source 与 Sink 对比速查

| 特性 | USRP Source | USRP Sink |
|------|-------------|-----------|
| 方向 | 接收（输出流） | 发射（输入流） |
| 流端口 | 1 个输出（多通道） | 1 个输入（多通道） |
| GRC 数据类型参数 | Output Type | Input Type |
| AGC | 支持（ChX: AGC） | 无 |
| 前端校正 | DC/IQ 自动/手动（GRC 面板） | 无 GRC 面板（有 API） |
| 突发/分组模式 | `issue_stream_cmd` / `finite_acquisition` | SOB/EOB 标签 或 TSB Tagged Stream |
| 独有 GRC 参数 | — | TSB tag name |
| `async_msgs` | 溢出（overflow）事件 | 异步状态消息 |
| 终端 `O` | 溢出（消费太慢） | — |
| 终端 `U` | — | 欠载（供数太慢） |
| 构造函数第三参数 | `issue_stream_cmd_on_start` | `tsb_tag_name` |

---

## 9. 最佳实践与常见问题

### 9.1 硬件流图禁忌

- **切勿**在含 USRP Source/Sink 的流图中使用 **Throttle** 块（GRC 生成时会警告）
- 硬件块自身以真实采样率驱动，Throttle 会导致速率不匹配

### 9.2 采样率处理

- 请求采样率可能与实际设置值不同，用 `get_samp_rate()` 查询实际值
- 若需精确速率，可用 `filter.pfb.arb_resampler_ccf` 做任意重采样：

```python
desired_rate = 1e6
usrp.set_samp_rate(desired_rate)
actual_rate = usrp.get_samp_rate()
resample_factor = desired_rate / actual_rate
resampler = filter.pfb.arb_resampler_ccf(resample_factor)
```

### 9.3 溢出/欠载排查

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| `O`（溢出） | USB/网络瓶颈、CPU 不足、下游处理过重 | 降低采样率、简化流图、增大缓冲 |
| `U`（欠载） | 上游供数不足、突发间隙配置不当 | 检查突发标签、优化上游块 |
| `L`（late commands） | 定时命令晚于预期执行时刻 | 检查 `tx_time` / `command time` 设置 |

GNU Radio 3.10.5+ 可通过 `async_msgs` 端口连接 `Message Debug` 块，程序化监控溢出事件。

### 9.4 多 USRP 同步

1. 配置 **Clock Source**（10 MHz 参考）与 **Time Source**（PPS）
2. 使用 `set_time_next_pps(uhd.time_spec(0.0))` 在下一 PPS 边沿对齐时间
3. 外部 PPS 信号应由 10 MHz 参考时钟驱动

### 9.5 日志配置

在 `gnuradio.conf` 中可配置 UHD 日志间隔（默认 750 ms）：

```ini
[uhd]
logging_interval_ms=750
```

---

## 10. 参考链接

| 资源 | 链接 |
|------|------|
| GNU Radio Wiki - USRP Source | https://wiki.gnuradio.org/index.php?title=USRP_Source |
| GNU Radio Wiki - USRP Sink | https://wiki.gnuradio.org/index.php?title=USRP_Sink |
| GNU Radio UHD Interface 文档 | https://gnuradio.readthedocs.io/en/latest/page_uhd.html |
| UHD 官方手册 | http://files.ettus.com/manual/ |
| GNU Radio Message Passing | https://wiki.gnuradio.org/index.php/Message_Passing |
| 硬件考虑教程 | https://wiki.gnuradio.org/index.php/Guided_Tutorial_Hardware_Considerations |

---

*文档生成日期：2026-06-24*
