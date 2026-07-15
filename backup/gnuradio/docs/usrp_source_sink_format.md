# GNU Radio USRP Source / USRP Sink 数据格式说明

本文说明 GNU Radio 中 **UHD: USRP Source**（接收）与 **UHD: USRP Sink**（发射）在 flowgraph 流端口上的数据格式。二者通过 UHD 驱动与 USRP 硬件交互，**CPU 格式**（GNU Radio 缓冲区中的样点类型）与 **Wire Format / OTW 格式**（主机与 USRP 之间的传输格式）是两层概念，UHD 会在两者之间自动转换。

---

## 总览对比

| 块 | 流端口方向 | GRC 参数名 | 可选 CPU 格式 | 默认 |
|----|-----------|-----------|--------------|------|
| **USRP Source** | **输出**（out） | Output Type | fc32 / sc16 / item32 | **fc32** |
| **USRP Sink** | **输入**（in） | Input Type | fc32 / sc16 / item32 | **fc32** |

| 属性 | 说明 |
|------|------|
| **domain** | `stream`（样点流） |
| **vlen** | 1（每个流 item 为 **1 个复数 IQ 样点**） |
| **multiplicity** | 等于 `Num Channels`（多通道时每个通道独立一个端口） |
| **样点语义** | 复基带 IQ，采样率为块内配置的 **Sample Rate [Hz]** |
| **Throttle** | 两块的 `flags` 均含 `throttle`，**无需**再挂 Throttle 块；硬件收发速率由 USRP 决定 |

除流端口外，二者还有可选 **message** 端口：

- **command**（输入）：运行时通过 PMT 消息改频率、增益等
- **async_msgs**（输出）：异步状态/事件消息

---

## 两种格式：CPU Format 与 Wire Format

```
USRP 硬件 (ADC/DAC, 整数 IQ)
        │  OTW / Wire Format (sc16 / sc12 / sc8 …)
        │  ← UHD 自动类型转换与幅度缩放 →
        ▼
GNU Radio buffer (CPU Format: fc32 / sc16 / item32)
        │
   Source 输出 / Sink 输入
```

| 层级 | GRC 参数 | 作用 |
|------|---------|------|
| **CPU Format** | Output Type / Input Type | flowgraph 中块与块之间传递的数据类型 |
| **Wire Format (OTW)** | Wire Format | 主机与 USRP 之间链路上的传输格式；可选 Automatic / sc16 / sc12 / sc8 |

Wire Format 用更低比特可换带宽，但动态范围与量化噪声会变差；并非所有 USRP 型号都支持全部 OTW 选项。

---

## USRP Source — 输出格式

USRP Source 从 USRP **接收** RF 样点，经 UHD 转为所选 CPU 格式后，写入 **输出流端口**。

### 端口定义（GRC block YAML）

```yaml
outputs:
- domain: stream
  dtype: ${type.type}      # fc32 | sc16 | s32
  multiplicity: ${nchan}   # 通道数
```

### 可选 Output Type 详解

#### 1. Complex float32（fc32）— 推荐默认

| 项目 | 内容 |
|------|------|
| GRC 端口颜色 | 蓝色 |
| C++ 类型 | `std::complex<float>` |
| Python / NumPy | `numpy.complex64` / `dtype='complex64'` |
| 每样点字节数 | **8 B**（I、Q 各 32-bit float） |
| 内存布局 | 小端序；`(real, imag)` 即 `(I, Q)` |
| 幅度范围 | 浮点；UHD 通常将 ADC 满量程映射到约 **[-1.0, +1.0]**（可通过 `stream_args` 的 `fullscale` 等调整） |

每个输出 item 表示：**在 Sample Rate 时刻 t 的一个复基带 IQ 样点**，尚未做 OFDM 解调等上层处理。

#### 2. Complex int16（sc16）

| 项目 | 内容 |
|------|------|
| GRC dtype | `sc16` |
| C++ 类型 | `std::complex<int16_t>` |
| 每样点字节数 | **4 B**（I、Q 各 16-bit 有符号整数） |
| 幅度 | 整数 IQ；UHD 与 fc32 互转时会做缩放 |

适合对 host 内存/带宽敏感、且可接受定点精度的链路。

#### 3. VITA word32（item32 / s32）

| 项目 | 内容 |
|------|------|
| GRC 显示名 | VITA word32 |
| GRC dtype | `s32` |
| 含义 | 32-bit 字打包的 VITA IF 样点格式（legacy / 特殊用途） |
| 每样点字节数 | **4 B** |

一般 SDR 应用优先使用 **fc32**；item32 多用于与特定 VITA 生态对接。

### 多通道输出

`Num Channels = N` 时，有 **N 个独立输出端口** `out0 … out(N-1)`，类型相同，各对应一路 RF 通道的 IQ 流。

### 输出流标签（Tags）

Source 在 work 过程中可能产生：

| Tag key | 含义 |
|---------|------|
| `rx_time` | 接收时间戳，PMT 元组 `(uint64 秒, double 小数秒)` |

在 `start()` 及溢出恢复等时刻也会出现该 tag。

---

## USRP Sink — 输入格式

USRP Sink 从 **输入流端口** 读取样点，经 UHD 转为 OTW 格式后 **发射** 到 USRP。

### 端口定义（GRC block YAML）

```yaml
inputs:
- domain: stream
  dtype: ${type.type}      # fc32 | sc16 | s32
  multiplicity: ${nchan}
```

### 可选 Input Type

与 Source 的 Output Type **一一对应**（fc32 / sc16 / item32），语义相同，只是方向为 **输入**：

- **fc32**：上游块应输出 `complex64` IQ，幅度约 **[-1.0, +1.0]** 对应 DAC 满量程
- **sc16**：上游输出 16-bit 整数 IQ
- **item32**：VITA word32 打包格式

**类型必须匹配**：Sink 的 Input Type 须与上游块的输出 dtype 一致（GRC 中端口颜色一致），否则无法连线或运行时报类型错误。

### 多通道输入

每路 RF 通道对应一个输入端口；每端口仍为 **vlen=1 的复数 IQ 流**。

### 输入流标签（Tags，Burst / 定时发射）

Sink 会识别并消费以下 tag（其余 tag 忽略）：

| Tag key | 用途 |
|---------|------|
| `tx_sob` | Start of Burst，`PMT_T` |
| `tx_eob` | End of Burst，`PMT_T` |
| `tx_time` | 定时发射时刻，`(uint64 秒, double 小数秒)` |
| `tx_freq` | 发射前调频；double 或 `(channel, freq)` |
| `tx_command` | 绑定样点的 UHD 命令 |
| `tsb_tag_name` 指定的长度 tag | Tagged Stream 突发长度（整数，单位：**样点数**） |

若设置了 **TSB tag name**（如 `tx_pkt_len`），则走 Tagged Stream 突发模式，`tx_sob` / `tx_eob` 会被忽略。

---

## 与 ISAC / OFDM 仿真对接时的建议

本仓库 `simulator_ofdm.grc` 等 Sionna 块输出为 **fc32 复基带**（如 `sionna_ofdm_tx` → `sionna_static_target`）。若接入真实 USRP：

```
Sionna OFDM Tx (fc32 IQ) ──► USRP Sink (Input Type: fc32)
USRP Source (Output Type: fc32) ──► Sionna DD Rx / 其他 fc32 收端
```

注意：

1. **Sample Rate** 与 OFDM 参数（如 30.72 MHz）一致
2. **Center Frequency** 与 `center_freq` 等射频参数一致
3. fc32 幅度过大可能导致 DAC  clipping；过小则 SNR 变差
4. 使用 Source 时 **不要** 对同一链路再挂 Throttle
5. 半双工感知场景需协调 TX/RX 时序与 `tx_sob` / `tx_eob` / `tx_time` 或 TSB 突发

---

## Python 实例化（与 GRC 等价）

```python
from gnuradio import uhd

# 接收：complex float32 输出，OTW 自动
src = uhd.usrp_source(
    "",  # device address
    uhd.stream_args(cpu_format="fc32", channels=[0]),
)

# 发射：complex float32 输入
snk = uhd.usrp_sink(
    "",
    uhd.stream_args(cpu_format="fc32", channels=[0]),
)
```

指定 Wire Format 示例：

```python
uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0])
```

---

## 参考资料

- [GNU Radio — USRP Source](https://wiki.gnuradio.org/index.php/USRP_Source)
- [GNU Radio — USRP Sink](https://wiki.gnuradio.org/index.php/USRP_Sink)
- [GNU Radio — Signal Data Types](https://wiki.gnuradio.org/index.php/Signal_Data_Types)
- [UHD — Configuring Devices and Streamers](https://files.ettus.com/manual/page_configuration.html)
- [UHD — Converters (CPU ↔ OTW)](https://files.ettus.com/manual/page_converters.html)
- 本机 GRC 块定义：`/usr/share/gnuradio/grc/blocks/uhd_usrp_source.block.yml`、`uhd_usrp_sink.block.yml`
