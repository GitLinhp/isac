# 参照源码（GNU Radio / gr-radar）

本目录存放 **GNU Radio** 与 **gr-radar** 上游仓库，供对照 `usrp_echotimer_cc` 行为并在 ISAC 中用 **UHD USRP Sink + USRP Source** 实现等价功能。

## 仓库

| 目录 | 远程 | 说明 |
|------|------|------|
| `gr-radar/` | https://github.com/kit-cel/gr-radar | KIT gr-radar，`usrp_echotimer_cc` 实现 |
| `gnuradio/` | https://github.com/gnuradio/gnuradio (`maint-3.10`, sparse) | `gr-uhd` USRP Source/Sink 参照 |

克隆（已在项目中执行过）：

```bash
cd reference
git clone --depth 1 https://github.com/kit-cel/gr-radar.git gr-radar
git clone --depth 1 --branch maint-3.10 --filter=blob:none --sparse https://github.com/gnuradio/gnuradio.git gnuradio
cd gnuradio && git sparse-checkout set gr-uhd include/gnuradio/uhd lib/uhd
```

## USRP Echotimer → USRP Sink/Source 映射

参照 `gr-radar/lib/usrp_echotimer_cc_impl.cc`：

| Echotimer 行为 | ISAC 替代 |
|----------------|-----------|
| 单块 TX+RX、按 `packet_len` tagged stream | TX: `BurstIqTagTxBlock` → `uhd.usrp_sink`（Style1 `tx_sob`/`tx_time`/`tx_eob`） |
| `STREAM_MODE_NUM_SAMPS_AND_DONE` 定时收包 | RX: `burst_iq_tag_tx` 每 CPI `issue_stream_cmd` → `uhd.usrp_source(..., False)` |
| `wait_tx` / `wait_rx` 未来时刻调度 | `time_lead_s` + `tx_time` tag；Snippet `set_time_now` 对时 |
| `num_delay_samps` 样点域前移 + 尾补零 | `EchotimerRxCompensatorBlock` |
| CPI 首样点 `packet_len` tag | `PacketLenTaggerBlock` |
| CPI 首样点 `rx_time` tag | UHD Source 自带 + compensator 归一化到 CPI 边界 |
| 增益 / 延迟 GUI 回调 | GRC 变量 + `set_TX_gain` / `set_RX_gain` / `set_num_delay_samp` |

## 流图

- **替代实现**：`gnuradio/tests/single_base/usrp_ofdm_sink_source_dd/`
- **原 echotimer 基线（已改为 Sink/Source）**：`gnuradio/tests/single_base/usrp_ofdm_echotimer_dd/`

## 关键约束（UHD 4.9 + X410）

- `uhd.usrp_source(..., False)`：`issue_stream_cmd_on_start=False`，禁止自动连续 RX
- **禁止**在 `tb.start()` 之前调用 `issue_stream_cmd()`（会 SIGSEGV）
- Scheduled RX 命令仅在流图运行后由 `burst_iq_tag_tx.work()` 下发
