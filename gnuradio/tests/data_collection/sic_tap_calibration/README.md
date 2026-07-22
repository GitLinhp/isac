# 固定 FIR 自干扰对消（SIC）标定流程

标定与数据采集已拆分为两个独立流图：

| 步骤 | 流图 | 目录 |
|------|------|------|
| 无目标 IQ 标定 | [`sic_tap_calibration.grc`](sic_tap_calibration.grc) | `dataset/run_001/calibration/` |
| 有目标距离谱采集 | [`../range_profile_collection/`](../range_profile_collection/) | `../range_profile_collection/dataset/run_001/range_profiles` |

采集流图在 Echotimer RX 与 CP Remover 之间插入 SIC 链：

```text
TX_ref → fir_filter_ccc(h) ──┐
                              ├─ sub_cc → selector → CP Remover
RX ──────────────────────────┘
```

`y_clean = RX - conv(TX_ref, h)`，抽头 `h` 来自本目录的离线标定。

## 1. 标定录制

1. 天线固定、**无目标**，TX/RX Gain 与 [`range_profile_collection`](../range_profile_collection/) 一致。
2. 运行 [`sic_tap_calibration.grc`](sic_tap_calibration.grc)，勾选 **Cal Record Enable**。
3. 运行 2–5 s 后正常停止（勿 `kill -9`，便于 file_sink 刷盘）。
4. 输出文件（complex64，无文件头）：
   - `dataset/run_001/calibration/sic_cal_tx.dat` — `multiply_const` 后 TX 参考
   - `dataset/run_001/calibration/sic_cal_rx.dat` — Echotimer RX

## 2. 估计 FIR 抽头

```bash
python script/implementation/estimate_sic_taps.py \
  --tx gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/calibration/sic_cal_tx.dat \
  --rx gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/calibration/sic_cal_rx.dat \
  --num-taps 64 \
  --output gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/sic_taps.npy
```

脚本会打印 coarse lag、LS 段内对消 dB 与峰值抽头索引。

## 3. 启用 SIC 并验证

1. 确认 [`range_profile_collection`](../range_profile_collection/range_profile_collection.grc) 中 `sic_taps_path` 指向上述 `sic_taps.npy`。
2. **重启**采集流图（启动时加载抽头），勾选 **SIC Enable**（默认已开启）。
3. 对比关闭 SIC 时的 Range Profile：
   - 0 m 直达峰应明显下降
   - ~R_max 镜像峰应同步减弱
4. 频域 sink 仍显示**原始 RX**，便于对照。

## 4. 重标定时机

以下情况需重新录制并运行 `estimate_sic_taps.py`：

- 更改 TX/RX Gain、`num_delay_samp`、`factor`
- 天线位置 / 线缆 / 温漂明显
- 修改 `sic_num_taps`（GRC 变量须与 `--num-taps` 一致）

## 5. 变量说明（标定流图）

| GUI / 变量 | 说明 |
|------------|------|
| Cal Record Enable | 写入 `calibration/*.dat` |
| sic_cal_tx_path / sic_cal_rx_path | 标定 IQ 输出路径 |

采集流图变量见 [`range_profile_collection`](../range_profile_collection/)。

## 6. 重新 Generate 后

若用 GRC **Generate** 覆盖 `range_profile_collection.py`，需保留其中的 `_load_sic_taps_from_path` 与 tag propagation 补丁，或从当前已补丁版本复制相应段落。
