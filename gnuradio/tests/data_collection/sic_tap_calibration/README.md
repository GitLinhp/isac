# 频域 Divide 模板 SIC 标定流程

标定与数据采集已拆分为独立流图：

| 步骤 | 流图 | 目录 |
|------|------|------|
| 无目标 Divide 标定 | [`sic_tap_calibration.grc`](sic_tap_calibration.grc) | `dataset/run_001/calibration/` |
| 有目标距离谱采集 | [`../range_profile_collection/`](../range_profile_collection/) | `../range_profile_collection/dataset/run_001/range_profiles` |
| MUSIC 实时采集 | [`../usrp_ofdm_echotimer_dd_data_collection_test/`](../usrp_ofdm_echotimer_dd_data_collection_test/) | `../usrp_ofdm_echotimer_dd_data_collection_test/dataset/run_001/range_profiles` |

MUSIC 采集流图在 **OFDM Divide 与距离 FFT 之间**插入模板对消：

```text
… → ofdm_divide → [H(f) − template] → 距离 FFT → Range Profile / MUSIC
```

`H_clean(f) = H_meas(f) − t(f)`，模板 `t` 来自无目标场景 Divide 输出的离线均值。

> **已弃用（MUSIC 流图）**：时域 FIR SIC（`sic_cal_tx/rx.dat` → `estimate_sic_taps.py` → `sic_taps.npy`）。旧脚本仍保留供参考。

## 1. 标定录制

1. 天线固定、**无目标**。
2. **标定流图参数须与采集流图一致**：`transpose_len=2`、`TX_gain`/`RX_gain`、`factor`、`num_delay_samp`（当前默认均为 30 / 0.004 / 278，与 MUSIC 流图对齐）。
3. 运行 [`sic_tap_calibration.grc`](sic_tap_calibration.grc)。
4. 流图先稳态运行 **2–5 s**（**Cal Record Enable 关闭**），确认 Range Profile 在刷新。
5. 删除旧 `sic_cal_divide.dat`（**只删文件，保留 `calibration/` 目录**）。
6. 勾选 **Cal Record Enable**，继续录制 **≥2 s** 后**正常停止**（勿 `kill -9`）。
7. 终端应出现 `[SicDivideRecorder] recording stopped, cpis_written=...`（数千 CPI 量级）。
8. 输出文件（complex64，无文件头，每个 CPI 含 `transpose_len` 个 `(4096,)` 向量展平）：
   - `dataset/run_001/calibration/sic_cal_divide.dat` — `ofdm_divide` 输出 `H(f)`

**期望文件大小**：录制 2 s 时约 **>10 MB**（约 48k CPI/s × 2 向量 × 4096 × 8 B）；若 `cpis_written=0`，见终端诊断 `tags=` / `cpis_queued=` 并检查预热与 Range Profile。

## 2. 估计 Divide 模板

```bash
python script/implementation/estimate_sic_template.py \
  --input gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/calibration/sic_cal_divide.dat \
  --output gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/sic_template.npy
```

脚本计算 `t = mean(H)`，经与流图一致的距离 FFT 验证 **0 m 峰下降 dB**。

**验收标准**（保存 `sic_template.npy` 前须满足）：

- `Cancellation (range profile)` 应 **> 3 dB**（`--min-cancel-db`，默认 3）；否则 exit 1，勿用于采集。
- 启动 MUSIC 流图时应看到 `[SIC] loaded template (4096,) from ...`。

可选参数：`--vlen`（默认 4096）、`--transpose-len`（默认 2）、`--max-vectors`（默认 5000）、`--min-cancel-db`（默认 3）。

## 3. 启用 SIC 并验证

1. 确认 MUSIC 流图 [`usrp_ofdm_echotimer_dd_data_collection_test.grc`](../usrp_ofdm_echotimer_dd_data_collection_test/usrp_ofdm_echotimer_dd_data_collection_test.grc) 中 `sic_template_path` 指向上述 `sic_template.npy`。
2. **重启**采集流图（`main()` 内 `load_sic_template(tb)` 在 `tb.start()` 前加载），勾选 **SIC Enable**（默认已开启）。
3. 对比关闭 SIC 时的 Range Profile：0 m 直达峰及镜像峰应明显下降。

## 4. 重标定时机

- 更改 TX/RX Gain、`num_delay_samp`、`factor`、`transpose_len`
- 天线 / 线缆 / 温漂明显

## 5. 变量说明

### 标定流图

| GUI / 变量 | 说明 |
|------------|------|
| Cal Record Enable | 启用 `SicDivideRecorder` 写入 `sic_cal_divide.dat` |
| sic_cal_divide_path | Divide `H(f)` 录制路径 |
| sic_template_path | 离线估计输出路径（供参考） |

### MUSIC 采集流图

| GUI / 变量 | 说明 |
|------------|------|
| SIC Enable | 启用 Divide 域模板对消 |
| sic_template_path | `sic_template.npy` 路径 |
| sic_template_vlen | 模板长度，须为 `fft_len * zeropadding_fac`（4096） |

## 6. 重新 Generate 后

若用 GRC **Generate** 覆盖 `*.py`，需保留 `main()` 中：

- 标定流图：`Path(.../calibration).mkdir(parents=True, exist_ok=True)`
- 标定流图：`radar_ofdm_divide_vcvc_0 → SicDivideRecorder`
- MUSIC 流图：`load_sic_template(tb)`（在 `tb.start()` 之前）
- MUSIC 流图：`divide → SicDivideSubtract → 距离 FFT`（勿恢复旧时域 FIR 链）
