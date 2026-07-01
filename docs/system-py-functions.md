# `src/isac/system.py` 函数说明

模块职责：封装 ISAC 仿真 **`System`**（加载配置、组件、射线追踪场景）、通信/感知基线、单基地感知评估、数据集生成（轨迹 / 蒙特卡洛）及 HDF5 落盘辅助逻辑。

---

## 模块级私有函数

| 函数                                                    | 功能概要                                                                                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `_random_unit_vector_3d(rng)`                         | 在单位球上均匀采样三维单位方向向量 `(3,)`。                                                                                 |
| `_roi_uniform_scalar(low, high, rng)`                 | ROI 单轴采样：`low==high` 返回固定值，否则在 `[low, high]` 上均匀采样（可用给定 RNG）。                                   |
| `_scene_slug_from_rt_simulator(scene)`                    | 从 RT 仿真器读取 `rt_simulator_params.filename`，规范化用于输出文件名的 slug（空、`None`、字面 `"none"` 等归一为 `"scene"`）。 |
| `_stack_ragged_cir_samples(cir_a_list, cir_tau_list)` | 各样本 CIR 路径条数不一致时，按各维取最大值零填充后堆叠为 `(N, …)`，供 HDF5 存储。                                         |

---

## `class System`

### 初始化与信道辅助

| 方法                          | 功能概要                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `__init__(args)`            | 加载配置、`SystemParams`、`SystemComponents`，设置 `sionna.phy.config.device`。                                   |
| `_paths_cfr_numpy()`        | 调用射线追踪 `paths.cfr`，按当前 RG 的子载波频率与符号参数得到 **CFR（numpy）**。                               |
| `_paths_cir_numpy()`        | 调用 `paths.cir`，参数与 OFDM 符号对齐；返回 **`cir_a`**（最后一维 `[Re, Im]`）与 **`tau`**（秒）。 |

### 通信 / 感知基线与评估

| 方法                                                    | 功能概要                                                                                                                                                                                 |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_communication_baseline()`                        | 通信基线：比特源 → 映射 → RG →**理想直通时域** → 解调 → 硬判决 BER（不经射线信道）。                                                                                          |
| `run_sensing_baseline(domain)`                        | 感知基线：修改场景中部分物体速度、发射链路、`frequency`/`time` 域信道 → CFR → 时延-多普勒谱 → CFAR 可视化 → MUSIC 寻峰（演示管线）。                                             |
| `run_sensing_monostatic_eval(domain, velocity_model)` | **单基地感知主流程**：发射链 → 射线追踪信道 → 信道估计 / 时延-多普勒 → MUSIC → 与几何真值对齐选峰 → 记录 RMSE；返回估计距离、估计径向速度、峰值 dB 指示（`torch.Tensor`）。 |

### 数据集 HDF5

| 方法                  | 功能概要                                                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `save_dataset(...)` | 将 CFR、`cir_a`、`cir_tau`、位置/速度、基站位置及 OFDM 元数据封装为 **`Dataset`**（``isac.datasets``）并写入 HDF5（必含 CFR+CIR）。 |

### 蒙特卡洛数据集采集（已迁至 `run_dataset_collection.py`）

轨迹推进相关 API 已移除；批量 episode 由 `target_generation.generate_targets_monte_carlo` 与 [`script/model_training/run_dataset_collection.py`](script/model_training/run_dataset_collection.py) 完成。

### 蒙特卡洛：仅数据 vs 采样+感知

| 方法                                                  | 功能概要                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_dataset_monte_carlo(...)`                 | 在给定**positions** 或 ROI 内采样位置，并按策略采样速度；每样本 **`target.update`** → 几何真值 → **CFR+CIR**（必要时经 `_stack_ragged_cir_samples`）；输出 `{slug}_mc_dataset_kinematics.csv`、`_mc_sionna_dataset.h5`；可选 GIF；**tqdm**。局部 `csv_float2`。 |
| `generate_monte_carlo_with_monostatic_sensing(...)` | 与上相同的采样/更新逻辑，但每样本**`run_sensing_monostatic_eval`**，写 `_mc_dataset_sensing_metrics.csv`；默认 **`save_h5=False`**；可选 HDF5/GIF；**tqdm**。                                                                                                               |

### 数据加载与 ROI 采样

| 方法                                       | 功能概要                                                                                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_dataset_loading(dataset_filename)` | 从默认或给定路径 **`Dataset.load`**（``isac.datasets``），打印 CFR/CIR/位置/速度等 shape 与元数据（用于快速校验）。                                                             |
| `generate_monte_carlo_points(...)`       | 在三维 ROI 内**uniform / gaussian** 采样，经 **`scene.is_position_valid`** 拒绝采样剔除障碍物与安全距内无效点，返回 `(num_samples, 3)`；采样不足则抛错。 |

---

## 局部嵌套函数（非模块 API）

在 `generate_dataset_monte_carlo`、`generate_monte_carlo_with_monostatic_sensing` 中重复出现的：

- **`fmt_vec3`**：三维向量格式化为日志字符串。
- **`csv_float2`**：标量/`torch.Tensor` 转为保留两位小数的 CSV 字符串。

---

## 相关感知组件（`SystemComponents`）

| 字段 / 类 | 功能概要 |
| --------- | -------- |
| `ls_channel_estimator` / `LSChannelEstimator(rg)` | OFDM 配置存在时随 `rg` 构建；`__call__(x, y)` 做 LS 频域信道估计 `h = y·conj(x)/(|x|²+ε)`。`System.sensing` 与数据集脚本通过 `components.ls_channel_estimator(x_rg, y_rg)` 调用。 |
| `channel` / `RTChannel` / `STChannel` | `Channel` 子类：RT 多径（时/频域）或 RCS 点目标（仅时域）；`SystemComponents.channel` 按 TOML `[channel].type` 构建其一。 |

---

*文档根据 `system.py` 当前实现整理；若后续增删方法，请以源码为准。*
