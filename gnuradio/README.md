# GNU Radio / Sionna 流图

本目录包含 ISAC 项目的 GNU Radio Companion (GRC) 流图与自定义 Sionna 块。

## 前置条件

在 ISAC conda 环境中于**仓库根目录**安装可编辑包：

```bash
pip install -e .
```

`isac` 由 pip 提供；[`bootstrap.py`](bootstrap.py) 仅引导 `blocks/`、`core/`、`flowgraphs/`、`tools/` 到 `sys.path`（GRC 块仍使用 `from sionna_tx import ...`）。

## 目录结构

| 目录 | 内容 |
|------|------|
| [`bootstrap.py`](bootstrap.py) | 统一 `sys.path` 引导 + `isac` 可导入校验 |
| [`core/`](core/) | 配置合并（`gr_config`）、System 上下文（`gr_system`）、流图 UI（`flowgraph_perf`） |
| [`blocks/`](blocks/) | 自定义 Sionna GR 块实现及 `.block.yml` 定义 |
| [`flowgraphs/`](flowgraphs/) | 活跃 `.grc` 流图、入口脚本、grcc 生成物 |
| [`tools/`](tools/) | 块安装脚本、离线验证脚本 |
| [`legacy/`](legacy/) | 历史流图备份，仅供对照，勿用于日常开发 |

## 两条产品线

| 流图 | 信道 | 配置 TOML |
|------|------|-----------|
| `flowgraphs/sensing_baseline.grc` | RT 射线追踪（`SionnaRTChannel`） | `config/simulation/sensing/sensing_baseline.toml` |
| `flowgraphs/simulator_ofdm.grc` | 静态点目标（`SionnaStaticTarget`） | `config/simulation/sensing/sensing_monostatic.toml` |

## 常用命令

```bash
# 安装 / 更新 GRC 块定义
bash gnuradio/tools/install_grc_blocks.sh

# 在 flowgraphs/ 目录下 grcc 生成 Python
# grcc sensing_baseline.grc
# grcc simulator_ofdm.grc

# 带 GPU 预热与感知 UI 的入口（须在 ISAC conda 环境中运行）
/home/caict/radioconda/envs/ISAC/bin/python gnuradio/flowgraphs/run_sensing_baseline_grc.py
/home/caict/radioconda/envs/ISAC/bin/python gnuradio/flowgraphs/run_simulator_ofdm.py

# 离线验证
python gnuradio/tools/verify_baseline_grc.py
python gnuradio/tools/verify_dd_axis.py
```

运行流图前请确保工作目录为仓库根目录（入口脚本会自动 `chdir`），以便相对 TOML 路径可解析。
