#!/usr/bin/env bash
# 将 gnuradio/blocks/*.block.yml 安装到 GRC 本地块目录，使 GRC GUI / grcc 加载最新块定义。
set -euo pipefail
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCKS_DIR="${TOOLS_DIR}/../blocks"
STATE="${GRC_HIER_PATH:-${HOME}/.local/state/gnuradio}"
mkdir -p "$STATE"
for f in sionna_bootstrap sionna_ofdm_tx sionna_dd_rx sionna_static_target sionna_rt_channel sionna_dd_spectrogram; do
  rm -f "$STATE/${f}.block.yml"
  cp -f "$BLOCKS_DIR/${f}.block.yml" "$STATE/${f}.block.yml"
  echo "installed $STATE/${f}.block.yml"
done
echo ""
echo "块定义已更新。若 GRC 仍显示 Bandwidth / R_max 等旧参数："
echo "  1) 完全退出 GNU Radio Companion 后重新打开"
echo "  2) 重新打开 flowgraphs/simulator_ofdm.grc（勿用 legacy/ 备份）"
echo "  3) 或菜单：View → Reload Blocks（若有）"
