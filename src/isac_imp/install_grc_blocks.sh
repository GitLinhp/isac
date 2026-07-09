#!/usr/bin/env bash
# 将 isac_imp/blocks 与遗留 gnuradio/blocks 的 *.block.yml 安装到 GRC 本地块目录。
set -euo pipefail

ISAC_IMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISAC_IMP_DIR}/../.." && pwd)"
ISAC_BLOCKS_DIR="${ISAC_IMP_DIR}/blocks"
GR_BLOCKS_DIR="${REPO_ROOT}/gnuradio/blocks"
STATE="${GRC_HIER_PATH:-${HOME}/.local/state/gnuradio}"

mkdir -p "$STATE"

# isac_imp：每个子目录下的 *.block.yml
shopt -s nullglob
for yml in "${ISAC_BLOCKS_DIR}"/*/*.block.yml; do
  base="$(basename "$yml")"
  cp -f "$yml" "${STATE}/${base}"
  echo "installed ${STATE}/${base}"
done
shopt -u nullglob

# 遗留 gnuradio/blocks（不含已迁走的 sionna_dd_spectrogram）
for f in sionna_bootstrap sionna_ofdm_tx sionna_dd_rx sionna_static_target sionna_rt_channel; do
  src="${GR_BLOCKS_DIR}/${f}.block.yml"
  if [[ ! -f "$src" ]]; then
    echo "warning: missing $src" >&2
    continue
  fi
  cp -f "$src" "${STATE}/${f}.block.yml"
  echo "installed ${STATE}/${f}.block.yml"
done

echo ""
echo "块定义已更新。若 GRC 仍显示旧参数："
echo "  1) 完全退出 GNU Radio Companion 后重新打开"
echo "  2) 菜单：View → Reload Blocks（若有）"
