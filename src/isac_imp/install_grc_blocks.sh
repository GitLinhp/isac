#!/usr/bin/env bash
# 将 isac_imp/blocks 下全部 *.block.yml 安装到 GRC 本地块目录。
# 仅覆盖本仓库 isac_imp/blocks 模块；Python 实现仍由 isac_imp 包导入提供。
set -euo pipefail

ISAC_IMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAC_BLOCKS_DIR="${ISAC_IMP_DIR}/blocks"
STATE="${GRC_HIER_PATH:-${HOME}/.local/state/gnuradio}"

if [[ ! -d "${ISAC_BLOCKS_DIR}" ]]; then
  echo "错误: 未找到块目录 ${ISAC_BLOCKS_DIR}" >&2
  exit 1
fi

mkdir -p "$STATE"

mapfile -t YMLS < <(find "${ISAC_BLOCKS_DIR}" -type f -name '*.block.yml' | sort)
if [[ ${#YMLS[@]} -eq 0 ]]; then
  echo "错误: ${ISAC_BLOCKS_DIR} 下未找到任何 *.block.yml" >&2
  exit 1
fi

count=0
for yml in "${YMLS[@]}"; do
  base="$(basename "$yml")"
  cp -f "$yml" "${STATE}/${base}"
  echo "installed ${STATE}/${base}"
  count=$((count + 1))
done

echo ""
echo "已安装 ${count} 个 isac_imp/blocks 模块定义到 ${STATE}"
echo "若 GRC 仍显示旧参数："
echo "  1) 完全退出 GNU Radio Companion 后重新打开"
echo "  2) 菜单：View → Reload Blocks（若有）"
