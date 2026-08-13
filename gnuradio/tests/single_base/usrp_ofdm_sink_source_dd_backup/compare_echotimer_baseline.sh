#!/usr/bin/env bash
# USRP Sink/Source 替代 echotimer 验证清单（参照 reference/gr-radar）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SRC="${ROOT}/src"
REF="${ROOT}/reference"
SINK_SOURCE_DIR="${ROOT}/gnuradio/tests/single_base/usrp_ofdm_sink_source_dd"
ECHOTIMER_DIR="${ROOT}/gnuradio/tests/single_base/usrp_ofdm_echotimer_dd"

echo "=== ISAC：USRP Sink/Source 替代 Echotimer ==="
echo "参照源码: ${REF}/gr-radar/lib/usrp_echotimer_cc_impl.cc"
echo
echo "1. 单元测试（无需硬件）"
PYTHONPATH="${SRC}" python3 -m pytest "${ROOT}/tests/test_burst_iq_tag_align.py" -q

echo
echo "2. Sink/Source 流图（idle_ms=10，便于观察 CPI 率）"
echo "   cd ${SINK_SOURCE_DIR}"
echo "   PYTHONPATH=${SRC} python3 usrp_ofdm_sink_source_dd.py"
echo "   终端: [scheduled_rx] CPI/s≈99，无 overflow"
echo
echo "3. Echotimer 等价流图（idle_ms=0，对齐原 echotimer 满速率）"
echo "   cd ${ECHOTIMER_DIR}"
echo "   PYTHONPATH=${SRC} python3 usrp_ofdm_echotimer_dd.py"
echo "   终端: [echotimer_rx] CPI/s 应接近 Sionna TX 重放速率"
echo
echo "4. 判定标准"
echo "   - packet_len + rx_time tag 正常（echotimer_rx_compensator）"
echo "   - num_delay_samp 滑块可调，距离谱峰值随校准变化"
echo "   - Range Profile 帧号递增，自环回 0 m 附近有峰值"
