#!/bin/bash
# Style1 burst OFDM sensing — run with ISAC environment (torch/sionna/gnuradio)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAC_BIN="/home/caict/radioconda/envs/ISAC/bin"
PYTHON="${ISAC_BIN}/python"
SCRIPT="${DIR}/style1_ofdm_sensing.py"

export PATH="${ISAC_BIN}:${PATH}"

if getcap "$PYTHON" 2>/dev/null | grep -q cap_sys_nice; then
    exec chrt -f 50 "$PYTHON" -u "$SCRIPT" "$@"
fi

exec "$PYTHON" -u "$SCRIPT" "$@"
