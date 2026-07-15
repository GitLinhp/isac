#!/usr/bin/env python3
"""Generate style1_ofdm_sensing.grc from ofdm_sensing.grc + Style1 USRP patches.

Regenerate Python with ISAC env (grcc uses PATH python; torch/sionna required):
  cd /home/caict/Desktop/example/ofdm_sensing
  python3 gen_style1_ofdm_sensing.py
  PATH=/home/caict/radioconda/envs/ISAC/bin:$PATH grcc style1_ofdm_sensing.grc

Run: ./run_style1_ofdm_sensing.sh
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = Path("/home/caict/Desktop/gunradio_test/uhd_test/ofdm_sensing.grc")
PHY_TX_EPY = ROOT / "style1_ofdm_phy_tx_epy.py"
OUT_GRC = ROOT / "style1_ofdm_sensing.grc"

NOTE_BLOCK = """\
- name: note_style1_ofdm
  id: note
  parameters:
    alias: ''
    comment: ''
    note: 'Style1 burst OFDM sensing: Sionna TX (tx_sob/tx_time/tx_eob) + GR OFDM RX.
      samp_rate=30.72 MS/s. OTA: start with low tx_amp/TX_gain. Use ISAC conda env
      (/home/caict/radioconda/envs/ISAC) for grcc and run. USRP sync=pc_clock. No
      GPU: set sionna_device to cpu. Burst interval: tx_repeat_period_ms (default 50
      ms). Success: QPSK clusters in constellation sink.'
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [8, 80]
    rotation: 0
    state: true
"""


def replace_epy_block(text: str, block_name: str, source: str, extra_params: str) -> str:
    epy_yaml = json.dumps(source)
    marker = f"- name: {block_name}\n  id: epy_block\n  parameters:\n"
    start = text.index(marker) + len(marker)
    states_marker = text.index("\n  states:", start)
    new_block = f"    _source_code: {epy_yaml}\n{extra_params}\n"
    return text[:start] + new_block + text[states_marker:]


def patch_usrp(text: str) -> str:
    text = text.replace("    sync: none\n", "    sync: pc_clock\n")
    text = text.replace(": TX/RX", ": '\"TX/RX\"'")
    text = text.replace("    gui_hint: ''\n    label: rx_gain", "    gui_hint: 0,1,1,1\n    label: rx_gain")
    text = text.replace("    gui_hint: ''\n    label: tx_gain", "    gui_hint: 0,0,1,1\n    label: tx_gain")
    text = re.sub(
        r"    minoutbuf: max\(65536, int\(\(2 \+ 1 \+ ofdm_syms_per_tag\) \* \(fft_len \+ cp_len\) \* 2\)\)",
        "    minoutbuf: '0'",
        text,
    )
    return text


def patch_metadata(text: str) -> str:
    text = text.replace("    id: ofdm_loopback_sionna", "    id: style1_ofdm_sensing")
    text = text.replace(
        "    description: Sionna PHY TX + GNU Radio native RX (hybrid loopback)",
        "    description: Style1 burst OFDM TX (Sionna) + GR OFDM RX sensing (X410 OTA)",
    )
    text = text.replace(
        "    title: OFDM Loopback (Sionna TX + GR RX)",
        "    title: Style1 Burst OFDM Sensing (X410)",
    )
    return text


def main() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    text = patch_metadata(text)
    text = patch_usrp(text)

    phy_tx_source = PHY_TX_EPY.read_text(encoding="utf-8")
    extra_params = """    affinity: ''
    alias: ''
    comment: Style1 Sionna PHY TX (tx_sob/tx_time/tx_eob)
    device: sionna_device
    fft_len: fft_len
    maxoutbuf: '0'
    minoutbuf: max(65536, int((2 + 1 + ofdm_syms_per_tag) * (fft_len + cp_len) * 2))
    n_carriers: n_carriers
    ofdm_syms_per_tag: ofdm_syms_per_tag
    repeat_period_ms: tx_repeat_period_ms
    subcarrier_spacing: subcarrier_spacing
    time_lead_s: '0.05'
    tx_amp: tx_amp"""
    text = replace_epy_block(text, "epy_sionna_phy_tx", phy_tx_source, extra_params)

    if "- name: note_style1_ofdm" not in text:
        text = text.replace("\nblocks:\n", f"\nblocks:\n{NOTE_BLOCK}", 1)

    OUT_GRC.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_GRC}")


if __name__ == "__main__":
    main()
