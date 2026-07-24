#!/usr/bin/env python3
"""Generate data_collection_cooperative_monostatic.grc (dual USRP + monostatic DSP)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).with_name("data_collection_cooperative_monostatic.grc")

HEADER = """options:
  parameters:
    author: ''
    catch_exceptions: 'True'
    category: '[GRC Hier Blocks]'
    cmake_opt: ''
    comment: ''
    copyright: ''
    description: Dual-USRP cooperative monostatic OFDM range profile collection; both
      devices TX+RX simultaneously (aligned with usrp_ofdm_echotimer_dd_seperate).
    gen_cmake: 'On'
    gen_linking: dynamic
    generate_options: qt_gui
    hier_block_src_path: '.:'
    id: data_collection_cooperative_monostatic
    max_nouts: '0'
    output_language: python
    placement: (0,0)
    qt_qss_theme: ''
    realtime_scheduling: ''
    run: 'True'
    run_command: '{python} -u {filename}'
    run_options: prompt
    sizing_mode: fixed
    thread_safe_setters: ''
    title: Cooperative monostatic sensing data collection
    window_size: (2000,2000)
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [8, 8]
    rotation: 0
    state: enabled

blocks:
"""

FOOTER = """
metadata:
  file_format: 1
  grc_version: 3.10.12.0
"""


def var(name: str, value: str, comment: str = "", coord: str = "[112, 8.0]") -> str:
    c = f'    comment: "{comment}"\n' if comment else "    comment: ''\n"
    return f"""- name: {name}
  id: variable
  parameters:
{c}    value: {value}
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: {coord}
    rotation: 0
    state: enabled
"""


def param(name: str, value: str, label: str, comment: str = "") -> str:
    return f"""- name: {name}
  id: parameter
  parameters:
    alias: ''
    comment: '{comment}'
    hide: none
    label: {label}
    short_id: ''
    type: str
    value: '{value}'
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [2384, 1092.0]
    rotation: 0
    state: enabled
"""


def range_var(
    name: str,
    value: str,
    label: str,
    gui_hint: str,
    coord: str,
    stop: str = "packet_len",
) -> str:
    return f"""- name: {name}
  id: variable_qtgui_range
  parameters:
    comment: ''
    gui_hint: {gui_hint}
    label: {label}
    min_len: '200'
    orient: QtCore.Qt.Horizontal
    rangeType: float
    start: '0'
    step: '1'
    stop: '{stop}'
    value: '{value}'
    widget: counter_slider
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: {coord}
    rotation: 0
    state: enabled
"""


def check_box_var(
    name: str,
    label: str,
    gui_hint: str,
    coord: str,
    value: str = "False",
    comment: str = "",
) -> str:
    c = f"    comment: {comment}\n" if comment else "    comment: ''\n"
    return f"""- name: {name}
  id: variable_qtgui_check_box
  parameters:
{c}    'false': 'False'
    gui_hint: {gui_hint}
    label: {label}
    'true': 'True'
    type: bool
    value: '{value}'
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: {coord}
    rotation: 0
    state: enabled
"""


def epy(
    name: str,
    module: str,
    cls: str,
    params: dict[str, str],
    coord: str,
    comment: str = "",
    io_cache: str = "",
) -> str:
    lines = [f"- name: {name}", "  id: epy_block", "  parameters:"]
    lines.append(
        f"    _source_code: 'from isac_imp.{module} import {cls}\n\n\n      blk = {cls}\n\n      '"
    )
    lines.append("    affinity: ''")
    lines.append("    alias: ''")
    if comment:
        lines.append(f"    comment: {comment}")
    for k, v in params.items():
        lines.append(f"    {k}: {v}")
    if "maxoutbuf" not in params:
        lines.append("    maxoutbuf: '0'")
    if "minoutbuf" not in params:
        lines.append("    minoutbuf: '0'")
    lines.append("  states:")
    if io_cache:
        lines.append(f"    _io_cache: {io_cache}")
    lines.append("    bus_sink: false")
    lines.append("    bus_source: false")
    lines.append("    bus_structure: null")
    lines.append(f"    coordinate: {coord}")
    lines.append("    rotation: 0")
    lines.append("    state: enabled")
    return "\n".join(lines)


IO_SIONNA = (
    '"(\'Sionna ResourceGrid TX\', \'SionnaResourceGridTxBlock\', '
    "[('fft_len', '2048'), ('transpose_len', '4'), ('subcarrier_spacing', '120000.0'), "
    "('cp_len', '512'), ('length_tag_key', \\\"'packet_len'\\\"), ('num_bits_per_symbol', '2'), "
    "('device', \\\"'cpu'\\\"), ('seed', '42')], [], [('0', 'complex', 2048)], "
    "'\\u65E0\\u8F93\\u5165\\uFF1B\\u8F93\\u51FA fftshift \\u9891\\u57DF OFDM "
    "\\u7B26\\u53F7\\u5411\\u91CF\\u6D41\\uFF08vlen=fft_len\\uFF09\\u3002', [])\""
)

IO_OFDM_RP = (
    '"(\'OFDM Range Profile\', \'OfdmRangeProfileBlock\', '
    "[('fft_len', '2048'), ('zeropadding_fac', '2'), ('transpose_len', '4')], "
    "[('0', 'complex', 2048), ('1', 'complex', 2048)], "
    "[('0', 'float', 4096), ('1', 'complex', 4096)], "
    "'\\u53CC\\u8F93\\u5165 TX/RX \\u9891\\u57DF\\u7B26\\u53F7 \\u2192 CPI dB "
    "\\u8DDD\\u79BB\\u8C31 + \\u53EF\\u9009 CPI \\u590D\\u6570\\u8DDD\\u79BB\\u8C31\\uFF08MUSIC\\uFF09\\u3002', [])\""
)

IO_RANGE_PLOT = (
    '"(\'Range Profile Plot\', \'RangeProfilePlotBlock\', '
    "[('vlen_in', '4096'), ('range_roi', '(0.0, 30.0)'), ('range_bin_step', '0.122')], "
    "[('0', 'float', 4096)], [], "
    "'\\u8DDD\\u79BB\\u8C31 dB \\u5411\\u91CF \\u2192 PyQtGraph \\u663E\\u793A\\uFF08ROI + autoscale Y\\uFF09\\u3002', "
    "['range_roi', 'range_bin_step'])\""
)

IO_RANGE_MUSIC = (
    '"(\'Range MUSIC\', \'RangeMusicBlock\', '
    "[('vlen_in', '4096'), ('range_bin_step', '0.122'), ('range_roi', '(0.0, 30.0)'), "
    "('num_sources', '1'), ('music_enable', 'False'), ('subarray_size', '16'), ('threshold', '0.1')], "
    "[('0', 'complex', 4096)], [], "
    "'\\u590D\\u6570 CPI \\u8DDD\\u79BB\\u8C31 \\u2192 1D MUSIC \\u8DDD\\u79BB\\u4F30\\u8BA1\\uFF08\\u65E0\\u6D41\\u8F93\\u51FA\\uFF09\\u3002', "
    "['music_enable', 'range_bin_step', 'range_roi'])\""
)


def io_limiter(dev: str) -> str:
    return (
        '"(\'Range Profile Record Limiter\', \'RangeProfileRecordLimiter\', '
        "[('vlen_in', '4096'), ('record_enable', 'False'), ('record_max_frames', '100'), "
        f"('record_output_dir_override', 'record_output_dir_{dev}'), "
        f"('file_sink_attr', 'blocks_file_sink_{dev}'), "
        f"('record_file_path_attr', 'record_file_path_{dev}')], "
        "[('0', 'complex', 4096)], [('0', 'complex', 4096)], "
        "'\\u590D\\u6570 CPI \\u8DDD\\u79BB\\u8C31 1:1 \\u900F\\u4F20\\uFF1Brecord_enable \\u65F6\\u8BA1\\u6570\\uFF0C\\u8FBE\\u5230\\u4E0A\\u9650\\u89E6\\u53D1 disable\\u3002', "
        "['record_enable', 'record_max_frames', 'record_output_dir_override', 'file_sink_attr', 'record_file_path_attr'])\""
    )


def block(id_: str, name: str, params: dict[str, str], coord: str, comment: str = "") -> str:
    lines = [f"- name: {name}", f"  id: {id_}", "  parameters:"]
    lines.append("    affinity: ''")
    lines.append("    alias: ''")
    if comment:
        lines.append(f"    comment: {comment}")
    for k, v in params.items():
        lines.append(f"    {k}: {v}")
    if "maxoutbuf" not in params:
        lines.append("    maxoutbuf: '0'")
    if "minoutbuf" not in params:
        lines.append("    minoutbuf: '0'")
    lines.append("  states:")
    lines.append("    bus_sink: false")
    lines.append("    bus_source: false")
    lines.append("    bus_structure: null")
    lines.append(f"    coordinate: {coord}")
    lines.append("    rotation: 0")
    lines.append("    state: enabled")
    return "\n".join(lines)


def main() -> None:
    parts: list[str] = [HEADER]

    # --- variables ---
    parts.append(var("fft_len", "'2048'"))
    parts.append(var("subcarrier_spacing", "120e3"))
    parts.append(var("zeropadding_fac", "'2'"))
    parts.append(var("transpose_len", "'4'"))
    parts.append(
        var(
            "samp_rate",
            "int(fft_len * subcarrier_spacing)",
            comment="Shared sample rate for both USRP devices and baseband chain",
        )
    )
    parts.append(var("n_carriers", "fft_len - 2"))
    parts.append(var("packet_len", "transpose_len * n_carriers // 4"))
    parts.append(var("length_tag_key", "'\"packet_len\"'"))
    parts.append(var("burst_len_samples", "transpose_len * (fft_len + fft_len//4)"))
    parts.append(var("min_out_buf_val", "packet_len*2"))
    parts.append(var("wait_to_start", "0.03"))
    parts.append(var("device", "'\"cpu\"'"))
    parts.append(var("factor", "0.008"))
    parts.append(var("range_roi", "(0.0, 30.0)"))
    parts.append(
        var(
            "range_bin_step",
            "3e8/(2*int(fft_len*subcarrier_spacing)*zeropadding_fac)",
        )
    )
    parts.append(var("R_max", "3e8/2/samp_rate*fft_len"))
    parts.append(var("freq0", "6.03e9", coord="[2224, 1080.0]"))
    parts.append(var("freq1", "5.97e9", coord="[2224, 1120.0]"))
    parts.append(var("record_enable", "False"))
    parts.append(var("record_max_frames", "'100'"))
    parts.append(
        var(
            "record_output_dir_dev0",
            'repo_data_dir("data", "experiment", "cooperative_monostatic", "dev0")',
        )
    )
    parts.append(
        var(
            "record_output_dir_dev1",
            'repo_data_dir("data", "experiment", "cooperative_monostatic", "dev1")',
        )
    )
    parts.append(var("record_file_path_dev0", '\'"/dev/null"\''))
    parts.append(var("record_file_path_dev1", '\'"/dev/null"\''))
    parts.append(var("record_output_index_dev0", "0 if record_enable else 1"))
    parts.append(var("record_output_index_dev1", "0 if record_enable else 1"))
    parts.append(
        check_box_var(
            "music_enable",
            "MUSIC Enable",
            "0,4,1,1",
            "[1328, 172.0]",
            comment="Enable 1D range MUSIC super-resolution on CPI complex range profile (dev0+dev1)",
        )
    )

    parts.append(
        param(
            "address0",
            "type=x4xx,serial=33ABFDE,mgmt_addr=192.168.1.101,addr=192.168.11.2,clock_source=external,time_source=external",
            "address0 (33ABFDE)",
            "USRP dev0",
        )
    )
    parts.append(
        param(
            "address1",
            "type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=external,time_source=external",
            "address1 (349B642)",
            "USRP dev1",
        )
    )

    parts.append(range_var("TX_gain0", "20", "TX Gain", "3,2,1,1", "[2504, 900.0]"))
    parts.append(range_var("RX_gain0", "20", "RX Gain", "3,3,1,1", "[2536, 900.0]"))
    parts.append(range_var("num_delay_samp0", "282", "Number of delayed samples", "3,0,1,1", "[2224, 900.0]"))
    parts.append(range_var("TX_gain1", "20", "TX Gain", "0,2,1,1", "[2536, 332.0]"))
    parts.append(range_var("RX_gain1", "20", "RX Gain", "0,3,1,1", "[2656, 332.0]"))
    parts.append(range_var("num_delay_samp1", "282", "Number of delayed samples", "0,0,1,1", "[2224, 332.0]"))

    parts.append(
        """- name: import_mics_test_record_flow_0
  id: import
  parameters:
    alias: ''
    comment: record limit + auto-stop hooks
    imports: |-
      from isac_imp.mics_test_record_flow import install_mics_test_record_flow
      from isac_imp.record_paths import repo_data_dir
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [1968, 16.0]
    rotation: 0
    state: enabled"""
    )

    parts.append(
        """- name: snippet_install_record_flow_0
  id: snippet
  parameters:
    alias: ''
    code: install_mics_test_record_flow(self)
    comment: Bind record limit handler after top block init
    priority: '10'
    section: main_after_init
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [1968, 96.0]
    rotation: 0
    state: enabled"""
    )

    # shared TX
    parts.append(
        epy(
            "sionna_resource_grid_tx",
            "sionna_resource_grid_tx",
            "SionnaResourceGridTxBlock",
            {
                "fft_len": "fft_len",
                "transpose_len": "transpose_len",
                "subcarrier_spacing": "subcarrier_spacing",
                "cp_len": "fft_len//4",
                "length_tag_key": "length_tag_key",
                "num_bits_per_symbol": "'2'",
                "device": "device",
                "seed": "'42'",
                "minoutbuf": "4*transpose_len",
            },
            "[728, 408.0]",
            io_cache=IO_SIONNA,
        )
    )
    parts.append(
        block(
            "fft_vxx",
            "fft_vxx_0",
            {
                "fft_size": "fft_len",
                "forward": "False",
                "window": "()",
                "shift": "True",
                "nthreads": "1",
                "type": "complex",
                "minoutbuf": "2*transpose_len",
            },
            "[600, 408.0]",
        )
    )
    parts.append(
        block(
            "digital_ofdm_cyclic_prefixer",
            "digital_ofdm_cyclic_prefixer_0",
            {
                "input_size": "fft_len",
                "cp_len": "fft_len//4",
                "rolloff": "'0'",
                "tagname": "length_tag_key",
                "minoutbuf": "int(2*transpose_len*(fft_len+fft_len/4))",
            },
            "[920, 408.0]",
        )
    )
    parts.append(
        block(
            "blocks_multiply_const_vxx",
            "blocks_multiply_const_vxx_0",
            {
                "const": "factor",
                "type": "complex",
                "vlen": "1",
                "minoutbuf": "int(2*transpose_len*(fft_len+fft_len/4))",
            },
            "[1120, 408.0]",
        )
    )

    def dev_chain(dev: str, cp_name: str, fft_name: str, freq_sink: str, echotimer: str, addr: str, freq: str, txg: str, rxg: str, ndelay: str, y: int) -> None:
        parts.append(
            block(
                "radar_usrp_echotimer_cc",
                echotimer,
                {
                    "samp_rate": "int(samp_rate)",
                    "center_freq": freq,
                    "num_delay_samps": f"int({ndelay})",
                    "args_tx": addr,
                    "channel_tx": "'0'",
                    "wire_tx": "''",
                    "clock_source_tx": "'external'",
                    "time_source_tx": "'external'",
                    "antenna_tx": "'TX/RX'",
                    "gain_tx": txg,
                    "timeout_tx": "'0.2'",
                    "wait_tx": "wait_to_start",
                    "lo_offset_tx": "'0'",
                    "args_rx": addr,
                    "channel_rx": "'0'",
                    "wire_rx": "''",
                    "clock_source_rx": "'external'",
                    "time_source_rx": "'external'",
                    "antenna_rx": "'RX1'",
                    "gain_rx": rxg,
                    "timeout_rx": "'0.2'",
                    "wait_rx": "wait_to_start",
                    "lo_offset_rx": "'0'",
                    "len_key": "'\"packet_len\"'",
                    "minoutbuf": "min_out_buf_val",
                },
                f"[1800, {y}]",
            )
        )
        parts.append(
            block(
                "radar_ofdm_cyclic_prefix_remover_cvc",
                cp_name,
                {
                    "fft_len": "fft_len",
                    "cp_len": "fft_len//4",
                    "len_key": "'\"packet_len\"'",
                    "minoutbuf": "2*transpose_len",
                },
                f"[1500, {y}]",
            )
        )
        parts.append(
            block(
                "fft_vxx",
                fft_name,
                {
                    "fft_size": "fft_len",
                    "forward": "True",
                    "window": "()",
                    "shift": "True",
                    "nthreads": "1",
                    "type": "complex",
                    "minoutbuf": "2*transpose_len",
                },
                f"[1300, {y}]",
            )
        )
        parts.append(
            block(
                "qtgui_freq_sink_x",
                freq_sink,
                {
                    "type": "complex",
                    "name": '""',
                    "srate": "samp_rate",
                    "fc": "'0'",
                    "bw": "samp_rate",
                    "fftsize": "fft_len",
                    "wintype": "window.WIN_BLACKMAN_hARRIS",
                    "norm_window": "'False'",
                    "freqhalf": "'True'",
                    "grid": "'False'",
                    "autoscale": "'False'",
                    "axislabels": "'True'",
                    "units": "dB",
                    "label": "Relative Gain",
                    "nconnections": "'1'",
                    "ctrlpanel": "'False'",
                    "showports": "'False'",
                    "update_time": "'0.10'",
                    "ymin": "'-140'",
                    "ymax": "'10'",
                },
                f"[1900, {y}]",
            )
        )
        parts.append(
            epy(
                f"ofdm_range_profile_{dev}",
                "ofdm_range_profile",
                "OfdmRangeProfileBlock",
                {
                    "fft_len": "fft_len",
                    "zeropadding_fac": "zeropadding_fac",
                    "transpose_len": "transpose_len",
                },
                f"[900, {y}]",
                io_cache=IO_OFDM_RP,
            )
        )
        parts.append(
            epy(
                f"range_profile_plot_{dev}",
                "range_profile_plot",
                "RangeProfilePlotBlock",
                {
                    "vlen_in": "fft_len*zeropadding_fac",
                    "range_roi": "range_roi",
                    "range_bin_step": "range_bin_step",
                },
                f"[700, {y}]",
                io_cache=IO_RANGE_PLOT,
            )
        )
        parts.append(
            epy(
                f"range_music_block_{dev}",
                "range_music_block",
                "RangeMusicBlock",
                {
                    "vlen_in": "fft_len*zeropadding_fac",
                    "range_bin_step": "range_bin_step",
                    "range_roi": "range_roi",
                    "num_sources": "'1'",
                    "music_enable": "music_enable",
                    "subarray_size": "'16'",
                    "threshold": "'0.1'",
                },
                f"[500, {y}]",
                io_cache=IO_RANGE_MUSIC,
            )
        )
        parts.append(
            epy(
                f"range_profile_record_limiter_{dev}",
                "range_profile_record_limiter",
                "RangeProfileRecordLimiter",
                {
                    "vlen_in": "fft_len*zeropadding_fac",
                    "record_enable": "record_enable",
                    "record_max_frames": "int(record_max_frames)",
                    "record_output_dir_override": f"record_output_dir_{dev}",
                    "file_sink_attr": f"'\"blocks_file_sink_{dev}\"'",
                    "record_file_path_attr": f"'\"record_file_path_{dev}\"'",
                },
                f"[300, {y}]",
                io_cache=io_limiter(dev),
            )
        )
        parts.append(
            block(
                "blocks_selector",
                f"blocks_selector_{dev}",
                {
                    "type": "complex",
                    "num_inputs": "1",
                    "output_index": f"record_output_index_{dev}",
                    "vlen": "fft_len*zeropadding_fac",
                },
                f"[100, {y}]",
            )
        )
        parts.append(
            block(
                "blocks_file_sink",
                f"blocks_file_sink_{dev}",
                {
                    "type": "complex",
                    "vlen": "fft_len*zeropadding_fac",
                    "file": f"record_file_path_{dev}",
                    "append": "False",
                    "unbuffered": "False",
                },
                f"[0, {y}]",
            )
        )
        parts.append(
            block(
                "blocks_null_sink",
                f"blocks_null_sink_rec_{dev}",
                {
                    "type": "complex",
                    "vlen": "fft_len*zeropadding_fac",
                },
                f"[0, {y + 40}]",
            )
        )

    dev_chain("dev0", "radar_ofdm_cyclic_prefix_remover_cvc_0_0", "fft_rx_dev0", "qtgui_freq_sink_x_dev0", "radar_usrp_echotimer_cc_0_0", "address0", "freq0", "TX_gain0", "RX_gain0", "num_delay_samp0", 900)
    dev_chain("dev1", "radar_ofdm_cyclic_prefix_remover_cvc_0", "fft_rx_dev1", "qtgui_freq_sink_x_dev1", "radar_usrp_echotimer_cc_0", "address1", "freq1", "TX_gain1", "RX_gain1", "num_delay_samp1", 400)

    parts.append(
        block(
            "qtgui_time_sink_x",
            "qtgui_time_sink_x_0",
            {
                "type": "complex",
                "name": '""',
                "srate": "samp_rate",
                "size": "fft_len + fft_len//4",
                "nconnections": "'1'",
                "update_time": "'0.10'",
                "ymin": "'-1'",
                "ymax": "'1'",
            },
            "[1784, 348.0]",
        )
    )

    connections = [
        ["sionna_resource_grid_tx", "0", "fft_vxx_0", "0"],
        ["fft_vxx_0", "0", "digital_ofdm_cyclic_prefixer_0", "0"],
        ["digital_ofdm_cyclic_prefixer_0", "0", "blocks_multiply_const_vxx_0", "0"],
        ["blocks_multiply_const_vxx_0", "0", "qtgui_time_sink_x_0", "0"],
        ["blocks_multiply_const_vxx_0", "0", "radar_usrp_echotimer_cc_0_0", "0"],
        ["blocks_multiply_const_vxx_0", "0", "radar_usrp_echotimer_cc_0", "0"],
        ["sionna_resource_grid_tx", "0", "ofdm_range_profile_dev0", "0"],
        ["sionna_resource_grid_tx", "0", "ofdm_range_profile_dev1", "0"],
    ]
    for dev, cp, fft, echotimer, freq_sink in [
        ("dev0", "radar_ofdm_cyclic_prefix_remover_cvc_0_0", "fft_rx_dev0", "radar_usrp_echotimer_cc_0_0", "qtgui_freq_sink_x_dev0"),
        ("dev1", "radar_ofdm_cyclic_prefix_remover_cvc_0", "fft_rx_dev1", "radar_usrp_echotimer_cc_0", "qtgui_freq_sink_x_dev1"),
    ]:
        connections += [
            [echotimer, "0", freq_sink, "0"],
            [echotimer, "0", cp, "0"],
            [cp, "0", fft, "0"],
            [fft, "0", f"ofdm_range_profile_{dev}", "1"],
            [f"ofdm_range_profile_{dev}", "0", f"range_profile_plot_{dev}", "0"],
            [f"ofdm_range_profile_{dev}", "1", f"range_music_block_{dev}", "0"],
            [f"ofdm_range_profile_{dev}", "1", f"range_profile_record_limiter_{dev}", "0"],
            [f"range_profile_record_limiter_{dev}", "0", f"blocks_selector_{dev}", "0"],
            [f"blocks_selector_{dev}", "0", f"blocks_file_sink_{dev}", "0"],
            [f"blocks_selector_{dev}", "1", f"blocks_null_sink_rec_{dev}", "0"],
        ]

    parts.append("\nconnections:")
    for src, sport, dst, dport in connections:
        parts.append(f"- [{src}, '{sport}', {dst}, '{dport}']")
    parts.append(FOOTER)

    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT} ({len(parts)} sections)")


if __name__ == "__main__":
    main()
