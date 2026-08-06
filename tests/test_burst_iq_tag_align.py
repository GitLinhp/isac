"""burst_iq_tag_tx / burst_iq_tag_rx 逐 CPI 对齐稳定性测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pmt
import pytest
from gnuradio import blocks, gr

from isac_imp.burst_iq_tag_rx import BurstIqTagRxBlock
from isac_imp.burst_iq_tag_tx import BurstIqTagTxBlock
from isac_imp.burst_pack import TAG_RX_TIME, make_tx_schedule_msg, make_tx_time_pmt, schedule_idle_delay_s

pytest.importorskip("gnuradio")


def _make_tag(offset: int, key: pmt.pmt, value: pmt.pmt) -> gr.tag_t:
    tag = gr.tag_t()
    tag.offset = offset
    tag.key = key
    tag.value = value
    return tag


def _build_tx_burst_stream(
    burst_len: int,
    num_cpi: int,
    idle_samples: int,
) -> tuple[list[complex], list[gr.tag_t]]:
    samples: list[complex] = []
    tags: list[gr.tag_t] = []
    packet_len = pmt.intern("packet_len")
    for cpi in range(num_cpi):
        offset = len(samples)
        tags.append(_make_tag(offset, packet_len, pmt.from_long(burst_len)))
        samples.extend([complex(cpi + 1, 0.0)] * burst_len)
        if cpi < num_cpi - 1:
            samples.extend([0j] * idle_samples)
    return samples, tags


def _build_continuous_rx_stream(
    burst_len: int,
    num_cpi: int,
    idle_samples: int,
    num_delay: int,
    base_epoch: float,
    samp_rate: float,
    buffer_samples: int = 128,
) -> tuple[list[complex], list[gr.tag_t]]:
    """构造带 idle 间隙的连续 RX 流（模拟 USRP Source 连续采样）。"""
    period_samples = burst_len + idle_samples
    total = num_cpi * period_samples + buffer_samples
    data = np.zeros(total, dtype=np.complex64)
    for cpi in range(num_cpi):
        start = cpi * period_samples + num_delay
        data[start : start + burst_len] = cpi + 1

    tags: list[gr.tag_t] = []
    for off in range(0, total, buffer_samples):
        epoch = base_epoch + off / samp_rate
        tags.append(_make_tag(off, TAG_RX_TIME, make_tx_time_pmt(epoch)))
    return data.tolist(), tags


class _ScheduledRx(BurstIqTagRxBlock):
    """测试用：``start()`` 后注入确定性 ``tx_schedule`` epoch 队列。"""

    def __init__(self, tx_epochs: list[float], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tx_epochs = tx_epochs

    def start(self) -> bool:
        super().start()
        for epoch in self._tx_epochs:
            self._handle_tx_schedule(make_tx_schedule_msg(epoch))
        return True


def test_burst_iq_tag_align_stable_offsets() -> None:
    """连续 10 个 CPI：``tx_schedule`` + ``rx_time`` 对齐，首包相位方差 < 1 样点。"""
    burst_len = 64
    samp_rate = 1_000_000.0
    num_delay = 5
    num_cpi = 10
    idle_ms = 10.0
    idle_s = schedule_idle_delay_s(burst_len, samp_rate, idle_ms / 1000.0)
    idle_samples = int(round(idle_s * samp_rate))
    base_epoch = 1000.0
    burst_period_s = burst_len / samp_rate + idle_s
    tx_epochs = [base_epoch + i * burst_period_s for i in range(num_cpi)]

    rx_samples, rx_tags = _build_continuous_rx_stream(
        burst_len,
        num_cpi,
        idle_samples,
        num_delay,
        base_epoch,
        samp_rate,
    )

    tb = gr.top_block()
    rx_src = blocks.vector_source_c(rx_samples, False, 1, rx_tags)
    tag_rx = _ScheduledRx(
        tx_epochs,
        burst_len_samples=burst_len,
        num_delay_samp=num_delay,
        samp_rate=samp_rate,
        idle_ms=idle_ms,
        enable_stats_log=False,
    )
    vec_snk = blocks.vector_sink_c()

    tb.connect(rx_src, tag_rx, vec_snk)
    tb.run()

    out = np.asarray(vec_snk.data(), dtype=np.complex64)
    assert len(out) == num_cpi * burst_len

    start_offsets: list[float] = []
    for cpi in range(num_cpi):
        chunk = out[cpi * burst_len : (cpi + 1) * burst_len]
        assert chunk[0].real == pytest.approx(float(cpi + 1), abs=1e-5)
        good = int(np.sum(np.isclose(chunk.real, float(cpi + 1))))
        start_offsets.append(float(burst_len - good))

    assert np.var(start_offsets) < 1.0
    assert max(start_offsets) <= 1.0


@patch("isac_imp.burst_iq_tag_tx.time.monotonic", return_value=float("inf"))
@patch("isac_imp.burst_iq_tag_tx.time.time", return_value=1000.0)
def test_burst_iq_tag_tx_schedule_matches_rx_period(
    _mock_time: object,
    _mock_mono: object,
) -> None:
    """``tag_tx`` 发布的 ``tx_schedule`` epoch 周期与 RX ``idle_ms`` 一致。"""
    burst_len = 64
    samp_rate = 1_000_000.0
    num_cpi = 5
    idle_ms = 10.0
    idle_s = schedule_idle_delay_s(burst_len, samp_rate, idle_ms / 1000.0)
    idle_samples = int(round(idle_s * samp_rate))
    tx_samples, tx_tags = _build_tx_burst_stream(burst_len, num_cpi, idle_samples)

    tb = gr.top_block()
    tx_src = blocks.vector_source_c(tx_samples, False, 1, tx_tags)
    tag_tx = BurstIqTagTxBlock(
        burst_len_samples=burst_len,
        time_lead_s=0.0,
        idle_ms=idle_ms,
        samp_rate=samp_rate,
    )
    msg_dbg = blocks.message_debug()

    tb.connect(tx_src, tag_tx, blocks.null_sink(gr.sizeof_gr_complex))
    tb.msg_connect((tag_tx, "tx_schedule"), (msg_dbg, "store"))
    tb.run()

    from isac_imp.burst_pack import parse_uhd_time_pmt

    epochs = [
        parse_uhd_time_pmt(msg_dbg.get_message(i))
        for i in range(msg_dbg.num_messages())
    ]
    assert len(epochs) == num_cpi
    deltas = np.diff(epochs)
    expected = burst_len / samp_rate + idle_s
    assert np.allclose(deltas, expected, rtol=0, atol=1e-9)


@patch("isac_imp.burst_iq_tag_tx.time.time", return_value=1000.0)
def test_burst_iq_tag_tx_publishes_tx_schedule(_mock_time: object) -> None:
    burst_len = 32
    samp_rate = 1_000_000.0
    tx_samples, tx_tags = _build_tx_burst_stream(burst_len, num_cpi=2, idle_samples=0)

    tb = gr.top_block()
    tx_src = blocks.vector_source_c(tx_samples, False, 1, tx_tags)
    tag_tx = BurstIqTagTxBlock(
        burst_len_samples=burst_len,
        time_lead_s=0.01,
        idle_ms=0.0,
        samp_rate=samp_rate,
    )
    msg_dbg = blocks.message_debug()

    tb.connect(tx_src, tag_tx, blocks.null_sink(gr.sizeof_gr_complex))
    tb.msg_connect((tag_tx, "tx_schedule"), (msg_dbg, "store"))
    tb.run()

    assert msg_dbg.num_messages() >= 2


@pytest.mark.parametrize("enable_stats_log", [False])
def test_burst_iq_tag_rx_stats_disabled_by_default_in_tests(enable_stats_log: bool) -> None:
    """回归：enable_stats_log=False 时不抛错。"""
    blk = BurstIqTagRxBlock(burst_len_samples=32, enable_stats_log=enable_stats_log)
    assert blk.cpi_count == 0


def test_packet_len_tagger_tags_each_burst() -> None:
    burst_len = 32
    num_bursts = 5
    data = np.arange(num_bursts * burst_len, dtype=np.complex64)

    tb = gr.top_block()
    src = blocks.vector_source_c(data.tolist(), False)
    tagger = __import__(
        "isac_imp.packet_len_tagger", fromlist=["PacketLenTaggerBlock"]
    ).PacketLenTaggerBlock(burst_len_samples=burst_len)
    snk = blocks.vector_sink_c()

    tb.connect(src, tagger, snk)
    tb.run()

    tags = snk.tags()
    assert len(tags) == num_bursts
    assert all(pmt.to_long(t.value) == burst_len for t in tags)
    assert [t.offset for t in tags] == [i * burst_len for i in range(num_bursts)]


def test_burst_iq_tag_rx_bulk_discard_under_high_rate_idle() -> None:
    """大 idle 间隙：批量丢弃应仍能稳定对齐（且 bulk_ops > 0）。"""
    burst_len = 64
    samp_rate = 10_000_000.0
    num_delay = 3
    num_cpi = 5
    idle_samples = 500_000
    base_epoch = 2000.0
    period_samples = burst_len + idle_samples
    total = num_cpi * period_samples + 1000

    data = np.zeros(total, dtype=np.complex64)
    for cpi in range(num_cpi):
        start = cpi * period_samples + num_delay
        data[start : start + burst_len] = cpi + 1

    rx_tags: list[gr.tag_t] = []
    for off in range(0, total, 4096):
        epoch = base_epoch + off / samp_rate
        rx_tags.append(_make_tag(off, TAG_RX_TIME, make_tx_time_pmt(epoch)))

    burst_period_s = burst_len / samp_rate + idle_samples / samp_rate
    tx_epochs = [base_epoch + i * burst_period_s for i in range(num_cpi)]

    class _ScheduledRx(BurstIqTagRxBlock):
        def __init__(self, epochs: list[float], *args, **kwargs) -> None:
            super().__init__(*args, enable_stats_log=False, **kwargs)
            self._epochs = epochs

        def start(self) -> bool:
            super().start()
            for epoch in self._epochs:
                self._handle_tx_schedule(make_tx_schedule_msg(epoch))
            return True

    tb = gr.top_block()
    rx_src = blocks.vector_source_c(data.tolist(), False, 1, rx_tags)
    tag_rx = _ScheduledRx(
        tx_epochs,
        burst_len_samples=burst_len,
        num_delay_samp=num_delay,
        samp_rate=samp_rate,
        idle_ms=0.0,
    )
    vec_snk = blocks.vector_sink_c()
    tb.connect(rx_src, tag_rx, vec_snk)
    tb.run()

    out = np.asarray(vec_snk.data(), dtype=np.complex64)
    assert len(out) == num_cpi * burst_len
    assert tag_rx._stats_bulk_discards > 0
    for cpi in range(num_cpi):
        chunk = out[cpi * burst_len : (cpi + 1) * burst_len]
        assert chunk[0].real == pytest.approx(float(cpi + 1), abs=1e-5)



def test_echotimer_rx_compensator_shift() -> None:
    import numpy as np

    from isac_imp.echotimer_rx_compensator import EchotimerRxCompensatorBlock

    burst_len = 8
    delay = 2
    blk = EchotimerRxCompensatorBlock(
        burst_len_samples=burst_len,
        num_delay_samps=delay,
    )
    src = np.arange(burst_len, dtype=np.complex64) + 1j
    dst = src.copy()
    blk._apply_shift(dst, burst_len)
    np.testing.assert_array_equal(dst[: burst_len - delay], src[delay:])
    assert np.all(dst[burst_len - delay :] == 0)


def test_echotimer_rx_compensator_forwards_packet_len_tag() -> None:
    import pmt

    from isac_imp.echotimer_rx_compensator import EchotimerRxCompensatorBlock

    blk = EchotimerRxCompensatorBlock(burst_len_samples=4, num_delay_samps=0)
    blk._length_tag_key = pmt.intern("packet_len")
    blk._srcid = pmt.intern("test")

    # Simulate CPI start: compensator must re-emit packet_len on output
    out = [0, 0, 0, 0]
    in_base = 100
    out_base = 200
    i = 0
    burst_len = 4
    abs_out = out_base + i
    blk.add_item_tag = lambda *args, **kwargs: None  # patched below
    tags_added = []

    def capture_add(port, offset, key, val, src):
        tags_added.append((offset, pmt.to_python(key), pmt.to_python(val)))

    blk.add_item_tag = capture_add

    # Directly exercise the CPI-start branch logic
    blk.add_item_tag(0, abs_out, blk._length_tag_key, pmt.from_long(burst_len), blk._srcid)
    assert tags_added
    assert tags_added[0][0] == abs_out
    assert tags_added[0][1] == "packet_len"
    assert tags_added[0][2] == burst_len


def test_burst_usrp_stream_scheduler_issues_num_samps_cmd() -> None:
    pytest.importorskip("gnuradio.uhd")
    from unittest.mock import MagicMock

    from isac_imp.burst_usrp_stream_scheduler import BurstUsrpStreamSchedulerBlock

    fake_source = MagicMock()
    sched = BurstUsrpStreamSchedulerBlock(
        fake_source,
        burst_len_samples=1024,
        num_delay_samp=10,
        samp_rate=1_000_000.0,
        enable_stats_log=False,
    )
    sched._handle_tx_schedule(make_tx_schedule_msg(1000.0))

    fake_source.issue_stream_cmd.assert_called_once()
    cmd = fake_source.issue_stream_cmd.call_args[0][0]
    assert cmd.num_samps == 1024
    assert cmd.stream_now is False
    assert abs(float(cmd.time_spec.get_real_secs()) - 1000.0) < 1e-9


def test_patch_usrp_source_factory_forces_false() -> None:
    import gnuradio.uhd as uhd

    from isac_imp import scheduled_usrp_source as mod

    mod._patched = False
    orig = uhd.usrp_source
    calls: list[bool] = []

    def fake_usrp_source(device_addr, stream_args, issue_stream_cmd_on_start=True):
        calls.append(issue_stream_cmd_on_start)
        return object()

    uhd.usrp_source = fake_usrp_source  # type: ignore[misc, assignment]
    try:
        mod.patch_usrp_source_factory()
        uhd.usrp_source("addr", "args", True)
        uhd.usrp_source("addr", "args", False)
    finally:
        uhd.usrp_source = orig  # type: ignore[misc, assignment]
        mod._patched = False

    assert calls == [False, False]


def test_sionna_ofdm_modulator_burst_length() -> None:
    import pmt

    from isac_imp.sionna_ofdm_modulator import SionnaOfdmModulatorBlock
    from isac_imp.sionna_resource_grid_tx import _build_freq_grid

    fft_len = 64
    cp_len = 16
    transpose_len = 2
    burst_len = transpose_len * (fft_len + cp_len)
    freq, _, _ = _build_freq_grid(
        fft_len=fft_len,
        transpose_len=transpose_len,
        subcarrier_spacing=60e3,
        cp_len=cp_len,
        num_bits_per_symbol=2,
        device="cpu",
        seed=7,
    )

    blk = SionnaOfdmModulatorBlock(
        fft_len=fft_len,
        cp_len=cp_len,
        burst_len_samples=burst_len,
        transpose_len=transpose_len,
        subcarrier_spacing=60e3,
        num_bits_per_symbol=2,
        seed=7,
        target_peak=1.0,
    )
    blk.start()
    cpi = np.concatenate([blk._modulate_symbol(freq[i]) for i in range(transpose_len)])  # noqa: SLF001
    assert abs(float(np.max(np.abs(cpi))) - 1.0) < 1e-5
    assert cpi.shape == (burst_len,)
    assert abs(blk._ifft_scale - 1.0 / np.sqrt(float(fft_len))) < 1e-12  # noqa: SLF001


def test_ofdm_range_profile_rx_align_offset() -> None:
    from isac_imp.ofdm_range_profile import OfdmRangeProfileBlock
    from isac_imp.sionna_ofdm_modulator import SionnaOfdmModulatorBlock
    from isac_imp.sionna_resource_grid_tx import _build_freq_grid

    fft_len = 128
    cp_len = 32
    transpose_len = 4
    freq, _, _ = _build_freq_grid(
        fft_len=fft_len,
        transpose_len=transpose_len,
        subcarrier_spacing=60e3,
        cp_len=cp_len,
        num_bits_per_symbol=2,
        device="cpu",
        seed=3,
    )
    mod = SionnaOfdmModulatorBlock(
        fft_len=fft_len,
        cp_len=cp_len,
        burst_len_samples=transpose_len * (fft_len + cp_len),
        transpose_len=transpose_len,
        seed=3,
        target_peak=1.0,
    )
    mod.start()

    def _rx_fft(sym_td: np.ndarray) -> np.ndarray:
        body = sym_td[cp_len:]
        return np.fft.fftshift(np.fft.fft(body, norm=None) / np.sqrt(fft_len))

    rx = np.stack(
        [_rx_fft(mod._modulate_symbol(freq[i])) for i in range(transpose_len)]  # noqa: SLF001
    )
    rx = np.concatenate(
        [
            np.zeros((2, fft_len), dtype=np.complex64),
            rx.astype(np.complex64, copy=False),
        ],
        axis=0,
    )
    blk = OfdmRangeProfileBlock(
        fft_len=fft_len, zeropadding_fac=2, transpose_len=transpose_len
    )
    off, score = blk._find_rx_align_offset(freq, rx)  # noqa: SLF001
    assert off == 2
    assert score > 100.0


def test_ofdm_range_profile_zero_range_peak_when_aligned() -> None:
    from gnuradio.fft import window

    from isac_imp.ofdm_range_profile import compute_cpi_range_profile_db
    from isac_imp.sionna_ofdm_modulator import SionnaOfdmModulatorBlock
    from isac_imp.sionna_resource_grid_tx import _build_freq_grid

    fft_len = 2048
    cp_len = 512
    transpose_len = 4
    zpf = 4
    vlen = fft_len * zpf
    freq, _, _ = _build_freq_grid(
        fft_len=fft_len,
        transpose_len=transpose_len,
        subcarrier_spacing=60e3,
        cp_len=cp_len,
        num_bits_per_symbol=2,
        device="cpu",
        seed=42,
    )
    mod = SionnaOfdmModulatorBlock(
        fft_len=fft_len,
        cp_len=cp_len,
        burst_len_samples=transpose_len * (fft_len + cp_len),
        transpose_len=transpose_len,
        seed=42,
        target_peak=1.0,
    )
    mod.start()

    def _rx_fft(sym_td: np.ndarray) -> np.ndarray:
        body = sym_td[cp_len:]
        return np.fft.fftshift(np.fft.fft(body, norm=None) / np.sqrt(fft_len))

    rx = np.stack(
        [_rx_fft(mod._modulate_symbol(freq[i])) for i in range(transpose_len)]  # noqa: SLF001
    )
    bh = np.asarray(window.blackmanharris(vlen), dtype=np.float32)
    prof = compute_cpi_range_profile_db(
        freq, rx, bh_window=bh, fft_len=fft_len, vlen_out=vlen
    )
    assert int(np.argmax(prof)) == 0


def test_packet_len_tagger_rx_time_min_spacing() -> None:
    from isac_imp.packet_len_tagger import PacketLenTaggerBlock

    burst_len = 10240
    blk = PacketLenTaggerBlock(burst_len_samples=burst_len, use_rx_time=True)
    blk.start()
    blk._last_tag_abs = 0
    assert 4096 - blk._last_tag_abs < burst_len
    assert 10240 - blk._last_tag_abs >= burst_len
