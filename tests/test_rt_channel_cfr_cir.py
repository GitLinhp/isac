"""RTChannel get_cir / get_cfr 与 cir/cfr 注入属性测试。"""

from pathlib import Path

import pytest
import sionna.phy.config
import torch

from isac import PROJECT_ROOT
from isac.system import System

_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@pytest.fixture
def rt_channel():
    sionna.phy.config.device = "cpu"
    system = System(_CONFIG, device="cpu")
    channel = system.components.channel
    rg = system.components.rg
    sym_dur = 1 / rg.ofdm_symbol_duration
    return channel, rg, sym_dur


def test_live_get_cfr_and_h_freq_shapes(rt_channel) -> None:
    channel, rg, sym_dur = rt_channel
    h = channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    assert isinstance(h, torch.Tensor)
    assert channel.h_freq.shape[-2:] == (rg.num_ofdm_symbols, rg.fft_size)


def test_injected_cfr_used_by_get_cfr_and_h_freq(rt_channel) -> None:
    channel, rg, sym_dur = rt_channel
    h_live = channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    h_inj = h_live.clone()
    channel.cfr = h_inj
    assert channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    ) is h_inj
    assert channel.h_freq is h_inj


def test_clear_injected_cfr_restores_live(rt_channel) -> None:
    channel, rg, sym_dur = rt_channel
    h_live = channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    channel.cfr = h_live.clone()
    channel.cfr = None
    h_again = channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    assert h_again.shape == h_live.shape


def test_injected_cfr_rejects_non_torch_out_type(rt_channel) -> None:
    channel, rg, sym_dur = rt_channel
    channel.cfr = channel.get_cfr(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    with pytest.raises(ValueError, match="out_type='torch'"):
        channel.get_cfr(
            num_time_steps=rg.num_ofdm_symbols,
            sampling_frequency=sym_dur,
            out_type="numpy",
        )


def test_injected_cir_returned_by_get_cir(rt_channel) -> None:
    channel, rg, sym_dur = rt_channel
    a, tau = channel.get_cir(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    channel.cir = (a, tau)
    a2, tau2 = channel.get_cir(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=sym_dur,
    )
    assert a2 is a
    assert tau2 is tau
