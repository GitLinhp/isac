"""OfdmDivideCpiBlock 与 compute_symbol_divide_padded 单元测试。"""

from __future__ import annotations

import numpy as np

from isac_imp.ofdm_range_profile import (
    OfdmDivideCpiBlock,
    compute_symbol_divide_padded,
)
from isac_imp.range_profile_record_limiter import (
    DivideCpiRecordLimiter,
    allocate_next_record_path,
)


def test_compute_symbol_divide_padded_shape_and_values() -> None:
    fft_len = 8
    vlen_out = 16
    tx = np.arange(fft_len, dtype=np.complex64) + 1j
    rx = np.ones(fft_len, dtype=np.complex64)

    h = compute_symbol_divide_padded(tx, rx, fft_len=fft_len, vlen_out=vlen_out)

    assert h.shape == (vlen_out,)
    np.testing.assert_allclose(h[:fft_len], tx / rx)
    np.testing.assert_allclose(h[fft_len:], 0.0)


def test_ofdm_divide_cpi_block_default_io_size() -> None:
    blk = OfdmDivideCpiBlock()
    assert blk.input_signature().sizeof_stream_item(0) == 2048 * 8
    assert blk.output_signature().sizeof_stream_item(0) == 32768 * 8


def test_cpi_flatten_order_matches_sic_divide_recorder() -> None:
    fft_len = 4
    zeropadding_fac = 2
    transpose_len = 3
    vlen_out = fft_len * zeropadding_fac

    symbols = []
    for k in range(transpose_len):
        tx = np.full(fft_len, k + 1, dtype=np.complex64)
        rx = np.full(fft_len, 1, dtype=np.complex64)
        symbols.append(
            compute_symbol_divide_padded(
                tx, rx, fft_len=fft_len, vlen_out=vlen_out
            )
        )

    flattened = np.stack(symbols, axis=0).ravel()
    expected = np.concatenate(symbols)

    np.testing.assert_array_equal(flattened, expected)
    assert flattened.size == transpose_len * vlen_out


def test_divide_cpi_record_limiter_default_vlen() -> None:
    limiter = DivideCpiRecordLimiter()
    assert limiter.input_signature().sizeof_stream_item(0) == 32768 * 8


def test_allocate_next_record_path_divide_profiles(tmp_path) -> None:
    out_dir = tmp_path / "dev0"
    path = allocate_next_record_path(str(out_dir), base_name="divide_profiles")
    assert path == str(out_dir / "divide_profiles_001")
