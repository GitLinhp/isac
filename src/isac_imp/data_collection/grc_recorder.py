"""GNU Radio Companion 专用包装块（避免 grcc 生成非法 meta_static= 调用）。"""

from __future__ import annotations

from isac_imp.data_collection.range_profile_recorder import DevRangeProfileRecorder


class Dev1RangeProfileRecorderBlock(DevRangeProfileRecorder):
    """dev1 录制块；GRC 请 import 为 blk。"""

    def __init__(
        self,
        vlen: int = 4096,
        record_enable: bool = False,
        output_dir: str = "dataset/run_001",
        label: str = "",
        flush_every: int = 1200,
        fft_len: int = 2048,
        samp_rate: float = 122880000.0,
        R_max: float = 2500.0,
        range_bin_step: float = 0.61,
        zeropadding_fac: int = 2,
        transpose_len: int = 4,
        freq0: float = 6.03e9,
        freq1: float = 5.97e9,
        num_delay_samp0: int = 161,
        num_delay_samp1: int = 161,
    ) -> None:
        super().__init__(
            device_id="dev1",
            vlen=vlen,
            record_enable=record_enable,
            output_dir=output_dir,
            label=label,
            flush_every=flush_every,
            fft_len=fft_len,
            samp_rate=samp_rate,
            R_max=R_max,
            range_bin_step=range_bin_step,
            zeropadding_fac=zeropadding_fac,
            transpose_len=transpose_len,
            freq0=freq0,
            freq1=freq1,
            num_delay_samp0=num_delay_samp0,
            num_delay_samp1=num_delay_samp1,
        )


class Dev0RangeProfileRecorderBlock(DevRangeProfileRecorder):
    """dev0 录制块；GRC 请 import 为 blk。"""

    def __init__(
        self,
        vlen: int = 4096,
        record_enable: bool = False,
        output_dir: str = "dataset/run_001",
        label: str = "",
        flush_every: int = 1200,
        fft_len: int = 2048,
        samp_rate: float = 122880000.0,
        R_max: float = 2500.0,
        range_bin_step: float = 0.61,
        zeropadding_fac: int = 2,
        transpose_len: int = 4,
        freq0: float = 6.03e9,
        freq1: float = 5.97e9,
        num_delay_samp0: int = 161,
        num_delay_samp1: int = 161,
    ) -> None:
        super().__init__(
            device_id="dev0",
            vlen=vlen,
            record_enable=record_enable,
            output_dir=output_dir,
            label=label,
            flush_every=flush_every,
            fft_len=fft_len,
            samp_rate=samp_rate,
            R_max=R_max,
            range_bin_step=range_bin_step,
            zeropadding_fac=zeropadding_fac,
            transpose_len=transpose_len,
            freq0=freq0,
            freq1=freq1,
            num_delay_samp0=num_delay_samp0,
            num_delay_samp1=num_delay_samp1,
        )
