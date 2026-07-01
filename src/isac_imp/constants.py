from gnuradio import gr
import pmt

TAG_SOB = pmt.intern("tx_sob")
TAG_EOB = pmt.intern("tx_eob")
TAG_TIME = pmt.intern("tx_time")
TPP_DONT = gr.TPP_DONT  # type: ignore[attr-defined]


def make_tx_time_pmt(epoch_s: float):
    """构造 UHD tx_time tag 的 PMT 值：(整数秒, 小数秒)。"""
    sec = int(epoch_s)
    frac = epoch_s - sec
    return pmt.make_tuple(  # type: ignore[attr-defined]
        pmt.from_uint64(sec),  # type: ignore[attr-defined]
        pmt.from_double(frac),  # type: ignore[attr-defined]
    )
