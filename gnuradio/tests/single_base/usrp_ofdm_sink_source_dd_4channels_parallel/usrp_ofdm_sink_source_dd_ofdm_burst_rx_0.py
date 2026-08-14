from isac_imp import ofdm_burst_rx as _obr


class OfdmBurstRx4Block(_obr.OfdmBurstRxBlock):
    """GRC epy 探测用：强制 4 输入。父类勿直接 import，否则 extract 会误选 1 口。"""

    def __init__(
        self,
        fft_len=2048,
        cp_len=512,
        num_symbols=4,
        zeropadding_fac=4,
        num_delay_samp=0,
        device="cuda",
        length_tag_key="packet_len",
        log_interval_s=1.0,
        range_roi=(0.0, 30.0),
        range_bin_step=0.305,
        music_enable=True,
        num_sources=1,
        subarray_size=16,
        threshold=0.1,
        plot_title="Range Profile",
        num_channels=4,
        plot_title_prefix="Range Profile",
    ):
        super().__init__(
            fft_len=fft_len,
            cp_len=cp_len,
            num_symbols=num_symbols,
            zeropadding_fac=zeropadding_fac,
            num_delay_samp=num_delay_samp,
            device=device,
            length_tag_key=length_tag_key,
            log_interval_s=log_interval_s,
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            music_enable=music_enable,
            num_sources=num_sources,
            subarray_size=subarray_size,
            threshold=threshold,
            plot_title=plot_title,
            num_channels=4,
            plot_title_prefix=plot_title_prefix,
        )


blk = OfdmBurstRx4Block
