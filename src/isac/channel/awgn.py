"""接收端 SNR (dB) 定标的 AWGN 信道块。"""

from typing import Optional, Union

import torch
from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import complex_normal, expand_to_rank


class AWGN(Block):
    r"""向输入叠加复 AWGN，噪声方差由接收端 SNR (dB) 定标。

    与 Sionna :class:`~sionna.phy.channel.AWGN` 的噪声生成方式相同，但第二参数为
    接收端 SNR (dB) 而非噪声方差 ``no``：

    ``no = mean(|x|²) / 10^(snr/10)``

    噪声每复维方差为 ``no``，实/虚部各 ``no/2``。
    ``snr`` 可为标量或可广播至 ``x`` 形状的张量。
    """

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

    def call(
        self,
        x: torch.Tensor,
        snr: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """对 ``x`` 叠加 AWGN；``snr`` 为接收端 SNR (dB)。"""
        sig_p = torch.mean(torch.abs(x) ** 2)
        if sig_p <= 0:
            raise ValueError("接收端信号功率须为正，无法按 snr_db 定标噪声。")

        if not isinstance(snr, torch.Tensor):
            snr_t = torch.tensor(snr, dtype=self.dtype, device=x.device)
        else:
            snr_t = snr.to(dtype=self.dtype, device=x.device)

        snr_linear = 10.0 ** (snr_t / 10.0)
        no = sig_p / snr_linear

        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, dtype=self.dtype, device=x.device)
        else:
            no = no.to(dtype=self.dtype, device=x.device)

        noise = complex_normal(
            x.shape,
            precision=self.precision,
            device=self.device,
            generator=self.torch_rng,
        )

        no = expand_to_rank(no, x.dim(), axis=-1)
        noise = noise * no.sqrt().to(dtype=self.cdtype)

        return x + noise
