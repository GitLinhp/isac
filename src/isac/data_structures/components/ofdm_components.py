"""
OFDM / 物理层相关组件构建（与 ``ofdm_params`` 对应）
"""

from dataclasses import dataclass

import numpy as np
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator,
)

from ..params import SystemParams


@dataclass
class OFDMComponents:
    """OFDM组件"""

    binary_source: BinarySource
    mapper: Mapper
    demapper: Demapper
    rg: ResourceGrid
    rg_mapper: ResourceGridMapper
    rg_demapper: ResourceGridDemapper
    modulator: OFDMModulator
    demodulator: OFDMDemodulator

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str,
    ) -> "OFDMComponents":
        """根据 ``SystemParams.ofdm`` 与 ``SystemParams.qam`` 构建 OFDM 相关组件。"""
        rx_tx_association = np.array([[1]])
        sm = StreamManagement(rx_tx_association, 1)

        binary_source = BinarySource(device=device)
        mapper = Mapper("qam", system_params.qam.num_bits_per_symbol, device=device)
        demapper = Demapper(
            "app",
            "qam",
            system_params.qam.num_bits_per_symbol,
            hard_out=True,
            device=device,
        )
        rg = ResourceGrid(
            num_ofdm_symbols=system_params.ofdm.num_symbols,
            fft_size=system_params.ofdm.num_subcarriers,
            subcarrier_spacing=system_params.ofdm.subcarrier_spacing,
            cyclic_prefix_length=system_params.ofdm.cyclic_prefix_length,
            dc_null=False,
            device=device,
        )
        rg_mapper = ResourceGridMapper(rg, device=device)
        rg_demapper = ResourceGridDemapper(rg, sm, device=device)
        modulator = OFDMModulator(
            cyclic_prefix_length=system_params.ofdm.cyclic_prefix_length,
            device=device,
        )
        demodulator = OFDMDemodulator(
            fft_size=system_params.ofdm.num_subcarriers,
            l_min=system_params.ofdm.l_min,
            cyclic_prefix_length=system_params.ofdm.cyclic_prefix_length,
            device=device,
        )
        return cls(
            binary_source=binary_source,
            mapper=mapper,
            demapper=demapper,
            rg=rg,
            rg_mapper=rg_mapper,
            rg_demapper=rg_demapper,
            modulator=modulator,
            demodulator=demodulator,
        )
