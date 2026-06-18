"""
OFDM / 物理层相关组件构建
"""

import math
from dataclasses import dataclass
from typing import Optional

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

from ...params.ofdm import OFDMParams
from ....zc_source import ZCSource


@dataclass
class OFDMComponents:
    """OFDM 组件"""

    binary_source: BinarySource
    mapper: Mapper
    demapper: Demapper
    rg: ResourceGrid
    rg_mapper: ResourceGridMapper
    rg_demapper: ResourceGridDemapper
    modulator: OFDMModulator
    demodulator: OFDMDemodulator
    zc_source: Optional[ZCSource] = None

    @classmethod
    def build_from_params(
        cls,
        ofdm_params: OFDMParams,
        device: str,
    ) -> "OFDMComponents":
        rx_tx_association = np.array([[1]])
        sm = StreamManagement(rx_tx_association, 1)

        binary_source = BinarySource(device=device)
        mapper = Mapper("qam", ofdm_params.num_bits_per_symbol, device=device)
        demapper = Demapper(
            "app",
            "qam",
            ofdm_params.num_bits_per_symbol,
            hard_out=True,
            device=device,
        )
        rg = ResourceGrid(
            num_ofdm_symbols=ofdm_params.num_symbols,
            fft_size=ofdm_params.num_subcarriers,
            subcarrier_spacing=ofdm_params.subcarrier_spacing,
            cyclic_prefix_length=ofdm_params.cyclic_prefix_length,
            dc_null=False,
            device=device,
        )

        src = ofdm_params.source
        zc_inst: Optional[ZCSource] = None
        if src.type == "zc":
            n_data = rg.num_data_symbols
            u = src.root_index
            if math.gcd(u, n_data) != 1:
                raise ValueError(
                    "ofdm.source: ZC requires gcd(root_index, rg.num_data_symbols)=1; "
                    f"got root_index={u}, num_data_symbols={n_data}"
                )
            zc_inst = ZCSource(
                root_index=u,
                normalize=src.normalize,
                device=device,
            )

        rg_mapper = ResourceGridMapper(rg, device=device)
        rg_demapper = ResourceGridDemapper(rg, sm, device=device)
        modulator = OFDMModulator(
            cyclic_prefix_length=ofdm_params.cyclic_prefix_length,
            device=device,
        )
        demodulator = OFDMDemodulator(
            fft_size=ofdm_params.num_subcarriers,
            l_min=ofdm_params.l_min,
            cyclic_prefix_length=ofdm_params.cyclic_prefix_length,
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
            zc_source=zc_inst,
        )
