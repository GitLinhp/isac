"""ISAC 端到端仿真编排：发射、接收与感知流水线 API。"""

from sionna.phy import config as sn_config
import torch

from .data_structures import SystemParams
from .data_structures.system_components import SystemComponents


class System:
    """ISAC 仿真顶层编排：配置加载、组件构建与标准链路 API。

    持有 ``params``（结构化配置）与 ``components``（OFDM/信道/感知子模块）。

    典型通信链::

        transmit() → channel(...) → receive(y_time)
    """

    def __init__(
        self,
        config: dict,
        *,
        device: str = "cuda:0",
    ) -> None:
        """初始化系统。

        参数:
        -------
        - config : dict
            已解析的配置字典（通常由 ``load_config`` 在外部加载）
        - device : str
            Sionna / Torch 计算设备
        """
        self.device = device
        self.config: dict = config

        sn_config.device = self.device
        self.params = SystemParams.from_dict(self.config)
        self.components = SystemComponents.build_from_params(
            self.params, device=self.device
        )

    # ==================== 发射 ====================
    def transmit(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """生成发射波形。

        按 ``params.source.type`` 分支：

        - ``binary``：随机比特 → QAM 映射
        - ``zc``：Zadoff-Chu 序列（无比特 ``b``）

        返回:
        -------
        - b : torch.Tensor | None
            发射比特；``zc`` 源时为 ``None``
        - x_rg : torch.Tensor
            频域 OFDM 资源网格
        - x_time : torch.Tensor
            时域 OFDM 波形
        """
        src_type = self.params.source.type
        comps = self.components
        rg = comps.rg

        if src_type == "binary":
            b = comps.binary_source(
                [
                    1,
                    1,
                    1,
                    rg.num_data_symbols * int(self.params.source.num_bits_per_symbol),
                ]
            )
            x = comps.mapper(b)
        elif src_type == "zc":
            b = None
            x = comps.zc_source([1, 1, 1, rg.num_data_symbols])
        else:
            raise ValueError(f"unsupported source.type: {src_type!r}")

        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)

        return b, x_rg, x_time

    # ==================== 接收 ====================
    def receive(
        self,
        y_time: torch.Tensor,
        no: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """时域接收与译码。

        参数:
        -------
        - y_time : torch.Tensor
            时域接收信号
        - no : torch.Tensor | float
            AWGN 噪声方差，供 QAM 软解映射；默认 ``0.0`` 表示无噪

        返回:
        -------
        - b_hat : torch.Tensor
            译码比特
        """
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=self.device, dtype=torch.float32)

        comps = self.components
        y_rg = comps.demodulator(y_time)
        y = comps.rg_demapper(y_rg)
        b_hat = comps.demapper(y, no=no)

        return b_hat
