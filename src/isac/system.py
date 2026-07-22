"""ISAC 端到端仿真编排：发射、接收与感知流水线 API。"""

from pathlib import Path

from sionna.phy import config as sn_config
import torch

from .data_structures import SystemParams
from .data_structures.system_components import SystemComponents
from .data_structures.types import MetricMode, SensingEstimate, SensMode
from .utils import load_config


class System:
    """ISAC 仿真顶层编排：配置加载、组件构建与标准链路 API。

    持有 ``params``（结构化配置）与 ``components``（OFDM/信道/感知子模块）。

    典型通信链::

        transmit() → channel(...) → receive(y_time)

    典型感知链::

        transmit() → channel(...) → sensing(x_rg, y_rg)
    """

    def __init__(
        self,
        config_file: str | Path,
        *,
        device: str = "cuda:0",
    ) -> None:
        """由配置文件路径初始化系统。

        参数:
        -------
        - config_file : str | Path
            TOML 配置路径（经 ``load_config`` 解析，相对 ``config/`` 或仓库根）
        - device : str
            Sionna / Torch 计算设备
        """
        self.config_file = str(config_file)
        self.device = device
        self.config: dict = load_config(config_file)

        sn_config.device = self.device
        self.params = SystemParams.from_dict(self.config)
        self.components = SystemComponents.build_from_params(
            self.params, device=self.device
        )

    @classmethod
    def from_dict(
        cls,
        config: dict,
        *,
        device: str = "cuda:0",
        config_file: str | None = None,
    ) -> "System":
        """由已解析的配置字典构建（供 GRC OFDM 覆盖等需改写 dict 的场景）。"""
        obj = object.__new__(cls)
        obj.config_file = config_file
        obj.device = device
        obj.config = config
        sn_config.device = obj.device
        obj.params = SystemParams.from_dict(obj.config)
        obj.components = SystemComponents.build_from_params(
            obj.params, device=obj.device
        )
        return obj

    # ==================== 发射 ====================
    def transmit(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """生成发射波形。

        按 ``params.source.type`` 分支：

        - ``binary``：随机比特 → QAM 映射
        - ``zc``：Zadoff-Chu 序列；``b`` 为全 1 占位比特（形状由 ``num_bits_per_symbol`` 决定）

        若 ``params.source.cache_file`` 已配置为缓存目录：三文件齐全则直接加载，
        否则生成后写入 ``b.npy`` / ``x_rg.npy`` / ``x_time.npy``。

        返回:
        -------
        - b : torch.Tensor | None
            发射比特；``zc`` 源时为全 1 占位比特
        - x_rg : torch.Tensor
            频域 OFDM 资源网格
        - x_time : torch.Tensor
            时域 OFDM 波形
        """
        cache = self.components.transmit_cache
        if cache is not None and cache.is_complete():
            return cache.load()

        src_type = self.params.source.type
        comps = self.components
        rg = comps.rg
        n_bps = int(self.params.source.num_bits_per_symbol)

        if src_type == "binary":
            b = comps.binary_source(
                [
                    1,
                    1,
                    rg.num_streams_per_tx,
                    rg.num_data_symbols * n_bps,
                ]
            )
            x = comps.mapper(b)
        elif src_type == "zc":
            b = torch.ones(
                1,
                1,
                rg.num_streams_per_tx,
                rg.num_data_symbols * n_bps,
                device=self.device,
            )
            x = comps.zc_source([1, 1, 1, rg.num_data_symbols])
        else:
            raise ValueError(f"unsupported source.type: {src_type!r}")

        # b: [batch, 1, streams, num_data_symbols * bits_per_symbol]

        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)

        if cache is not None:
            cache.save(b, x_rg, x_time)

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

    # ==================== 感知 ====================
    def sensing(
        self,
        x_rg: torch.Tensor,
        y_rg: torch.Tensor,
        *,
        metric_mode: MetricMode = "rv",
        sens_mode: SensMode = "monostatic",
        visualize_file: Path | str | None = None,
        to_db: bool = False,
    ) -> tuple[torch.Tensor, SensingEstimate]:
        """频域接收 → 信道估计 → 时延多普勒谱 → MUSIC 检峰。

        参数:
        -------
        - x_rg : torch.Tensor
            发射侧频域 OFDM 资源网格
        - y_rg : torch.Tensor
            接收侧频域 OFDM 资源网格
        - metric_mode : {"dd", "rv"}
            谱图与 MUSIC 日志 metric
        - sens_mode : {"monostatic", "bistatic"}
            物理换算尺度
        - visualize_file : Path | str | None
            DD 谱输出路径；``None`` 时跳过可视化
        - to_db : bool
            可视化是否使用 dB 刻度

        返回:
        -------
        - h_dd : torch.Tensor
            裁切后的时延多普勒谱
        - estimate : SensingEstimate
            MUSIC 峰换算后的距离/速度估计
        """
        comps = self.components

        # 显示感知性能
        comps.sensing_performance()

        # 信道估计
        h_freq = comps.ls_channel_estimator(x_rg, y_rg)

        # 时延多普勒谱
        h_dd = comps.delay_doppler_spectrum(h_freq, sens_mode=sens_mode)

        # 可视化时延多普勒谱
        if visualize_file is not None:
            comps.delay_doppler_spectrum.visualize(
                file_name=visualize_file,
                metric_mode=metric_mode,
                sens_mode=sens_mode,
                to_db=to_db,
            )

        # MUSIC 峰估计
        peaks = comps.music_estimator(h_dd)

        # 距离/速度估计
        estimate = comps.sensing_estimator(
            peaks,
            sens_mode=sens_mode,
            metric_mode=metric_mode,
        )

        # 返回时延多普勒谱与距离/速度估计
        return h_dd, estimate
