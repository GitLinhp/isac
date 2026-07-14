"""ISAC 端到端仿真编排：发射、接收与感知流水线 API。"""

from pathlib import Path

import numpy as np
from sionna.phy import config as sn_config
import torch

from . import PROJECT_ROOT
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

    def resolve_cache_path(self) -> Path | None:
        """解析 ``source.cache_file`` 为绝对目录路径；未配置时返回 ``None``。

        目录内固定存放 ``b.npy`` / ``x_rg.npy`` / ``x_time.npy``。
        """
        raw = self.params.source.cache_file if self.params.source is not None else None
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @staticmethod
    def cache_npy_paths(cache_dir: Path) -> dict[str, Path]:
        """缓存目录内三个 ``.npy`` 路径。"""
        return {
            "b": cache_dir / "b.npy",
            "x_rg": cache_dir / "x_rg.npy",
            "x_time": cache_dir / "x_time.npy",
        }

    def cache_complete(self, cache_dir: Path) -> bool:
        """三个 ``.npy`` 均存在时视为缓存命中。"""
        return all(p.is_file() for p in self.cache_npy_paths(cache_dir).values())

    def _load_transmit_cache(
        self, cache_dir: Path
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """从缓存目录加载 ``b`` / ``x_rg`` / ``x_time``。"""
        paths = self.cache_npy_paths(cache_dir)
        b_np = np.load(paths["b"])
        x_rg_np = np.load(paths["x_rg"])
        x_time_np = np.load(paths["x_time"])
        b: torch.Tensor | None
        if b_np.size == 0:
            b = None
        else:
            b = torch.from_numpy(b_np).to(device=self.device)
        x_rg = torch.from_numpy(x_rg_np).to(device=self.device)
        x_time = torch.from_numpy(x_time_np).to(device=self.device)
        return b, x_rg, x_time

    def _save_transmit_cache(
        self,
        cache_dir: Path,
        b: torch.Tensor | None,
        x_rg: torch.Tensor,
        x_time: torch.Tensor,
    ) -> None:
        """将 ``b`` / ``x_rg`` / ``x_time`` 分别写入缓存目录下的 ``.npy``。"""
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = self.cache_npy_paths(cache_dir)
        b_np = np.array([]) if b is None else b.detach().cpu().numpy()
        np.save(paths["b"], b_np)
        np.save(paths["x_rg"], x_rg.detach().cpu().numpy())
        np.save(paths["x_time"], x_time.detach().cpu().numpy())

    def load_transmit_cache(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """从 ``source.cache_file`` 目录加载全部发射缓存；未配置或不完整则报错。

        供需要 ``b`` / ``x_rg`` / ``x_time`` 的场景；GRC 收发端应优先用
        ``load_transmit_x_rg`` / ``load_transmit_x_time`` 按需加载。
        """
        cache_dir = self.resolve_cache_path()
        if cache_dir is None:
            raise ValueError("source.cache_file 未配置，无法 load_transmit_cache")
        if not self.cache_complete(cache_dir):
            raise FileNotFoundError(
                f"发射波形缓存不完整: {cache_dir} "
                f"(需要 b.npy / x_rg.npy / x_time.npy)；请先离线运行 transmit() 生成"
            )
        return self._load_transmit_cache(cache_dir)

    def load_transmit_x_rg(self) -> torch.Tensor:
        """仅从缓存目录加载 ``x_rg.npy``（供 GRC RX）。"""
        cache_dir = self.resolve_cache_path()
        if cache_dir is None:
            raise ValueError("source.cache_file 未配置，无法 load_transmit_x_rg")
        path = self.cache_npy_paths(cache_dir)["x_rg"]
        if not path.is_file():
            raise FileNotFoundError(
                f"发射资源网格缓存不存在: {path}；请先离线运行 transmit() 生成"
            )
        return torch.from_numpy(np.load(path)).to(device=self.device)

    def load_transmit_x_time(self) -> torch.Tensor:
        """仅从缓存目录加载 ``x_time.npy``（供 GRC TX）。"""
        cache_dir = self.resolve_cache_path()
        if cache_dir is None:
            raise ValueError("source.cache_file 未配置，无法 load_transmit_x_time")
        path = self.cache_npy_paths(cache_dir)["x_time"]
        if not path.is_file():
            raise FileNotFoundError(
                f"发射时域缓存不存在: {path}；请先离线运行 transmit() 生成"
            )
        return torch.from_numpy(np.load(path)).to(device=self.device)

    # ==================== 发射 ====================
    def transmit(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """生成发射波形。

        按 ``params.source.type`` 分支：

        - ``binary``：随机比特 → QAM 映射
        - ``zc``：Zadoff-Chu 序列（无比特 ``b``）

        若 ``params.source.cache_file`` 已配置为缓存目录：三文件齐全则直接加载，
        否则生成后写入 ``b.npy`` / ``x_rg.npy`` / ``x_time.npy``。

        返回:
        -------
        - b : torch.Tensor | None
            发射比特；``zc`` 源时为 ``None``
        - x_rg : torch.Tensor
            频域 OFDM 资源网格
        - x_time : torch.Tensor
            时域 OFDM 波形
        """
        cache_path = self.resolve_cache_path()
        if cache_path is not None and self.cache_complete(cache_path):
            return self._load_transmit_cache(cache_path)

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
            b = torch.ones(
                1,
                1,
                1,
                rg.num_data_symbols * int(self.params.source.num_bits_per_symbol),
            )
            x = comps.zc_source([1, 1, 1, rg.num_data_symbols])
        else:
            raise ValueError(f"unsupported source.type: {src_type!r}")

        # b: [batch, 1, 1, 1024]

        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)

        if cache_path is not None:
            self._save_transmit_cache(cache_path, b, x_rg, x_time)

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
