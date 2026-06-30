# 标准库
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import torch

# 第三方库
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid
from sionna.rt import (
    Scene,
    load_scene,
    PlanarArray,
    Camera,
    PathSolver,
    ITURadioMaterial,
    Paths,
)
import sionna.rt.scene

# 本地模块
from .rt_transceiver import RTTransceiver
from .rt_target import RTTarget
from .scene_filter import SceneFilter
from ...data_structures.params.channel_params.rt_scene_params import (
    RtSceneParams,
    AntennaArrayParams,
)
from .rx_target_tx_geometric import RxTargetTxGeometric
from ... import PROJECT_ROOT
from . import RT_SCENES_DIR


class RTSimulator:
    """射线追踪仿真器：组合 Sionna ``Scene`` 与 ISAC 收发机/目标配置。

    ``self.scene`` 为 Sionna 场景；``transceivers``、``rt_targets`` 等 ISAC 状态
    保留在本包装类上。
    """

    def __init__(
        self,
        scene_params: RtSceneParams,
        *,
        frequency: Optional[float] = None,
        bandwidth: Optional[float] = None,
    ):
        self.scene_params = scene_params
        self.transceivers: Dict[str, RTTransceiver] = {}
        self.rt_targets: Dict[str, RTTarget] = {}
        self.target_material: Dict[str, ITURadioMaterial] = {}
        self._paths = None
        self._rx_target_tx_geometric: Optional[RxTargetTxGeometric] = None
        self._scene_filter: Optional[SceneFilter] = None
        self._scene_filter_margin: Optional[float] = None

        # 初始化场景
        self._init_scene()
        if frequency is not None:
            self.scene.frequency = float(frequency)
        if bandwidth is not None:
            self.scene.bandwidth = float(bandwidth)
        self._init_scene_filter()
        self._init_camera()
        self._init_antenna_array()
        self._init_transceivers()
        self._init_target_material()
        self._init_targets()
        self.path_solver = PathSolver()

    def validate_transceivers_not_in_obstacles(
        self, *, safe_margin: float = 0.0
    ) -> None:
        """场景初始化时校验收发机 ``position`` 未落入任何障碍物包围盒。"""
        if not self.transceivers:
            return
        scene_filter = SceneFilter(self.scene, safe_margin=safe_margin)
        for tc_name, tc in self.transceivers.items():
            pos = np.asarray(tc.position, dtype=np.float64).reshape(3)
            if scene_filter(pos):
                continue
            for obs in scene_filter.obstacles:
                box_min = np.asarray(obs["min"], dtype=np.float64)
                box_max = np.asarray(obs["max"], dtype=np.float64)
                x, y, z = pos
                if (
                    box_min[0] <= x <= box_max[0]
                    and box_min[1] <= y <= box_max[1]
                    and box_min[2] <= z <= box_max[2]
                ):
                    raise ValueError(
                        f"收发机 {tc_name!r} 位置 {pos.tolist()} 落入障碍物 "
                        f"{obs['name']!r} 的包围盒 "
                        f"(min={box_min.tolist()}, max={box_max.tolist()})。"
                        "请调整 [rt_scene.transceivers.*.position] 或场景布局。"
                    )

    # ==================== 初始化方法 ====================
    def _init_scene(self) -> None:
        """加载 Sionna 场景到 ``self.scene``。"""
        self.scene = load_scene(
            filename=self._get_scene_filename(self.scene_params.filename),
            merge_shapes=self.scene_params.merge_shapes,
        )

    def _init_scene_filter(self) -> None:
        """初始化场景过滤器"""
        self._scene_filter = SceneFilter(self.scene, safe_margin=0.0)
        self._scene_filter_margin = 0.0

    def _init_camera(self) -> None:
        """初始化相机

        注意：Sionna 的 Camera 类不允许同时指定 orientation 和 look_at。
        如果 look_at 不为 None，则优先使用 look_at，忽略 orientation。
        """
        camera = self.scene_params.camera
        if camera is None:
            return
        if camera.look_at is not None:
            self.scene.camera = Camera(position=camera.position, look_at=camera.look_at)
        elif camera.orientation is not None:
            self.scene.camera = Camera(
                position=camera.position, orientation=camera.orientation
            )
        else:
            self.scene.camera = Camera(position=camera.position)

    def _init_antenna_array(self) -> None:
        """初始化天线阵列"""
        if self.scene_params.antenna_arrays is None:
            return

        tx_array_params = self.scene_params.antenna_arrays["tx_array"]
        rx_array_params = self.scene_params.antenna_arrays["rx_array"]

        self.scene.tx_array = self._create_planar_array(tx_array_params)
        self.scene.rx_array = self._create_planar_array(rx_array_params)

    def _init_transceivers(self) -> None:
        """初始化收发器"""
        for name, transceiver_params in self.scene_params.transceivers.items():
            transceiver = RTTransceiver(
                name=name,
                position=transceiver_params.position,
                look_at=transceiver_params.look_at,
                transceiver_type=transceiver_params.type,
                power_dbm=transceiver_params.power_dbm,
            )

            if transceiver.tx is not None:
                self.scene.add(transceiver.tx)
            if transceiver.rx is not None:
                self.scene.add(transceiver.rx)

            self.transceivers[name] = transceiver

    def _init_target_material(self) -> None:
        """初始化目标材料"""
        if self.scene_params.target_materials is None:
            return

        for name, target_material_params in self.scene_params.target_materials.items():
            material = ITURadioMaterial(
                name=name,
                itu_type=target_material_params.type,
                thickness=target_material_params.thickness,
                color=target_material_params.color,
            )
            self.target_material[name] = material

    def _init_targets(self) -> None:
        """初始化目标（使用 ``rt_targets`` 避免与 ``Scene.targets`` 方法混淆）。"""
        if self.scene_params.targets is None:
            return

        for name, targets_params in self.scene_params.targets.items():
            target = RTTarget(
                name=name,
                fname=targets_params.fname,
                radio_material=self.target_material[targets_params.material],
            )

            self.scene.edit(add=target)
            target(
                position=targets_params.position,
                velocity=targets_params.velocity,
            )

            self.rt_targets[name] = target

    # ==================== 辅助方法 ====================
    @staticmethod
    def _snapshot_pos_vel(
        role_cn: str, name: str, obj: object
    ) -> dict[str, np.ndarray]:
        """从场景实体读取位置/速度，导出为 NumPy 快照。

        供 ``targets_states``、``rx_states``、``tx_states`` 等属性统一格式化输出；
        ``position`` 必填，``velocity`` 缺省时视为静止（零向量）。

        参数:
        -------
        - role_cn: str
            实体角色中文名，用于校验失败时的错误信息（如 ``"目标"``、``"接收机"``）。
        - name: str
            实体名称，写入错误信息。
        - obj: object
            含 ``position`` / 可选 ``velocity`` 属性的 Sionna 或封装对象。

        返回:
        -------
        - dict[str, np.ndarray]
            ``{"pos": ndarray(3,), "vel": ndarray(3,)}``，dtype 均为 ``float64``。
        """
        pos_raw = getattr(obj, "position", None)
        if pos_raw is None:
            raise ValueError(f"{role_cn} '{name}' 未设置 position，无法导出状态。")
        pos_flat = np.asarray(pos_raw, dtype=np.float64).ravel()
        if pos_flat.size != 3:
            raise ValueError(
                f"{role_cn} '{name}' 的 position 必须为三维向量，当前长度为 {pos_flat.size}。"
            )
        pos = np.array(pos_flat, dtype=np.float64, copy=True)

        vel_raw = getattr(obj, "velocity", None)
        if vel_raw is None:
            vel = np.zeros(3, dtype=np.float64)
        else:
            vel_flat = np.asarray(vel_raw, dtype=np.float64).ravel()
            if vel_flat.size != 3:
                raise ValueError(
                    f"{role_cn} '{name}' 的 velocity 必须为三维向量，当前长度为 {vel_flat.size}。"
                )
            vel = np.array(vel_flat, dtype=np.float64, copy=True)

        return {"pos": pos, "vel": vel}

    def _collect_transceiver_states(
        self,
        *,
        role_attr: str,
        role_cn: str,
        empty_warning: str,
        empty_runtime_msg: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for tc in self.transceivers.values():
            ent = getattr(tc, role_attr, None)
            if ent is None:
                continue
            out[ent.name] = self._snapshot_pos_vel(role_cn, ent.name, ent)
        if not out:
            print(empty_warning)
            raise RuntimeError(empty_runtime_msg)
        return out

    def _get_scene_filename(self, filename: str) -> Optional[Any]:
        """解析场景文件路径。

        按以下顺序查找：
        1. 包内 ``isac/channel/rt/scenes/{filename}/{filename}.xml`` 本地文件
        2. 字面量 ``"None"`` → 空场景
        3. ``sionna.rt.scene`` 内置场景属性
        """
        local_xml = (RT_SCENES_DIR / filename / f"{filename}.xml").resolve()
        if local_xml.is_file():
            return str(local_xml)
        if filename == "None":
            return None
        return getattr(sionna.rt.scene, filename)

    def _create_planar_array(self, array_params: AntennaArrayParams) -> PlanarArray:
        """创建平面天线阵列"""
        return PlanarArray(
            num_rows=array_params.num_rows,
            num_cols=array_params.num_cols,
            vertical_spacing=array_params.vertical_spacing,
            horizontal_spacing=array_params.horizontal_spacing,
            pattern=array_params.pattern,
            polarization=array_params.polarization,
        )

    @property
    def targets_states(self) -> dict[str, dict[str, np.ndarray]]:
        """所有 ``rt_targets`` 的位置/速度 NumPy 快照。"""
        if not self.rt_targets:
            raise RuntimeError(
                "rt_targets 为空，无法进行径向真值对齐；请检查 rt_targets 配置。"
            )

        out: dict[str, dict[str, np.ndarray]] = {}
        for name, target in self.rt_targets.items():
            out[name] = self._snapshot_pos_vel("目标", name, target)
        return out

    @property
    def rx_states(self) -> dict[str, dict[str, np.ndarray]]:
        """所有接收机的位置/速度 NumPy 快照。"""
        return self._collect_transceiver_states(
            role_attr="rx",
            role_cn="接收机",
            empty_warning=(
                "rx_states：未找到任何接收机，无法进行径向真值对齐；请检查 transceivers 配置。"
            ),
            empty_runtime_msg=(
                "rx_states() 为空，请在 transceivers 中配置至少一个接收机。"
            ),
        )

    @property
    def tx_states(self) -> dict[str, dict[str, np.ndarray]]:
        """所有发射机的位置/速度 NumPy 快照。"""
        return self._collect_transceiver_states(
            role_attr="tx",
            role_cn="发射机",
            empty_warning=(
                "tx_states：未找到任何发射机；请检查 transceivers 中是否配置 tx。"
            ),
            empty_runtime_msg=(
                "tx_states 为空，请在 transceivers 中配置至少一个发射机。"
            ),
        )

    @property
    def rx_target_tx_geometric(self) -> RxTargetTxGeometric:
        """对当前场景所有 (接收机, 目标, 发射机) 三元组计算几何量。"""
        self._rx_target_tx_geometric = RxTargetTxGeometric.from_states(
            self.targets_states,
            self.rx_states,
            self.tx_states,
            device=None,
        )
        return self._rx_target_tx_geometric

    @property
    def paths(self) -> Paths:
        """获取路径，每次调用时自动重新计算最新结果。"""
        cfg = self.scene_params.path_solver
        if cfg is None:
            return self._paths
        self._paths = self.path_solver(
            scene=self.scene,
            max_depth=cfg.max_depth,
            max_num_paths_per_src=cfg.max_num_paths_per_src,
            samples_per_src=cfg.samples_per_src,
            los=cfg.los,
            specular_reflection=cfg.specular_reflection,
            diffuse_reflection=cfg.diffuse_reflection,
            refraction=cfg.refraction,
            diffraction=cfg.diffraction,
            edge_diffraction=cfg.edge_diffraction,
            synthetic_array=cfg.synthetic_array,
        )
        return self._paths

    def cfr_per_tx(
        self,
        rg: ResourceGrid,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离 OFDM 频域信道 ``(S, F)``，与 ``RTChannel.h_freq`` 一致。"""
        freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a, tau = self.paths.cir(
            num_time_steps=rg.num_ofdm_symbols,
            sampling_frequency=1 / rg.ofdm_symbol_duration,
            normalize_delays=False,
            out_type="torch",
        )
        h = cir_to_ofdm_channel(
            freqs,
            torch.unsqueeze(a, dim=0),
            torch.unsqueeze(tau, dim=0),
            normalize=False,
        )
        s, f = rg.num_ofdm_symbols, rg.fft_size
        if h.ndim != 7 or h.shape[-2:] != (s, f):
            raise ValueError(
                "cir_to_ofdm_channel 须为 7D (batch, rx, rx_ant, tx, tx_ant, S, F)，"
                f"收到 {tuple(h.shape)}，末两维期望 ({s}, {f})"
            )
        tx_names = list(self.tx_states.keys())
        num_tx = int(h.shape[3])
        if num_tx != len(tx_names):
            raise ValueError(
                f"OFDM 信道的 tx 维 ({num_tx}) 与 tx_states 数量 ({len(tx_names)}) 不一致"
            )
        out: dict[str, torch.Tensor] = {}
        for i, name in enumerate(tx_names):
            slab = h[0, 0, 0, i, 0]
            if device is not None or dtype != slab.dtype:
                slab = slab.to(device=device, dtype=dtype)
            out[name] = slab
        return out

    def cfr_numpy(self, rg: ResourceGrid) -> np.ndarray:
        """在 OFDM 子载波频率网格上取射线追踪 CFR（numpy）。"""
        freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        return self.paths.cfr(
            frequencies=freqs,
            sampling_frequency=1 / rg.ofdm_symbol_duration,
            num_time_steps=rg.num_ofdm_symbols,
            out_type="numpy",
        )

    def cir_numpy(self, rg: ResourceGrid) -> tuple[np.ndarray, np.ndarray]:
        """与 ``RTChannel`` OFDM 采样一致的路径 CIR（numpy）。"""
        a_cpx, tau = self.paths.cir(
            num_time_steps=rg.num_ofdm_symbols,
            sampling_frequency=1 / rg.ofdm_symbol_duration,
            normalize_delays=False,
            out_type="numpy",
        )
        tau_np = np.asarray(tau, dtype=np.float64)
        a_np = np.asarray(a_cpx)
        cir_a = np.stack(
            [
                np.asarray(a_np.real, dtype=np.float64),
                np.asarray(a_np.imag, dtype=np.float64),
            ],
            axis=-1,
        )
        return cir_a, tau_np

    # ==================== 预览方法 ====================
    def preview(self, with_paths: bool = True) -> None:
        """预览场景

        参数:
        -------
        - with_paths: bool
            是否渲染路径

        返回:
        -------
        - None
        """
        self.scene.preview(
            camera=self.scene.camera,
            paths=self.paths if with_paths else None,
        )

    def render(self, with_paths: bool = True) -> plt.Figure:
        """渲染场景

        参数:
        -------
        - with_paths: bool
            是否渲染路径

        返回:
        -------
        - plt.Figure
        """
        if not with_paths:
            return self.scene.render(camera=self.scene.camera)
        paths = self.paths
        a = paths.cir(out_type="numpy")[0]
        if a.size == 0:
            return self.scene.render(camera=self.scene.camera)
        return self.scene.render(camera=self.scene.camera, paths=paths)

    def render_to_file(self, filename: str, with_paths: bool = True) -> None:
        """渲染场景到文件

        参数:
        -------
        - filename: str
            输出文件名
        - with_paths: bool
            是否渲染路径

        返回:
        -------
        - None
        """
        out_path = (PROJECT_ROOT / "out" / filename).resolve()
        self.scene.render_to_file(
            camera=self.scene.camera,
            filename=str(out_path),
            paths=self.paths if with_paths else None,
        )
