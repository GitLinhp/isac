# 标准库
import weakref
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
from .scene_filter import SceneFilter, validate_transceivers_not_in_obstacles
from ...data_structures.params.channel_params.rt_scene_params import (
    RtSceneParams,
    AntennaArrayParams,
)
from .rx_target_tx_geometric import RxTargetTxGeometric
from ... import PROJECT_ROOT


class RTScene(Scene):
    """射线追踪场景类（`Scene` 的扩展）。

    通过 `load_scene` 加载 Sionna 场景后，将内部状态接到本实例上，因此 `RTScene`
    可直接作为 `Scene` 使用。
    """

    path_solver = PathSolver()

    def __init__(
        self,
        scene_params: RtSceneParams,
        *,
        frequency: Optional[float] = None,
        bandwidth: Optional[float] = None,
    ):
        self.scene_params = scene_params
        self._paths = None
        self._rx_target_tx_geometric: Optional[RxTargetTxGeometric] = None
        self._scene_filter: Optional[SceneFilter] = None
        self._scene_filter_margin: Optional[float] = None

        loaded = load_scene(
            filename=self._get_scene_filename(self.scene_params.filename),
            merge_shapes=self.scene_params.merge_shapes,
        )
        self.__dict__.update(loaded.__dict__)
        # 将已加载场景中的对象重新绑定到当前 RTScene 实例。
        # 否则第一次调用 `edit(add=...)` 时，Sionna 会在回填已有对象（如 building_1）
        # 过程中报错：对象已被另一个 scene 使用。
        for obj in self._scene_objects.values():
            obj._scene = weakref.ref(self)
        self.scene_params = scene_params
        self._paths = None
        self._rx_target_tx_geometric = None

        self._init_camera()  # 初始化相机
        self._init_antenna_array()  # 初始化天线阵列
        self._init_transceivers()  # 初始化收发器
        self._init_target_material()  # 初始化目标材料
        self._init_targets()  # 初始化目标

        validate_transceivers_not_in_obstacles(self)

        if frequency is not None:
            self.frequency = float(frequency)
        if bandwidth is not None:
            self.bandwidth = float(bandwidth)

    # ==================== 初始化方法 ====================
    def _init_camera(self) -> None:
        """初始化相机

        注意：Sionna 的 Camera 类不允许同时指定 orientation 和 look_at。
        如果 look_at 不为 None，则优先使用 look_at，忽略 orientation。
        """
        camera = self.scene_params.camera
        if camera is None:
            return
        if camera.look_at is not None:
            self.camera = Camera(position=camera.position, look_at=camera.look_at)
        else:
            self.camera = Camera(
                position=camera.position, orientation=camera.orientation
            )

    def _init_antenna_array(self) -> None:
        """初始化天线阵列"""
        if self.scene_params.antenna_arrays is None:
            return

        tx_array_params = self.scene_params.antenna_arrays["tx_array"]
        rx_array_params = self.scene_params.antenna_arrays["rx_array"]

        self.tx_array = self._create_planar_array(tx_array_params)
        self.rx_array = self._create_planar_array(rx_array_params)

    def _init_transceivers(self) -> None:
        """初始化收发器"""
        self.transceivers: Dict[str, RTTransceiver] = {}
        for name, transceiver_params in self.scene_params.transceivers.items():
            transceiver = RTTransceiver(
                name=name,
                position=transceiver_params.position,
                look_at=transceiver_params.look_at,
                transceiver_type=transceiver_params.type,
                power_dbm=transceiver_params.power_dbm,
            )

            # 将收发器添加到场景
            if transceiver.tx is not None:
                self.add(transceiver.tx)
            if transceiver.rx is not None:
                self.add(transceiver.rx)

            self.transceivers[name] = transceiver

    def _init_target_material(self) -> None:
        """初始化目标材料"""
        self.target_material: Dict[str, ITURadioMaterial] = {}
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
        """初始化目标（使用 ``rt_targets`` 避免覆盖 ``Scene.targets`` 方法）。"""
        self.rt_targets: Dict[str, RTTarget] = {}
        if self.scene_params.targets is None:
            return

        for name, targets_params in self.scene_params.targets.items():

            # 创建目标对象（使用 RTTarget 进行封装）
            target = RTTarget(
                name=name,
                fname=targets_params.fname,
                radio_material=self.target_material[targets_params.material],
            )

            # 先添加到场景，然后设置属性
            self.edit(add=target)
            target.update(
                position=targets_params.position,
                velocity=targets_params.velocity,
            )

            self.rt_targets[name] = target

    # ==================== 辅助方法 ====================
    @staticmethod
    def _snapshot_pos_vel(
        role_cn: str, name: str, obj: object
    ) -> dict[str, np.ndarray]:
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
        1. 项目 ``scenes/{filename}/{filename}.xml`` 本地文件
        2. 字面量 ``"None"`` → 空场景
        3. ``sionna.rt.scene`` 内置场景属性

        参数:
        -------
            - filename (str): 场景文件名字符串

        返回:
        -------
            - XML 路径字符串、内置场景路径，或 None（空场景）
        """
        local_xml = (PROJECT_ROOT / "scenes" / filename / f"{filename}.xml").resolve()
        if local_xml.is_file():
            return str(local_xml)
        if filename == "None":
            return None
        return getattr(sionna.rt.scene, filename)

    def _create_planar_array(self, array_params: AntennaArrayParams) -> PlanarArray:
        """创建平面天线阵列

        参数:
        -------
            - array_params (AntennaArrayParams): 天线阵列配置

        返回:
        -------
            - PlanarArray: 平面天线阵列对象
        """
        return PlanarArray(
            num_rows=array_params.num_rows,
            num_cols=array_params.num_cols,
            vertical_spacing=array_params.vertical_spacing,
            horizontal_spacing=array_params.horizontal_spacing,
            pattern=array_params.pattern,
            polarization=array_params.polarization,
        )

    def build_scene_filter(self, safe_margin: float = 2.0) -> SceneFilter:
        """创建或刷新场景过滤器。"""
        if safe_margin < 0:
            raise ValueError("safe_margin 不能为负数。")
        self._scene_filter = SceneFilter(self, safe_margin=safe_margin)
        self._scene_filter_margin = safe_margin
        return self._scene_filter

    def is_position_valid(self, position: np.ndarray, safe_margin: float = 2.0) -> bool:
        """判断点位是否有效（不在障碍物包围盒内）。"""
        if (
            self._scene_filter is None
            or self._scene_filter_margin is None
            or self._scene_filter_margin != safe_margin
        ):
            self.build_scene_filter(safe_margin=safe_margin)
        return self._scene_filter.is_valid(position)

    @property
    def targets_states(self) -> dict[str, dict[str, np.ndarray]]:
        """Snapshot of all ``rt_targets`` positions and velocities as NumPy vectors.

        返回 ``rt_targets`` 中每个目标的当前 ``position`` / ``velocity``，组装为嵌套字典：
        ``{target_name: {'pos': ndarray(3,), 'vel': ndarray(3,)}}``，dtype 为 ``float64``。
        若某目标未设置 ``velocity``，则 ``vel`` 为全零向量。
        若 ``rt_targets`` 为空：打印警告后抛出 ``RuntimeError``（不再返回空字典）。
        ``pos`` / ``vel`` 均为独立副本，原地修改不会影响场景对象内部状态。

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Outer keys: target names. Inner keys: ``"pos"``, ``"vel"`` — each ``np.ndarray`` of shape ``(3,)``.
        """
        if not self.rt_targets:
            print(
                "targets_states：rt_targets 为空，无法进行径向真值对齐；请检查 RT 目标配置。"
            )

        out: dict[str, dict[str, np.ndarray]] = {}

        for name, target in self.rt_targets.items():
            out[name] = self._snapshot_pos_vel("目标", name, target)

        return out

    @property
    def rx_states(self) -> dict[str, dict[str, np.ndarray]]:
        """All receivers under ``transceivers`` as NumPy pos/vel snapshots (same layout as ``get_targets_states``).

        按 ``transceivers`` 插入顺序遍历，对每个含 ``rx`` 的条目以 ``Receiver.name`` 为键写入
        ``{'pos': ndarray(3,), 'vel': ndarray(3,)}``。``velocity`` 缺失时为全零；数组均为独立副本。
        若未找到任何接收机：打印警告后抛出 ``RuntimeError``。
        """
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
        """All transmitters under ``transceivers`` as NumPy pos/vel snapshots (same layout as ``get_rx_states``).

        按 ``transceivers`` 插入顺序遍历，对每个含 ``tx`` 的条目以 ``Transmitter.name`` 为键写入
        ``{'pos': ndarray(3,), 'vel': ndarray(3,)}``。``velocity`` 缺失时为全零；数组均为独立副本。
        若未找到任何发射机：打印警告后抛出 ``RuntimeError``。
        """
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

    # ==================== 几何属性 ====================
    @property
    def rx_target_tx_geometric(self) -> RxTargetTxGeometric:
        """对当前场景所有 (接收机, 目标, 发射机) 三元组计算路径类型、几何路径长度与 RX 视线径向速度。

        返回张量形状为 ``(n_rx, n_target, n_tx)``。

        构造时 ``_rx_target_tx_geometric`` 为 ``None``；每次读取根据最新场景状态重新计算并写回缓存，
        与 ``paths`` 在启用 PathSolver 时的刷新方式一致。

        内部经 ``RxTargetTxGeometric.from_states`` 放置张量；device 为 ``None`` 时与 ``from_states`` 一致默认为 CPU。
        若需与其它模块的计算设备对齐，可对返回对象中的张量调用 ``.to(device)``。

        内部调用 ``targets_states``、``rx_states``、``tx_states``。
        """
        self._rx_target_tx_geometric = RxTargetTxGeometric.from_states(
            self.targets_states,
            self.rx_states,
            self.tx_states,
            device=None,
        )
        return self._rx_target_tx_geometric

    # ==================== 路径属性 ====================
    @property
    def paths(self) -> Paths:
        """获取路径，每次调用时自动重新计算最新结果

        返回值:
        -------
            - Paths对象: 最新计算得到的路径
        """
        # 每次调用都重新计算paths
        cfg = self.scene_params.path_solver
        if cfg is None:
            return self._paths
        self._paths = self.path_solver(
            scene=self,
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

    @property
    def output_slug(self) -> str:
        """输出文件名用：将 ``scene_params.filename`` 规范为合法片段（未配置或字面 ``None`` 时用 ``scene``）。"""
        raw = getattr(self.scene_params, "filename", None)
        if raw is None:
            return "scene"
        s = str(raw).strip()
        if not s or s.lower() == "none":
            return "scene"
        return s

    @staticmethod
    def stack_ragged_cir_samples(
        cir_a_list: list[np.ndarray],
        cir_tau_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """路径条数随几何变化时，将各样本 CIR 在每一维上取上界后零填充，再堆成 ``(N,...)``。"""
        if not cir_a_list or len(cir_a_list) != len(cir_tau_list):
            raise ValueError("cir_a_list 与 cir_tau_list 须同长度且非空")
        n = len(cir_a_list)
        ndims_a = {x.ndim for x in cir_a_list}
        if len(ndims_a) != 1:
            raise ValueError(f"CIR_a 各样本秩不一致: {ndims_a}")
        nd_a = cir_a_list[0].ndim
        max_shape_a = list(cir_a_list[0].shape)
        for arr in cir_a_list[1:]:
            if arr.ndim != nd_a:
                raise ValueError("CIR_a 逐样本秩不一致")
            max_shape_a = [max(max_shape_a[i], arr.shape[i]) for i in range(nd_a)]

        ndims_t = {x.ndim for x in cir_tau_list}
        if len(ndims_t) != 1:
            raise ValueError(f"CIR_tau 各样本秩不一致: {ndims_t}")
        nd_t = cir_tau_list[0].ndim
        max_shape_t = list(cir_tau_list[0].shape)
        for arr in cir_tau_list[1:]:
            if arr.ndim != nd_t:
                raise ValueError("CIR_tau 逐样本秩不一致")
            max_shape_t = [max(max_shape_t[i], arr.shape[i]) for i in range(nd_t)]

        out_a = np.zeros((n,) + tuple(max_shape_a), dtype=np.float64)
        out_tau = np.zeros((n,) + tuple(max_shape_t), dtype=np.float64)
        for i, (a, t) in enumerate(zip(cir_a_list, cir_tau_list, strict=True)):
            out_a[(i,) + tuple(slice(0, s) for s in a.shape)] = a
            out_tau[(i,) + tuple(slice(0, s) for s in t.shape)] = t
        return out_a, out_tau

    def cfr_per_tx(
        self,
        rg: ResourceGrid,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离 OFDM 频域信道 ``(S, F)``，与 ``RTChannel.h_freq`` 一致。

        假定单 RX / 单 RX 天线（``h[0, 0, 0, tx, 0]``）。速度符号由 ``doppler_to_velocity`` 统一换算。
        """
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
        """与 ``RTChannel.cfr`` 一致：在 OFDM 子载波频率网格上取射线追踪 CFR（numpy）。"""
        freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        return self.paths.cfr(
            frequencies=freqs,
            sampling_frequency=1 / rg.ofdm_symbol_duration,
            num_time_steps=rg.num_ofdm_symbols,
            out_type="numpy",
        )

    def cir_numpy(self, rg: ResourceGrid) -> tuple[np.ndarray, np.ndarray]:
        """与 ``RTChannel`` OFDM 采样一致的路径 CIR（numpy）：``cir_a`` 最后一维 ``[Re,Im]``，`tau` 为时延 (s)。"""
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

    # ==================== 显示方法 ====================
    def preview(self) -> None:
        """显示场景"""
        super().preview(paths=self.paths)

    def render(self, with_paths: bool = True) -> plt.Figure:
        """渲染场景

        参数:
        -------
            - with_paths (bool): 是否叠加射线路径。为 False 时仅渲染场景几何。
        """
        if not with_paths:
            return super().render(camera=self.camera)

        paths = self.paths
        a = paths.cir(out_type="numpy")[0]
        if a.size == 0:
            return super().render(camera=self.camera)
        else:
            return super().render(camera=self.camera, paths=paths)

    def render_to_file(self, filename: str) -> None:
        """渲染场景到文件"""
        out_path = (PROJECT_ROOT / "out" / filename).resolve()
        super().render_to_file(
            camera=self.camera,
            filename=str(out_path),
            paths=self.paths,
        )
