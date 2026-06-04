# 标准库
import weakref
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np

# 第三方库
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
from ...data_structures.params.rt_scene_params import RtSceneParams, AntennaArrayParams
from ...data_structures.rx_target_tx_geometric import RxTargetTxGeometric
from ...utils import get_logger
from ... import PROJECT_ROOT

logger = get_logger(__name__)


def _snapshot_pos_vel(role_cn: str, name: str, obj: object) -> dict[str, np.ndarray]:
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
    scene: "RTScene",
    *,
    role_attr: str,
    role_cn: str,
    empty_warning: str,
    empty_runtime_msg: str,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for tc in scene.transceivers.values():
        ent = getattr(tc, role_attr, None)
        if ent is None:
            continue
        out[ent.name] = _snapshot_pos_vel(role_cn, ent.name, ent)
    if not out:
        logger.warning(empty_warning)
        raise RuntimeError(empty_runtime_msg)
    return out


class SceneFilter:
    """场景障碍物过滤器：基于场景对象 AABB（轴对齐包围盒）进行点有效性判定。"""

    def __init__(self, scene: "RTScene", safe_margin: float = 2.0):
        """
        初始化障碍物过滤器。

        参数:
        -------
        scene: RTScene
            射线追踪场景对象，其中包含所有三维实体对象。
        safe_margin: float
            包围盒外扩的安全距离，用于判定障碍物时的冗余缓冲，防止穿透边界取样。
        """
        self.safe_margin = safe_margin
        # 存储所有障碍物的包围盒信息，每项为dict，含min/max坐标
        self.obstacles: list[dict[str, np.ndarray]] = []

        for name, obj in scene.objects.items():
            name_lower = name.lower()
            # 地面/地形/楼层/无人机自身不作为障碍物（排除判定）。
            if (
                "ground" in name_lower
                or "terrain" in name_lower
                or "floor" in name_lower
                or "uav" in name_lower
            ):
                continue

            try:
                # 检查对象是否有 mesh（mi_mesh），并能正确获得 bbox
                if hasattr(obj, "mi_mesh") and obj.mi_mesh is not None:
                    bbox = obj.mi_mesh.bbox()
                    # 障碍物包围盒外扩 safe_margin，防止边缘采样点过近
                    self.obstacles.append(
                        {
                            "min": np.array(bbox.min, dtype=np.float64) - safe_margin,
                            "max": np.array(bbox.max, dtype=np.float64) + safe_margin,
                        }
                    )
            except Exception:
                # 若对象（如部分自定义物体）无效bbox或接口异常则跳过
                continue

    def is_valid(self, position: np.ndarray) -> bool:
        """
        判断指定三维点位置是否在所有障碍物包围盒之外。

        参数:
        -------
        position: np.ndarray
            要判断的三维坐标，形如 [x, y, z]

        返回:
        -------
        bool
            True: 点有效（未落入任意障碍物内），False: 点无效（落入某障碍物内）
        """
        x, y, z = np.asarray(position, dtype=np.float64).reshape(-1)
        for obs in self.obstacles:
            # 若点坐标xyz全部同时落在某个障碍物min~max范围内，则判定为无效点
            if (
                obs["min"][0] <= x <= obs["max"][0]
                and obs["min"][1] <= y <= obs["max"][1]
                and obs["min"][2] <= z <= obs["max"][2]
            ):
                return False
        return True


class RTScene(Scene):
    """射线追踪场景类（`Scene` 的扩展）。

    通过 `load_scene` 加载 Sionna 场景后，将内部状态接到本实例上，因此 `RTScene`
    可直接作为 `Scene` 使用。
    """

    path_solver = PathSolver()

    def __init__(
        self,
        scene_params: RtSceneParams,
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
            self.camera = Camera(position=camera.position, orientation=camera.orientation)

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
                trajectory_params=targets_params.trajectory,
            )

            # 先添加到场景，然后设置属性
            self.edit(add=target)
            target.update(
                position=targets_params.position,
                velocity=targets_params.velocity,
            )

            self.rt_targets[name] = target

    # ==================== 辅助方法 ====================
    def _get_scene_filename(self, filename: str) -> Optional[Any]:
        """获取场景文件名对象

        参数:
        -------
            - filename (str): 场景文件名字符串

        返回:
        -------
            - 场景文件名对象或 None
        """
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
        若 ``rt_targets`` 为空：``logger.warning`` 后抛出 ``RuntimeError``（不再返回空字典）。
        ``pos`` / ``vel`` 均为独立副本，原地修改不会影响场景对象内部状态。

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Outer keys: target names. Inner keys: ``"pos"``, ``"vel"`` — each ``np.ndarray`` of shape ``(3,)``.
        """
        if not self.rt_targets:
            logger.warning(
                "targets_states：rt_targets 为空，无法进行径向真值对齐；请检查 RT 目标配置。"
            )

        out: dict[str, dict[str, np.ndarray]] = {}

        for name, target in self.rt_targets.items():
            out[name] = _snapshot_pos_vel("目标", name, target)

        return out

    @property
    def rx_states(self) -> dict[str, dict[str, np.ndarray]]:
        """All receivers under ``transceivers`` as NumPy pos/vel snapshots (same layout as ``get_targets_states``).

        按 ``transceivers`` 插入顺序遍历，对每个含 ``rx`` 的条目以 ``Receiver.name`` 为键写入
        ``{'pos': ndarray(3,), 'vel': ndarray(3,)}``。``velocity`` 缺失时为全零；数组均为独立副本。
        若未找到任何接收机：``logger.warning`` 后抛出 ``RuntimeError``。
        """
        return _collect_transceiver_states(
            self,
            role_attr="rx",
            role_cn="接收机",
            empty_warning=(
                "rx_states：未找到任何接收机，无法进行径向真值对齐；请检查 transceivers 配置。"
            ),
            empty_runtime_msg=("rx_states() 为空，请在 transceivers 中配置至少一个接收机。"),
        )

    @property
    def tx_states(self) -> dict[str, dict[str, np.ndarray]]:
        """All transmitters under ``transceivers`` as NumPy pos/vel snapshots (same layout as ``get_rx_states``).

        按 ``transceivers`` 插入顺序遍历，对每个含 ``tx`` 的条目以 ``Transmitter.name`` 为键写入
        ``{'pos': ndarray(3,), 'vel': ndarray(3,)}``。``velocity`` 缺失时为全零；数组均为独立副本。
        若未找到任何发射机：``logger.warning`` 后抛出 ``RuntimeError``。
        """
        return _collect_transceiver_states(
            self,
            role_attr="tx",
            role_cn="发射机",
            empty_warning=("tx_states：未找到任何发射机；请检查 transceivers 中是否配置 tx。"),
            empty_runtime_msg=("tx_states 为空，请在 transceivers 中配置至少一个发射机。"),
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
