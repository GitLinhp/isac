# 标准库
from pathlib import Path
from typing import Optional, Any
from matplotlib.figure import Figure
import numpy as np
from dataclasses import asdict

# 第三方库
from sionna.rt import (
    load_scene,
    PlanarArray,
    Camera,
    PathSolver,
    ITURadioMaterial,
    Paths,
)
import sionna.rt.scene
from mitsuba import Bitmap

# 本地模块
from ...data_structures.params import RTSimulatorParams
from .rt_scene_filter import RTSceneFilter
from .rt_transceiver import RTTransceiver
from .rt_target import RTTarget
from .rx_target_tx_geometric import RxTargetTxGeometric
from ... import PROJECT_ROOT
from . import RT_SCENES_DIR


class RTSimulator:
    """射线追踪仿真器：组合 Sionna ``Scene`` 与 ISAC 配置（同级拥有，非从属）。

    与 ``self.scene`` 同级：`scene_filter``、``camera``、``tx_array``、``rx_array``、
    ``transceivers``、``target_material``、``rt_targets``。
    ``frequency`` / ``bandwidth`` 仍写入 Sionna ``scene``；路径求解与渲染前由
    ``_sync_scene()`` 将相机、阵列与实体注册同步到 ``scene``。
    """

    def __init__(
        self,
        rt_simulator_params: RTSimulatorParams,
        device: Optional[str] = "cpu",
        *,
        frequency: Optional[float] = None,
        bandwidth: Optional[float] = None,
    ):
        """初始化射线追踪仿真器

        参数:
        -------
        - rt_simulator_params: RTSimulatorParams
            射线追踪仿真器参数
        - device: Optional[str]
            设备类型，可选 ``"cpu"`` 或 ``"cuda"``，默认 ``"cpu"``
        - frequency: Optional[float]
            频率，默认 ``None``
        - bandwidth: Optional[float]
            带宽，默认 ``None``
        """
        self.rt_simulator_params = rt_simulator_params
        self.device = device

        self._init_scene()  # 初始化场景
        if frequency is not None:
            self.scene.frequency = float(frequency)  # 设置频率
        if bandwidth is not None:
            self.scene.bandwidth = float(bandwidth)  # 设置带宽
        self._init_scene_filter()  # 初始化场景过滤器
        self._init_camera()  # 初始化相机
        self._init_antenna_array()  # 初始化天线阵列
        self._init_transceivers()  # 初始化收发器
        self._init_target_material()  # 初始化目标材质
        self._init_targets()  # 初始化目标
        self.path_solver = PathSolver()  # 初始化路径求解器

    # ==================== 初始化方法 ====================
    def _init_scene(self) -> None:
        """加载 Sionna 场景到 ``self.scene``。"""
        self.scene = load_scene(
            filename=self._resolve_fname(self.rt_simulator_params.filename),
            merge_shapes=self.rt_simulator_params.merge_shapes,
        )

    def _init_scene_filter(self) -> None:
        """初始化场景过滤器（与 ``scene`` 同级，读取 mesh 包围盒）。"""
        if getattr(self, "scene") is not None:
            cfg = self.rt_simulator_params.scene_filter
            safe_margin = cfg.safe_margin if cfg is not None else 1.0
            self.scene_filter = RTSceneFilter(self.scene, safe_margin=safe_margin)
        else:
            raise ValueError("scene 未初始化，无法初始化 scene_filter。")

    def _init_camera(self) -> None:
        """初始化相机（赋给 ``self.camera``）。

        注意：Sionna 的 Camera 类不允许同时指定 orientation 和 look_at。
        如果 look_at 不为 None，则优先使用 look_at，忽略 orientation。
        """
        camera = self.rt_simulator_params.camera
        if camera is None:
            raise ValueError("camera 未配置，无法初始化相机。")
        if camera.look_at is not None:
            self.camera = Camera(position=camera.position, look_at=camera.look_at)
        elif camera.orientation is not None:
            self.camera = Camera(
                position=camera.position, orientation=camera.orientation
            )
        else:
            self.camera = Camera(position=camera.position)

    def _init_antenna_array(self) -> None:
        """初始化天线阵列（赋给 ``self.tx_array`` / ``self.rx_array``）。"""
        if self.rt_simulator_params.antenna_arrays is None:
            raise ValueError("antenna_arrays 未配置，无法初始化天线阵列。")

        tx_array_params = self.rt_simulator_params.antenna_arrays["tx_array"]
        rx_array_params = self.rt_simulator_params.antenna_arrays["rx_array"]
        self.tx_array = PlanarArray(**asdict(tx_array_params))  # 创建发射天线阵列
        self.rx_array = PlanarArray(**asdict(rx_array_params))  # 创建接收天线阵列
        self.scene.tx_array = self.tx_array  # 将发射天线阵列添加到场景
        self.scene.rx_array = self.rx_array  # 将接收天线阵列添加到场景

    def _init_transceivers(self) -> None:
        """初始化收发器（仅填充 ``self.transceivers``）。"""
        self.transceivers: dict[str, RTTransceiver] = {}

        if self.rt_simulator_params.transceivers is None:
            return

        for name, transceiver_params in self.rt_simulator_params.transceivers.items():
            transceiver = RTTransceiver(
                name=name,
                position=transceiver_params.position,
                look_at=transceiver_params.look_at,
                transceiver_type=transceiver_params.type,
                power_dbm=transceiver_params.power_dbm,
            )

            if not self.scene_filter(transceiver.position):
                raise ValueError(f"收发器 {name} 位置无效，落入障碍物包围盒。")

            # 将收发器添加到场景
            if transceiver.tx is not None:
                self.scene.add(transceiver.tx)
            if transceiver.rx is not None:
                self.scene.add(transceiver.rx)

            self.transceivers[name] = transceiver  # 将收发器添加到字典

    def _init_target_material(self) -> None:
        """初始化目标材料"""
        self.target_materials: dict[str, ITURadioMaterial] = {}

        if self.rt_simulator_params.target_materials is None:
            return

        for (
            name,
            target_material_params,
        ) in self.rt_simulator_params.target_materials.items():
            material = ITURadioMaterial(
                name=name,
                itu_type=target_material_params.type,
                thickness=target_material_params.thickness,
                color=target_material_params.color,
            )
            self.target_materials[name] = material

    def _init_targets(self) -> None:
        """初始化目标（``rt_targets``字典）"""
        self.rt_targets: dict[str, RTTarget] = {}

        if self.rt_simulator_params.targets is None:
            return

        for name, targets_params in self.rt_simulator_params.targets.items():
            # 创建目标对象
            target = RTTarget(
                name=name,
                fname=targets_params.fname,
                radio_material=self.target_materials[targets_params.material],
            )

            if not self.scene_filter(target.position):
                raise ValueError(f"目标 {name} 位置无效，落入障碍物包围盒。")

            # 先添加到场景，然后设置属性
            self.scene.edit(add=target)

            # 设置目标位置和速度
            target(
                position=targets_params.position,
                velocity=targets_params.velocity,
                scaling=targets_params.scaling,
            )
            self.rt_targets[name] = target  # 将目标添加到字典

    # ==================== 辅助方法 ====================
    def _resolve_fname(self, filename: str) -> Optional[Any]:
        """解析场景文件路径。

        按以下顺序查找：
        1. 包内 ``isac/channel/rt/scenes/{filename}/{filename}.xml`` 本地文件
        2. 字面量 ``"None"`` → 空场景
        3. ``sionna.rt.scene`` 内置场景属性
        """
        local_xml = (RT_SCENES_DIR / filename / f"{filename}.xml").resolve()
        if local_xml.is_file():
            return str(local_xml)
        elif filename == "None":
            return None
        else:
            try:
                return getattr(sionna.rt.scene, filename)
            except AttributeError:
                raise ValueError(f"场景文件 '{filename}' 不存在。")

    @staticmethod
    def _snapshot_pos_vel(obj: object) -> list[np.ndarray]:
        """从场景实体读取位置/速度，导出为 NumPy 快照。

        供 ``targets_states``、``rx_states``、``tx_states`` 等属性统一格式化输出；
        ``position`` 必填，``velocity`` 缺省时视为静止（零向量）。

        参数:
        -------
        - obj: object
            含 ``position`` / 可选 ``velocity`` 属性的 Sionna 或封装对象。

        返回:
        -------
        - list[np.ndarray]
            ``[pos, vel]`` 二元列表，各为 ``ndarray(3,)``，dtype 均为 ``float64``。
        """
        pos = np.array(getattr(obj, "position"), dtype=np.float64, copy=True)
        vel = np.array(getattr(obj, "velocity"), dtype=np.float64, copy=True)
        return [pos, vel]

    def _collect_transceiver_states(
        self,
        *,
        role_attr: str,
        empty_runtime_msg: str,
    ) -> dict[str, list[np.ndarray]]:
        """从 ``self.transceivers`` 按角色收集位置/速度快照。

        遍历各 ``RTTransceiver``，读取 ``role_attr``（``"tx"`` 或 ``"rx"``）对应的
        Sionna 实体；未配置该角色的收发器跳过。结果以实体 ``name`` 为键，供
        ``rx_states`` / ``tx_states`` 复用。

        参数:
        -------
        - role_attr: str
            ``RTTransceiver`` 上的角色属性名，取 ``"tx"`` 或 ``"rx"``。
        - empty_runtime_msg: str
            未收集到任何实体时 ``RuntimeError`` 的错误信息。

        返回:
        -------
        - dict[str, list[np.ndarray]]
            键为 Sionna 实体名，值为 ``_snapshot_pos_vel`` 返回的 ``[pos, vel]``。

        异常:
        -------
        - RuntimeError
            ``transceivers`` 中无任何收发器具备 ``role_attr`` 对应实体时抛出。
        """
        states: dict[str, list[np.ndarray]] = {}

        # 遍历所有收发器，收集位置/速度快照
        for tc in self.transceivers.values():
            ent = getattr(tc, role_attr)
            # 如果实体为 None，跳过
            if ent is None:
                continue
            states[ent.name] = self._snapshot_pos_vel(ent)

        # 如果未收集到任何实体，抛出错误
        if not states:
            raise RuntimeError(empty_runtime_msg)

        return states

    # ==================== 属性 ====================
    @property
    def targets_states(self) -> dict[str, list[np.ndarray]]:
        """所有 ``rt_targets`` 的位置/速度 NumPy 快照。"""
        if not self.rt_targets:
            raise RuntimeError(
                "rt_targets 为空，无法进行径向真值对齐；请检查 rt_targets 配置。"
            )

        targets_states: dict[str, list[np.ndarray]] = {}
        for name, target in self.rt_targets.items():
            targets_states[name] = self._snapshot_pos_vel(target)
        return targets_states

    @property
    def rx_states(self) -> dict[str, list[np.ndarray]]:
        """所有接收机的位置/速度 NumPy 快照。"""

        if not self.transceivers:
            raise RuntimeError(
                "transceivers 为空，无法进行径向真值对齐；请检查 transceivers 配置。"
            )

        return self._collect_transceiver_states(
            role_attr="rx",
            empty_runtime_msg=(
                "rx_states() 为空，请在 transceivers 中配置至少一个接收机。"
            ),
        )

    @property
    def tx_states(self) -> dict[str, list[np.ndarray]]:
        """所有发射机的位置/速度 NumPy 快照。"""

        if not self.transceivers:
            raise RuntimeError(
                "transceivers 为空，无法进行径向真值对齐；请检查 transceivers 配置。"
            )

        return self._collect_transceiver_states(
            role_attr="tx",
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
            device=self.device,
        )
        return self._rx_target_tx_geometric

    def paths(self, *, update: bool = False) -> Paths:
        """获取路径；``update=True`` 时强制重算，否则返回缓存（无缓存时首次求解）。"""
        cfg = self.rt_simulator_params.path_solver
        if cfg is None:
            if not hasattr(self, "_paths"):
                raise RuntimeError("path_solver 未配置且尚无缓存 paths")
            return self._paths
        if update or not hasattr(self, "_paths"):
            self._paths = self.path_solver(scene=self.scene, **asdict(cfg))
        return self._paths

    # ==================== 场景可视化方法 ====================

    def _resolve_clip_at(self, clip_at: Optional[float]) -> Optional[float]:
        if clip_at is not None:
            return clip_at
        cfg = self.rt_simulator_params.render
        if cfg is not None and cfg.clip_at is not None:
            return cfg.clip_at
        return None

    def preview(self, with_paths: bool = True, clip_at: Optional[float] = None) -> None:
        """预览场景

        参数:
        -------
        - with_paths: bool
            是否渲染路径

        返回:
        -------
        - None
        """
        clip_at = self._resolve_clip_at(clip_at)
        self.scene.preview(
            paths=self.paths(update=True) if with_paths else None, clip_at=clip_at
        )

    def render(
        self, with_paths: bool = True, clip_at: Optional[float] = None
    ) -> Figure | Bitmap:
        """渲染场景

        参数:
        -------
        - with_paths: bool
            是否渲染路径

        返回:
        -------
        - Figure
        """
        clip_at = self._resolve_clip_at(clip_at)
        camera = self.camera
        if not with_paths:
            return self.scene.render(camera=camera, clip_at=clip_at)
        else:
            paths = self.paths(update=True)
            a: np.ndarray = paths.cir(out_type="numpy")[0]

            if a.size == 0:  # 不存在有效路径，只渲染场景
                return self.scene.render(camera=camera, clip_at=clip_at)
            else:  # 存在有效路径，渲染场景和路径
                return self.scene.render(camera=camera, paths=paths, clip_at=clip_at)

    def render_to_file(
        self,
        filename: str,
        with_paths: bool = True,
        clip_at: Optional[float] = None,
        output_dir: Path | None = None,
    ) -> Path:
        """渲染场景到文件

        参数:
        -------
        - filename: str
            输出文件名
        - with_paths: bool
            是否渲染路径
        - output_dir: Path | None
            输出目录，默认 ``PROJECT_ROOT / "out"``

        返回:
        -------
        - Path
            写入的文件路径
        """
        clip_at = self._resolve_clip_at(clip_at)
        base = output_dir or (PROJECT_ROOT / "out")
        out_path = (base / filename).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.scene.render_to_file(
            camera=self.camera,
            filename=str(out_path),
            paths=self.paths(update=True) if with_paths else None,
            clip_at=clip_at,
        )
        return out_path
