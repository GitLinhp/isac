"""目标点生成：轨迹运动学回放、ROI 内蒙特卡洛位置采样与速度采样。"""

from __future__ import annotations

from typing import Literal

import numpy as np

# 轴对齐三维 ROI：``((x_min, x_max), (y_min, y_max), (z_min, z_max))``；每轴允许 ``min == max`` 表示固定坐标。
RoiBox3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def random_unit_vector_3d(rng: np.random.Generator) -> np.ndarray:
    """单位球均匀分布的随机方向向量（形状 ``(3,)``）。"""
    v = rng.normal(size=3).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return random_unit_vector_3d(rng)
    return v / n


def roi_uniform_scalar(
    low: float, high: float, rng: np.random.Generator | None
) -> float:
    """ROI 单轴坐标：`low == high` 时为固定值，否则 ``Uniform(low, high)``。"""
    if low == high:
        return float(low)
    if rng is None:
        return float(np.random.uniform(low, high))
    return float(rng.uniform(low, high))


def sample_monte_carlo_velocities(
    n_samples: int,
    gen: np.random.Generator,
    velocities: np.ndarray | None,
    speed_range: tuple[float, float],
    velocity_sampling: Literal["sphere_uniform", "axis_box"],
    velocity_roi_vx: tuple[float, float] | None,
    velocity_roi_vy: tuple[float, float] | None,
    velocity_roi_vz: tuple[float, float] | None,
) -> np.ndarray:
    if velocities is not None:
        vel_arr = np.asarray(velocities, dtype=np.float64)
        if vel_arr.shape != (n_samples, 3):
            raise ValueError(
                f"velocities 形状须为 ({n_samples}, 3)，当前为 {vel_arr.shape}"
            )
        return vel_arr
    vel_arr = np.zeros((n_samples, 3), dtype=np.float64)
    smin, smax = float(speed_range[0]), float(speed_range[1])
    if not np.isfinite(smin) or not np.isfinite(smax) or smin < 0 or smax <= smin:
        raise ValueError("speed_range 须满足 0 <= min < max 且为有限值")
    if velocity_sampling == "sphere_uniform":
        for i in range(n_samples):
            spd = gen.uniform(smin, smax)
            vel_arr[i] = spd * random_unit_vector_3d(gen)
    elif velocity_sampling == "axis_box":
        if (
            velocity_roi_vx is None
            or velocity_roi_vy is None
            or velocity_roi_vz is None
        ):
            raise ValueError(
                "velocity_sampling='axis_box' 时必须提供 "
                "velocity_roi_vx, velocity_roi_vy, velocity_roi_vz"
            )
        for axis_key, bounds in (
            ("vx", velocity_roi_vx),
            ("vy", velocity_roi_vy),
            ("vz", velocity_roi_vz),
        ):
            lo, hi = float(bounds[0]), float(bounds[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(
                    f"velocity_roi_{axis_key} 非法，须满足有限且 min < max"
                )
        for i in range(n_samples):
            vel_arr[i, 0] = gen.uniform(velocity_roi_vx[0], velocity_roi_vx[1])
            vel_arr[i, 1] = gen.uniform(velocity_roi_vy[0], velocity_roi_vy[1])
            vel_arr[i, 2] = gen.uniform(velocity_roi_vz[0], velocity_roi_vz[1])
    else:
        raise ValueError("velocity_sampling 须为 'sphere_uniform' 或 'axis_box'")
    return vel_arr


def generate_targets_from_trajectory(
    rt_scene: object,
    time_delta: float,
    steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """纯 Trajectory 运动学回放，得到每步位置/速度表（不调用 ``target.update``、不追射线）。"""
    scene = rt_scene
    if not scene.rt_targets:
        raise RuntimeError("当前场景中没有可用的 RT 目标（scene.rt_targets 为空）")
    _, target = next(iter(scene.rt_targets.items()))
    trajectory = target.trajectory
    trajectory.distance = 0.0
    total_distance = float(trajectory.total_distance())
    velocity = float(trajectory.velocity)
    if total_distance <= 0:
        print("轨迹总长度为 0，跳过轨迹回放")
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    pos_rows: list[np.ndarray] = []
    vel_rows: list[np.ndarray] = []
    step = 0
    while True:
        if trajectory.distance >= total_distance:
            break
        if steps is not None and step >= int(steps):
            break
        curr = trajectory.current_position_and_direction()
        if curr is None:
            raise RuntimeError("trajectory.current_position_and_direction 返回 None")
        pos, direction = curr
        velocity_vec = direction * velocity
        trajectory.distance = min(
            total_distance,
            float(trajectory.distance + velocity * float(time_delta)),
        )
        pos_rows.append(np.asarray(pos, dtype=np.float64).reshape(-1).copy())
        vel_rows.append(np.asarray(velocity_vec, dtype=np.float64).reshape(-1).copy())
        step += 1

    trajectory.distance = 0.0
    if not pos_rows:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pos_rows, dtype=np.float64), np.asarray(
        vel_rows, dtype=np.float64
    )


def generate_monte_carlo_points(
    rt_scene: object,
    roi: RoiBox3D,
    num_samples: int,
    sampling_mode: str = "uniform",
    safe_margin: float = 2.0,
    max_trials_factor: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """在 ROI 内执行蒙特卡洛采样并过滤障碍物点。

    参数:
    -------
    - roi: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``；每轴 **允许 `min == max`** 表示该轴固定。
    - num_samples: 目标采样点数量
    - sampling_mode: `uniform` 或 `gaussian`
    - safe_margin: 障碍物包围盒额外安全距离
    - max_trials_factor: 最大尝试倍数，防止拒绝采样死循环
    - rng: 若为 ``None``，内部仍用全局 ``np.random.*``（兼容旧调用）；否则所有随机数由此生成器产出。

    返回:
    -------
    - np.ndarray: 形状 `(num_samples, 3)` 的有效采样点
    """
    roi_x, roi_y, roi_z = roi
    bounds = {"x": roi_x, "y": roi_y, "z": roi_z}
    for axis, (low, high) in bounds.items():
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(
                f"ROI 维度 `{axis}` 非法：须为有限值且 min <= max（min==max 表示该轴固定）。"
            )

    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0。")
    if safe_margin < 0:
        raise ValueError("safe_margin 不能为负数。")
    if max_trials_factor <= 0:
        raise ValueError("max_trials_factor 必须大于 0。")
    if sampling_mode not in ("uniform", "gaussian"):
        raise ValueError("sampling_mode 仅支持 'uniform' 或 'gaussian'。")

    scene = rt_scene
    scene_filter = scene.build_scene_filter(safe_margin=safe_margin)
    print(f"蒙特卡洛采样开始: mode={sampling_mode}, samples={num_samples}")
    print(f"障碍物包围盒数量: {len(scene_filter.obstacles)}")

    x_low, x_high = roi_x
    y_low, y_high = roi_y
    z_low, z_high = roi_z

    roi_center = np.array(
        [(x_low + x_high) / 2.0, (y_low + y_high) / 2.0, (z_low + z_high) / 2.0],
        dtype=np.float64,
    )
    roi_std = np.array(
        [
            (x_high - x_low) / 6.0,
            (y_high - y_low) / 6.0,
            (z_high - z_low) / 6.0,
        ],
        dtype=np.float64,
    )

    max_trials = num_samples * max_trials_factor
    trials = 0
    accepted: list[np.ndarray] = []

    while len(accepted) < num_samples and trials < max_trials:
        trials += 1
        if sampling_mode == "uniform":
            point = np.array(
                [
                    roi_uniform_scalar(x_low, x_high, rng),
                    roi_uniform_scalar(y_low, y_high, rng),
                    roi_uniform_scalar(z_low, z_high, rng),
                ],
                dtype=np.float64,
            )
        else:
            if rng is None:
                point = np.random.normal(loc=roi_center, scale=roi_std).astype(
                    np.float64
                )
            else:
                point = rng.normal(loc=roi_center, scale=roi_std).astype(np.float64)
            point[0] = np.clip(point[0], x_low, x_high)
            point[1] = np.clip(point[1], y_low, y_high)
            point[2] = np.clip(point[2], z_low, z_high)

        if scene.is_position_valid(point, safe_margin=safe_margin):
            accepted.append(point)

    if len(accepted) < num_samples:
        raise RuntimeError(
            "蒙特卡洛采样未达到目标数量。"
            f"已采样 {len(accepted)}/{num_samples}，尝试次数 {trials}/{max_trials}。"
            "请增大 ROI、减小 safe_margin 或提高 max_trials_factor。"
        )

    acceptance_rate = len(accepted) / max(trials, 1)
    print(
        f"蒙特卡洛采样完成: accepted={len(accepted)}, trials={trials}, "
        f"acceptance_rate={acceptance_rate * 100.0:.2f}%"
    )
    return np.asarray(accepted, dtype=np.float64)


def generate_targets_monte_carlo(
    rt_scene: object,
    *,
    roi: RoiBox3D,
    num_samples: int,
    sampling_mode: str = "uniform",
    safe_margin: float = 2.0,
    max_trials_factor: int = 20,
    velocities: np.ndarray | None = None,
    speed_range: tuple[float, float] = (0.1, 50.0),
    velocity_sampling: Literal["sphere_uniform", "axis_box"] = "sphere_uniform",
    velocity_roi_vx: tuple[float, float] | None = None,
    velocity_roi_vy: tuple[float, float] | None = None,
    velocity_roi_vz: tuple[float, float] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ROI 内采样位置并生成速度表。"""
    if rng is not None:
        gen = rng
    elif seed is not None:
        gen = np.random.default_rng(seed)
    else:
        gen = np.random.default_rng()
    pos_arr = generate_monte_carlo_points(
        rt_scene,
        roi,
        num_samples,
        sampling_mode=sampling_mode,
        safe_margin=safe_margin,
        max_trials_factor=max_trials_factor,
        rng=gen,
    )
    n_samples = int(num_samples)
    vel_arr = sample_monte_carlo_velocities(
        n_samples,
        gen,
        velocities,
        speed_range,
        velocity_sampling,
        velocity_roi_vx,
        velocity_roi_vy,
        velocity_roi_vz,
    )
    return pos_arr, vel_arr


def generate_target_episodes(
    rt_scene: object,
    *,
    source: Literal["monte_carlo", "trajectory"],
    time_delta: float | None = None,
    steps: int | None = None,
    roi: RoiBox3D | None = None,
    num_samples: int | None = None,
    sampling_mode: str = "uniform",
    safe_margin: float = 2.0,
    max_trials_factor: int = 20,
    velocities: np.ndarray | None = None,
    speed_range: tuple[float, float] = (0.1, 50.0),
    velocity_sampling: Literal["sphere_uniform", "axis_box"] = "sphere_uniform",
    velocity_roi_vx: tuple[float, float] | None = None,
    velocity_roi_vy: tuple[float, float] | None = None,
    velocity_roi_vz: tuple[float, float] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按来源生成全部 Episode 的位置与速度数组。"""
    if source == "trajectory":
        if time_delta is None:
            raise ValueError("source='trajectory' 时必须提供 time_delta")
        return generate_targets_from_trajectory(rt_scene, time_delta, steps)
    if roi is None or num_samples is None:
        raise ValueError("source='monte_carlo' 时必须提供 roi 与 num_samples")
    return generate_targets_monte_carlo(
        rt_scene,
        roi=roi,
        num_samples=num_samples,
        sampling_mode=sampling_mode,
        safe_margin=safe_margin,
        max_trials_factor=max_trials_factor,
        velocities=velocities,
        speed_range=speed_range,
        velocity_sampling=velocity_sampling,
        velocity_roi_vx=velocity_roi_vx,
        velocity_roi_vy=velocity_roi_vy,
        velocity_roi_vz=velocity_roi_vz,
        seed=seed,
        rng=rng,
    )
