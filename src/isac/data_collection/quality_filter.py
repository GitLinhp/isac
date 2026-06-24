"""数据采集侧质量过滤：配置、单样本评估与蒙特卡洛拒绝采样。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np
import torch
from tqdm import tqdm

from isac.channel.rt.rx_target_tx_geometric import RxTargetTxGeometric
from isac.sensing.sample_quality import (
    RejectReason,
    SampleQualityConfig,
    SampleQualityResult,
    evaluate_sample_quality,
)
from isac.sensing.sensing_performance import SensingPerformance
from isac.utils import target_generation as tg

DEFAULT_RX_IDX = 0
DEFAULT_TARGET_IDX = 0
DEFAULT_TX_IDX = 0


@dataclass(frozen=True)
class QualityFilterConfig:
    """质量过滤 CLI 选项与阈值。"""

    enabled: bool = True
    require_los: bool = True
    min_los_ratio: float = 0.3
    min_peak_prominence_db: float = 6.0
    max_bin_offset: int = 3
    quality_max_trials_factor: int = 50
    rx_idx: int = DEFAULT_RX_IDX
    tx_idx: int = DEFAULT_TX_IDX
    target_idx: int = DEFAULT_TARGET_IDX

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> QualityFilterConfig:
        return cls(
            enabled=bool(args.quality_filter),
            require_los=bool(args.require_los),
            min_los_ratio=float(args.min_los_ratio),
            min_peak_prominence_db=float(args.min_peak_prominence_db),
            max_bin_offset=int(args.max_bin_offset),
            quality_max_trials_factor=int(args.quality_max_trials_factor),
        )

    def sample_quality_config(self) -> SampleQualityConfig:
        return SampleQualityConfig(
            require_los=self.require_los,
            min_los_ratio=self.min_los_ratio,
            min_peak_prominence_db=self.min_peak_prominence_db,
            max_bin_offset=self.max_bin_offset,
            rx_idx=self.rx_idx,
            tx_idx=self.tx_idx,
        )


@dataclass
class QualityFilterStats:
    """采集期拒绝采样统计。"""

    accepted: int = 0
    rejected: int = 0
    reject_counts: dict[str, int] = field(default_factory=dict)

    def record_reject(self, reason: RejectReason) -> None:
        self.rejected += 1
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def record_accept(self) -> None:
        self.accepted += 1

    def summary_line(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.reject_counts.items()))
        return (
            f"质量过滤: accepted={self.accepted}, rejected={self.rejected}"
            + (f" ({parts})" if parts else "")
        )


@dataclass(frozen=True)
class MonteCarloSamplingParams:
    """蒙特卡洛位置/速度采样参数（与 ``target_generation`` 对齐）。"""

    sampling_mode: str
    safe_margin: float
    max_trials_factor: int
    speed_range: tuple[float, float]
    velocity_sampling: str


def assess_collection_sample(
    scene: object,
    rg: object,
    device: str | torch.device,
    sensing_performance: SensingPerformance,
    quality_cfg: QualityFilterConfig,
) -> SampleQualityResult:
    """更新位姿后的 RT 场景：几何真值 + CFR → ``evaluate_sample_quality``。"""
    geom = RxTargetTxGeometric.from_states(
        scene.targets_states,
        scene.rx_states,
        scene.tx_states,
        device=device,
    )
    rx_i = quality_cfg.rx_idx
    tgt_i = quality_cfg.target_idx
    tx_i = quality_cfg.tx_idx
    true_range = geom.range_tensor[rx_i, tgt_i, tx_i]
    true_velocity = geom.vel_tensor[rx_i, tgt_i, tx_i]
    cfr = scene.cfr_numpy(rg)
    return evaluate_sample_quality(
        scene,
        cfr,
        float(true_range.item()),
        float(true_velocity.item()),
        sensing_performance,
        cfg=quality_cfg.sample_quality_config(),
        device=torch.device(device),
    )


def run_monte_carlo_with_quality_filter(
    *,
    scene: object,
    target: object,
    rg: object,
    device: str | torch.device,
    sensing_performance: SensingPerformance,
    quality_cfg: QualityFilterConfig,
    num_samples: int,
    seed: int,
    roi_box3d: tuple,
    sampling: MonteCarloSamplingParams,
    update_target_pose: Callable[[object, np.ndarray, np.ndarray], None],
    process_accepted: Callable[[int, np.ndarray, np.ndarray], None],
) -> QualityFilterStats:
    """拒绝采样直至凑满 ``num_samples`` 个可检测样本。"""
    num_target = int(num_samples)
    max_trials = num_target * int(quality_cfg.quality_max_trials_factor)
    rng = np.random.default_rng(int(seed))
    quality_stats = QualityFilterStats()

    accepted = 0
    trials = 0
    pbar = tqdm(total=num_target, desc="MC 数据集(质量过滤)", unit="sample")

    while accepted < num_target and trials < max_trials:
        trials += 1
        pos_batch = tg.generate_monte_carlo_points(
            scene,
            roi_box3d,
            1,
            sampling_mode=sampling.sampling_mode,
            safe_margin=sampling.safe_margin,
            max_trials_factor=sampling.max_trials_factor,
            rng=rng,
        )
        vel_batch = tg.sample_monte_carlo_velocities(
            1,
            rng,
            None,
            sampling.speed_range,
            sampling.velocity_sampling,
            None,
            None,
            None,
        )
        pos = pos_batch[0]
        vel = vel_batch[0]

        update_target_pose(target, pos, vel)
        result = assess_collection_sample(
            scene,
            rg,
            device,
            sensing_performance,
            quality_cfg,
        )
        if not result.passed:
            quality_stats.record_reject(result.reason or "low_peak_prominence")
            continue

        quality_stats.record_accept()
        process_accepted(accepted, pos, vel)
        accepted += 1
        pbar.update(1)

    pbar.close()
    if accepted < num_target:
        raise RuntimeError(
            f"质量过滤后仅采集 {accepted}/{num_target} 个样本，"
            f"尝试 {trials}/{max_trials} 次。请放宽 ROI/阈值或增大 --quality_max_trials_factor。"
        )
    print(quality_stats.summary_line())
    return quality_stats
