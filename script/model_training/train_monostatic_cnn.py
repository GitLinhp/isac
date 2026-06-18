"""训练单基地时延–多普勒 CNN：HDF5 CFR → 距离/速度回归。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from isac import PROJECT_ROOT
from isac.learning import MonostaticDelayDopplerCNN, MonostaticSensingTorchDataset
from isac.learning.monostatic_cnn import MonostaticCNNConfig
from isac.learning.torch_dataset import _normalize_torch_device

_DEFAULT_DATASET_DIR = PROJECT_ROOT / "out" / "dataset_collection"


def _resolve_default_dataset() -> Path | None:
    """在 ``out/dataset_collection/`` 下选取最新的 ``*.h5``。"""
    if not _DEFAULT_DATASET_DIR.is_dir():
        return None
    candidates = sorted(
        _DEFAULT_DATASET_DIR.glob("*.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def argument_parser() -> argparse.Namespace:
    default_dataset = _resolve_default_dataset()
    parser = argparse.ArgumentParser(description="训练单基地时延–多普勒 CNN")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(default_dataset) if default_dataset else None,
        help=(
            "HDF5 数据集路径；省略时自动使用 "
            f"{_DEFAULT_DATASET_DIR} 下最新的 *.h5"
        ),
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="sensing_monostatic_canyon.toml",
        help="OFDM/感知参数 TOML（须与数据集采集配置一致）",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--offset", type=int, default=128, help="DD 谱 ROI 半宽 (bin)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备；Sionna 须使用 cuda:0 而非 cuda",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "out" / "monostatic_cnn" / "model.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _evaluate(
    model: MonostaticDelayDopplerCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    range_sq = 0.0
    vel_sq = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_range = batch["range_m"].to(device)
            y_vel = batch["velocity_mps"].to(device)
            y_norm = model.forward_normalized(x)
            target_norm = MonostaticDelayDopplerCNN.normalize_labels(
                y_range,
                y_vel,
                max_range_m=model.cfg.max_range_m,
                max_velocity_mps=model.cfg.max_velocity_mps,
            )
            total_loss += criterion(y_norm, target_norm).item() * x.size(0)
            pred = model.denormalize(y_norm)
            range_sq += torch.sum((pred[:, 0] - y_range) ** 2).item()
            vel_sq += torch.sum((pred[:, 1] - y_vel) ** 2).item()
            n += x.size(0)
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, (range_sq / n) ** 0.5, (vel_sq / n) ** 0.5


def main() -> None:
    args = argument_parser()
    if not args.dataset:
        raise SystemExit(
            "未指定 --dataset，且 "
            f"{_DEFAULT_DATASET_DIR} 下未找到 *.h5。\n"
            "请先采集数据，例如：\n"
            "  python script/model_training/run_dataset_collection.py\n"
            "或显式指定：\n"
            "  python script/train_monostatic_cnn.py "
            "--dataset out/dataset_collection/scene_mc_sionna_dataset.h5"
        )
    torch.manual_seed(args.seed)
    device = _normalize_torch_device(args.device)

    config_path = (
        Path(args.config_file)
        if Path(args.config_file).is_absolute()
        else PROJECT_ROOT / "config" / args.config_file
    )

    full_ds = MonostaticSensingTorchDataset(
        args.dataset,
        config_file=config_path,
        offset=args.offset,
        device=device,
    )
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_channels = 2 if full_ds.use_phase else 1
    cfg = MonostaticCNNConfig(
        in_channels=in_channels,
        max_range_m=full_ds.max_range_m,
        max_velocity_mps=full_ds.max_velocity_mps,
    )
    model = MonostaticDelayDopplerCNN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        f"数据集: {args.dataset} | 训练 {n_train} / 验证 {n_val} | "
        f"max_range={cfg.max_range_m:.1f} m, max_vel=±{cfg.max_velocity_mps:.1f} m/s"
    )

    best_val_rmse = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y_range = batch["range_m"].to(device)
            y_vel = batch["velocity_mps"].to(device)

            optimizer.zero_grad()
            y_norm = model.forward_normalized(x)
            target_norm = MonostaticDelayDopplerCNN.normalize_labels(
                y_range,
                y_vel,
                max_range_m=cfg.max_range_m,
                max_velocity_mps=cfg.max_velocity_mps,
            )
            loss = criterion(y_norm, target_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1

        val_loss, val_rmse_r, val_rmse_v = _evaluate(
            model, val_loader, criterion, device
        )
        val_rmse = (val_rmse_r**2 + val_rmse_v**2) ** 0.5
        print(
            f"epoch {epoch:03d} | train_loss={train_loss / max(n_batch, 1):.4f} | "
            f"val_loss={val_loss:.4f} | RMSE_range={val_rmse_r:.3f} m | "
            f"RMSE_vel={val_rmse_v:.3f} m/s"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "offset": args.offset,
                    "use_phase": full_ds.use_phase,
                },
                out_path,
            )

    print(f"最佳验证综合 RMSE={best_val_rmse:.3f}，模型已保存至 {out_path.resolve()}")


if __name__ == "__main__":
    main()
