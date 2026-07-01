import argparse
import warnings

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import load_config, set_random_seed

# Sionna/DrJit 射线追踪在大量 episode 时会触发 AST 装饰器次数告警，不影响数值结果
warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（蒙特卡洛、导出格式）。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    # --- 系统与随机性 ---
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_monostatic.toml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="计算设备类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（蒙特卡洛位置/速度采样）",
    )

    # --- 蒙特卡洛采样参数 ---
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="采样条数",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[0.0, 80.0, -40.0, 40.0],
        help="平面 ROI 四元组",
    )
    parser.add_argument(
        "--position_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="位置采样分布（均匀或高斯）",
    )
    parser.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[0.1, 10.0],
        help="速度模值范围 (m/s)",
    )
    parser.add_argument(
        "--speed_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="速度模值采样分布（均匀或高斯）",
    )

    return parser.parse_args()


def main() -> None:
    """蒙特卡洛采集 episode → 写出 CSV / HDF5。"""
    # 1. 解析 CLI、固定随机种子
    args = argument_parser()
    set_random_seed(args.seed)

    # 2. 构建仿真系统
    config = load_config(args.config_file)
    system = System(
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 3. 取 RT 场景与待驱动的目标
    scene = system.components.rt_simulator
    target_name, target = next(iter(scene.rt_targets.items()))
    print(target_name, target)

    target(
        position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0)
    )


if __name__ == "__main__":
    main()
