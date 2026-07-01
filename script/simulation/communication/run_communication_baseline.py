import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import set_random_seed
from sionna.phy.utils import compute_ber


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 通信基线")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/communication/communication_baseline.toml",
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
        help="随机种子",
    )

    return parser.parse_args()


def main() -> None:
    args = argument_parser()  # 解析命令行参数
    set_random_seed(args.seed)  # 设置随机种子
    system = System(  # 创建系统实例
        config_file=args.config_file,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 通信基线当前仅打印 BER；产物目录与其他脚本一致，便于后续扩展写入。
    script_out_dir = PROJECT_ROOT / "out" / "communication_baseline"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    # 发射
    b, _, x_time = system.transmit()

    # 时域信道
    y_time = x_time  # 时域信道

    # 接收
    b_hat = system.receive(y_time)

    # 计算误码率
    ber = compute_ber(b, b_hat)
    print("BER: {:.3e}".format(ber))


if __name__ == "__main__":
    main()
