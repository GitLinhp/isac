import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import set_random_seed
from sionna.phy.utils import compute_ber
import torch


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 通信基线")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="communication_baseline.toml",
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
    system = System(args)  # 创建系统实例

    # 通信基线当前仅打印 BER；产物目录与其他脚本一致，便于后续扩展写入。
    script_out_dir = PROJECT_ROOT / "out" / "communication_baseline"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = system.args.batch_size  # 获取批处理大小

    b: torch.Tensor = system.components.binary_source(  # 生成比特流
        [
            batch_size,
            1,
            1,
            system.components.rg.num_data_symbols
            * system.params.qam.num_bits_per_symbol,
        ]
    )
    b = b.to(system.device)  # 将比特流移动到设备上
    x = system.components.mapper(b)  # 映射比特流到频域
    x_rg = system.components.rg_mapper(x)  # 频域映射
    x_time = system.components.modulator(x_rg)  # 调制到时域

    y_time = x_time  # 时域信道

    y_rg = system.components.demodulator(y_time)  # 解调到频域
    y = system.components.rg_demapper(y_rg)  # 频域解映射
    b_hat = system.components.demapper(
        y, no=torch.tensor(0.0, device=system.device)
    )  # 解码
    ber = compute_ber(b, b_hat)  # 计算误码率

    print("BER: {:.3e}".format(ber))


if __name__ == "__main__":
    main()
