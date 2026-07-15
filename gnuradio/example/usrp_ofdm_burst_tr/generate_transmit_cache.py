#!/usr/bin/env python3
"""离线预生成发射波形缓存目录（三个 .npy），供 usrp_ofdm_burst_tr GRC TX/RX 使用。

仅调用 ``System.transmit()``：若 TOML ``source.cache_file`` 目录下三文件不齐全则生成并写入；
已齐全时默认直接加载。使用 ``--force`` 可删除旧 ``.npy`` 后重新生成。

目录约定（``cache_file`` 为目录路径）::

    <cache_file>/b.npy
    <cache_file>/x_rg.npy
    <cache_file>/x_time.npy

示例::

    python example/usrp_ofdm_burst_tr/generate_transmit_cache.py --device cpu --force
"""

from __future__ import annotations

import argparse

from isac.system import System
from isac.utils import set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线预生成发射缓存目录(b.npy / x_rg.npy / x_time.npy)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/implementaion/ofdm_burst_source_large_sacle.toml",
        help="配置文件路径（相对 config/ 或仓库根）",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cuda:0", "cpu"],
        help="计算设备",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若缓存已存在则先删除三个 .npy 再重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    set_random_seed(args.seed)

    system = System(args.config_file, device=args.device)
    cache_dir = system.resolve_cache_path()
    if cache_dir is None:
        raise SystemExit(
            "TOML 未配置 source.cache_file，无法写出发射缓存；"
            "请在 [source] 中设置 cache_file（缓存目录）"
        )

    paths = system.cache_npy_paths(cache_dir)
    if args.force:
        removed = []
        for name, path in paths.items():
            if path.is_file():
                path.unlink()
                removed.append(name)
        if removed:
            print(f"已删除旧缓存: {cache_dir} ({', '.join(removed)})")

    existed = system.cache_complete(cache_dir)
    b, x_rg, x_time = system.transmit()

    action = "加载" if existed and not args.force else "生成并写入"
    print(f"{action} cache_dir: {cache_dir}")
    print(f"  b.npy:      {paths['b']}  shape={None if b is None else tuple(b.shape)}")
    print(f"  x_rg.npy:   {paths['x_rg']}  shape={tuple(x_rg.shape)}")
    print(
        f"  x_time.npy: {paths['x_time']}  shape={tuple(x_time.shape)}  "
        f"size={x_time.numel()}"
    )
    if not system.cache_complete(cache_dir):
        raise SystemExit(f"缓存目录未写全: {cache_dir}")


if __name__ == "__main__":
    main()
