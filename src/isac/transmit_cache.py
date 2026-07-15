"""发射波形磁盘缓存：``b.npy`` / ``x_rg.npy`` / ``x_time.npy``。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import PROJECT_ROOT


@dataclass
class TransmitCache:
    """发射波形缓存：目录内固定存放 ``b.npy`` / ``x_rg.npy`` / ``x_time.npy``。

    由 ``source.cache_file`` 解析得到绝对目录；相对路径相对 ``PROJECT_ROOT``。
    """

    cache_dir: Path
    device: str = "cuda:0"

    @classmethod
    def from_cache_file(
        cls,
        raw: str | Path,
        *,
        device: str = "cuda:0",
    ) -> "TransmitCache":
        """由 ``source.cache_file`` 原始路径构建；相对路径相对 ``PROJECT_ROOT``。"""
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return cls(cache_dir=path, device=device)

    @classmethod
    def require(cls, cache: "TransmitCache | None") -> "TransmitCache":
        """断言已配置发射缓存；``None``（未设 ``source.cache_file``）时抛错。

        参数:
        -------
        - cache : TransmitCache | None
            通常为 ``system.components.transmit_cache``

        返回:
        -------
        - TransmitCache
            非 ``None`` 的同一实例

        异常:
        -------
        - ValueError
            TOML 未配置 ``source.cache_file``
        """
        if cache is None:
            raise ValueError(
                "TOML 未配置 source.cache_file，无法写出发射缓存；"
                "请在 [source] 中设置 cache_file（缓存目录）"
            )
        return cache

    def npy_paths(self) -> dict[str, Path]:
        """缓存目录内三个 ``.npy`` 路径。"""
        return {
            "b": self.cache_dir / "b.npy",
            "x_rg": self.cache_dir / "x_rg.npy",
            "x_time": self.cache_dir / "x_time.npy",
        }

    def is_complete(self) -> bool:
        """三个 ``.npy`` 均存在时视为缓存命中。"""
        return all(p.is_file() for p in self.npy_paths().values())

    def prepare(
        self, *, force: bool = False
    ) -> tuple[dict[str, Path], bool, list[str]]:
        """离线生成前准备：可选强制删除旧 ``.npy``，再探测是否齐全。

        参数:
        -------
        - force : bool
            ``True`` 时先 ``remove_files()``，再检测完整性（通常为未命中）

        返回:
        -------
        - paths : dict[str, Path]
            ``b`` / ``x_rg`` / ``x_time`` 路径
        - existed : bool
            准备完成后三文件是否齐全（可走加载分支）
        - removed : list[str]
            ``force`` 时实际删除的键名；否则为空列表
        """
        removed: list[str] = []
        if force:
            removed = self.remove_files()
        return self.npy_paths(), self.is_complete(), removed

    def load(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """从缓存目录加载 ``b`` / ``x_rg`` / ``x_time``（假定文件齐全）。"""
        paths = self.npy_paths()
        b_np = np.load(paths["b"])
        x_rg_np = np.load(paths["x_rg"])
        x_time_np = np.load(paths["x_time"])
        b: torch.Tensor | None
        if b_np.size == 0:
            b = None
        else:
            b = torch.from_numpy(b_np).to(device=self.device)
        x_rg = torch.from_numpy(x_rg_np).to(device=self.device)
        x_time = torch.from_numpy(x_time_np).to(device=self.device)
        return b, x_rg, x_time

    def load_all(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """加载全部发射缓存；不完整则报错。

        供需要 ``b`` / ``x_rg`` / ``x_time`` 的场景；GRC 收发端应优先用
        ``load_x_rg`` / ``load_x_time`` 按需加载。
        """
        if not self.is_complete():
            raise FileNotFoundError(
                f"发射波形缓存不完整: {self.cache_dir} "
                f"(需要 b.npy / x_rg.npy / x_time.npy)；请先离线运行 transmit() 生成"
            )
        return self.load()

    def load_x_rg(self) -> torch.Tensor:
        """仅加载 ``x_rg.npy``（供 GRC RX）。"""
        path = self.npy_paths()["x_rg"]
        if not path.is_file():
            raise FileNotFoundError(
                f"发射资源网格缓存不存在: {path}；请先离线运行 transmit() 生成"
            )
        return torch.from_numpy(np.load(path)).to(device=self.device)

    def load_x_time(self) -> torch.Tensor:
        """仅加载 ``x_time.npy``（供 GRC TX）。"""
        path = self.npy_paths()["x_time"]
        if not path.is_file():
            raise FileNotFoundError(
                f"发射时域缓存不存在: {path}；请先离线运行 transmit() 生成"
            )
        return torch.from_numpy(np.load(path)).to(device=self.device)

    def save(
        self,
        b: torch.Tensor | None,
        x_rg: torch.Tensor,
        x_time: torch.Tensor,
    ) -> None:
        """将 ``b`` / ``x_rg`` / ``x_time`` 分别写入缓存目录下的 ``.npy``。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        paths = self.npy_paths()
        b_np = np.array([]) if b is None else b.detach().cpu().numpy()
        np.save(paths["b"], b_np)
        np.save(paths["x_rg"], x_rg.detach().cpu().numpy())
        np.save(paths["x_time"], x_time.detach().cpu().numpy())

    def remove_files(self) -> list[str]:
        """删除已有三个 ``.npy``；返回被删除的文件名键（``b`` / ``x_rg`` / ``x_time``）。"""
        removed: list[str] = []
        for name, path in self.npy_paths().items():
            if path.is_file():
                path.unlink()
                removed.append(name)
        return removed
