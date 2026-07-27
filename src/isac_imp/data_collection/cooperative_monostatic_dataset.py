"""Cooperative monostatic 实验数据 HDF5 读写与构建。"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch optional
    torch = None
    Dataset = object  # type: ignore[misc, assignment]

from isac_imp.record_target_metadata import COOPERATIVE_TARGET_CSV, CSV_COLUMNS

DEFAULT_COOPERATIVE_VLEN = 32768
DEFAULT_LABEL_JITTER_M = 0.02
DEFAULT_FFT_LEN = 2048
DEFAULT_ZEROPADDING_FAC = 4
DEFAULT_TRANSPOSE_LEN = 4
_DEFAULT_RDCC_NBYTES = 32 * 1024 * 1024

DATASET_KEY_PROFILES_DEV0 = "profiles_dev0"
DATASET_KEY_PROFILES_DEV1 = "profiles_dev1"
DATASET_KEY_TARGET_POSITION = "target_position"
DATASET_KEY_SESSION_INDEX = "session_index"
DATASET_KEY_FRAME_INDEX = "frame_index"

META_KEY_DESCRIPTION = "description"
META_KEY_DATA_KIND = "data_kind"
META_KEY_VLEN = "vlen"
META_KEY_FFT_LEN = "fft_len"
META_KEY_ZEROPADDING_FAC = "zeropadding_fac"
META_KEY_TRANSPOSE_LEN = "transpose_len"
META_KEY_SOURCE_DIR = "source_dir"
META_KEY_NUM_SESSIONS = "num_sessions"
META_KEY_FRAMES_PER_SESSION = "frames_per_session"
META_KEY_COORDINATE_UNIT = "coordinate_unit"
META_KEY_LABEL_AXES = "label_axes"

_COMPLEX_DTYPE = np.complex64


def _iter_divide_cpi_frames(
    path: str | Path,
    *,
    vlen: int,
) -> Iterator[np.ndarray]:
    """逐帧读取 divide_profiles 二进制（complex64）。"""
    block = _read_divide_cpi_file(path, vlen=vlen)
    for frame in block:
        yield frame


def _count_divide_cpi_frames(path: str | Path, *, vlen: int) -> int:
    item_bytes = vlen * np.dtype(_COMPLEX_DTYPE).itemsize
    size = os.path.getsize(path)
    if size % item_bytes != 0:
        raise ValueError(
            f"file size {size} is not a multiple of frame size {item_bytes}; "
            f"path={path!r}"
        )
    return size // item_bytes


def _read_divide_cpi_file(path: str | Path, *, vlen: int) -> np.ndarray:
    """整文件读取 divide_profiles，返回 shape ``(n_frames, vlen)`` complex64。"""
    n_frames = _count_divide_cpi_frames(path, vlen=vlen)
    if n_frames == 0:
        return np.empty((0, vlen), dtype=_COMPLEX_DTYPE)
    data = np.fromfile(path, dtype=_COMPLEX_DTYPE, count=n_frames * vlen)
    return data.reshape(n_frames, vlen)


def _target_position_from_row(row: dict[str, str], *, target_z_m: float) -> np.ndarray:
    return np.array(
        [
            float(row["target_x_cm"]) / 100.0,
            float(row["target_y_cm"]) / 100.0,
            float(target_z_m),
        ],
        dtype=np.float64,
    )


def _read_cooperative_rows(parent_dir: Path) -> list[dict[str, str]]:
    csv_path = parent_dir / COOPERATIVE_TARGET_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as csv_f:
        rows = [row for row in csv.DictReader(csv_f) if any(row.get(col) for col in CSV_COLUMNS)]
    if not rows:
        raise ValueError(f"no data rows in {csv_path}")
    return rows


@dataclass
class CooperativeMonostaticDatasetWriter:
    """Cooperative monostatic HDF5 写入器（预分配 + 按会话批量写入）。"""

    path: Path
    vlen: int
    compression: str | None = None
    _total_frames: int = 0
    _file: h5py.File | None = field(default=None, repr=False)
    _profile_datasets: tuple[h5py.Dataset, h5py.Dataset] | None = field(
        default=None, repr=False
    )
    _target_position_ds: h5py.Dataset | None = field(default=None, repr=False)
    _session_index_ds: h5py.Dataset | None = field(default=None, repr=False)
    _frame_index_ds: h5py.Dataset | None = field(default=None, repr=False)
    _writer_count: int = 0
    _finalized: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.vlen = int(self.vlen)
        if self.compression in (None, "none"):
            self.compression = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        total_frames: int,
        vlen: int = DEFAULT_COOPERATIVE_VLEN,
        compression: str | None = None,
        chunk_rows: int | None = None,
    ) -> CooperativeMonostaticDatasetWriter:
        writer = cls(path=path, vlen=vlen, compression=compression)
        writer._open_preallocated(int(total_frames), chunk_rows=chunk_rows)
        return writer

    def __len__(self) -> int:
        return self._writer_count

    def __repr__(self) -> str:
        return (
            f"CooperativeMonostaticDatasetWriter(count={self._writer_count}, "
            f"path={self.path})"
        )

    def __enter__(self) -> CooperativeMonostaticDatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None and not self._finalized:
            self._file.close()
            self._file = None

    def append_session(
        self,
        profiles_dev0: np.ndarray,
        profiles_dev1: np.ndarray,
        target_position_m: np.ndarray,
        session_index: int,
    ) -> None:
        dev0 = np.asarray(profiles_dev0, dtype=_COMPLEX_DTYPE)
        dev1 = np.asarray(profiles_dev1, dtype=_COMPLEX_DTYPE)
        if dev0.ndim != 2 or dev1.ndim != 2:
            raise ValueError("profiles must be 2-D arrays with shape (n_frames, vlen)")
        if dev0.shape != dev1.shape:
            raise ValueError(
                f"dev0/dev1 shape mismatch: {dev0.shape} vs {dev1.shape}"
            )
        n_frames, width = dev0.shape
        if width != self.vlen:
            raise ValueError(f"expected vlen={self.vlen}, got width={width}")

        pos = np.asarray(target_position_m, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"expected target_position shape (3,), got {pos.shape}")

        if self._file is None:
            raise RuntimeError("writer not opened; use CooperativeMonostaticDatasetWriter.open")

        start = self._writer_count
        end = start + n_frames
        if end > self._total_frames:
            raise ValueError(
                f"append_session would exceed preallocated frames: {end} > {self._total_frames}"
            )

        assert self._profile_datasets is not None
        assert self._target_position_ds is not None
        assert self._session_index_ds is not None
        assert self._frame_index_ds is not None

        self._profile_datasets[0][start:end] = dev0
        self._profile_datasets[1][start:end] = dev1
        self._target_position_ds[start:end] = pos
        self._session_index_ds[start:end] = int(session_index)
        self._frame_index_ds[start:end] = np.arange(n_frames, dtype=np.int32)
        self._writer_count = end

    def _open_preallocated(
        self,
        total_frames: int,
        *,
        chunk_rows: int | None,
    ) -> None:
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._total_frames = total_frames
        chunk = chunk_rows if chunk_rows is not None else min(256, total_frames)

        self._file = h5py.File(self.path, "w", rdcc_nbytes=_DEFAULT_RDCC_NBYTES)
        profile_kwargs: dict[str, Any] = {
            "chunks": (chunk, self.vlen),
        }
        if self.compression:
            profile_kwargs["compression"] = self.compression

        dev0_ds = self._file.create_dataset(
            DATASET_KEY_PROFILES_DEV0,
            shape=(total_frames, self.vlen),
            dtype=_COMPLEX_DTYPE,
            **profile_kwargs,
        )
        dev1_ds = self._file.create_dataset(
            DATASET_KEY_PROFILES_DEV1,
            shape=(total_frames, self.vlen),
            dtype=_COMPLEX_DTYPE,
            **profile_kwargs,
        )
        self._profile_datasets = (dev0_ds, dev1_ds)
        self._target_position_ds = self._file.create_dataset(
            DATASET_KEY_TARGET_POSITION,
            shape=(total_frames, 3),
            dtype=np.float64,
            chunks=(min(1024, total_frames), 3),
        )
        self._session_index_ds = self._file.create_dataset(
            DATASET_KEY_SESSION_INDEX,
            shape=(total_frames,),
            dtype=np.int32,
            chunks=(min(1024, total_frames),),
        )
        self._frame_index_ds = self._file.create_dataset(
            DATASET_KEY_FRAME_INDEX,
            shape=(total_frames,),
            dtype=np.int32,
            chunks=(min(1024, total_frames),),
        )

    def finalize(self, **metadata_attrs: Any) -> None:
        if self._file is None:
            raise ValueError("CooperativeMonostaticDatasetWriter 无帧数据")
        if self._finalized:
            return

        if META_KEY_DESCRIPTION not in metadata_attrs:
            metadata_attrs[META_KEY_DESCRIPTION] = (
                f"Cooperative monostatic experiment dataset "
                f"({self._writer_count} CPI frames)"
            )
        metadata_attrs.setdefault(META_KEY_DATA_KIND, "divide_cpi")
        metadata_attrs.setdefault(META_KEY_VLEN, self.vlen)
        metadata_attrs.setdefault(META_KEY_COORDINATE_UNIT, "meter")
        metadata_attrs.setdefault(META_KEY_LABEL_AXES, "target_xy")

        for key, value in metadata_attrs.items():
            self._file.attrs[key] = value

        self._file.close()
        self._file = None
        self._finalized = True


@dataclass
class CooperativeMonostaticDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """Cooperative monostatic HDF5 只读数据集。"""

    profiles_dev0: np.ndarray
    profiles_dev1: np.ndarray
    target_position: np.ndarray
    session_index: np.ndarray
    frame_index: np.ndarray
    attrs: dict[str, Any]

    def _validate_index(self, idx: int) -> None:
        n = len(self)
        if idx < 0 or idx >= n:
            raise IndexError(f"index {idx} out of range for {n} frames")

    def __len__(self) -> int:
        return int(self.profiles_dev0.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None:
            raise ImportError("torch is required for CooperativeMonostaticDataset.__getitem__")
        self._validate_index(idx)
        return {
            "profiles_dev0": torch.from_numpy(self.profiles_dev0[idx]),
            "profiles_dev1": torch.from_numpy(self.profiles_dev1[idx]),
            "target_position": torch.from_numpy(self.target_position[idx]).float(),
            "session_index": torch.tensor(self.session_index[idx], dtype=torch.int64),
            "frame_index": torch.tensor(self.frame_index[idx], dtype=torch.int64),
        }

    def __repr__(self) -> str:
        return (
            f"CooperativeMonostaticDataset(n={len(self)}, "
            f"vlen={self.profiles_dev0.shape[1]})"
        )

    @classmethod
    def load(cls, filepath: str | Path) -> CooperativeMonostaticDataset:
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            return cls(
                profiles_dev0=_require_dataset(f, DATASET_KEY_PROFILES_DEV0)[:],
                profiles_dev1=_require_dataset(f, DATASET_KEY_PROFILES_DEV1)[:],
                target_position=_require_dataset(f, DATASET_KEY_TARGET_POSITION)[:],
                session_index=_require_dataset(f, DATASET_KEY_SESSION_INDEX)[:],
                frame_index=_require_dataset(f, DATASET_KEY_FRAME_INDEX)[:],
                attrs=dict(f.attrs),
            )


def summarize_cooperative_monostatic_h5(filepath: str | Path) -> dict[str, Any]:
    """轻量读取 HDF5 shape 与根属性，不加载数组到内存。"""
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        profiles_dev0 = _require_dataset(f, DATASET_KEY_PROFILES_DEV0)
        profiles_dev1 = _require_dataset(f, DATASET_KEY_PROFILES_DEV1)
        target_position = _require_dataset(f, DATASET_KEY_TARGET_POSITION)
        return {
            "path": filepath,
            "file_size_bytes": filepath.stat().st_size,
            "profiles_dev0_shape": tuple(profiles_dev0.shape),
            "profiles_dev1_shape": tuple(profiles_dev1.shape),
            "target_position_shape": tuple(target_position.shape),
            "num_sessions": int(f.attrs.get("num_sessions", 0)),
            "frames_per_session": int(f.attrs.get("frames_per_session", 0)),
            "total_frames": int(profiles_dev0.shape[0]),
            "attrs": dict(f.attrs),
        }


def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    if key not in f:
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


@dataclass(frozen=True)
class _CooperativeSessionSpec:
    session_index: int
    row: dict[str, str]
    dev0_path: Path
    dev1_path: Path
    n_frames: int
    target_position_m: np.ndarray


def _prepare_cooperative_sessions(
    parent_dir: Path,
    rows: list[dict[str, str]],
    *,
    vlen: int,
    target_z_m: float,
) -> list[_CooperativeSessionSpec]:
    """校验 CSV 引用的二进制文件并汇总各会话帧数。"""
    missing: list[str] = []
    for row in rows:
        for key in ("dev0_file", "dev1_file"):
            rel = row[key]
            if not (parent_dir / rel).is_file():
                missing.append(rel)
    if missing:
        raise FileNotFoundError(
            "referenced divide_profiles files missing: " + ", ".join(sorted(set(missing)))
        )

    sessions: list[_CooperativeSessionSpec] = []
    for session_index, row in enumerate(rows):
        dev0_path = parent_dir / row["dev0_file"]
        dev1_path = parent_dir / row["dev1_file"]
        n_dev0 = _count_divide_cpi_frames(dev0_path, vlen=vlen)
        n_dev1 = _count_divide_cpi_frames(dev1_path, vlen=vlen)
        if n_dev0 != n_dev1:
            raise ValueError(
                f"dev0/dev1 frame count mismatch for session {session_index}: "
                f"{row['dev0_file']} ({n_dev0}) vs {row['dev1_file']} ({n_dev1})"
            )
        sessions.append(
            _CooperativeSessionSpec(
                session_index=session_index,
                row=row,
                dev0_path=dev0_path,
                dev1_path=dev1_path,
                n_frames=n_dev0,
                target_position_m=_target_position_from_row(row, target_z_m=target_z_m),
            )
        )
    return sessions


def _uniform_chunk_rows(sessions: list[_CooperativeSessionSpec]) -> int | None:
    if not sessions:
        return None
    frames = sessions[0].n_frames
    if all(session.n_frames == frames for session in sessions):
        return frames
    return 256


def build_cooperative_monostatic_h5(
    parent_dir: str | Path,
    output_path: str | Path,
    *,
    vlen: int = DEFAULT_COOPERATIVE_VLEN,
    compression: str | None = None,
    target_z_m: float = 0.0,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    transpose_len: int = DEFAULT_TRANSPOSE_LEN,
    show_progress: bool = True,
) -> Path:
    """将 ``parent_dir`` 下 CSV + divide_profiles 二进制转为 HDF5。"""
    parent_dir = Path(parent_dir).resolve()
    output_path = Path(output_path)
    rows = _read_cooperative_rows(parent_dir)
    sessions = _prepare_cooperative_sessions(
        parent_dir,
        rows,
        vlen=vlen,
        target_z_m=target_z_m,
    )
    total_frames = sum(session.n_frames for session in sessions)
    frames_per_session = sessions[0].n_frames if sessions else 0
    num_sessions = len(sessions)
    chunk_rows = _uniform_chunk_rows(sessions)

    with CooperativeMonostaticDatasetWriter.open(
        output_path,
        total_frames=total_frames,
        vlen=vlen,
        compression=compression,
        chunk_rows=chunk_rows,
    ) as writer:
        session_iter = tqdm(
            sessions,
            desc="转换 HDF5",
            unit="session",
            disable=not show_progress,
        )
        for session in session_iter:
            session_iter.set_postfix(
                session=f"{session.session_index + 1}/{num_sessions}",
                target=f"({session.row['target_x_cm']},{session.row['target_y_cm']})cm",
                refresh=False,
            )
            dev0_block = _read_divide_cpi_file(session.dev0_path, vlen=vlen)
            dev1_block = _read_divide_cpi_file(session.dev1_path, vlen=vlen)
            writer.append_session(
                dev0_block,
                dev1_block,
                session.target_position_m,
                session.session_index,
            )

        writer.finalize(
            vlen=vlen,
            fft_len=fft_len,
            zeropadding_fac=zeropadding_fac,
            transpose_len=transpose_len,
            source_dir=str(parent_dir),
            num_sessions=num_sessions,
            frames_per_session=frames_per_session,
            target_z_m=target_z_m,
        )

    return output_path


def session_train_val_split(
    session_indices: np.ndarray,
    val_ratio: float,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """按 session 划分 train/val 帧索引，避免同目标泄漏。"""
    session_indices = np.asarray(session_indices, dtype=np.int64)
    if session_indices.ndim != 1:
        raise ValueError(
            f"session_indices 须为一维，收到 shape {session_indices.shape}"
        )
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio 须在 (0, 1)，收到 {val_ratio}")

    unique_sessions = np.unique(session_indices)
    rng = np.random.default_rng(seed)
    perm = unique_sessions.copy()
    rng.shuffle(perm)
    n_val_sessions = max(1, int(round(len(unique_sessions) * val_ratio)))
    val_session_set = set(int(s) for s in perm[:n_val_sessions])

    all_indices = np.arange(session_indices.size, dtype=np.int64)
    val_mask = np.isin(session_indices, list(val_session_set))
    val_indices = all_indices[val_mask]
    train_indices = all_indices[~val_mask]
    return train_indices, val_indices


class CooperativeMonostaticRangeProfileDataset(
    Dataset if torch is not None else object  # type: ignore[misc]
):
    """Cooperative monostatic HDF5 lazy 读取 Dataset（可选在线 divide CPI → ROI）。"""

    def __init__(
        self,
        h5_path: str | Path,
        frame_indices: np.ndarray | Sequence[int],
        *,
        proc_params: dict[str, Any] | None = None,
        range_roi: tuple[float, float] = (0.0, 5.0),
        transform_on_load: bool = True,
        label_jitter_m: float = 0.0,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for CooperativeMonostaticRangeProfileDataset")

        from isac_imp.cooperative_monostatic_pipeline import grc_cooperative_processing_params

        self.h5_path = Path(h5_path)
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.proc_params = proc_params or grc_cooperative_processing_params()
        self.range_roi = (float(range_roi[0]), float(range_roi[1]))
        self.transform_on_load = transform_on_load
        self.label_jitter_m = float(label_jitter_m)
        if self.label_jitter_m < 0.0:
            raise ValueError(f"label_jitter_m 须 >= 0，收到 {label_jitter_m}")
        self._label_rng = np.random.default_rng()
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None:
            raise ImportError(
                "torch is required for CooperativeMonostaticRangeProfileDataset.__getitem__"
            )
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for {len(self)} frames")

        global_idx = int(self.frame_indices[idx])
        f = self._get_h5()
        cpi_dev0 = f[DATASET_KEY_PROFILES_DEV0][global_idx]
        cpi_dev1 = f[DATASET_KEY_PROFILES_DEV1][global_idx]
        target_pos = f[DATASET_KEY_TARGET_POSITION][global_idx]
        session_idx = int(f[DATASET_KEY_SESSION_INDEX][global_idx])

        if self.transform_on_load:
            from isac.models.preprocess import divide_cpi_dual_to_roi_range_profiles_np

            roi0, roi1 = divide_cpi_dual_to_roi_range_profiles_np(
                cpi_dev0,
                cpi_dev1,
                proc_params=self.proc_params,
                range_roi=self.range_roi,
            )
            dual_profiles = torch.from_numpy(np.stack([roi0, roi1], axis=0))
        else:
            dual_profiles = torch.stack(
                [
                    torch.from_numpy(np.asarray(cpi_dev0, dtype=_COMPLEX_DTYPE)),
                    torch.from_numpy(np.asarray(cpi_dev1, dtype=_COMPLEX_DTYPE)),
                ],
                dim=0,
            )

        xy = np.array([float(target_pos[0]), float(target_pos[1])], dtype=np.float32)
        if self.label_jitter_m > 0.0:
            jitter = self._label_rng.uniform(
                -self.label_jitter_m,
                self.label_jitter_m,
                size=2,
            ).astype(np.float32)
            xy = xy + jitter

        return {
            "dual_profiles": dual_profiles.to(dtype=torch.complex64),
            "target_xy": torch.from_numpy(xy),
            "session_index": torch.tensor(session_idx, dtype=torch.int64),
        }
