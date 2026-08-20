"""单站 OFDM 测距实验数据 HDF5 读写与构建。"""

from __future__ import annotations

import os
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

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    META_KEY_COORDINATE_UNIT,
    META_KEY_DATA_KIND,
    META_KEY_DESCRIPTION,
    META_KEY_FFT_LEN,
    META_KEY_FRAMES_PER_SESSION,
    META_KEY_LABEL_AXES,
    META_KEY_NUM_SESSIONS,
    META_KEY_SOURCE_DIR,
    META_KEY_VLEN,
    META_KEY_ZEROPADDING_FAC,
    _count_divide_cpi_frames,
    _read_divide_cpi_file,
)
from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    MONO_RANGE_TARGET_CSV_COLUMNS,
    _read_mono_range_target_rows,
)

DEFAULT_SINGLE_BS_RANGE_VLEN = 16384
DEFAULT_SINGLE_BS_FFT_LEN = 4096
DEFAULT_SINGLE_BS_ZEROPADDING_FAC = 4
_VLEN_CANDIDATES = (8192, 16384)
_DEFAULT_RDCC_NBYTES = 32 * 1024 * 1024
_COMPLEX_DTYPE = np.complex64
_DATA_KIND_DIVIDE_CPI = "divide_cpi"

DATASET_KEY_PROFILES = "profiles"
DATASET_KEY_TARGET_RANGE = "target_range"
DATASET_KEY_SESSION_INDEX = "session_index"
DATASET_KEY_FRAME_INDEX = "frame_index"


def infer_single_bs_range_vlen(path: str | Path) -> int:
    """根据文件大小在候选 vlen 中推断帧长。"""
    size = os.path.getsize(path)
    matches = [
        vlen
        for vlen in _VLEN_CANDIDATES
        if size % (vlen * np.dtype(_COMPLEX_DTYPE).itemsize) == 0
        and size // (vlen * np.dtype(_COMPLEX_DTYPE).itemsize) > 0
    ]
    if not matches:
        raise ValueError(
            f"cannot infer vlen from file size {size} for {path!r}; "
            f"tried candidates {_VLEN_CANDIDATES}"
        )
    if len(matches) > 1:
        # Prefer the larger candidate when both divide (e.g. empty edge cases).
        return max(matches)
    return matches[0]


def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    if key not in f:
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


def _read_mono_range_rows(parent_dir: Path) -> list[dict[str, str]]:
    csv_path = parent_dir / COOPERATIVE_TARGET_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    rows = _read_mono_range_target_rows(csv_path)
    if not rows:
        raise ValueError(f"no data rows in {csv_path}")
    missing_cols = [
        col for col in MONO_RANGE_TARGET_CSV_COLUMNS if col not in rows[0]
    ]
    if missing_cols:
        raise ValueError(f"CSV missing columns {missing_cols}: {csv_path}")
    return rows


def _target_range_from_row(row: dict[str, str], *, ndigits: int = 2) -> float:
    return round(float(row["target_range_m"]), ndigits)


@dataclass
class SingleBsRangeDatasetWriter:
    """单站测距 HDF5 写入器（预分配 + 按会话批量写入）。"""

    path: Path
    vlen: int
    compression: str | None = None
    _total_frames: int = 0
    _file: h5py.File | None = field(default=None, repr=False)
    _profiles_ds: h5py.Dataset | None = field(default=None, repr=False)
    _target_range_ds: h5py.Dataset | None = field(default=None, repr=False)
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
        vlen: int = DEFAULT_SINGLE_BS_RANGE_VLEN,
        compression: str | None = None,
        chunk_rows: int | None = None,
    ) -> SingleBsRangeDatasetWriter:
        writer = cls(path=path, vlen=vlen, compression=compression)
        writer._open_preallocated(int(total_frames), chunk_rows=chunk_rows)
        return writer

    def __len__(self) -> int:
        return self._writer_count

    def __repr__(self) -> str:
        return (
            f"SingleBsRangeDatasetWriter(count={self._writer_count}, path={self.path})"
        )

    def __enter__(self) -> SingleBsRangeDatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None and not self._finalized:
            self._file.close()
            self._file = None

    def append_session(
        self,
        profiles: np.ndarray,
        target_range_m: float,
        session_index: int,
    ) -> None:
        block = np.asarray(profiles, dtype=_COMPLEX_DTYPE)
        if block.ndim != 2:
            raise ValueError("profiles must be 2-D with shape (n_frames, vlen)")
        n_frames, width = block.shape
        if width != self.vlen:
            raise ValueError(f"expected vlen={self.vlen}, got width={width}")

        if self._file is None:
            raise RuntimeError("writer not opened; use SingleBsRangeDatasetWriter.open")

        start = self._writer_count
        end = start + n_frames
        if end > self._total_frames:
            raise ValueError(
                f"append_session would exceed preallocated frames: "
                f"{end} > {self._total_frames}"
            )

        assert self._profiles_ds is not None
        assert self._target_range_ds is not None
        assert self._session_index_ds is not None
        assert self._frame_index_ds is not None

        self._profiles_ds[start:end] = block
        self._target_range_ds[start:end] = float(target_range_m)
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
        profile_kwargs: dict[str, Any] = {"chunks": (chunk, self.vlen)}
        if self.compression:
            profile_kwargs["compression"] = self.compression

        self._profiles_ds = self._file.create_dataset(
            DATASET_KEY_PROFILES,
            shape=(total_frames, self.vlen),
            dtype=_COMPLEX_DTYPE,
            **profile_kwargs,
        )
        self._target_range_ds = self._file.create_dataset(
            DATASET_KEY_TARGET_RANGE,
            shape=(total_frames,),
            dtype=np.float64,
            chunks=(min(1024, total_frames),),
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
            raise ValueError("SingleBsRangeDatasetWriter 无帧数据")
        if self._finalized:
            return

        if META_KEY_DESCRIPTION not in metadata_attrs:
            metadata_attrs[META_KEY_DESCRIPTION] = (
                f"USRP OFDM single-BS range experiment dataset "
                f"({self._writer_count} CPI frames)"
            )
        metadata_attrs.setdefault(META_KEY_DATA_KIND, _DATA_KIND_DIVIDE_CPI)
        metadata_attrs.setdefault(META_KEY_VLEN, self.vlen)
        metadata_attrs.setdefault(META_KEY_COORDINATE_UNIT, "meter")
        metadata_attrs.setdefault(META_KEY_LABEL_AXES, "target_range")

        for key, value in metadata_attrs.items():
            self._file.attrs[key] = value

        self._file.close()
        self._file = None
        self._finalized = True


@dataclass
class SingleBsRangeDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """单站测距 HDF5 只读数据集。"""

    profiles: np.ndarray
    target_range: np.ndarray
    session_index: np.ndarray
    frame_index: np.ndarray
    attrs: dict[str, Any]

    def _validate_index(self, idx: int) -> None:
        n = len(self)
        if idx < 0 or idx >= n:
            raise IndexError(f"index {idx} out of range for {n} frames")

    def __len__(self) -> int:
        return int(self.profiles.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None:
            raise ImportError("torch is required for SingleBsRangeDataset.__getitem__")
        self._validate_index(idx)
        return {
            "profiles": torch.from_numpy(self.profiles[idx]),
            "target_range": torch.tensor(self.target_range[idx], dtype=torch.float32),
            "session_index": torch.tensor(self.session_index[idx], dtype=torch.int64),
            "frame_index": torch.tensor(self.frame_index[idx], dtype=torch.int64),
        }

    def __repr__(self) -> str:
        return (
            f"SingleBsRangeDataset(n={len(self)}, vlen={self.profiles.shape[1]})"
        )

    @classmethod
    def load(cls, filepath: str | Path) -> SingleBsRangeDataset:
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            return cls(
                profiles=_require_dataset(f, DATASET_KEY_PROFILES)[:],
                target_range=_require_dataset(f, DATASET_KEY_TARGET_RANGE)[:],
                session_index=_require_dataset(f, DATASET_KEY_SESSION_INDEX)[:],
                frame_index=_require_dataset(f, DATASET_KEY_FRAME_INDEX)[:],
                attrs=dict(f.attrs),
            )


def summarize_usrp_ofdm_single_bs_range_h5(filepath: str | Path) -> dict[str, Any]:
    """轻量读取 HDF5 shape 与根属性，不加载数组到内存。"""
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        profiles = _require_dataset(f, DATASET_KEY_PROFILES)
        target_range = _require_dataset(f, DATASET_KEY_TARGET_RANGE)
        return {
            "path": filepath,
            "file_size_bytes": filepath.stat().st_size,
            "data_kind": str(f.attrs.get(META_KEY_DATA_KIND, _DATA_KIND_DIVIDE_CPI)),
            "profiles_shape": tuple(profiles.shape),
            "target_range_shape": tuple(target_range.shape),
            "num_sessions": int(f.attrs.get(META_KEY_NUM_SESSIONS, 0)),
            "frames_per_session": int(f.attrs.get(META_KEY_FRAMES_PER_SESSION, 0)),
            "total_frames": int(profiles.shape[0]),
            "attrs": dict(f.attrs),
        }


@dataclass(frozen=True)
class _SingleBsSessionSpec:
    session_index: int
    row: dict[str, str]
    data_path: Path
    n_frames: int
    target_range_m: float


def _prepare_single_bs_sessions(
    parent_dir: Path,
    rows: list[dict[str, str]],
    *,
    vlen: int,
    label_ndigits: int = 2,
) -> list[_SingleBsSessionSpec]:
    missing: list[str] = []
    for row in rows:
        rel = row["data_file"]
        if not (parent_dir / rel).is_file():
            missing.append(rel)
    if missing:
        raise FileNotFoundError(
            "referenced divide_profiles files missing: "
            + ", ".join(sorted(set(missing)))
        )

    sessions: list[_SingleBsSessionSpec] = []
    for session_index, row in enumerate(rows):
        data_path = parent_dir / row["data_file"]
        n_frames = _count_divide_cpi_frames(data_path, vlen=vlen)
        sessions.append(
            _SingleBsSessionSpec(
                session_index=session_index,
                row=row,
                data_path=data_path,
                n_frames=n_frames,
                target_range_m=_target_range_from_row(row, ndigits=label_ndigits),
            )
        )
    return sessions


def _uniform_chunk_rows(sessions: list[_SingleBsSessionSpec]) -> int | None:
    if not sessions:
        return None
    frames = sessions[0].n_frames
    if all(session.n_frames == frames for session in sessions):
        return frames
    return 256


def _resolve_vlen(
    parent_dir: Path,
    rows: list[dict[str, str]],
    *,
    vlen: int | None,
) -> int:
    if vlen is not None:
        return int(vlen)
    first_path = parent_dir / rows[0]["data_file"]
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    return infer_single_bs_range_vlen(first_path)


def build_usrp_ofdm_single_bs_range_h5(
    parent_dir: str | Path,
    output_path: str | Path,
    *,
    vlen: int | None = None,
    compression: str | None = None,
    fft_len: int = DEFAULT_SINGLE_BS_FFT_LEN,
    zeropadding_fac: int = DEFAULT_SINGLE_BS_ZEROPADDING_FAC,
    label_ndigits: int = 2,
    show_progress: bool = True,
) -> Path:
    """将 ``parent_dir`` 下 CSV + divide_profiles 二进制转为 HDF5。"""
    parent_dir = Path(parent_dir).resolve()
    output_path = Path(output_path)
    rows = _read_mono_range_rows(parent_dir)
    resolved_vlen = _resolve_vlen(parent_dir, rows, vlen=vlen)
    sessions = _prepare_single_bs_sessions(
        parent_dir,
        rows,
        vlen=resolved_vlen,
        label_ndigits=label_ndigits,
    )
    total_frames = sum(session.n_frames for session in sessions)
    frames_per_session = sessions[0].n_frames if sessions else 0
    num_sessions = len(sessions)
    chunk_rows = _uniform_chunk_rows(sessions)

    with SingleBsRangeDatasetWriter.open(
        output_path,
        total_frames=total_frames,
        vlen=resolved_vlen,
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
                range_m=f"{session.target_range_m:.2f}",
                refresh=False,
            )
            block = _read_divide_cpi_file(session.data_path, vlen=resolved_vlen)
            writer.append_session(
                block,
                session.target_range_m,
                session.session_index,
            )

        writer.finalize(
            **{
                META_KEY_VLEN: resolved_vlen,
                META_KEY_FFT_LEN: fft_len,
                META_KEY_ZEROPADDING_FAC: zeropadding_fac,
                META_KEY_SOURCE_DIR: str(parent_dir),
                META_KEY_NUM_SESSIONS: num_sessions,
                META_KEY_FRAMES_PER_SESSION: frames_per_session,
            }
        )

    return output_path
