"""range_profile_record_limiter 路径分配单元测试。"""

from pathlib import Path

from isac_imp.range_profile_record_limiter import (
    RangeProfileRecordLimiter,
    _flowgraph_has_record_limiter,
    allocate_next_record_path,
)


def test_allocate_next_record_path_empty_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_001"
    path = allocate_next_record_path(str(out_dir))
    assert path == str(out_dir / "range_profiles_001")


def test_allocate_next_record_path_legacy_file_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_001"
    out_dir.mkdir(parents=True)
    (out_dir / "range_profiles").write_bytes(b"\x00" * 32)
    path = allocate_next_record_path(str(out_dir))
    assert path == str(out_dir / "range_profiles_001")


def test_allocate_next_record_path_increments_numbered_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_001"
    out_dir.mkdir(parents=True)
    (out_dir / "range_profiles_001").write_bytes(b"\x00" * 16)
    (out_dir / "range_profiles_003").write_bytes(b"\x00" * 16)
    path = allocate_next_record_path(str(out_dir))
    assert path == str(out_dir / "range_profiles_004")


def test_allocate_next_record_path_ignores_legacy_000_placeholder(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_001"
    out_dir.mkdir(parents=True)
    (out_dir / "range_profiles_000").write_bytes(b"\x00" * 16)
    path = allocate_next_record_path(str(out_dir))
    assert path == str(out_dir / "range_profiles_001")


def test_flowgraph_has_record_limiter_with_suffixed_block() -> None:
    class TopBlock:
        range_profile_record_limiter_0 = object()

        def set_record_enable(self, value: bool) -> None:
            pass

    assert _flowgraph_has_record_limiter(TopBlock()) is True


def test_flowgraph_has_record_limiter_without_suffix() -> None:
    class TopBlock:
        range_profile_record_limiter = object()

        def set_record_enable(self, value: bool) -> None:
            pass

    assert _flowgraph_has_record_limiter(TopBlock()) is True


def test_flowgraph_has_record_limiter_via_record_sink_attrs() -> None:
    class TopBlock:
        record_output_dir = "/tmp/out"
        blocks_file_sink_0 = object()

        def set_record_enable(self, value: bool) -> None:
            pass

    assert _flowgraph_has_record_limiter(TopBlock()) is True


def test_record_enable_allocates_next_path(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "monostatic"
    out_dir.mkdir()
    (out_dir / "range_profiles_001").write_bytes(b"\x00" * 16)

    opened_paths: list[str] = []

    class FakeFileSink:
        def open(self, path: str) -> None:
            opened_paths.append(path)

    class TopBlock:
        record_output_dir = str(out_dir)
        record_file_path = str(out_dir / "range_profiles")
        blocks_file_sink_0 = FakeFileSink()

        def set_record_enable(self, value: bool) -> None:
            self.limiter.record_enable = value

        def __init__(self) -> None:
            self.limiter = RangeProfileRecordLimiter(
                vlen_in=4, record_enable=False, record_max_frames=10
            )

    tb = TopBlock()
    tb.set_record_enable(True)

    assert opened_paths == [str(out_dir / "range_profiles_002")]
    assert tb.record_file_path == str(out_dir / "range_profiles_002")


def test_record_enable_allocates_override_path_and_sink(tmp_path: Path) -> None:
    out_dir = tmp_path / "dev1"
    out_dir.mkdir()

    opened_paths: list[str] = []

    class FakeFileSink:
        def open(self, path: str) -> None:
            opened_paths.append(path)

    class TopBlock:
        record_output_dir = str(tmp_path / "ignored")
        record_file_path_dev1 = "/dev/null"
        blocks_file_sink_dev1 = FakeFileSink()

        def set_record_enable(self, value: bool) -> None:
            self.limiter.record_enable = value

        def __init__(self) -> None:
            self.limiter = RangeProfileRecordLimiter(
                vlen_in=4,
                record_enable=False,
                record_max_frames=10,
                record_output_dir_override=str(out_dir),
                file_sink_attr="blocks_file_sink_dev1",
                record_file_path_attr="record_file_path_dev1",
            )

    tb = TopBlock()
    tb.set_record_enable(True)

    assert opened_paths == [str(out_dir / "range_profiles_001")]
    assert tb.record_file_path_dev1 == str(out_dir / "range_profiles_001")
