"""RTTarget mesh fname 解析单元测试。"""

import pytest
import sionna.rt.scene

from isac.channel.rt import rt_target as rt_target_module
from isac.channel.rt.rt_target import RTTarget


def test_resolve_fname_local_ply(monkeypatch, tmp_path):
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    ply = scenes_dir / "custom_car.ply"
    ply.write_text("")

    monkeypatch.setattr(rt_target_module, "RT_SCENES_DIR", scenes_dir)

    resolved = RTTarget.resolve_fname("t1", "custom_car")
    assert resolved == str(ply.resolve())


def test_resolve_fname_local_with_extension(monkeypatch, tmp_path):
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    obj = scenes_dir / "van.obj"
    obj.write_text("")

    monkeypatch.setattr(rt_target_module, "RT_SCENES_DIR", scenes_dir)

    resolved = RTTarget.resolve_fname("t1", "van.obj")
    assert resolved == str(obj.resolve())


def test_resolve_fname_falls_back_to_sionna(monkeypatch, tmp_path):
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    monkeypatch.setattr(rt_target_module, "RT_SCENES_DIR", scenes_dir)

    resolved = RTTarget.resolve_fname("t2", "low_poly_car")
    assert resolved == getattr(sionna.rt.scene, "low_poly_car")
    assert isinstance(resolved, str)
    assert resolved.endswith(".ply")


def test_resolve_fname_existing_absolute_path(tmp_path):
    ply = tmp_path / "mesh.ply"
    ply.write_text("")

    resolved = RTTarget.resolve_fname("t3", str(ply))
    assert resolved == str(ply.resolve())


def test_resolve_fname_unknown_raises(monkeypatch, tmp_path):
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    monkeypatch.setattr(rt_target_module, "RT_SCENES_DIR", scenes_dir)

    with pytest.raises(ValueError, match="无法解析"):
        RTTarget.resolve_fname("t4", "no_such_mesh")
