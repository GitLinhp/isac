"""run_train_sensing_cnn CLI 参数测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TRAIN_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "script"
    / "model_training"
    / "run_train_sensing_cnn.py"
)


def _load_train_module():
    module_name = "run_train_sensing_cnn_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TRAIN_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_argument_parser_regularization_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_train_sensing_cnn.py"])
    mod = _load_train_module()
    args = mod.argument_parser()
    assert args.dropout == 0.2
    assert args.base_channels == 32
    assert args.early_stopping_patience == 15
    assert args.sens_mode == "monostatic"
    assert args.output is None


def test_argument_parser_rejects_invalid_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_train_sensing_cnn.py", "--dropout", "1.0"],
    )
    mod = _load_train_module()
    with pytest.raises(ValueError, match="dropout"):
        mod.main()
