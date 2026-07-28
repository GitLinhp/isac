"""``--swap-train-eval-h5`` 路径解析单测。"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "model_training"
    / "run_train_cooperative_monostatic_cnn.py"
)


@pytest.fixture(scope="module")
def train_mod():
    module_name = "run_train_cooperative_monostatic_cnn_swap_test"
    spec = importlib.util.spec_from_file_location(module_name, _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ns(train_mod, **kwargs: object) -> Namespace:
    defaults = {
        "h5_path": train_mod.DEFAULT_H5,
        "eval_h5_path": train_mod.DEFAULT_TEST_H5,
        "test_h5_path": train_mod.DEFAULT_TEST_H5,
        "swap_train_eval_h5": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_resolve_no_swap_keeps_defaults(train_mod) -> None:
    train_h5, eval_h5, test_h5 = train_mod._resolve_dataset_h5_paths(_ns(train_mod))
    assert train_h5 == train_mod.DEFAULT_H5.resolve()
    assert eval_h5 == train_mod.DEFAULT_TEST_H5.resolve()
    assert test_h5 == train_mod.DEFAULT_TEST_H5.resolve()


def test_resolve_swap_exchanges_train_and_eval(train_mod) -> None:
    train_h5, eval_h5, test_h5 = train_mod._resolve_dataset_h5_paths(
        _ns(train_mod, swap_train_eval_h5=True)
    )
    assert train_h5 == train_mod.DEFAULT_TEST_H5.resolve()  # Run2
    assert eval_h5 == train_mod.DEFAULT_H5.resolve()  # Run1
    # default test == default eval (Run2) → follows swapped eval (Run1)
    assert test_h5 == eval_h5


def test_resolve_swap_custom_test_not_auto_linked(train_mod) -> None:
    custom_test = Path("/tmp/custom_test_only.h5")
    train_h5, eval_h5, test_h5 = train_mod._resolve_dataset_h5_paths(
        _ns(
            train_mod,
            swap_train_eval_h5=True,
            test_h5_path=custom_test,
        )
    )
    assert train_h5 == train_mod.DEFAULT_TEST_H5.resolve()
    assert eval_h5 == train_mod.DEFAULT_H5.resolve()
    assert test_h5 == custom_test.resolve()


def test_resolve_swap_respects_explicit_paths(train_mod) -> None:
    a = Path("/tmp/dataset_a.h5")
    b = Path("/tmp/dataset_b.h5")
    train_h5, eval_h5, _ = train_mod._resolve_dataset_h5_paths(
        _ns(
            train_mod,
            h5_path=a,
            eval_h5_path=b,
            test_h5_path=b,
            swap_train_eval_h5=True,
        )
    )
    assert train_h5 == b.resolve()
    assert eval_h5 == a.resolve()
