"""共享 CFAR CLI：MUSIC / ESPRIT 评测脚本复用参数组与 detector 构建。"""

from __future__ import annotations

import argparse

from isac.sensing.detection.cfar import CFARDetector
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_CFAR_DETECTOR,
    DEFAULT_CFAR_GUARD,
    DEFAULT_CFAR_PFA,
    DEFAULT_CFAR_TRAILING,
    DEFAULT_CFAR_TYPE,
    default_range_cfar_detector,
)


def add_cfar_arguments(
    parser: argparse.ArgumentParser,
    *,
    method_label: str = "peak selection",
) -> None:
    """向评测脚本 argparse 追加 1D 距离谱 CFAR 开关与参数。"""
    parser.add_argument(
        "--enable-cfar",
        action="store_true",
        help=f"apply 1D CFAR threshold before 1D {method_label}",
    )
    parser.add_argument(
        "--cfar-type",
        type=str,
        default=DEFAULT_CFAR_TYPE,
        choices=("ca", "os"),
        help="CFAR type (default: ca)",
    )
    parser.add_argument(
        "--cfar-guard",
        type=int,
        default=DEFAULT_CFAR_GUARD,
        help="CFAR guard cells (default: 2)",
    )
    parser.add_argument(
        "--cfar-trailing",
        type=int,
        default=DEFAULT_CFAR_TRAILING,
        help="CFAR trailing/reference cells (default: 4)",
    )
    parser.add_argument(
        "--cfar-pfa",
        type=float,
        default=DEFAULT_CFAR_PFA,
        help="CFAR false-alarm rate (default: 1e-4)",
    )
    parser.add_argument(
        "--cfar-detector",
        type=str,
        default=DEFAULT_CFAR_DETECTOR,
        choices=("linear", "squarelaw"),
        help="CFAR detector domain (default: linear)",
    )
    parser.add_argument(
        "--cfar-k",
        type=int,
        default=None,
        help="OS-CFAR rank k (required when --cfar-type os)",
    )
    parser.add_argument(
        "--cfar-offset",
        type=float,
        default=None,
        help="manual CFAR threshold scale (<1 looser, >1 stricter); default auto from pfa",
    )


def build_cfar_detector_from_args(args: argparse.Namespace) -> CFARDetector | None:
    if not getattr(args, "enable_cfar", False):
        return None
    cfar_type = str(args.cfar_type).strip().lower()
    if cfar_type == "os" and args.cfar_k is None:
        raise ValueError("--cfar-k is required when --cfar-type os")
    return default_range_cfar_detector(
        cfar_type=cfar_type,
        guard=int(args.cfar_guard),
        trailing=int(args.cfar_trailing),
        pfa=float(args.cfar_pfa),
        detector=str(args.cfar_detector),
        k=int(args.cfar_k) if args.cfar_k is not None else None,
        offset=float(args.cfar_offset) if args.cfar_offset is not None else None,
    )


def print_cfar_config(cfar_detector: CFARDetector | None) -> None:
    if cfar_detector is None:
        print("CFAR: off")
        return
    k_label = f", k={cfar_detector.k}" if cfar_detector.k is not None else ""
    offset_label = (
        f", offset={cfar_detector.offset}"
        if cfar_detector.offset is not None
        else ""
    )
    print(
        "CFAR: on "
        f"(type={cfar_detector.cfar_type}, guard={cfar_detector.guard}, "
        f"trailing={cfar_detector.trailing}, pfa={cfar_detector.pfa}, "
        f"detector={cfar_detector.detector}{k_label}{offset_label})"
    )
