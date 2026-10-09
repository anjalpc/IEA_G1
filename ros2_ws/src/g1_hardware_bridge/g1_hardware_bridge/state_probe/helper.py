"""CLI and lazy binding resolution for physical state inspection."""

import argparse
import importlib
import math


CANDIDATE_BINDINGS = {
    "rt/lowstate": (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "LowState_",
    ),
    "rt/secondary_imu": (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "IMUState_",
    ),
    "rt/sportmodestate": (
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
        "SportModeState_",
    ),
}


def _nonempty_string(value):
    value = str(value).strip()
    if not value:
        raise argparse.ArgumentTypeError(
            "value must be nonempty"
        )
    return value


def _positive_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be a number"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        )
    return value


def _positive_integer(value):
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from error
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "value must be positive"
        )
    return value


def parse_args(args=None):
    """Parse explicit, bounded physical-state probe arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--interface",
        required=True,
        type=_nonempty_string,
    )
    parser.add_argument(
        "--duration-seconds",
        default=10.0,
        type=_positive_float,
    )
    parser.add_argument(
        "--minimum-samples",
        default=20,
        type=_positive_integer,
    )
    return parser.parse_args(args)


def resolve_candidate_types(import_module=importlib.import_module):
    """Resolve approved message types without importing command APIs."""
    resolved = {}
    modules = {}

    for topic, (module_name, type_name) in CANDIDATE_BINDINGS.items():
        try:
            if module_name not in modules:
                modules[module_name] = import_module(module_name)
            resolved[topic] = getattr(
                modules[module_name],
                type_name,
            )
        except (AttributeError, ImportError, KeyError):
            resolved[topic] = None

    return resolved
