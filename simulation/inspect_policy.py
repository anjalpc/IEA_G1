#!/usr/bin/env python3
"""Inspect an ONNX policy without sending commands to a robot or simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def describe(value) -> dict:
    return {"name": value.name, "shape": value.shape, "type": value.type}


def concrete_shape(shape: list) -> list[int]:
    return [dimension if isinstance(dimension, int) and dimension > 0 else 1
            for dimension in shape]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--zero-inference", action="store_true")
    parser.add_argument("--require-g1-velocity-contract", action="store_true")
    args = parser.parse_args()

    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)
    session = ort.InferenceSession(
        str(args.policy), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    report = {
        "path": str(args.policy.resolve()),
        "providers": session.get_providers(),
        "inputs": [describe(value) for value in inputs],
        "outputs": [describe(value) for value in outputs],
    }

    if args.require_g1_velocity_contract:
        if len(inputs) != 1 or concrete_shape(inputs[0].shape) != [1, 480]:
            raise ValueError("G1 velocity policy must accept one [1, 480] input")
        if len(outputs) != 1 or concrete_shape(outputs[0].shape) != [1, 29]:
            raise ValueError("G1 velocity policy must produce one [1, 29] output")

    if args.zero_inference:
        feed = {
            value.name: np.zeros(concrete_shape(value.shape), dtype=np.float32)
            for value in inputs
        }
        values = session.run(None, feed)
        report["zero_inference"] = [
            {"shape": list(value.shape),
             "finite": bool(np.isfinite(value).all()),
             "minimum": float(value.min()),
             "maximum": float(value.max())}
            for value in values
        ]
        if not all(item["finite"] for item in report["zero_inference"]):
            raise ValueError("policy produced non-finite output for zero input")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
