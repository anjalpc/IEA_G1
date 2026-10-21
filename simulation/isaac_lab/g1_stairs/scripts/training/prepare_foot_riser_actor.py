"""Append two zero-weight foot-to-riser observation columns to model120."""

from pathlib import Path
import hashlib

import torch


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / (
    "experiments/g1_stairs/logs/rsl_rl/"
    "unitree_iea_g1_29dof_stairs_v0/"
    "albert_flat_gait50_20261021_081452/model_120.pt"
)
OUTPUT = ROOT / (
    "experiments/g1_stairs/outputs/initialization/"
    "albert_model120_actor487_foot_riser.pt"
)
EXPECTED_SOURCE_SHA256 = "e88ede277943075e2d8ead33d4f1e1bd56164e41111f71975b0698b1ae16b661"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("model120 hash mismatch")

    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    first = "actor.0.weight"
    if tuple(state[first].shape) != (512, 485):
        raise RuntimeError(f"unexpected actor input shape: {tuple(state[first].shape)}")

    expanded = dict(state)
    expanded[first] = torch.cat(
        (state[first], state[first].new_zeros(512, 2)), dim=1
    )
    if not torch.equal(expanded[first][:, :485], state[first]):
        raise AssertionError("existing actor weights changed")
    if torch.count_nonzero(expanded[first][:, 485:]):
        raise AssertionError("new observation weights are not zero")
    for key in state:
        if key != first and not torch.equal(expanded[key], state[key]):
            raise AssertionError(f"unexpected checkpoint mutation: {key}")

    checkpoint["model_state_dict"] = expanded
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT)
    print(f"Checkpoint: {OUTPUT}")
    print(f"SHA256: {sha256(OUTPUT)}")
    print("PASS: model120 behavior preserved; actor 485 -> 487")


if __name__ == "__main__":
    main()
