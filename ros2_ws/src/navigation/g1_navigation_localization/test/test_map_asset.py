"""Contract for the deterministic simulation reference map."""

from pathlib import Path
import math

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
MAP_DIRECTORY = PACKAGE_ROOT / "maps"
MAP_YAML = MAP_DIRECTORY / "navigation_arena.yaml"
MAP_IMAGE = MAP_DIRECTORY / "navigation_arena.pgm"


def test_reference_map_files_exist():
    """Localization ships both metadata and occupancy image."""
    assert MAP_YAML.is_file(), str(MAP_YAML)
    assert MAP_IMAGE.is_file(), str(MAP_IMAGE)
    assert MAP_IMAGE.stat().st_size > 100


def test_reference_map_metadata_is_portable_and_explicit():
    """The map contains no machine-specific path or implicit threshold."""
    metadata = yaml.safe_load(
        MAP_YAML.read_text(encoding="utf-8")
    )

    assert metadata["image"] == "navigation_arena.pgm"
    assert metadata["mode"] == "trinary"
    assert metadata["resolution"] == 0.05
    assert metadata["negate"] == 0
    assert metadata["occupied_thresh"] == 0.65
    assert metadata["free_thresh"] == 0.25

    origin = metadata["origin"]
    assert len(origin) == 3
    assert all(math.isfinite(float(value)) for value in origin)


def test_reference_map_is_a_portable_pgm_image():
    """Map Server receives an actual PGM, not an arbitrary blob."""
    header = MAP_IMAGE.read_bytes()[:2]

    assert header in {b"P2", b"P5"}
