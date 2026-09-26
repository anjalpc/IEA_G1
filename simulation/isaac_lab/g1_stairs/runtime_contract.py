"""Exact runtime-version contract for reproducible Isaac Lab experiments."""

from __future__ import annotations

EXPECTED_ISAAC_LAB_VERSION = "2.3.0"
EXPECTED_ISAAC_LAB_TAG = "v2.3.0"
EXPECTED_ISAAC_LAB_COMMIT = (
    "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
)
EXPECTED_RSL_RL_VERSION = "3.0.1"


class RuntimeVersionMismatch(RuntimeError):
    """Raised when an installed training dependency violates the freeze."""


def normalize_release_tag(value: str | None) -> str | None:
    """Normalize an exact Git release tag such as ``v2.3.0``."""
    if not value:
        return None
    value = value.strip()
    return value[1:] if value.startswith("v") else value


def validate_runtime_versions(
    *,
    isaac_lab_source_commit: str | None,
    isaac_lab_source_tag: str | None,
    rsl_rl: str | None,
    expected_isaac_lab_commit: str = EXPECTED_ISAAC_LAB_COMMIT,
    expected_isaac_lab_tag: str = EXPECTED_ISAAC_LAB_TAG,
    expected_rsl_rl: str = EXPECTED_RSL_RL_VERSION,
) -> None:
    """Require host-supplied source provenance and the frozen RSL runtime."""
    mismatches = []
    if isaac_lab_source_commit != expected_isaac_lab_commit:
        mismatches.append(
            f"Isaac Lab commit expected {expected_isaac_lab_commit}, found "
            f"{isaac_lab_source_commit or 'unavailable'}"
        )
    if isaac_lab_source_tag != expected_isaac_lab_tag:
        mismatches.append(
            f"Isaac Lab tag expected {expected_isaac_lab_tag}, found "
            f"{isaac_lab_source_tag or 'unavailable'}"
        )
    if rsl_rl != expected_rsl_rl:
        mismatches.append(
            f"RSL-RL expected {expected_rsl_rl}, found "
            f"{rsl_rl or 'unavailable'}"
        )
    if mismatches:
        raise RuntimeVersionMismatch(
            "runtime version preflight failed: " + "; ".join(mismatches)
        )

ARCHIVED_PRE_SPECIALIST_RUNTIME_SECTIONS = (
    "approach_teacher",
    "halfway_reset",
    "physical_step",
    "structured_step",
)


def reject_archived_pre_specialist_runtime(resolved: dict) -> None:
    """Fail closed when an archived controller/reward runtime is requested."""
    enabled = []
    for section in ARCHIVED_PRE_SPECIALIST_RUNTIME_SECTIONS:
        value = resolved.get(section)
        if isinstance(value, dict) and value.get("enabled", False):
            enabled.append(section)

    if enabled:
        raise ValueError(
            "archived pre-specialist runtime section(s) are not executable on "
            "the specialist branch: " + ", ".join(enabled)
        )
