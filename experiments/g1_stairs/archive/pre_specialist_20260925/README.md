# Pre-specialist G1 stair archive

This directory preserves the historical experiment generations that preceded
the clean specialist stair policy. The archived files are evidence and source
history, not active runtime code.

## Scope

- experiment YAMLs 001 through 050, including their smoke/diagnostic variants;
- the 043 physical-step runtime;
- the 044 through 050 structured-step runtime lineage;
- approach-teacher and teacher-action-blend code;
- the halfway-reset experiment path;
- the old controller-port qualification scripts;
- version-specific preflight scripts and tests tied to those generations.

## Active architecture boundary

The active package remains under `simulation/isaac_lab/g1_stairs/` and keeps:

- specialist staging/runtime configuration;
- robot and actuator contracts;
- terrain geometry/state/observations;
- reward registry/runtime;
- generic action, contact, reward, and posture diagnostics;
- checkpoint evaluation, video replay, run manifests, and `simctl`;
- reference-motion infrastructure that may be reused by the specialist skill.

Archived runtime sections intentionally fail closed if an old config is sent
to the active training/replay entry points. Historical recorded runs retain
their own resolved configuration and Git provenance.
