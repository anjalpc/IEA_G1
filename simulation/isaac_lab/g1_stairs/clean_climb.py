"""Clean-climb events on the stair plan (patch 57), for the height curriculum.

A climb is clean when every plan step lands on its tread at the first
touchdown and no foot hits a riser or lands on a tread nose. The rules are
the ones trace_report.py gates on, applied per policy step in training:

- extra step: a plan hit that paid no acquisition because a re-plant
  (missed touchdown, patch 52) or an off-beat step came before it;
- toe strike: horizontal force > STRIKE_N on a foot whose toe is within
  TOE_RISER_BAND_M of a riser line;
- nose landing: a touchdown on a tread with the heel over the tread's front
  edge and horizontal force > STRIKE_N on that foot within NOSE_FRAMES
  frames (touchdown frame included).

Pure tensors; no Isaac imports. Course frame (x along the course).
"""
from __future__ import annotations

import torch

STRIKE_N = 50.0
TOE_RISER_BAND_M = 0.06
NOSE_FRAMES = 4

VIOLATIONS = ("extra_step", "toe_strike", "nose_landing")
EXTRA_STEP, TOE_STRIKE, NOSE_LANDING = range(len(VIOLATIONS))


def _riser_lines(first_riser_x, tread_depth, step_count):
    """[N,S] x of each riser face."""
    index = torch.arange(int(step_count), device=first_riser_x.device, dtype=first_riser_x.dtype)
    return first_riser_x[:, None] + index[None, :] * tread_depth[:, None]


def toe_strikes(sole_bounds, horizontal_force, first_riser_x, tread_depth, step_count, rows):
    """[N] a foot of a row in ``rows`` pushes > STRIKE_N horizontally with its
    toe within TOE_RISER_BAND_M of a riser line.

    ``sole_bounds`` [N,2,6] (heel x, toe x, ...), ``horizontal_force`` [N,2] N.
    """
    toe = sole_bounds[..., 1]
    lines = _riser_lines(first_riser_x, tread_depth, step_count)
    near = ((toe[..., None] - lines[:, None, :]).abs() < TOE_RISER_BAND_M).any(dim=-1)
    return (near & (horizontal_force > STRIKE_N)).any(dim=-1) & rows


def nose_landings(touchdown, sole_bounds, horizontal_force, window, first_riser_x,
                  tread_depth, step_count, rows):
    """Return ([N] nose landing this frame, [N,2] frames left to watch).

    A touchdown (``touchdown`` [N,2]) of a sole whose centre is on a tread
    while its heel is behind that tread's front edge opens a window of
    NOSE_FRAMES frames, this one included; a push > STRIKE_N inside it is a
    nose landing. ``window`` [N,2] long is the state from the last frame.
    """
    heel, toe = sole_bounds[..., 0], sole_bounds[..., 1]
    centre = 0.5 * (heel + toe)
    first = first_riser_x[:, None]
    tread = tread_depth[:, None]
    on_treads = (centre >= first) & (centre < first + int(step_count) * tread)
    edge = first + tread * torch.floor((centre - first) / tread)
    opened = touchdown & on_treads & (heel < edge) & rows[:, None]
    window = torch.where(opened, torch.full_like(window, NOSE_FRAMES), window)
    watching = (window > 0) & rows[:, None]
    pushed = watching & (horizontal_force > STRIKE_N)
    window = torch.where(pushed | ~rows[:, None], torch.zeros_like(window),
                         (window - 1).clamp_min(0))
    return pushed.any(dim=-1), window
