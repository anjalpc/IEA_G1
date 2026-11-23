# Floor regression data

- floor_refinement_failure.npz: deterministic sampled points from the manual-reposition failure captured in floor-failure-1795089053787169361/failed.npz.
- floor_flicker_sequence.npz: six sampled frames from depth-floor-flicker-20261119/replay-frames, including four rejected frames and two accepted frames. These are regression examples, not independent validation data.
- Coordinates are metres in pelvis; up vectors come from timestamp-matched pelvis IMU. Raw captures remain under ros2_ws/log/glim_gpu_benchmark. Fixtures contain no RGB images.

## Open-hand self returns — 2026-11-23

hand_self_returns.npz contains only 214 selected walking-time obstacle points,
exact-image-time left/right wrist transforms and the nominal wrist-local hand
bounds. Source: manual-depth-source-1795184167261279537, marking frames 16–20.
Points use base_footprint coordinates; hand poses map wrist to that frame.
These frames were used to select the candidate margin, so this is a regression
fixture, not independent validation of arbitrary hand poses or nearby objects.
The full source bag and baseline/12 cm box controls remain in ignored logs.
