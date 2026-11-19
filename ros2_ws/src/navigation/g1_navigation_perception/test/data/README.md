# Floor regression data

- floor_refinement_failure.npz: deterministic sampled points from the manual-reposition failure captured in floor-failure-1795089053787169361/failed.npz.
- floor_flicker_sequence.npz: six sampled frames from depth-floor-flicker-20261119/replay-frames, including four rejected frames and two accepted frames. These are regression examples, not independent validation data.
- Coordinates are metres in pelvis; up vectors come from timestamp-matched pelvis IMU. Raw captures remain under ros2_ws/log/glim_gpu_benchmark. Fixtures contain no RGB images.
