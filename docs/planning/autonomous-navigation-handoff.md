# Autonomous navigation handoff — 2026-11-16

- Repository: /home/herbie/IEA_G1. Migration is complete; do not repeat setup.
- Follow AGENTS.md and the terminal-workflow memory: one bounded operation at a time.
- CPU and GPU Docker targets built and passed fresh-container library checks. GPU selection is automatic; CPU GLIM remains the default estimator.
- RTX RViz pauses with the external monitor connected; disconnecting it removes the pauses and reconnecting restores them. AMD RViz is smooth with both monitors. Do not enable PRIME rendering globally.
- Stationary odometry: approximately 10 Hz. Delay averages: RViz closed 139 ms; RViz without cloud 140 ms; live cloud 145 ms. Maximum intervals: 111 ms without RViz, 112 ms with cloud. These are stationary observations, not walking qualification.
- Current recording-only stack passed localization, fresh planning, actual planned-path command delivery, cancellation, zero commands, and ownership release.
- Upstream silence triggered adapter STOP and release. Adapter termination triggered independent arbiter COMMAND_TIMEOUT at approximately 263 ms after the last command.
- Adapter was manually restarted in a separate foreground terminal. Recovery stream passed; final owner 0, backend 1, ready true, stopped true, reason RELEASED.
- Runtime: PC2 hardware_state and Livox remain running; laptop dry-run launch, AMD RViz, and separately restarted adapter remain running. Check live processes before starting anything; never reuse old PIDs.
- Physical locomotion was not enabled. Next: review physical readiness without activating the Sport helper or authorizing motion.
- Evidence and experimental GPU profiles: ros2_ws/log/glim_gpu_benchmark/ (ignored local artifacts).
- Docker, launcher, documentation, and wired eno1 changes remain uncommitted. Preserve unrelated dashboard edits.

## Physical zero-velocity qualification — 2026-11-16

- All five deployed locomotion files matched laptop SHA-256 checksums.
- PC2 deployment uses SCP. Never run Git on PC2; Git and Graphify belong on the laptop host.
- Operator confirmed harness and emergency-stop readiness before Sport activation.
- Physical backend 2 started successfully with no laptop command producers.
- A silent test lease expired with LEASE_START_TIMEOUT: owner 0, backend 2, ready true, stopped true. This confirms zero-velocity RPC acceptance, not independently measured velocity.
- No nonzero physical velocity command was sent.
- Arbiter shutdown completed; process inspection confirmed no remaining locomotion arbiter or Sport helper.
- Current runtime supersedes earlier notes: laptop dry-run and separate adapter are stopped; PC2 physical arbiter/helper are stopped. PC2 sensor publishers and laptop AMD RViz remain running.
- Physical walking remains unqualified; do not infer walking authorization from these zero-velocity checks.

## Supervised forward smoke — 2026-11-16

- Requested command: forward 0.20 m/s for a 1.0-second publication window, zero lateral/yaw, then explicit zero and lease release.
- CLI forward-limit override appeared in PC2 process arguments but the effective parameter remained 0.30. Root cause remains unresolved; do not rely on that override.
- Dedicated physical_forward_020.yaml was prepared on the laptop and deployed by SCP to /tmp/iea-g1-physical-forward-020.yaml on PC2. Effective cap 0.20 was verified before execution.
- Local probe and four software timing/cleanup tests are in ros2_ws/log/glim_gpu_benchmark/. Tests passed; preview ran before explicit execution.
- Operator observed G1 step forward and stop stably. Actual velocity, travel distance, and physical stop latency were not independently measured.
- Final state: owner 0, backend 2, ready true, stopped true, reason RELEASED.
- Physical arbiter was stopped afterward; process inspection returned NONE for arbiter and Sport helper.
- Autonomous path following and moving-state odometry qualification remain pending. Do not treat this single supervised test as autonomous navigation qualification.
- Workflow: Codex supplies one short paste-ready operation; user performs edits, tests, deployment, and runtime actions. Use grep in user-facing commands. No Git on PC2; deploy through SCP.

## Moving-state qualification — 2026-11-16 — NOT PASSED

- Recorded repeat: 0.20 m/s, one-second command window; operator observed stable stopping. Final state RELEASED. Physical arbiter/helper subsequently stopped; process check NONE.
- Capture: ros2_ws/log/glim_gpu_benchmark/moving-state-forward-020-1s-run (136.45 s, 701.2 MiB).
- GLIM raw, validated, and planar streams stayed continuous near 10 Hz. Pelvis IMU maximum gap was 40 ms.
- Physical-body staging and canonical odometry lost seven samples: 0.800 s gap beginning 0.662 s after first accepted nonzero command.
- Pelvis acceleration norms 22.33, 17.08, and 16.12 m/s² exceeded the configured 15 m/s² limit. Quaternion, angular velocity, frame, and covariance checks passed for these samples.
- Offline replay through the unchanged health gate reproduced invalidation and repeated 20-sample warmup resets; readiness recovered at relative time 1.571 s.
- Next: design and regression-test motion-appropriate pelvis IMU qualification. Do not weaken thresholds merely to pass this recording. Autonomous path following remains blocked.
- Every new laptop ROS terminal needs explicit domain 42 and the physical wired CycloneDDS profile; the generic profile prevented sensor discovery after lunch.

## Moving-state continuity fix and live retest — 2026-11-16

- Isolated high-acceleration samples are rejected without refreshing freshness or resetting an already-qualified fresh stream. Rejected samples never enter the staging history.
- Existing sample bounds, startup warmup, freshness, synchronization, and hard-fault checks remain enforced. Contract and configuration validation updated/passed.
- Regression tests passed; complete state-estimation package: 133 passed. Supported ROS workspace build/test gate passed.
- Original-bag replay: 60/60 staging and authority samples, max gap 100.3 ms, max IMU offset 11.2 ms, zero rejected samples used.
- Live retest bag: ros2_ws/log/glim_gpu_benchmark/moving-state-forward-020-1s-fixed.
- Accepted nonzero-to-zero interval 1.0003 s. All five odometry stages produced 60 samples with max gap 100.8 ms and no nonadvancing timestamps or gaps above 150 ms.
- Pelvis acceleration peaked at 21.0 m/s². Canonical estimated delta: +0.1332 m X, +0.0092 m Y; maximum recorded header age 141.4 ms. Independent distance accuracy remains unmeasured.
- Final physical state RELEASED; operator confirmed stability. Arbiter/helper shutdown verified NONE. Read-only odometry and PC2 sensors remain running.
- Continuity passed for this bounded supervised test. Moving saved-map localization and autonomous path following remain pending.

## Wi-Fi mapping checkpoint — 2026-11-18

- Full physical RViz model works with 29 measured body joints; BrainCo finger visuals are static, with relaxed thumbs. No finger telemetry or control was added.
- Trial scan ceiling is 1.50 m for upright, arms-down mapping; raw clouds remain unchanged.
- Direct Wi-Fi fanout produced cloud p95 age 700 ms and odometry gaps up to 800 ms. A single cloud relay substantially improved delivery; reliable local GLIM cloud input removed gaps in the subsequent bounded tests. This does not isolate every causal factor.
- Manual-forward bag: ros2_ws/log/glim_gpu_benchmark/wifi-relay-manual-forward-check. All five odometry stages: 258 samples, max header gap 100.758 ms, no gaps over 150 ms or nonadvancing stamps. Canonical p95/max age: 155.333/170.687 ms. Scan max arrival gap/age: 211.692/260.876 ms.
- Estimated movement: 0.326 m, yaw change 1.266 degrees; physical distance not independently measured.
- Reproducibility work: tracked Wi-Fi DDS profiles, glim_mid360_wifi profile, topic_tools runtime/container dependency, and physical_wifi_mapping.launch.py. Bringup source tests passed; new-entry-point runtime verification and image rebuild remain pending.
- Current runtime uses the manual relay plus mapping with the earlier log-directory reliable profile. Stop both before testing the new combined launch. PC2 sensors remain read-only. No project locomotion backend was started.
- Next: install/test the combined entry point, save a persistent PC2 profile through SCP, update graphify, and perform longer supervised mapping/route tests. Autonomous path following and measured communications-loss stopping remain unqualified.

## Reproducible Wi-Fi entry point validated — 2026-11-18

- physical_wifi_mapping.launch.py was built and started successfully. Runtime checks confirmed one transport publisher/subscriber, one canonical cloud publisher named g1_wifi_cloud_relay, GLIM reliable depth-5 cloud input, and the installed glim_mid360_wifi profile.
- PC2 Wi-Fi XML was deployed persistently by SCP to /home/unitree/IEA_G1/livox_ws/config/cyclonedds_pc2_wifi.xml. Existing sensor processes may still reference the equivalent temporary file until restart.
- Docker rebuild initially failed because September Nav2 binary pins were unavailable. All 35 pins were refreshed to available November rebuilds of upstream 1.1.20 after a simulation with no additions/removals. The working container was not upgraded.
- Rebuilt GPU image verified all 35 exact pins and the relay dependency. Fresh-image full ROS build/test gate passed after correcting the isolated validation layout to copy, rather than symlink, the simulation directory. No test assertion was weakened.
- Current live mapping still runs in the older container. Next: controlled laptop-container restart into the validated image, rebuild/source its overlay, verify stationary live operation, then longer supervised mapping/coverage tests. Unrestricted autonomous navigation and measured Wi-Fi-loss stopping remain unqualified.

## Walking acceleration dips — offline fix validated — 2026-11-18

- Bag: ros2_ws/log/glim_gpu_benchmark/wifi-room-mapping-20261118-retry.
- Raw Wi-Fi clouds and GLIM remained continuous throughout the recorded route.
- Brief pelvis acceleration dips below 4 m/s² reset qualification; subsequent spikes restarted warmup. Staging gaps reached 1.3 s and canonical gaps reached 3.5 s.
- Qualified fresh streams now discard both low and high out-of-range acceleration without refreshing freshness or buffering rejected measurements. Startup qualification, 100 ms freshness, 50 ms synchronization, and hard-fault checks remain enforced.
- Regression tests cover brief dips, sustained invalid acceleration, rejected-sample exclusion, and unchanged timestamp matching.
- Full-bag offline replay: staging 2377 samples; authority accepted 2358 samples; both maximum gap 103.472471 ms, zero gaps over 150 ms, zero nonadvancing timestamps. Counts include fresh startup qualification.
- State-estimation package tests, configuration validation, and full ROS workspace build/test gate passed.
- Pending: graphify update and supervised live manual-walking retest after restarting laptop mapping with the corrected code. Robot may remain off until then. No PC2 deployment or autonomous motion is required for this fix.

## Walking acceleration fix — live continuity passed — 2026-11-18

- Bag: ros2_ws/log/glim_gpu_benchmark/wifi-walking-dip-fix-retest.
- Operator manually walked the robot around and returned to the starting point.
- All nine analyzed streams had zero header gaps over 150 ms and zero nonadvancing timestamps.
- Canonical odometry: 1437 samples, maximum header gap 106.214 ms, maximum arrival gap 156.278 ms, p95 age 168.684 ms, maximum age 223.194 ms.
- Scan: 1436 samples, maximum arrival gap 216.219 ms, p95 age 263.686 ms, maximum age 278.627 ms.
- The prior 1.3-second staging and 3.5-second canonical interruptions did not recur in this supervised run.
- Recording stopped cleanly. Mapping and PC2 sensors remain running; no project locomotion backend was started.
- Next: inspect and save the updated map, then validate saved-map localization and planning before autonomous motion. Independent accuracy and communications-loss stopping remain unqualified.

## Depth coverage checkpoint — 2026-11-18

- Map candidate saved: building_map_wifi_walkfix_20261118.yaml and .pgm.
- Operator observed a low box disappearing from nearby raw LiDAR but appearing farther away; exact visibility boundary remains unmeasured.
- D435i paired snapshots show valid depth on the visible low-box surface at operator-reported 50 cm and 20 cm forward distances. Box partly cropped; side/rear crane-bar coverage remains unproven.
- Snapshots preserved under ros2_ws/log/glim_gpu_benchmark/depth-coverage/.
- D435i driver recovered from startup errors; depth published near 30 Hz with occasional 204 ms intervals. Camera motion-module faults were reported; its IMU is not used for navigation.
- Live base_footprint to d435_depth_optical_frame TF available: translation approximately [0.072, 0.004, 1.257] m. This is observed TF, not independent calibration proof.
- Depth is not connected to Nav2. Next: offline depth back-projection and floor-relative obstacle-height checks before designing live perception integration.
- Operator plans normal shutdown. Recheck processes and sensor readiness next session; do not assume any launch remains running.

## Offline depth geometry — 2026-11-19

- Added pure depth back-projection and signed floor-plane heights in g1_navigation_perception; no ROS node or live integration.

- Six geometry tests passed. Tested conversion matches all 394894 saved 50 cm capture points.

- Refined-plane comparison: baseline 0 raised points; low-box capture 18786 raised points, median height 0.165 m, within the selected forward region and 0.08–0.20 m height band.

- Floor plane was fitted offline; snapshot-time robot TF and independent calibration remain unverified. This is not live obstacle-avoidance qualification.

- Next: verify synchronized depth calibration and camera-to-base transforms before live perception integration.

## Grounded depth and TF experiment — 2026-11-19

- Temporary PC2 profile: depth only, 848x480 at 15 Hz; color, infrared and camera IMU disabled. CLI overrides were ineffective; a temporary YAML applied correctly. Persistent camera YAML unchanged.

- Initial full-stream run reported frame-resource exhaustion. Reduced-stream delivery measured approximately 15 Hz; long-duration reliability remains unqualified.

- Suspended capture retained separately: depth-tf-stationary-20261119. Grounded captures: depth-tf-grounded-20261119, depth-tf-box12cm-50cm-20261119, depth-tf-box12cm-20cm-20261119, under ros2_ws/log/glim_gpu_benchmark/.

- Grounded snapshot: exact depth/CameraInfo stamp match, image-time camera-to-pelvis TF lookup succeeded, nearest pelvis IMU offset 6.166 ms.

- Floor estimate across three frames: camera distance 1.2897–1.2911 m. Single-frame floor/IMU normal disagreement 2.18 degrees; inferred pelvis height 0.8205 m. Configured 0.785 m remains unchanged pending calibration investigation.

- Operator measured box height 0.12 m; placements approximately 0.50 m and 0.20 m from toe tips to nearest box face.

- Common camera-floor region: forward 0–1.2 m, lateral +/-0.4 m. Counts at 0.04–0.16 m height: baseline 583; box50 12328; box20 15981.

- Low-return median forward positions relative to camera floor projection: baseline 1.118 m; box50 0.534 m; box20 0.336 m. These are not toe-referenced distance measurements.

- Stationary forward-box visibility supported. Moving obstacle retention, thin bars, side/rear coverage, measured clearance and autonomous avoidance remain unqualified. Depth is not connected to Nav2.

## Depth reference-frame and nearby-floor checks — 2026-11-19

- Recorded camera chain matches the expected model: torso_link to d435_link uses the pinned Unitree mount; d435_link to depth_frame is identity; depth optical rotation is (-0.5, 0.5, -0.5, 0.5). No extra transform identified; physical extrinsic accuracy remains unproven.

- Upright target: approximate horizontal tape distance 1.00–1.01 m; fitted face perpendicular distance 1.0126 m, p95 plane residual 12.66 mm. This rough check does not establish full depth calibration. Scale unchanged.

- Distant-region plane intercepts were not independent camera-height measurements. Unfiltered distant regions contained many elevated returns; the earlier masked preview showed only selected near-plane points.

- Nearby baseline floor region: 0.4–0.8 m forward, lateral +/-0.4 m; 99.5 percent of points within 20 mm of the original reference plane.

- Using the refined nearby floor and a 0–0.8 m forward region: baseline zero raised returns; box50 11445 returns with height p10/p50/p90 5.9/10.8/12.0 cm; box20 15281 returns at 6.4/11.7/12.7 cm. Selection height band 4–18 cm; percentiles describe selected surfaces, not independent box-height accuracy.

- Cross-capture comparison assumes unchanged robot pose. No calibration, standing-height constant or live navigation input was changed. Next: read-only depth-obstacle output with timestamp, calibration and transform checks.

## Read-only depth preview — 2026-11-19

- Robot stayed in place; operator moved only the box. Eight geometry tests and recorded transform height-invariance check passed.

- depth_image_proc installed in the current container; diagnostic cloud measured approximately 15 Hz and displayed in RViz with fixed frame pelvis.

- Added depth_preview.launch.py and declared package/Docker dependencies. Perception package build/tests and installed launch passed. Updated-image rebuild and broader workspace gate pending.

- Full cloud only: no floor filtering, obstacle qualification or connection to Nav2. Next: complete checkpoint validation and save it before adding diagnostic obstacle filtering.

- Checkpoint validation: full ROS workspace 994 tests, 0 errors, 0 failures, 2 skipped; configuration validator and graphify update passed. Updated Docker-image rebuild remains pending.

## Automatic floor-selection checkpoint — 2026-11-19

- Twelve geometry tests and the perception package build/test gate passed.

- Selection uses recorded points, timestamped camera-to-pelvis TF and nearest pelvis IMU gravity; no saved floor plane is supplied.

- Middle-frame candidate pelvis distances: baseline 0.8273 m; box50 0.8298 m; box20 0.8223 m. Raised counts in the selected region: 0, 12167, 15631 respectively. Selected-surface height p90: 12.6 cm at box50 and 12.9 cm at box20.

- Baseline selection elapsed 11.4 ms in one offline observation; not a runtime benchmark. Results saved in ros2_ws/log/glim_gpu_benchmark/automatic-floor-comparison-20261119.json.

- Distance bounds 0.65–0.95 m were experimental replay settings, not universal walking limits. Ground identity, live freshness/TF handling and moving-operation robustness remain unqualified.

- Next: read-only live diagnostic filtering with explicit unknown status and stale-output clearing. No Nav2 obstacle input or autonomous motion has been enabled.


## Depth preview CPU checkpoint — 2026-11-19

Live diagnostic floor selection now recovers the recorded near-wall view.
The original full-region selection is preserved; height-prior filtering is
a fallback only. Manual movement recovered without a new floor reference,
but a roughly 0.41-second stale interruption remains unresolved.

Next: profile CPU stages on saved recordings, then benchmark a batched GPU
candidate including transfers and synchronization. Compare plane/obstacle
outputs as well as latency. Retain CPU fallback and existing timestamp
checks. No Nav2 obstacle input or robot command integration is authorized
by this diagnostic checkpoint.

Evidence: ros2_ws/log/glim_gpu_benchmark/depth-wall-floor-20261119-133753
contains three wall-frame NPZ snapshots and the source recording.
Earlier floor regression fixtures are in the perception package tests.
Updated thesis notes outside the repository remain to be appended.

## CUDA and decimated depth checkpoint — 2026-11-19

Offline CUDA geometry: median 15.554 ms versus CPU 59.717 ms on three wall
frames, including GPU transfers. Sixteen recorded module comparisons passed;
nine baseline/box masks matched exactly. Live CUDA initially still stalled.
PC2-local camera timing was steady; laptop delivery improved after
magnitude-2 depth decimation to 424x240 at 15 Hz. Box appearance/removal passed.

Next: build/validate the updated GPU image without replacing the live
container, install/deploy the opt-in camera profile via SCP (no PC2 Git),
check installed launcher and profile, append thesis evidence outside the
repository, then review, commit and push. Do not connect this diagnostic
cloud to Nav2 until separately validated.

### Reproducibility verification completed

The updated GPU image built and passed a disposable-container CUDA floor
fitting check with CuPy 14.2.0, NumPy 2.2.6 and CUDA runtime 12.9.
Both affected package gates passed and pip reported no broken requirements.
Thesis addendum and copied evidence:
~/Documents/ChatGPT/G1_Navigation_Thesis/2026-11-19_Depth_GPU_Checkpoint.md

Remaining: deploy the saved camera profile via SCP, verify the reusable
launcher in a fresh runtime, and validate Nav2 obstacle/stopping integration.
The currently running container was not replaced by the image build.
