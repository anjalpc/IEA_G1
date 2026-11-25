# IEA G1 system contract

This is the local quick reference for the physical robot, simulation, SDK,
BrainCo hands, and stair-climbing RL task. It is intended to eliminate routine
dependence on upstream websites. The machine-readable YAML files linked below
remain authoritative when this summary and a configuration disagree.

Last reviewed: 2026-11-03

## Robot identity

| Property | Project value |
|---|---|
| Manufacturer | Unitree |
| Family | G1 |
| Variant | G1 EDU U7 |
| Machine type (`mode_machine`) | **5** |
| Body degrees of freedom | 29 |
| Hip pitch / roll gear ratio | 14.3 / 22.5 |
| Wrist motor generation | 4010 |
| Waist | 3 active joints |
| Hand system | Dual BrainCo Revo2 Basic |
| Hand channels | 6 per hand; 12 total |
| Locomotion policy outputs | 29 body joints only |

`mode_machine` is a physical hardware identifier read from
`unitree_hg/LowState`; it is not a software-selectable locomotion mode. A
mode-15 policy or robot model must not be substituted for this mode-5 robot.
Mode 15 uses 22.5/22.5 hip gearing and newer 5010 wrist motors.

Authoritative local configuration:

- [`simulation/mujoco/config/robot.yaml`](../../simulation/mujoco/config/robot.yaml)
- [`simulation/mujoco/config/joint_mapping.yaml`](../../simulation/mujoco/config/joint_mapping.yaml)
- [`simulation/isaac_lab/g1_stairs/sources.yaml`](../../simulation/isaac_lab/g1_stairs/sources.yaml)

## Timing, frames, and policy interface

| Contract | Value |
|---|---|
| Physics rate | 200 Hz (`dt = 0.005 s`) |
| Policy/control rate | 50 Hz (`step_dt = 0.02 s`) |
| Policy input | `[1, 480]` for the pinned reference velocity policy |
| Policy output | `[1, 29]` |
| World frame | Right-handed, +Z up |
| IMU quaternion order | `wxyz` |
| Angular velocity frame | Body |
| Initial velocity command | Zero |
| Body message family | `unitree_hg` |
| Body command topic | `rt/lowcmd` |
| Body state topic | `rt/lowstate` |
| Motor mode | 1 |

The trained stair policy must retain the 29-output action order below and must
export to ONNX before MuJoCo sim-to-sim evaluation.

### MuJoCo navigation-state source

`g1_simulation_bridge` publishes measured MuJoCo state for the
workstation navigation stack:

| Topic | Type | Cadence | Frames |
|---|---|---:|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | 200 Hz | Simulation time |
| `/g1/state/imu_raw` | `sensor_msgs/msg/Imu` | 50 Hz | `pelvis` |
| `/g1/state/odometry_raw` | `nav_msgs/msg/Odometry` | 50 Hz | `odom` to `pelvis` |
| `/joint_states` | `sensor_msgs/msg/JointState` | 50 Hz | Three movable waist joints |

The source resolves `floating_base_joint`, `imu_quat`, `imu_gyro`,
`imu_acc`, and the three waist joints by MuJoCo name rather than
fixed array offsets. The joint-state order is
`waist_yaw_joint`, `waist_roll_joint`, then `waist_pitch_joint`.
Only finite scalar hinge positions are accepted. The bridge does not
publish default, commanded, or fabricated joint values.

MuJoCo `wxyz` quaternions are converted to ROS `xyzw`. Global
free-joint linear velocity is rotated into the pelvis frame, while
body angular velocity remains in the pelvis frame.

Clock, raw IMU, and raw odometry use volatile, best-effort,
depth-one QoS. `/joint_states` uses volatile, reliable, depth-one
QoS for compatibility with `robot_state_publisher`. All state
messages released in one cycle use the same MuJoCo sample and
simulation timestamp, not wall time or render cadence. Period,
frames, and the nonzero covariance floor are configured in
`ros2_ws/src/g1_simulation_bridge/config/simulation.yaml`.

This source publishes neither TF nor fused state. The
`g1_state_estimation` package remains the validation and health
boundary before future fusion and Nav2 consumption.

### Simulated MID-360 navigation source

When enabled in
`ros2_ws/src/g1_simulation_bridge/config/simulation.yaml`,
`g1_simulation_bridge` also publishes deterministic MuJoCo lidar
observations:

| Contract | Value |
|---|---|
| Topic | `/g1/sensors/lidar/points` |
| Type | `sensor_msgs/msg/PointCloud2` |
| Frame | `mid360_link` |
| Cadence | 10 Hz of simulation time |
| Fields | Little-endian float32 `x`, `y`, `z` |
| QoS | Best effort, volatile, depth one |
| Range | 0.25 m through 40.0 m |

The source composes the live `torso_link` pose with the official
fixed MID-360 mount. Rays originate in `mid360_link`, are rotated
into the MuJoCo world for intersection tests, and accepted returns
are published back in `mid360_link`.

The initial navigation-integration grid uses 0.5-degree azimuth
spacing and elevation rows at -20, 0, and +20 degrees, producing
2160 candidate rays per cloud. This deterministic grid is not the
physical Livox firing pattern and must not be presented as sensor
fidelity validation.

Robot geometries are identified from the MuJoCo `pelvis` body
subtree and explicitly skipped. Self-filtering is mandatory whenever
the source is enabled. Environment geometries may be unnamed; their
numeric MuJoCo identity and body ownership remain authoritative.

Cloud scheduling uses completed physics-step simulation time, not
wall time or viewer cadence. Computational raycasting may make
simulation progress slower than real time while preserving the
10 Hz simulation-time cadence.

The project-owned navigation integration arena is declared in
`simulation/mujoco/navigation/navigation_arena.py` and is composed
after loading the vendored Unitree scene. It contains the named
geometries `navigation_wall_north`, `navigation_wall_east`, and
`navigation_pillar_southwest`. These deterministic objects provide
finite vertical lidar returns without modifying vendor files. They
are an integration fixture, not a model of a physical building.

The ROS MuJoCo backend initializes the G1 under an enabled elastic
overhead harness and holds the imported initial joint pose with
joint-space PD control. It does not run the Unitree balance or walking
policy. The robot may therefore appear suspended with an L-shaped arm
posture. This runtime proves sensors, TF, mapping, and interaction
contracts; it does not prove unsupported standing or autonomous
walking.

The simulated source and physical Livox source are mutually
exclusive. Exactly one publisher may own
`/g1/sensors/lidar/points` in a running system. Never enable the
simulated source while the physical MID-360 driver is publishing.

`g1_simulation_bridge` publishes neither `/scan` nor lidar TF.
`g1_navigation_perception` owns the lazy conversion to `/scan`, and
`g1_robot_description` owns the fixed `torso_link -> mid360_link`
transform.


### Physical MID-360 and guarded GLIM odometry

Physical navigation sensing is split across PC2 and the workstation. PC2 runs
only the isolated Foxy `livox_ros_driver2` sensor driver. The workstation runs
CPU GLIM, health qualification, mapping, and navigation. The interaction stack
continues owning the D435i, handshake, card, arms, and hands.

PC2 publishes these observations on ROS application domain `42`:

| Topic | Type | Frame | Measured cadence |
|---|---|---|---:|
| `/g1/sensors/lidar/points` | `sensor_msgs/msg/PointCloud2` | `mid360_link` | 10 Hz |
| `/g1/sensors/lidar/imu_raw` | `sensor_msgs/msg/Imu` | `livox_frame` | 200 Hz |

Project-owned wired CycloneDDS configurations live under
`g1_navigation_bringup/config/physical`. They bind the workstation to
`eno1` at `192.168.123.99` and PC2 to `eth0` at
`192.168.123.164`, disable multicast, and use explicit peers. Both sides use
automatic participant indices through `63`; PC2 lists both the workstation
and itself so cross-machine discovery and local diagnostics use predictable
unicast ports. The PC2 configuration is restricted to ROS domain `42`. It
must not apply its multicast-disabled application settings to the child
native Unitree domain-0 LowState helper. The physical interaction entry point
pins this PC2 wired configuration explicitly and does not inherit an ambient
`CYCLONEDDS_URI`.

The workstation uses pinned GLIM 1.2.2 packages with CPU odometry selected by default. GLIM consumes the
raw cloud and MID-360 IMU, uses absolute nanosecond per-point timestamps,
applies the Livox acceleration scale `9.80665`, and runs odometry estimation
with local and global GLIM mapping disabled. SLAM Toolbox or AMCL, not GLIM,
will eventually own `map -> odom`.

GLIM publishes an isolated `glim_odom -> livox_frame` estimator tree. The
`g1_glim_health` boundary requires advancing and fresh cloud, IMU, and
odometry samples; exact frames; finite poses; unit quaternions; a 20-sample
warmup; and bounded 10 Hz pose steps. Limits are 0.35 s odometry age, 0.35 s
cloud age, 0.10 s IMU age, 0.15 m translation per update, and 15 degrees
rotation per update.

Only while every check passes does the boundary copy the unchanged sample to
`/g1/navigation/glim/odometry_validated`. This is an intermediate output
only.

`g1_glim_planar` consumes only that qualified intermediate output. It
captures the first accepted pose as a local origin, corrects the measured
upside-down MID-360 axes, removes vertical motion, and publishes
start-relative `odom -> base_footprint` messages on
`/g1/navigation/glim/odometry_planar`. Forward remains positive X, while
sensor Y and Z are reversed. No empirical distance or angle scaling is
applied.

The planar topic remains staged. It must not be remapped directly to
`/g1/state/odometry`. The `g1_physical_body` staging node combines that
planar pose with the qualified physical pelvis IMU. It requires a 20-sample
IMU warmup, retains at most 100 timestamped IMU samples and 20 pending
odometry samples, and selects the nearest IMU header stamp without exceeding
the strict 0.05-second synchronization limit. It publishes the resulting
`odom -> pelvis` message only on
`/g1/navigation/physical/odometry_staged`, with calibrated standing pelvis
height `0.785 m`; it publishes neither TF nor canonical odometry.

The physical odometry launch also starts the physical robot-state runtime.
It consumes measured `/joint_states` and publishes only the kinematic subtree
rooted at `pelvis`. The physical runtime applies the verified upside-down
MID-360 mount: 180 degrees about local X followed by the official
0.04014257279586953-radian pitch.

The physical odometry launch must not publish
`odom -> base_footprint -> pelvis` or start Nav2, SLAM, localization,
locomotion, or robot-command nodes.

The separate opt-in `physical_odometry_authority.launch.py` includes that staging stack and adds exactly one final authority boundary. It requires exact `odom -> pelvis` staged frames, finite state, a unit quaternion, strictly advancing timestamps, a 20-sample warmup, and freshness within `0.25 s`. Motion is limited to at most `0.15 m` or 15 degrees per update, relaxed only according to elapsed source time up to `0.5 m/s` and 60 degrees/s. A source timestamp gap greater than `2.0 s` fails closed. A finite, correctly framed sample rejected for an excessive motion step or timestamp gap becomes a non-published comparison baseline; the boundary then requires its complete warmup again before restoring authority. Invalid frames, non-finite state, invalid orientation, and non-advancing timestamps never become baselines. Only while every check passes does it publish canonical planar `odom -> base_footprint` messages on `/g1/state/odometry` and the matching `odom -> base_footprint -> pelvis` TF chain. The canonical message has zero height and no roll or pitch; measured pelvis height and tilt remain on the second TF edge. This launch still starts no mapping, Nav2, localization, locomotion, or robot-command node.

The separate opt-in `physical_mapping.launch.py` composes the qualified
physical odometry authority, MID-360 point-cloud-to-scan projection, and
asynchronous SLAM Toolbox mapping. It uses wall-clock time. SLAM Toolbox
consumes `/scan`, publishes `/map`, and exclusively owns `map -> odom`.
The launch starts no localization, Nav2 planning or control, locomotion
lease, velocity command, or robot-command node.

During physical motion, keep RViz PointCloud2 disabled. Rendering the full
MID-360 cloud alongside CPU GLIM and SLAM caused multi-second estimator
stalls during validation. Map, LaserScan, TF, and RobotModel displays remain
suitable for live operation. Stopping the foreground launch removes mapping
and canonical odometry authority together.

The separate opt-in `physical_localization.launch.py` composes the
qualified physical odometry authority, MID-360 scan projection, Map Server,
and AMCL. It requires the absolute path of a saved physical occupancy-map
YAML and uses wall-clock time. AMCL exclusively owns `map -> odom`.

Physical mapping and physical localization are mutually exclusive because
SLAM Toolbox and AMCL would otherwise compete for `map -> odom`. The
localization launch starts no SLAM Toolbox mapping, Nav2 planning or control,
locomotion lease, velocity command, or robot-command node. The simulation
`navigation_arena` map must never be used as the physical building map.

PC2 system time must be synchronized before starting the Livox driver. Starting it across a large system-clock correction can defeat scan aggregation and produce approximately 2 kHz packet-sized, 96-point clouds. A healthy MID-360 stream is approximately 10 Hz with about 20,000 points per cloud. Restart the driver after time synchronization and do not start GLIM until that shape is restored.

GLIM is configured with `publish_imu2lidar` disabled. Therefore GLIM owns
only its isolated `glim_odom -> livox_frame` estimator tree, while the
physical robot-state runtime owns the separate
`pelvis -> torso_link -> mid360_link` kinematic tree. No transform connects
`livox_frame` to `mid360_link` at this stage; this deliberately prevents
either sensor frame from acquiring a duplicate parent.

The validated GLIM build publishes finite but all-zero covariance. These
arrays remain unknown; the project neither invents covariance nor weakens the
generic navigation-state validator. The dedicated GLIM health boundary
provides source-specific liveness and plausibility checks, but its validated
topic remains an intermediate input to the final physical authority boundary.

A suspended stationary proof produced approximately 10 Hz qualified output,
about 1.4 mm translation drift over 59.8 seconds, and about 0.0034 degrees of
yaw drift. A grounded proof measured 0.4678 m for a marked 0.50 m forward
movement, with 0.0296 m lateral displacement, 0.0651 m vertical displacement,
and 2.733 degrees of unintended rotation. A physical 90-degree left turn
produced 81.184 degrees of GLIM rotation, 0.1197 m translation, and bounded
tilt.

These results qualify direction and approximate metric scale for staged
odometry. Subsequent authority and mapping proofs authorize
`/g1/state/odometry` and read-only mapping only. They do not prove extended
walking robustness or authorize localization, Nav2, or locomotion.

After pelvis-IMU warmup, an isolated out-of-range acceleration sample with otherwise
valid fields is discarded while fresh qualified history remains available.
Discarding it neither refreshes the last-good receipt time nor counts it
toward warmup. Only accepted samples enter the timestamp-matching history.
The 4–15 m/s² per-sample acceleration bounds, 20-sample startup warmup,
100 ms freshness limit, and 50 ms synchronization limit remain unchanged.
Malformed data, invalid orientation, excessive angular
velocity, and nonadvancing timestamps retain their invalidating behavior.
Out-of-range acceleration at startup or after freshness expires still
invalidates qualification. Repeated rejected samples cannot extend freshness.
No rejected measurement, fabricated attitude, or widened synchronization
window is used to fill a gap.

### Physical LowState source

On PC2, `g1_hardware_bridge` owns one read-only physical source for measured
body-joint and pelvis IMU state. The isolated native helper subscribes once to
Unitree `rt/lowstate` on DDS domain `0`; it creates no command publisher and
never writes robot state.

Each accepted native tick produces one atomic sample containing:

- all 29 measured body positions, plus retained legacy waist fields;
- pelvis IMU quaternion in Unitree `wxyz` order;
- pelvis angular velocity;
- pelvis linear acceleration.

The supervising `g1_hardware_state` ROS node runs on application domain `42`.
It converts the quaternion to ROS `xyzw` order and publishes:

| Topic | Type | Frame | QoS |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | standard joint names | reliable, volatile, depth one |
| `/g1/state/imu_raw` | `sensor_msgs/msg/Imu` | `imu_in_pelvis` | best effort, volatile, depth one |

The hardware indices for mode-machine `5` are:

| SDK index | Joint |
|---:|---|
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |

Full-body joint output follows `sdk_order` in
`simulation/mujoco/config/joint_mapping.yaml`. Legacy payloads retain
waist yaw, roll, then pitch. All motor and IMU values must be
finite, the quaternion must have unit length within the declared tolerance,
and the native tick must advance. Invalid IMU data rejects the entire atomic
sample rather than releasing partial joint state.

Publication is rate-limited to 50 Hz. The native monitor thread writes into a
depth-one latest-sample handoff; the ROS executor thread consumes that handoff
and publishes both messages with one supervising-node receipt timestamp.
Overwritten intermediate native samples are intentionally dropped rather than
queued as stale state. There are no default, commanded, interpolated,
integrated, or replayed values.

Physical IMU covariance has not been measured. The raw IMU therefore publishes
zero covariance arrays to mean unknown covariance. `g1_state_estimation`
continues rejecting that raw source as `COVARIANCE_REQUIRED`; physical
calibration or estimator-specific covariance qualification is required before
physical navigation may consume it.

The native helper is deliberately separated from application ROS so Unitree
DDS domain `0` cannot collide with ROS application domain `42`. The parent
treats 0.5 seconds without an accepted sample as stale, stops and reaps the
helper, waits 3.0 seconds for native DDS cleanup, and then restarts it. A valid
sample resets the consecutive-restart count. After three unsuccessful
restarts, the parent requests orderly main-loop shutdown; ROS shutdown and
helper cleanup occur only in the outer `finally` path. Downstream consumers
therefore receive neither frozen nor fabricated state.

### Physical navigation-state probe

`g1_hardware_bridge` provides the manual `physical_state_probe` command for
time-bounded, read-only inspection on PC2. It uses native Unitree DDS domain
`0`, creates subscribers only, and never starts automatically from bringup.

The approved candidates are `rt/lowstate`, `rt/secondary_imu`, and
`rt/sportmodestate`. A run emits exactly one `PHYSICAL_STATE_PROBE=` JSON line
containing sanitized metadata only. Raw measurements, network interfaces,
addresses, identifiers, and exception details are forbidden from the output.

A `VALID` result proves only that the candidate was available, finite,
sufficiently sampled, and advancing during the bounded run. It does not
authorize that stream as navigation input. In particular,
`rt/sportmodestate` may disappear when Unitree's built-in motion service is
disabled. Commanded velocity and IMU-only dead reckoning remain forbidden
substitutes for measured planar odometry.

### Validated navigation-state boundary

`g1_state_estimation` subscribes to `/g1/state/imu_raw` and
`/g1/state/odometry_raw`. Only while both streams are valid and fresh,
it publishes:

| Output | Contract |
|---|---|
| `/g1/state/odometry` | Unchanged validated `odom` to `pelvis` odometry |
| `/tf` | `odom` to `base_footprint` and dynamic `base_footprint` to `pelvis` transforms |

The planar transform preserves the validated pelvis `x`, `y`, yaw,
and source timestamp. It sets translation `z` to zero and removes
roll and pitch so walking-body oscillation does not become Nav2's
robot-base pose. Invalid, missing, or stale input suppresses both
navigation-facing odometry and TF.

The default stale timeout is `0.1 s`, equal to five expected samples
at 50 Hz, and is configured in
`ros2_ws/src/navigation/g1_state_estimation/config/state_estimation.yaml`.

The two TF edges come from the same validated odometry sample and use
the same timestamp. `g1_state_estimation` owns both edges because the
pelvis height, roll, and pitch change while walking. Robot-description
and state-publisher components begin at `pelvis` and own only its
downstream body, joint, and sensor frames; they must not republish
either navigation-state edge.

This boundary does not fuse sensors, generate odometry, run SLAM, or
command the robot. `base_footprint` is the authoritative planar robot
frame for Nav2.

### Navigation robot description

`g1_robot_description` installs the kinematic subset required to
connect `pelvis` to the moving waist, torso/head, Intel RealSense
D435i, Livox MID-360, and official IMU reference frames. Its source
is Unitree's official
`g1_29dof.urdf`, pinned at commit
`5994d4faef0a9cadd3287f8de0199a67eeb2a259`. The upstream BSD-3-Clause
license and exact source digest are retained with the package.

The description has one root, `pelvis`, and contains no `world`,
`map`, `odom`, or `base_footprint` link. It contains no visual,
collision, inertial, limb, or rubber-hand model. This prevents it from
claiming that the installed BrainCo hands are Unitree rubber hands and
avoids duplicating the existing MuJoCo meshes.

The retained sensor frames are `d435_link`, `mid360_link`,
`imu_in_torso`, and `imu_in_pelvis`. Unitree's standard Intel
RealSense D435i and Livox MID-360 head-area mounts are represented as
fixed children of `torso_link`, matching the pinned official
kinematic model. The MID-360 transform is
`xyz="0.0002835 0.00003 0.40618"` and
`rpy="0 0.04014257279586953 0"`.

This package consumes no hardware. Its installed
`robot_state.launch.py` starts the upstream ROS 2
`robot_state_publisher`, loads the standard `g1_navigation.urdf`, and
consumes the three measured waist positions from `/joint_states`.

The installed `physical_robot_state.launch.py` derives a runtime-only
physical description from the pinned full URDF and its 34 visual meshes.
Unitree rubber-hand links are omitted. BrainCo visuals follow measured wrists
with fixed fingers and relaxed thumbs; finger positions are not measured.
It requires exactly one
`mid360_joint`, preserves its translation and pitch, and changes its
roll from zero to pi for the verified upside-down physical mount. It
does not modify the standard source URDF used by simulation.

Both runtimes publish the three moving waist edges on `/tf` and the
fixed head, D435i, MID-360, and IMU edges on `/tf_static`. Each begins
at the URDF root `pelvis`; neither publishes or duplicates
`odom -> base_footprint -> pelvis`. Exactly one robot-state runtime may
run at a time.

`use_sim_time` defaults to `false` for the future physical workstation
path. Simulation must explicitly launch with `use_sim_time:=true`.
The runtime never starts `joint_state_publisher` or substitutes
default joint positions.

On physical PC2, the upstream Foxy `realsense2_camera_node` at
`/d435/d435` is the sole process allowed to open D435i serial
`254322073626`. Project-owned `config/d435i.yaml` enables depth, color,
infrared 1 and 2, gyro, accelerometer, synchronized timestamps, CameraInfo,
and driver TF while leaving point-cloud generation disabled. The D435i IMU
is supplementary sensor data and is not an input to navigation state
estimation.

`g1_camera_bridge` never opens the D435i. It subscribes with best-effort,
volatile sensor-data QoS to `/d435/color/image_raw` and
`/d435/depth/image_rect_raw`. Valid `rgb8` color is JPEG-encoded onto the
existing `/g1/camera/rgb/compressed` compatibility topic while preserving
the driver timestamp and `d435_color_optical_frame`. Valid depth is
republished unchanged on `/g1/camera/depth/image_raw`, preserving the
driver timestamp, `d435_depth_optical_frame`, dimensions, `16UC1` encoding,
step, and payload. Color older than 0.25 seconds, depth older than 0.5
seconds, and structurally invalid frames fail closed and are not published.
Both compatibility outputs use best-effort, volatile sensor-data QoS.

Physical proof showed that the legacy Unitree `VideoClient` service is
unavailable even while the RealSense driver is stopped. Supported bringup
therefore does not launch `rgb_helper`; the supervised D435 driver is the
single physical camera source for both RGB and depth. The RealSense driver
publishes its optical-frame tree below `d435_link`. The physical
MID-360 driver must publish in `mid360_link` or provide an explicit measured
transform from its driver frame; the description does not guess or silently
alias a mismatch.

### Navigation perception boundary

`g1_navigation_perception` runs on the workstation and owns
navigation-specific preprocessing for both MID-360 and future D435i
observations. Physical sensor acquisition remains on PC2.

The initial MID-360 contract is:

| Direction | Topic | Type | Frame |
|---|---|---|---|
| Input | `/g1/sensors/lidar/points` | `sensor_msgs/msg/PointCloud2` | `mid360_link` |
| Output | `/scan` | `sensor_msgs/msg/LaserScan` | `base_footprint` |

The package launches the upstream ROS
`pointcloud_to_laserscan_node`; it contains no custom projection
algorithm. The input subscription uses best-effort, volatile QoS and
is created lazily only while `/scan` has a subscriber. Therefore the
input topic may be absent from the ROS graph when no mapping or
navigation consumer is active.

The 1.50 m ceiling is a provisional upright, arms-down mapping setting:
measured head height is approximately 1.35 m and the secured hook is above
1.60 m. It is not validated for raised arms or every walking posture.
Robot-body and support filtering remain separate; raw clouds are unchanged.

The trial projection envelope is 0.05 m through 1.50 m above
`base_footprint`, with a 0.25 m minimum and 40 m maximum range,
360-degree coverage, 0.5-degree angular bins, and a 0.1 s scan time.
These are conservative integration values, not validated physical
calibration. They must be checked against recorded MID-360 data,
self-returns, obstacle coverage, and timestamped TF before physical
mapping.

This package does not provide a physical or simulated point-cloud
publisher. It does not run Livox hardware, SLAM Toolbox, AMCL, Nav2,
or map persistence.

### Navigation mapping boundary

`g1_navigation_mapping` runs upstream asynchronous SLAM Toolbox on the
workstation. Its frame and topic contract is:

| Direction | Name | Ownership |
|---|---|---|
| Input | `/scan` | `g1_navigation_perception` |
| Input transform | `odom -> base_footprint` | `g1_state_estimation` |
| Output | `/map` | SLAM Toolbox |
| Output transform | `map -> odom` | SLAM Toolbox |

The mapping package does not publish `odom -> base_footprint`, acquire a
locomotion lease, or send robot commands. Simulation launches it with
`use_sim_time:=true`; the future physical workstation path must use wall
clock time.

### Navigation localization boundary

`g1_navigation_localization` loads a selected occupancy-map YAML with
upstream Nav2 Map Server and localizes the robot with upstream AMCL. Its
frame and topic contract is:

| Direction | Name | Ownership |
|---|---|---|
| Input | `/scan` | `g1_navigation_perception` |
| Input transform | `odom -> base_footprint` | `g1_state_estimation` |
| Input map | occupancy YAML + PGM | Map Server |
| Output | `/amcl_pose` | AMCL |
| Output transform | `map -> odom` | AMCL |

Mapping and localization modes are mutually exclusive. SLAM Toolbox owns
`map -> odom` while creating a map; AMCL owns the same transform while using
a saved map. They must never run together.

Localization requires an explicit initial pose in the `map` frame. The
production localization package does not hide or guess that pose. Operators
provide it through `/initialpose`; the deterministic simulation proof seeds
the known arena start at `(0, 0, 0)`.

The committed `navigation_arena.yaml` and `navigation_arena.pgm` files are a
reference fixture for the deterministic MuJoCo navigation arena only. They
are not a physical-building map and must not be deployed as one.

`localization_simulation.launch.py` composes the headless simulation,
validated state, robot description, lidar projection, saved-map
localization, Nav2, and leased simulation control. It excludes SLAM Toolbox
and requires the map YAML path explicitly.

The deterministic MuJoCo navigation arena is an integration fixture, not a
model of a physical building. A nonempty stationary map proves scan
generation, projection, timestamps, TF connectivity, and SLAM ingestion. It
does not prove loop closure, map quality while walking, localization recovery,
humanoid balance, or autonomous locomotion.

The current simulation G1 is supported by an elastic overhead harness and
initial-pose joint PD control. That limitation remains separate from the
successful SLAM data-path proof.

### Simulation-only navigation carrier

The harness-supported G1 cannot currently execute a balance or walking
policy. Closed-loop Nav2 integration therefore uses an explicit,
simulation-only planar carrier. This fixture validates navigation data flow,
planning, command authority, watchdogs, and simulated pose feedback; it does
not validate humanoid gait, balance, contact dynamics, or hardware motion.

The standard `simulation.yaml` configuration keeps
`planar_carrier_enabled` false. The opt-in
`navigation_simulation.yaml` configuration enables it. The carrier accepts
only arbiter-approved commands on the private
`/g1/simulation/locomotion_command` topic.

The carrier integrates planar X, Y, and yaw using MuJoCo simulation time.
It preserves the physics-controlled pelvis Z value and updates the elastic
harness anchor in X and Y so the support follows the test fixture. Lateral
motion is disabled, forward speed is limited to 0.15 m/s, yaw rate is limited
to 0.40 rad/s, and stale commands stop after 0.15 s.

`navigation_control.yaml` retains the non-moving recording backend.
`navigation_simulation_control.yaml` explicitly selects the simulation
backend. Neither configuration selects the Unitree Sport backend. The
locomotion arbiter remains the sole owner of backend dispatch and retains its
independent 0.25 s stop watchdog.

The complete workstation integration fixture is started with
`g1_navigation_bringup/launch/navigation_simulation.launch.py`. It composes
the package-owned simulation, state estimation, robot description,
perception, mapping, Nav2, and navigation-control launches. Every include has
a scoped launch-configuration context so generic arguments such as
`params_file` cannot leak between packages.

### Navigation command authority

Nav2 publishes untrusted velocity output on
`/g1/navigation/cmd_vel_raw`. That topic has no authority to move the
robot or simulation backend.

`g1_navigation_control` requests `OWNER_NAVIGATION` only after receiving
Nav2 command demand. The first command is dropped while lease acquisition
is pending. After a lease is granted, the adapter validates the planar
velocity, attaches the opaque lease identifier, and publishes
`g1_locomotion_interfaces/msg/LocomotionCommand` on
`/g1/locomotion/command`.

Navigation is forward-only in this initial integration. Commands are bounded
to 0.15 m/s forward velocity and 0.40 rad/s yaw rate; lateral velocity and
unsupported axes are rejected. The locomotion arbiter independently validates
the same command before dispatch.

An invalid command, a command-stream timeout, or adapter shutdown causes the
adapter to publish an explicit zero command under the active lease, clear its
local authority, and request lease release. The adapter timeout is 0.15 s,
while the locomotion arbiter retains an independent 0.25 s fallback watchdog.
This ordering gives the adapter time to release normally while preserving an
arbiter-owned stop if the adapter fails.

## Body joint order

`Policy` is the neural-network action index. `SDK` is the index in Unitree
`LowCmd.motor_cmd` and `LowState.motor_state`.

| Policy | SDK | Joint |
|---:|---:|---|
| 0 | 0 | `left_hip_pitch_joint` |
| 1 | 6 | `right_hip_pitch_joint` |
| 2 | 12 | `waist_yaw_joint` |
| 3 | 1 | `left_hip_roll_joint` |
| 4 | 7 | `right_hip_roll_joint` |
| 5 | 13 | `waist_roll_joint` |
| 6 | 2 | `left_hip_yaw_joint` |
| 7 | 8 | `right_hip_yaw_joint` |
| 8 | 14 | `waist_pitch_joint` |
| 9 | 3 | `left_knee_joint` |
| 10 | 9 | `right_knee_joint` |
| 11 | 15 | `left_shoulder_pitch_joint` |
| 12 | 22 | `right_shoulder_pitch_joint` |
| 13 | 4 | `left_ankle_pitch_joint` |
| 14 | 10 | `right_ankle_pitch_joint` |
| 15 | 16 | `left_shoulder_roll_joint` |
| 16 | 23 | `right_shoulder_roll_joint` |
| 17 | 5 | `left_ankle_roll_joint` |
| 18 | 11 | `right_ankle_roll_joint` |
| 19 | 17 | `left_shoulder_yaw_joint` |
| 20 | 24 | `right_shoulder_yaw_joint` |
| 21 | 18 | `left_elbow_joint` |
| 22 | 25 | `right_elbow_joint` |
| 23 | 19 | `left_wrist_roll_joint` |
| 24 | 26 | `right_wrist_roll_joint` |
| 25 | 20 | `left_wrist_pitch_joint` |
| 26 | 27 | `right_wrist_pitch_joint` |
| 27 | 21 | `left_wrist_yaw_joint` |
| 28 | 28 | `right_wrist_yaw_joint` |

Safety invariants: commands require an explicit motor mode; unknown names,
duplicate SDK indices, and non-finite values are rejected.

## BrainCo hands

The BrainCo hands affect whole-body mass, inertia, collision, and arm-swing
dynamics, but their 12 channels are not locomotion-policy actions. During the
initial stair task they remain in a neutral pose.

Each vendored hand declares 22 links and 0.390 kg of link mass. Five fixed tip
links per side have no separate inertial mass and must be checked after Isaac
URDF import/fixed-joint merging.

| Channel | Function | Right URDF joint | Left URDF joint | Range (rad) |
|---:|---|---|---|---:|
| 0 | Thumb flexion | `right_thumb_proximal_joint` | `left_thumb_proximal_joint` | 0–1.0472 |
| 1 | Thumb auxiliary | `right_thumb_metacarpal_joint` | `left_thumb_metacarpal_joint` | 0–1.55334 |
| 2 | Index | `right_index_proximal_joint` | `left_index_proximal_joint` | 0–1.41 |
| 3 | Middle | `right_middle_proximal_joint` | `left_middle_proximal_joint` | 0–1.41 |
| 4 | Ring | `right_ring_proximal_joint` | `left_ring_proximal_joint` | 0–1.41 |
| 5 | Pinky | `right_pinky_proximal_joint` | `left_pinky_proximal_joint` | 0–1.41 |

Hand DDS uses hardware domain 0. Commands are normalized to 0–1 at the project
boundary. Both hands are physically verified. The right hand records firmware
`1.0.16.U` and device ID 127. The left hand was validated through the guarded
physical card-presentation workflow, including gripping and releasing the
card; its firmware and device ID were not recorded and remain `null` rather
than being inferred. While the left hand awaits hardware service, the physical
interaction stack uses the right-hand card profile. Its semantic hand gate is
restricted to the right side of the D435 RGB image; a full-image hand gate is
not permitted.

The current MuJoCo attachment transforms in
[`simulation/mujoco/model_composition.py`](../../simulation/mujoco/model_composition.py) are
simulation-derived. Physical wrist-adapter translation, orientation, and mass
must be measured before the Isaac asset is considered deployment-grade.

Authoritative hand contract:
[`simulation/mujoco/config/brainco_revo2.yaml`](../../simulation/mujoco/config/brainco_revo2.yaml).

## SDK, runtime, and DDS isolation

| Component | Pinned version/revision |
|---|---|
| Unitree SDK2 | `9754cd153af3da471b0fe5f3aa535e426fb11db3` |
| Unitree SDK2 Python (development image) | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` |
| Unitree SDK2 Python (physical PC2) | `814556d15970dd2ecf1c9984e845ca02ab07e206` |
| Unitree MuJoCo | `ae6a8403e272733e9996ef59990880330496177f` |
| Unitree RL Lab deployment subset | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` |
| MuJoCo | 3.3.6 |
| ONNX Runtime | 1.20.1 Python; pinned Unitree native runtime for deployment |
| NumPy | 1.26.4 |
| ROS development environment | ROS 2 Humble |
| Physical PC2 environment | ROS 2 Foxy |

The physical high-level locomotion helper loads only
`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient` inside its isolated PC2
process. The physical robot runs Unitree software 1.5.4. Helper startup
requests the official internal `WALKRUN` controller through API `7111` with
payload `2` and fails closed unless Unitree returns code `0`. This handoff
occurs before the helper accepts health, MOVE, or STOP requests. MOVE maps to
`SetVelocity(vx, vy, vyaw, bounded_duration)` and STOP maps to
`SetVelocity(0.0, 0.0, 0.0, bounded_duration)` so the RPC return code remains
observable. It must never call `Damp`, `ZeroTorque`, `Sit`, or
posture-transition methods. A successful STOP response confirms acceptance of
the zero-velocity RPC; it is not independent physical proof that velocity is
zero. These contracts use the pinned physical PC2 Python SDK revision above.
The installed `g1_sport_helper` executable accepts the physical interface,
bounded velocity duration, and RPC timeout. It communicates through strict
line-delimited requests and must issue the zero-velocity STOP RPC whenever
its parent stream closes or request processing exits.
The application-side supervisor permits exactly one live helper, uses a
finite positive response timeout, and treats timeout or closed output as
helper failure. Shutdown closes the parent input stream first so helper EOF
can issue STOP, then escalates to terminate and kill only when bounded waits
fail, and always reaps the child.

The separately installed `g1_physical_locomotion_arbiter` executable is the
only entry point that injects the supervised physical Sport backend factory.
The generic `g1_locomotion_arbiter` executable and workstation navigation
configuration remain recording-only by default. PC2 physical operation uses
the package-owned `g1_locomotion_control/config/physical_pc2.yaml` profile,
which explicitly selects `backend: sport`, interface `eth0`, 0.30 m/s forward,
0.01 m/s lateral, and 0.40 rad/s yaw limits. The physical helper velocity
duration is 1.0 s, the helper RPC timeout is 1.0 s, and the parent response
timeout is 1.5 s. Each streamed command refreshes the velocity duration; it
does not limit total navigation time. A newly granted physical lease may remain
stopped for up to 5.0 s while its first command is delivered; expiry reports
`LEASE_START_TIMEOUT`. After the first accepted command, the arbiter uses its
independent 0.25 s command watchdog and normally issues an explicit STOP first.
The 1.0 s SDK duration remains the final firmware-side fallback if the parent
and helper disappear. Construction fails closed unless the parent response
timeout is strictly greater than the child RPC timeout.

DDS/network separation is mandatory:

| Environment | Domain | Interface |
|---|---:|---|
| Physical Unitree/BrainCo traffic | 0 | Deployment-specific physical NIC |
| Native MuJoCo simulation | 1 | Loopback (`lo`) |
| Project ROS application stack | 42 | Configured CycloneDDS interface |

The development application domain uses CycloneDDS with automatic participant
indices and `MaxAutoParticipantIndex` set to `63`. This provides sufficient
capacity for the separate simulation, state-estimation, perception, mapping,
and Nav2 processes. It changes discovery capacity only; domain ownership and
network isolation remain unchanged.

Never run simulation on hardware domain 0. Never run two physical controllers
simultaneously. The explicit physical locomotion arbiter is the only project
component permitted to issue high-level Sport commands. The physical hardware
bridge remains the only component permitted to command low-level body or
peripheral hardware directly.

Dependency definition: [`docker/development/Dockerfile`](../../docker/development/Dockerfile).

The Docker `cpu` target uses `glim-packages.lock`; the optional `gpu` target
replaces the three CPU GLIM packages using `glim-gpu-packages.lock` and installs
CUDA runtime 12.2.140. NVIDIA driver support comes from the host.
GPU GLIM remains experimental: stationary replay showed no clear CPU-time
benefit. GPU availability does not select the GPU estimator or authorize motion.
Both variants retain the existing odometry health and DDS isolation boundaries.

## Model and policy assets

### Active local assets

- G1 mode-5-compatible reference: vendored Unitree 29-DoF MJCF and meshes under
  `simulation/mujoco/vendor/unitree_g1/`.
- BrainCo Revo2: vendored URDF/Xacro and meshes under
  `simulation/mujoco/vendor/brainco_revo2/`.
- Combined MuJoCo model: constructed at runtime by
  `simulation/mujoco/model_composition.py`.
- Reference balance/velocity policy: installed into the development image,
  not stored in Git.

### Isaac Lab asset decision

Use the official `g1_29dof_rev_1_0` mode-5 body and compose both BrainCo Revo2
hands. Do not use Unitree's ready-made
`g1_29dof_mode_15_brainco_hand.urdf` as the training body: its gearing and
wrist motors describe different hardware.

Exact upstream provenance and candidate paths are recorded locally in
[`simulation/isaac_lab/g1_stairs/sources.yaml`](../../simulation/isaac_lab/g1_stairs/sources.yaml).
Licenses and vendored revisions are documented in
[`simulation/mujoco/PROVENANCE.md`](../../simulation/mujoco/PROVENANCE.md).

## Stair-climbing task

| Parameter | Initial value |
|---|---:|
| Direction | Ascending, +X |
| Riser height | 0.20 m |
| Tread depth | 0.25 m |
| Steps | 3 |
| Total rise | 0.60 m |
| Stair run | 0.75 m |
| Width | 1.20 m |
| Approach length | 2.00 m |
| Upper platform length in training scene | 3.00 m |
| Required post-stair continuation | 2.00 m |

Initial fixed base pose:

- position: `(1.00, 0.00, 0.80)` m;
- roll/pitch/yaw: `(0, 0, 0)` rad;
- first-riser distance: 1.00 m;
- position/yaw randomization: disabled until nominal ascent is learned.

The initial executable training and visual-inspection task uses the exact
`0.20 m` riser height in every environment. The planned `0.05–0.20 m`
riser-height curriculum remains disabled until the nominal staircase passes
its smoke and evaluation gates. During this phase, observation corruption,
external-force resets, terrain progression, and velocity progression are also
disabled. Proprioception retains five control steps of history, while the local
terrain height scan uses only its current frame.

Training uses PPO, initialized from human stair-ascent motion tracking. The
policy observes proprioception, commands, previous actions, and a local terrain
height scan. Training-only privileged observations may not become undeclared
deployment inputs.

Phase 12 Experiment A used integrated forward +X progress and a one-time reward
for each newly reached stair level. Experiment B starts from a fresh policy and
adds only Isaac Lab's `flat_orientation_l2` term at weight `-1.0`. No inherited
flat-ground shaping, completion, safety, foot-control, constraint, power,
action-rate, or jerk rewards are active in Experiment B.

Acceptance gates:

- at least 99% completion on the exact nominal staircase;
- at least 95% completion over the approved randomized evaluation set;
- no torso, pelvis, knee, forearm, or hand contacts in nominal evaluation;
- stable forward continuation for at least 2 m on the top platform;
- ONNX export passes the 29-output contract;
- independent MuJoCo sim-to-sim validation passes before hardware work.

Task configuration:
[`simulation/isaac_lab/g1_stairs/task.yaml`](../../simulation/isaac_lab/g1_stairs/task.yaml).

Isaac training integration contract:

- task ID: `Unitree-IEA-G1-29dof-Stairs-v0` (the `Unitree` prefix is required
  by the pinned Unitree RL Lab launcher task filter);
- compatibility target: Isaac Sim 5.1.0, Isaac Lab 2.3.0, Unitree RL Lab
  revision `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, and RSL-RL 3.0.1 exactly;
- the AWS host supplies Isaac Lab source tag `v2.3.0` and immutable commit
  `3c6e67bb5c7ada942a6d1884ab69338f57596f77` to the container; canonical
  launches reject absent or mismatched provenance and never require `.git`
  metadata inside `/workspace/isaaclab`;
- Unitree RL Lab remains pristine. The project-specific G1 asset override lives
  in `simulation/isaac_lab/g1_stairs/robot_cfg.py` and loads the pinned mode-5
  `g1_29dof_rev_1_0.urdf` from the read-only Unitree ROS checkout;
- the supported infrastructure gate is the committed
  `experiments/g1_stairs/configs/004_experiment_b_smoke.yaml` run through
  `simctl`. It uses one training iteration to validate the complete runtime
  path and may complete without producing a resolved checkpoint.

## Experiment recording and independent evaluation

Version-controlled experiment definitions live under `experiments/g1_stairs/configs/`.
Generated run artifacts live under `experiments/g1_stairs/runs/` and
must not be committed to Git.

Every retained experiment run must contain:

- the recursively resolved experiment configuration;
- a run manifest recording Git, host, GPU, package, and configuration facts;
- the complete console log;
- training and reward-component metrics;
- independent per-episode evaluation outcomes;
- an evaluation summary;
- checkpoint-associated evaluation videos when recording is enabled.

The recorded-run launcher rejects dirty worktrees by default. The
`--allow-dirty` option is permitted only for local infrastructure testing and
must not be used for thesis experiment results.

Training reward and task termination are not accepted as independent evidence
of stair completion. The simulator-independent evaluator determines success
and assigns a deterministic primary failure reason.

Nominal success requires:

- completion of the approach;
- all three stair levels reached;
- entry onto the continuous upper platform;
- at least 2.00 m of forward platform continuation;
- at least 1.00 s of upright dwell;
- no earlier fall, forbidden contact, joint, torque, velocity, direction,
  slip, stall, invalid-state, or timeout failure.

The metrics contract produces:

- `training_metrics.csv`;
- `reward_components.csv`, including raw, weight, and weighted values;
- `evaluation_episodes.csv`;
- `evaluation_summary.json`, including a 95% Wilson success-rate interval.

Experiment A uses the project-owned `StairPPORunnerCfg` rather than relying on
unverified upstream defaults. Each RSL-RL iteration writes structured training,
active reward-component, constraint, motion, PPO, performance, GPU-memory, and
non-finite-state metrics into the recorded run directory. RSL-RL 3.0.1 does
not expose approximate KL or clip fraction through its runner loss dictionary;
those two diagnostics are explicitly deferred until PPO ablation work rather
than being represented as measured values.

The independent evaluator and metrics writer have simulator-independent unit
coverage. Their Isaac Lab adapter and live logging integration passed the
one-iteration AWS metrics smoke, so the nominal Experiment A launch gate is
open.

Experiment 021 is a bounded 5 cm, single-step initial-state curriculum. It
starts from the confirmed nominal G1 standing pose, applies deterministic
two-link sagittal IK only to the leading leg, places the root immediately
before the riser, zeros all velocities, and settles under the normal actuator
configuration. Training is fail-closed unless every environment physically
validates leading-foot upper-tread support, trailing-foot lower-floor support,
upright posture, finite observations, and absence of forbidden contact. Its
reward state begins at `FIRST_SUPPORT`; no first-support reward is available.
Only one-time `BASE_ABOVE`, `BOTH_FEET_SUPPORTED`, and `COMPLETED` transitions
augment the pinned locomotion-preservation rewards. Experiment 020 remains a
rejected historical run and is never used for policy initialization.

## Safety and promotion sequence

1. Validate configuration and all 29 body-joint mappings.
2. Reproduce the unmodified Unitree flat-ground policy in Isaac Lab.
3. Validate the composed mode-5 G1 plus BrainCo asset at rest.
4. Train and evaluate the stair policy in Isaac Lab.
5. Export ONNX and inspect input/output shapes and finite zero inference.
6. Run the policy in MuJoCo on DDS domain 1 with the elastic harness attached.
7. Test adverse cases and confirm fall/limit/stale-state transitions stop the controller.
8. Require explicit human review before releasing the simulated harness.
9. Design a separate, supervised physical promotion plan with an operator and emergency stop.

Simulation success is not evidence that physical execution is safe. The
physical controller must start from fresh state, preserve exclusive low-level
ownership, and transition to passive on stale state, fall, limit violation, or
non-finite command.

## Local validation commands

Run inside the project development container unless stated otherwise:

```bash
python3 simulation/mujoco/validate_config.py
python3 simulation/mujoco/run_g1_model.py --validate-only
python3 simulation/mujoco/inspect_policy.py "$UNITREE_BALANCE_POLICY" \
  --require-g1-velocity-contract --zero-inference
python3 -m unittest discover -s simulation/isaac_lab/g1_stairs/tests -v
python3 simulation/isaac_lab/g1_stairs/audit_assets.py
pytest -q
```

Native harness-assisted balance validation:

```bash
python3 simulation/mujoco/run_g1_model.py --native --balance
```

Keep the harness attached while loading or changing a controller. Harness
release remains an explicit operator decision.

## Maintenance rule

### Opt-in 043 physical single-step smoke (not hardware promotion)

`experiments/g1_stairs/configs/043a_physical_single_step_smoke.yaml` starts
from the original Unitree ONNX actor, the normal fixed lower-floor reset,
one 6 cm riser and a 25 cm first landing region. The unchanged 5.75 m tile
has a 3.5 m effective continuation platform. This diagnostic is not the
final three-step 20 cm thesis course. Historical 042 and earlier configs
are unchanged.

The command slows from 0.50 to 0.18 m/s; episode length is 20 s. Actor,
critic and action dimensions remain 485/687/29. G1 mode_machine 5, control
timing, policy/action order and PPO settings remain those of 042. BrainCo
hands are still absent from this diagnostic asset and required before
final training/deployment. No frozen-teacher loss is introduced here.

043 uses transformed ankle-roll collision sphere samples from the pinned
mode-5 URDF, vertical load relative to actual model weight, foot speed,
angular speed, orientation, tread containment, forbidden-body contact and
0.12 s support dwell. Footprint coverage is a geometric proxy, not measured
centre of pressure. First support requires the first tread rectangle;
subsequent support may move onto the continuous platform. Both feet must
establish support, not stand motionless together throughout the gait.
These thresholds are explicit initial diagnostic choices, not a claim of
validated dynamic feasibility or a guarantee of natural-looking motion.

The snapshot boundary follows [Isaac Lab v2.3.0's step order](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py):
termination, reward, then automatic reset. Foot poses and velocities use the
[pinned link-frame data API](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py),
not offsets mixed between the link and centre-of-mass frames.

Physical state updates once per control step through termination checks,
independently of reward weights/order. Explicit per-environment reset clears
state while retaining the last pre-autoreset snapshot for metrics. Event
payouts are 2/3/3/5 for first support/body transfer/trailing recovery/completion;
raw event terms divide by dt to cancel RewardManager's dt multiplication.
Positive preparation budgets are capped at 0.25/0.25/0.50 per episode.

The smoke is 32 environments, one iteration, seed 42. Tensor counterexample
tests and an actual Isaac smoke are mandatory before enabling longer runs.
`physical_step043_summary.yaml` records event counts and their environment-
step denominator; training metrics persist `step043_*` in `extra_json` and
TensorBoard. Zero completed episodes is not a measured zero success rate.
These are training-detector metrics, not independent proof of successful
climbing. Independent replay/video acceptance and walking-preservation
ablation remain separate work; the base config is launch-blocked.

`043b_approach_preservation_base.yaml` is the shared, launch-blocked contract
for actor-only approach-gait preservation. Its smoke and short-run children
change only execution scope. The smoke uses 32 environments for one iteration;
the short behavioral run uses 512 environments for 500 iterations. Both use
training seed 42 and remain diagnostics rather than climbing certification.
The shared contract adds a separate 480-D observation group matching the original
Unitree locomotion contract and evaluates the frozen original ONNX actor as a
training-only reference. A project-owned PPO extension adds gated Smooth-L1
distance between the current student actor mean and frozen teacher action to
the actor optimization loss. The aligned 480-D teacher observation and scalar
release gate are stored with each rollout sample; the term never enters the
environment reward, critic target, GAE or PPO advantage. The gate fades from
full strength at 0.40 m to zero at 0.25 m before the current riser. The deployed student remains the same
single 485-input, 29-output actor. These runs do not add early-lift, stance,
swing, placement, or support mechanisms and are not evidence of preserved gait
or learned climbing.

Update this document in the same change whenever any of these change:

- physical G1 machine type or motor hardware;
- joint order, policy shape, timing, gains, or observation contract;
- hand firmware, mass, adapter, mounting transform, or verification status;
- SDK, simulator, model, or policy revision;
- DDS domain/interface ownership;
- stair geometry, fixed start pose, curriculum, or evaluation gates.

After such a change, run `simulation/mujoco/validate_config.py`, the stair geometry
tests, and `graphify update .`.

### Opt-in Wi-Fi mapping

The opt-in physical_wifi_mapping.launch.py selects the domain-42 laptop Wi-Fi DDS profile and glim_mid360_wifi configuration, starts one topic_tools relay, and includes the existing read-only mapping launch. Wired defaults remain unchanged. PC2 publishes the unchanged raw cloud on /g1/transport/lidar/points_wifi; only the laptop relay subscribes to that transport topic and republishes /g1/sensors/lidar/points locally. GLIM uses reliable cloud input with depth 5. Health, freshness, timestamp, and motion limits remain unchanged.

Profiles bind laptop wlp3s0 at 192.168.188.117 and PC2 wlan0 at 192.168.188.160. Verify addresses before reuse. The PC2 profile applies only to ROS domain 42, preserving native Unitree domain 0. Profile files are config/physical/cyclonedds_laptop_wifi.xml and cyclonedds_pc2_wifi.xml in g1_navigation_bringup.

The bounded manual Wi-Fi check recorded 258 samples at all five odometry stages, maximum header gap 100.758 ms, no gaps over 150 ms or nonadvancing timestamps, canonical p95 age 155.333 ms and maximum age 170.687 ms. Scan delivery reached 211.692 ms gaps and 260.876 ms age. Estimated displacement was 0.326 m; independent distance accuracy was not measured. This is bounded continuity evidence, not unrestricted autonomous-navigation qualification.

### Nav2 binary-pin refresh — 2026-11-18

The Nav2 lock retains upstream version 1.1.20 and pins the available November binary rebuilds of the same 35 packages. The September binaries were no longer available from the active ROS apt index. Installation simulation reported 35 upgrades, no additions, and no removals. Existing runtime measurements used the September binaries; the rebuilt image requires separate validation before live use. No installed packages in the running container were upgraded by this lock refresh.

### Opt-in depth cloud preview — 2026-11-19



g1_navigation_perception/depth_preview.launch.py starts only the upstream depth_image_proc point_cloud_xyz_node. It consumes /d435/depth/image_rect_raw and /d435/depth/camera_info and publishes the full, unfiltered cloud on /g1/diagnostics/depth/points, retaining the image timestamp and optical frame. It starts no camera driver, odometry authority, mapping, planner, controller or command node. This diagnostic topic is not a Nav2 input and provides no obstacle qualification or stopping guarantee.



The development image declares ros-humble-depth-image-proc. The current container installation, perception package build/tests and installed preview launch passed. Rebuilding and validating the updated image remain pending. The temporary PC2 depth-only 848x480x15 profile used during these checks does not replace the persistent camera configuration.

### Experimental offline floor selection — 2026-11-19



The perception geometry module fits candidate planes and performs deterministic sampled selection with explicit caller-supplied distance bounds. Diagnostic criteria currently use a 10-degree tilt limit, 20 mm inlier tolerance, at least 30 inliers and 35 percent sampled support. Insufficient supported geometry returns None. These criteria do not establish semantic ground identity: a raised surface inside the permitted distance band can still be ambiguous. The selector is not connected to live navigation or command generation.


### Experimental live depth obstacle preview — 2026-11-19

The opt-in depth_obstacle_preview node publishes only diagnostic obstacles
and status, in pelvis coordinates. It is not connected to Nav2 or locomotion.
Inputs retain their image timestamps and require exact-time TF, pelvis IMU
matching within 50 ms, a maximum cloud age of 350 ms and at most 50 ms future
offset. Missing or rejected context produces empty diagnostic output.
A latest-cloud cache limits geometry processing to 10 Hz, with non-blocking
TF retries. Brief deferral retains only previously published output that
remains within its original freshness limit.

Floor fitting uses the local forward region 0.2–1.2 m and lateral +/-0.4 m.
The original selection runs first. If unsupported, a height-prior filter
excludes points incompatible with the configured distance bounds and tilt
limit before retrying selection. The 10-degree tilt, 20 mm inlier tolerance,
30-point minimum and 35-percent support criteria remain; fallback support
is measured over the filtered fitting inputs. All original points remain
available for obstacle classification. This prior does not prove semantic
ground identity, and raised surfaces may remain ambiguous.

Recorded wall captures recovered candidate planes near 0.823 m below the
pelvis. Earlier regression checks were preserved. A subsequent manual
back-and-forward check reported no FLOOR_UNKNOWN or TF rejection, but one
approximately 0.41-second stale/processing interruption remained.
These observations do not qualify autonomous obstacle avoidance.

Run numerical processing with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1. GPU acceleration has not yet been implemented or measured.

### Optional diagnostic CUDA floor fitting — 2026-11-19

The depth obstacle preview accepts compute_backend=cpu (default) or cuda.
CUDA selection imports the optional CuPy module and warms it before sensor
subscriptions start. Initialization or execution failure logs a warning
and selects the CPU implementation. Input timestamps, exact-time TF,
IMU matching, cloud-age checks and diagnostic-only endpoints are unchanged.
Only floor fitting is accelerated; transforms and obstacle filtering
remain on CPU. No Nav2 or locomotion connection is added.

An isolated benchmark environment uses CuPy 14.2.0, NumPy 2.2.6 and CUDA
runtime 12.9 on the RTX 3070 laptop. Three recorded wall frames measured
median geometry times of 59.717 ms CPU and 15.554 ms GPU, including GPU
transfers and synchronization but excluding ROS. Nine baseline/box
snapshots had identical obstacle masks; seven floor regression inputs
also agreed within a plane-component tolerance of 1e-5. Two additional
prototype rejection cases agreed. These are offline comparisons, not
live timing or autonomous-navigation qualification. The optional CuPy environment is now declared in the GPU image; its build and fresh-container CUDA floor-fitting smoke test passed.

### Depth Wi-Fi delivery checkpoint — 2026-11-19

The separate opt-in d435i_depth_wifi.yaml profile captures depth at
848x480x15, disables color, infrared publication, camera IMU and stream
synchronization, and enables magnitude-2 decimation. Verified driver output
is 424x240 with updated CameraInfo. The existing d435i.yaml remains available.
The reduced depth payload is approximately 24.4 Mbps before transport
overhead, versus 97.7 Mbps at the original dimensions and same frame rate.

Before decimation, PC2-local depth was steady while laptop-side images
showed gaps over 400 ms. After decimation, the shown stationary image
measurement was near 15 Hz with a maximum interval of 78 ms. The user
confirmed the 12 cm box appeared and disappeared on removal. Subsequent
logs reported sampled CUDA callbacks of 10–18 ms, output ages of 44–108 ms,
and maximum reported input gaps up to 134 ms, without rejection transitions
in that excerpt. This is bounded diagnostic evidence, not obstacle-avoidance
or unrestricted-navigation qualification.

The GPU Docker target declares an isolated /opt/iea-depth-gpu environment
using depth-gpu-requirements.lock. CPU-image Python dependencies are not
replaced. scripts/depth-obstacle-preview.sh selects the optional CUDA
environment when present, otherwise CPU; explicit cpu selection is supported.
The node also falls back with a warning if CUDA initialization/execution
fails. The updated GPU image built successfully and a fresh-container CUDA floor-fitting smoke test passed. Fresh-container live launch was subsequently verified; see the completed timing checkpoint below.
The launcher starts only the diagnostic filter; sensors, robot TF and the
upstream depth converter must already be running.


### Fresh-container launcher verification — 2026-11-19

The reusable launcher initially failed because nounset was enabled while
sourcing ROS setup. Nounset is now disabled only around ROS/overlay sourcing.
A fresh GPU container subsequently selected the CUDA backend and published
diagnostic candidates. Reported steady callbacks were approximately 10–12 ms.

The run also showed a startup stall of approximately five seconds, stale
processing and temporarily unavailable TF, followed by recovery. Later
reported input gaps included 398 ms and 331 ms. Therefore the launcher is
functionally verified, but continuous timing robustness is not qualified.
The diagnostic output remains disconnected from Nav2 and locomotion.

### Laptop DDS receive-buffer trial — 2026-11-19

During a 25-second diagnostic run, laptop UDP counters increased by 355
RcvbufErrors; PC2 reported zero. Both kernel rmem_max values were 212992.
These counters are system-wide and do not identify the affected socket.
Disabling Wi-Fi power saving did not remove the observed delivery gaps.

A provisional laptop-only trial requests a 4 MiB CycloneDDS receive buffer
and raises the host net.core.rmem_max ceiling to 8 MiB. Existing sockets
must be recreated by restarting the relevant laptop processes. The subsequent completed checkpoint below records persistence and measured improvement. No freshness limits are relaxed.


### Depth/GPU timing checkpoint completed — 2026-11-19

First-frame profiling located approximately 4983 ms inside CUDA floor fitting.
A fixed 2048-column batch excludes padding from all inlier/support counts.
Sixteen recorded CPU/GPU comparisons and nine obstacle masks still agreed.
Startup now warms both clear-floor and wall-fallback paths before subscriptions.
In the verified cold-container run, warm-up took approximately 8.3 and 3.2 s,
but first live geometry took 9.8 ms; expensive preparation no longer blocked
the live callbacks. Startup may briefly lack TF while its buffer fills.

The laptop had 355 UDP receive-buffer errors in 25 s with rmem_max=212992.
After requesting 4 MiB DDS buffers, raising the host ceiling to 8 MiB and
restarting the converter/preview, a 30 s check reported 68762 UDP datagrams
and zero InErrors, RcvbufErrors or SndbufErrors. These are system-wide
counters, not per-topic attribution, but timing improved in the same trial.

The ceiling is installed through /etc/sysctl.d/90-iea-g1-dds.conf;
the repository source is config/sysctl/90-iea-g1-dds.conf. Reproducing
this laptop DDS configuration requires installing that host setting first.
Wi-Fi power saving was disabled during the trial but did not independently
resolve the gaps; no persistent power-saving change was made.

The final manual movement-check excerpt reported callback times of
10.0–18.1 ms, output ages of 37.7–105.8 ms and input gaps up to 88.7 ms,
with no rejection transitions shown. Box appearance/removal was previously
confirmed at 424x240. This closes the bounded diagnostic checkpoint.
Nav2 obstacle integration, automatic stopping and unrestricted navigation
remain unqualified; the diagnostic output is not connected to control.

### Timestamped diagnostic depth health — 2026-11-20

The depth preview additionally publishes diagnostic_msgs/DiagnosticArray on
/g1/diagnostics/depth/health with reliable, volatile depth-one QoS.
Its header exactly matches the corresponding diagnostic obstacle cloud.
A single status named g1_depth_obstacle_preview reports OK only for
DIAGNOSTIC_CANDIDATE; all rejection reasons report ERROR with that reason
in message. Existing text status remains available.

Candidate health retains the source-image timestamp. Rejection/clear health
matches the empty rejection cloud and cannot represent healthy free space.
Consumers must check status, source age, matching cloud timestamp/frame and
receipt freshness; arrival order across topics is not guaranteed.
This addition is diagnostic only. No costmap input, command gate or physical
navigation behavior is changed by this step.

### Isolated depth costmap preview — 2026-11-20

depth_costmap_preview.launch.py starts only upstream nav2_costmap_2d and its
lifecycle manager. It accepts replay points on the separate diagnostic topic
/g1/diagnostics/depth/costmap_input and publishes a diagnostic costmap.
It starts no sensors, planner, controller, command adapter or locomotion backend.

The preview uses pelvis as both global and robot frame to test rasterization
without localization. Its height limits are body-relative, not floor heights.
Depth marks obstacles only; filtered obstacle points do not clear free space.
Marked cells may remain until the preview is reset. This is not a driving
costmap, obstacle-removal proof or stale-data motion gate.
The head angle must remain at the reference setting used for the recordings.

The isolated depth costmap preview uses lifecycle-manager bond_timeout=0.0:
the standalone upstream Costmap2DROS executable does not create a bond.
Lifecycle configure/activate services remain in use. This exception applies
only to the no-control preview, not physical navigation or locomotion.

The isolated depth costmap preview uses lifecycle-manager bond_timeout=0.0:
the standalone upstream Costmap2DROS executable does not create a bond.
Lifecycle configure/activate services remain in use. This exception applies
only to the no-control preview, not physical navigation or locomotion.

### Optional offline voxel clearing preview — 2026-11-20

The standalone costmap launch accepts params_file; its marking-only default
is unchanged. depth_voxel_preview.yaml selects the upstream VoxelLayer.
Filtered marks use /g1/diagnostics/depth/costmap_input with clearing false.
Full depth rays use /g1/diagnostics/depth/costmap_clear with marking false.
Both require a recorded camera-origin transform named depth_preview_camera.
This avoids using filtered obstacle points as free-space evidence.

The diagnostic voxel volume spans -0.9 to +0.7 m in pelvis coordinates,
using 16 vertical bins of 0.1 m. It is an experiment for the recorded 12 cm
box, not qualification of smaller obstacles or a physical driving profile.
No planner, controller, motion gate or locomotion backend is started.

### Offline voxel frame correction — 2026-11-20

The initial pelvis-aligned voxel replay marked 59 near-field box cells,
but retained 39 after the clear-floor scene. Offline geometry showed
39 old box voxels coinciding with newly observed floor voxels.
This partial-clearing result is retained as a failed removal test.

The optional voxel preview now uses depth_preview_ground for both costmap
and reference pose, with origin_z=-0.07 m. Vertical resolution remains
0.1 m with 16 bins. A single rigid floor-reference transform, derived from
the recorded clear-floor scene, must be used for both scenes. The recorded
camera origin is transformed through that same reference. This is an
offline representation test, not a new physical TF authority or calibration
valid for arbitrary walking poses. Marking-only defaults remain unchanged.


### Offline depth-to-Nav2 checkpoint — 2026-11-20

Timestamped diagnostic depth health was added and tested. It does not yet
gate commands. The standalone costmap preview starts only the upstream
costmap and lifecycle manager; no controller or locomotion backend.

Marking replay: clear-floor scene had zero nearby occupied cells; the
12 cm box scene had 59. Both scenes marked 100 percent of the expected
occupied cells at 5 cm horizontal resolution.

The first 3D-clearing replay retained 39 of 59 box cells. Geometry showed
39 old box voxels coinciding with observed floor voxels in the coarse,
pelvis-aligned grid. A corrected replay used one fixed floor reference
derived from the clear-floor recording for both scenes. With unchanged
vertical resolution and marking thresholds, the box marked 59 cells and
the subsequent clear-floor scene removed all 59, without an intervening
map reset. Failed and successful reports are both preserved.

These are frozen snapshots with test timestamps. They do not establish
live timing, dynamic walking-frame accuracy, or autonomous stopping.
The offline floor-reference transform must not be copied into the physical
TF authority as a calibration valid for all poses.

The user reports that manual head adjustment moves both camera and LiDAR.
The head setting remained unchanged through these recordings. The current
model assumes fixed sensor mounting; the numerical adjustment angle is
not independently recorded. Live sensor-to-body/floor alignment must be
verified before production costmap integration.

### Opt-in diagnostic hand containment candidate — 2026-11-23

The depth obstacle preview optionally applies hand containment before floor
selection. It is disabled by default; depth_hand_self_filter.yaml explicitly
enables nominal wrist-local boxes derived from the vendored BrainCo Revo2
visual geometry and the physical_robot_state hand mounts. Each face has a
0.02 m margin. Both pelvis-from-wrist transforms are required at the source
image timestamp before the pending cloud is consumed. Missing TF follows the
existing TF_UNAVAILABLE rejection path; invalid numerical geometry is rejected.
No latest-TF substitution, timestamp shift or fabricated finger telemetry is used.

This is a containment approximation for the recorded open/relaxed hand pose,
not a full-body filter. No raw cloud, clearing ray, LiDAR, navigation footprint,
controller or locomotion setting is changed. The default diagnostic output
remains /g1/diagnostics/depth/obstacles; enabling a separate experimental bridge
to Nav2 requires its own validation. Removed returns are not free-space evidence.

Recorded comparison: 39 marking clouds with available TF contained 214 close
returns, all excluded; one startup cloud lacked TF and was reported separately.
The selected low-box counts stayed 2738/2738 at approximately 50 cm from toes
and 3519/3519 at approximately 20 cm; baseline selected count remained zero.
A restamped, isolated Nav2 depth-only marking comparison reproduced self-marking
before filtering. After filtering none of 22 hand-only cells were occupied,
while 1442 background obstacle cells remained. Clearing, LiDAR, inflation and
controllers were excluded from that experiment. This does not qualify touching
objects, arbitrary finger poses, full-body exclusion or live navigation.
The margin was selected using these recordings; fresh validation remains required.

The robot_body_filter branch named ros2 was inspected on 2026-11-23. Its manifest
still requires catkin, roscpp and dynamic_reconfigure; ROS Index lists no Humble
release. No dependency on that package or source port is introduced here.

### Hand containment software checkpoint validation — 2026-11-23

The perception package passed 62 tests, including the recorded 214-point
regression, rotated/translated containment, invalid inputs, missing wrist TF,
and image-time lookup with a different future pose. The operator completed the
full workspace build/test gate without reported failures. The installed function
and installed profile reproduced all 39 available walking clouds and retained
all tested low-box points. Configuration validation passed. The live opt-in
profile has not yet been qualified; default filtering remains disabled.

### Packaged physical Wi-Fi candidate startup

The opt-in `g1_navigation_bringup/physical_wifi_candidate.launch.py` installs
the AMCL seed, navigation settings, behavior tree, and depth/velocity
helpers. The verified physical map is installed by g1_navigation_localization. It owns one Wi-Fi cloud relay, physical odometry
authority and robot-state publisher, depth projection and CUDA filter, scan
projection and diagnostic costmap, Map Server and AMCL, and the candidate
planner, controller, smoother, adapter, and navigator. The launch starts no
PC2 driver, physical arbiter, or command forwarder. The default AMCL profile
accepts an operator seed. The `gantry_start:=true` option selects the saved
physical reference at map x=0.747 m, y=0.718 m, yaw=1.18246 rad. Align the scan
with the fixed mapped walls before requesting a path.

The package-owned `g1_locomotion_control/config/physical_pc2_deployment.yaml`
and the candidate laptop adapter share deployment ceilings of 1.00 m/s
forward and 0.60 rad/s absolute yaw. The laptop candidate controller and
smoother limit operation to 0.25 m/s forward and 0.35 rad/s absolute yaw,
with zero commanded lateral velocity and the 0.30 rad/s minimum-turn tuning.
The PC2 watchdog remains 0.25 s and the adapter timeout remains 0.15 s.
The deployment ceilings are project settings. The 25 November bounded physical
trial recorded a 0.25 m/s peak forward command and a 0.217 rad/s peak absolute
yaw command, successful arrival, and stopped release. Independent validation
of motion at the deployment ceilings remains outside that trial.

The PC2 counterpart is `g1_bringup/pc2_navigation_wifi.launch.py`. It owns
Hardware State, the depth-only D435i, MID-360 Wi-Fi transport, and the Sport
arbiter. A child exit shuts down the launch. Verify PC2 clock sync before
starting Livox. Run the interaction launch separately because it uses the
same camera and hardware-state sources.

The 25 November two-launch check produced three plans and 68 nonzero
candidate commands, canceled as expected, recorded zero raw commands,
and confirmed an idle physical boundary with no checker failures.
