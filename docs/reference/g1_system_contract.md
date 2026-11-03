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
`enp2s0f0` at `192.168.123.99` and PC2 to `eth0` at
`192.168.123.164`, disable multicast, and use explicit peers. The PC2
configuration is restricted to ROS domain `42`. It must not apply its
multicast-disabled application settings to the child native Unitree
domain-0 LowState helper.

The workstation uses pinned CPU-only GLIM 1.2.2 packages. GLIM consumes the
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

The separate opt-in `physical_odometry_authority.launch.py` includes that staging stack and adds exactly one final authority boundary. It requires exact `odom -> pelvis` staged frames, finite state, a unit quaternion, strictly advancing timestamps, a 20-sample warmup, freshness within `0.25 s`, translation steps no larger than `0.15 m`, and rotation steps no larger than 15 degrees. Only while every check passes does it publish canonical planar `odom -> base_footprint` messages on `/g1/state/odometry` and the matching `odom -> base_footprint -> pelvis` TF chain. The canonical message has zero height and no roll or pitch; measured pelvis height and tilt remain on the second TF edge. This launch still starts no mapping, Nav2, localization, locomotion, or robot-command node.

PC2 system time must be synchronized before starting the Livox driver. Starting it across a large system-clock correction can defeat scan aggregation and produce approximately 2 kHz packet-sized, 96-point clouds. A healthy MID-360 stream is approximately 10 Hz with about 20,000 points per cloud. Restart the driver after time synchronization and do not start GLIM until that shape is restored.

GLIM is configured with `publish_imu2lidar` disabled. Therefore GLIM owns
only its isolated `glim_odom -> livox_frame` estimator tree, while the
physical robot-state runtime owns the separate
`pelvis -> torso_link -> mid360_link` kinematic tree. No transform connects
`livox_frame` to `mid360_link` at this stage; this deliberately prevents
either sensor frame from acquiring a duplicate parent.

The validated GLIM build publishes finite but all-zero covariance. These
arrays remain unknown; the project neither invents covariance nor weakens the
generic navigation-state validator. The dedicated boundary provides
source-specific liveness and plausibility checks but does not yet authorize
navigation-facing odometry or TF.

A suspended stationary proof produced approximately 10 Hz qualified output,
about 1.4 mm translation drift over 59.8 seconds, and about 0.0034 degrees of
yaw drift. A grounded proof measured 0.4678 m for a marked 0.50 m forward
movement, with 0.0296 m lateral displacement, 0.0651 m vertical displacement,
and 2.733 degrees of unintended rotation. A physical 90-degree left turn
produced 81.184 degrees of GLIM rotation, 0.1197 m translation, and bounded
tilt.

These results qualify direction and approximate metric scale for staging.
They do not yet prove extended walking robustness or authorize connection to
`/g1/state/odometry`, mapping, Nav2, or locomotion.

### Physical LowState source

On PC2, `g1_hardware_bridge` owns one read-only physical source for measured
waist and pelvis IMU state. The isolated native helper subscribes once to
Unitree `rt/lowstate` on DDS domain `0`; it creates no command publisher and
never writes robot state.

Each accepted native tick produces one atomic sample containing:

- waist yaw, roll, and pitch positions;
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

Joint output order is yaw, roll, then pitch. All motor and IMU values must be
finite, the quaternion must have unit length within the declared tolerance,
and the native tick must advance. Invalid IMU data rejects the entire atomic
sample rather than releasing partial joint state.

Publication is rate-limited to 50 Hz. Both messages from one callback use the
same supervising-node receipt timestamp. There are no default, commanded,
interpolated, integrated, or replayed values.

Physical IMU covariance has not been measured. The raw IMU therefore publishes
zero covariance arrays to mean unknown covariance. `g1_state_estimation`
continues rejecting that raw source as `COVARIANCE_REQUIRED`; physical
calibration or estimator-specific covariance qualification is required before
physical navigation may consume it.

The native helper is deliberately separated from application ROS so Unitree
DDS domain `0` cannot collide with ROS application domain `42`.
`g1_bringup/event_interaction.launch.py` supervises the parent node as
required: helper loss shuts down the supported physical stack instead of
leaving downstream consumers with frozen state.

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
physical description from the same URDF. It requires exactly one
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

The initial projection envelope is 0.05 m through 1.75 m above
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
than being inferred.

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
| Unitree SDK2 Python | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` |
| Unitree MuJoCo | `ae6a8403e272733e9996ef59990880330496177f` |
| Unitree RL Lab deployment subset | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` |
| MuJoCo | 3.3.6 |
| ONNX Runtime | 1.20.1 Python; pinned Unitree native runtime for deployment |
| NumPy | 1.26.4 |
| ROS development environment | ROS 2 Humble |
| Physical PC2 environment | ROS 2 Foxy |

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

Never run simulation on hardware domain 0. Never run two low-level physical
controllers simultaneously. The physical hardware bridge is the only project
component permitted to command the robot directly.

Dependency definition: [`docker/development/Dockerfile`](../../docker/development/Dockerfile).

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
