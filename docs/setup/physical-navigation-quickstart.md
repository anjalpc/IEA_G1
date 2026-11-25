# Physical navigation quickstart

## Standard startup — Wi-Fi navigation

Use this workflow to run candidate navigation on the G1 EDU U7
(`mode_machine` 5). Start **PC2 first, then the laptop**. The reference
workflows below are separate bringup modes.

Keep the robot stationary during startup. Stop any existing instances of
the shared sensor or navigation nodes before switching modes.

### 1. Start PC2 — sensors and Sport arbiter

On PC2, stop the interaction stack or standalone sensor processes before
starting the navigation-only launch. This launch owns Hardware State, the
depth-only D435i, MID-360, and the Sport arbiter.

**What:** Start the unified PC2 navigation stack.<br>
**Why:** Bring up the four PC2 processes with the tested Wi-Fi configuration.<br>
**Expected:** The four processes start and the arbiter publishes idle state.

```bash
cd /home/unitree/IEA_G1 &&
source /opt/ros/foxy/setup.bash &&
source livox_ws/install/setup.bash &&
source ros2_ws/install/setup.bash &&
test "$(timedatectl show -p NTPSynchronized --value)" = yes &&
ros2 launch g1_bringup pc2_navigation_wifi.launch.py
```

The launch uses the installed physical_pc2_deployment.yaml by default.
Set G1_PC2_DEPLOYMENT_PROFILE to the verified local profile path on a PC2
installation that has yet to install the packaged profile.

### 2. Start the laptop — navigation

In the sourced Humble development container with the CUDA depth environment
installed, stop any standalone laptop instances of the included nodes before
starting the packaged launch.

**What:** Start the laptop candidate navigation stack.<br>
**Why:** Load the verified map and start sensing, localization, and candidate navigation.<br>
**Expected:** Odometry and depth become healthy, and the navigation servers activate.

```bash
cd /workspace/IEA_G1/ros2_ws &&
source /opt/ros/humble/setup.bash &&
source install/setup.bash &&
ros2 launch g1_navigation_bringup physical_wifi_candidate.launch.py
```

### 3. Set the starting pose in RViz

Set **Fixed Frame** to `map`. Add **Map** on `/map` and **LaserScan**
on `/scan` with **Best Effort** reliability. With the robot stationary,
use **2D Pose Estimate** to seed its actual position and heading. Check that
the live scan lines up with fixed walls and corners.

If the robot is at the saved gantry position and heading, append
`gantry_start:=true` to the laptop launch command at startup. AMCL then
loads that starting pose automatically. Both starts load the same saved map.

### 4. Verify readiness before requesting a path

Open a separate terminal in the laptop development container. Sourcing
Humble loads the ROS CLI, and sourcing the workspace overlay makes the
installed G1 packages and message types available. The Wi-Fi variables
below let this terminal discover the PC2 sensors and laptop nodes on
domain 42. The running launch's environment is local to its own processes.

**What:** Check navigation lifecycle, depth status, and the raw velocity topic.<br>
**Why:** Confirm that navigation is ready and the physical forwarding path is idle.<br>
**Expected:** All three nodes are active, depth reports DIAGNOSTIC_CANDIDATE, and raw velocity has zero publishers.

```bash
cd /workspace/IEA_G1/ros2_ws &&
source /opt/ros/humble/setup.bash &&
source install/setup.bash &&
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp &&
export CYCLONEDDS_URI=file:///workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_bringup/config/physical/cyclonedds_laptop_wifi.xml

for node in /planner_server /controller_server /bt_navigator; do
  printf '%s ' "$node"
  timeout 8s ros2 lifecycle get "$node"
done
timeout 8s ros2 topic echo /g1/diagnostics/depth/status --once
ros2 topic info /g1/navigation/cmd_vel_raw
```

The operating limits are **0.25 m/s forward** and **0.35 rad/s absolute
yaw**. Physical forwarding is a separate supervised trial. This startup
is complete once the stationary checks pass.

### 5. Shut down

With the robot stopped and any active trial finished, press **Ctrl+C** in
the laptop launch terminal, then in the PC2 navigation launch terminal.
Close RViz when you no longer need the display.

## Map a new environment

Keep the unified PC2 navigation launch running. On the laptop, stop the
candidate navigation launch before starting mapping. SLAM Toolbox owns
`map → odom` while mapping. AMCL owns that transform during saved-map
navigation.

### 1. Start mapping

**What:** Start the Wi-Fi mapping launch in the laptop development container.<br>
**Why:** Build an occupancy map from the live LiDAR scan.<br>
**Expected:** After stationary warmup, RViz displays a growing map.

```bash
cd /workspace/IEA_G1/ros2_ws &&
source /opt/ros/humble/setup.bash &&
source install/setup.bash &&
ros2 launch g1_navigation_bringup physical_wifi_mapping.launch.py
```

In the Wi-Fi RViz window, set **Fixed Frame** to `map`. Add **Map** on
`/map` and **LaserScan** on `/scan`, with **Best Effort** reliability for
the scan. Keep PointCloud2 displays disabled while moving. Once the map
and scan appear, use your normal supervised manual procedure to map the
area. Stop the robot before saving.

### 2. Save the map for navigation in RViz

1. Leave SLAM Toolbox running. In RViz, choose **Panels → Add New Panel**
   and select **SlamToolboxPlugin**.
2. In the **Save Map** field, enter a new absolute path **without an
   extension**, such as
   `/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/new_lab`.
3. Click **Save Map**. Confirm that both `new_lab.yaml` and `new_lab.pgm`
   exist before stopping mapping. Keep the pair together. The YAML
   references the image and is the file you pass to navigation.

If the panel or **Save Map** is unavailable, use the terminal **Save the
physical map** procedure in the reference section with a unique filename.
The RViz button requires the running SLAM Toolbox map-saver service.

### 3. Save or reopen an editable SLAM session

| RViz option | What it does | When to use it |
| --- | --- | --- |
| **Save Map** | Exports the occupancy image and YAML for Map Server and AMCL. | Navigate in the finished map. |
| **Serialize Map** | Saves the SLAM pose graph and associated scan data. | Preserve the session so you can extend or refine the map later. |
| **Deserialize Map** | Loads a previously serialized pose graph into SLAM Toolbox. | Resume a mapping session you saved earlier. |

To continue mapping another day, use **Serialize Map** before stopping
SLAM Toolbox. Give the session its own absolute path and keep the generated
pose-graph and data files together. Later, start the mapping launch, stay
stationary, and use **Deserialize Map** with the saved session basename.
If the panel offers a starting pose or matching mode, choose one that
corresponds to where the robot actually is. Check that the loaded map and
live scan align before moving or adding more scans.

For later editing and navigation, save **both** the serialized session and
the occupancy map. After extending a session, use **Save Map** again to
export the updated occupancy map. Pass that new YAML to AMCL.

### 4. Switch to navigation with the new map

Stop the laptop mapping launch with **Ctrl+C** after saving. Leave the PC2
launch running. In the same sourced laptop terminal:

**What:** Start candidate navigation with the saved map YAML.<br>
**Why:** Switch from mapping to estimating the robot pose on the finished map.<br>
**Expected:** RViz shows the new map and AMCL is ready for an initial pose.

```bash
ros2 launch g1_navigation_bringup physical_wifi_candidate.launch.py   map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/new_lab.yaml
```

Replace `new_lab.yaml` with the file you saved. Leave `gantry_start` off for
a new environment. Use **2D Pose Estimate** at the actual position and
heading, then check scan alignment against fixed walls. Complete the
stationary readiness check above before requesting a candidate-only goal.

## Reference workflows

Use these procedures for wired mapping, standalone localization, or recording-only diagnostics. Stop the standard Wi-Fi launch before starting a reference workflow that owns the same nodes.

### Safety rules

- Use G1 EDU U7 `mode_machine` 5 only.
- Never run duplicate Livox or `hardware_state` processes.
- Never run mapping and localization simultaneously.
- Keep RViz PointCloud2 disabled while the robot moves.
- Stop if LiDAR clouds are not about 10 Hz and roughly 20,000 points.
- Use `Ctrl+C` in the foreground terminal to stop a stack.
- Every new terminal needs its own setup and exports.

### PC2 Terminal 1: measured robot state

Run this on PC2 and leave it in the foreground:

```bash
cd /home/unitree/IEA_G1/ros2_ws

source /opt/ros/foxy/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file:///home/unitree/IEA_G1/livox_ws/config/cyclonedds_pc2_wired.xml"

if pgrep -af '/g1_hardware_bridge/hardware_state' |
  grep -v pgrep
then
  echo "ALREADY RUNNING: hardware_state"
else
  /home/unitree/IEA_G1/ros2_ws/install/g1_hardware_bridge/lib/g1_hardware_bridge/hardware_state     --ros-args     --params-file     /home/unitree/IEA_G1/ros2_ws/install/g1_hardware_bridge/share/g1_hardware_bridge/config/hardware.yaml
fi
```

Healthy startup reports `Physical LowState source ready`. If native samples
stop, the supervisor waits for DDS cleanup and performs at most three
automatic helper restarts. Do not start a second direct LowState helper while
the supervised source is running.

### PC2 Terminal 2: MID-360 driver

First confirm that PC2 has an NTP-selected peer:

```bash
ntpq -pn

if ntpq -pn |
  awk '$1 ~ /^\*/ {found=1} END {exit !found}'
then
  echo "PASS: PC2 time has a selected NTP peer"
else
  echo "STOP: synchronize PC2 time before starting Livox"
fi
```

Only after the NTP check passes, start Livox in the foreground:

```bash
cd /home/unitree/IEA_G1/livox_ws

source /opt/ros/foxy/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file:///home/unitree/IEA_G1/livox_ws/config/cyclonedds_pc2_wired.xml"

if pgrep -af '/livox_ros_driver2/livox_ros_driver2_node' |
  grep -v pgrep
then
  echo "ALREADY RUNNING: Livox driver"
else
  ros2 run     livox_ros_driver2     livox_ros_driver2_node     --ros-args     -p xfer_format:=0     -p multi_topic:=0     -p data_src:=0     -p publish_freq:=10.0     -p output_data_type:=0     -p frame_id:=mid360_link     -p user_config_path:=/home/unitree/IEA_G1/livox_ws/config/mid360_pc2.json     -r /livox/lidar:=/g1/sensors/lidar/points     -r /livox/imu:=/g1/sensors/lidar/imu_raw
fi
```

### Laptop setup: run in every new ROS terminal

Use this block at the beginning of every new terminal inside the development
container:

```bash
cd /workspace/IEA_G1/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS2CLI_DISABLE_DAEMON=1
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file:///workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_bringup/config/physical/cyclonedds_laptop_wired.xml"
```

### Laptop preflight

After running the laptop setup block:

```bash
echo "=== REQUIRED TOPICS ==="

ros2 topic list -t |
grep -E   '^/g1/sensors/lidar/(points|imu_raw)|^/g1/state/imu_raw|^/joint_states'

echo "=== CLOUD RATE ==="

timeout --signal=INT --kill-after=2s 6s   stdbuf -oL -eL   ros2 topic hz   /g1/sensors/lidar/points

echo "=== CLOUD WIDTH ==="

timeout 8s   ros2 topic echo   /g1/sensors/lidar/points   sensor_msgs/msg/PointCloud2   --qos-reliability reliable   --field width   --once
```

Healthy LiDAR output is about 10 Hz with roughly 20,000 points per cloud.
Approximately 96 points at packet rate is invalid; restart Livox only after
PC2 time is synchronized.

### Laptop Terminal 1: physical mapping

Run the laptop setup block, then:

```bash
ros2 launch   g1_navigation_bringup   physical_mapping.launch.py
```

Keep the robot stationary until GLIM completes initial IMU-state estimation.

### Laptop Terminal 2: RViz

Run the laptop setup block, then:

```bash
rviz2
```

Set:

- Fixed Frame: `map`
- Map topic: `/map`
- LaserScan topic: `/scan`
- Add TF and RobotModel
- Do not add PointCloud2 while moving

### Save the physical map

While physical mapping is still running, open another laptop terminal, run
the laptop setup block, then:

```bash
map_directory=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical

mkdir -p "${map_directory}"

ros2 run nav2_map_server map_saver_cli   -f "${map_directory}/building_map"   --ros-args   -p save_map_timeout:=10.0   -p free_thresh_default:=0.25   -p occupied_thresh_default:=0.65
```

Verify that both files exist:

```bash
ls -lh   "${map_directory}/building_map.yaml"   "${map_directory}/building_map.pgm"
```

Stop physical mapping with `Ctrl+C` before starting localization.

### Laptop Terminal 1: saved-map localization

Run the laptop setup block, then:

```bash
ros2 launch   g1_navigation_bringup   physical_localization.launch.py   map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map_verified_cellfix.yaml
```

In RViz, use **2D Pose Estimate** to provide the robot's actual starting pose.
Do not start localization and mapping together.

### Laptop Terminal 1: planning-only navigation

Stop the localization-only launch before starting this stack.
Run the laptop setup block, then:

```bash
ros2 launch g1_navigation_bringup physical_planning.launch.py map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map_verified_cellfix.yaml autostart:=true
```

Use **2D Pose Estimate** in RViz before requesting a path.
This launch starts localization and global planning only.
It does not start a controller, `cmd_vel`, or robot commands.

If obstacles enclose the robot in the inflated costmap, planning
correctly aborts instead of crossing them.

### Laptop Terminal 1: recording-only control dry run

Stop the planning-only launch, run the laptop setup block, then start the dry run:

    ros2 launch g1_navigation_bringup physical_control_dry_run.launch.py map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map_verified_cellfix.yaml

This stack uses recording backend 1 and cannot move the physical robot. Set the initial pose with RViz 2D Pose Estimate. The planner server, controller server, and velocity smoother must each report active [3]. After every proof, locomotion state must return to owner 0, backend 1, and stopped true. Set the RViz LaserScan Reliability Policy to Best Effort. A Navigation command stream timed out warning after cancellation is the expected watchdog release.

### Clean shutdown

- Stop RViz with `Ctrl+C`.
- Stop the foreground laptop mapping or localization launch with `Ctrl+C`.
- Stop the foreground PC2 Livox process with `Ctrl+C`.
- Stop the foreground PC2 hardware-state process with `Ctrl+C`.

Confirm no duplicates remain on PC2:

```bash
pgrep -af   'g1_hardware_bridge.low_state.helper|hardware_state|livox_ros_driver2_node'   || echo "PASS: physical PC2 publishers stopped"
```

### Opt-in Wi-Fi mapping

Use config/physical/cyclonedds_laptop_wifi.xml and cyclonedds_pc2_wifi.xml from g1_navigation_bringup. Current interfaces/addresses are wlp3s0/192.168.188.117 and wlan0/192.168.188.160; verify them before startup. Deploy the PC2 XML by SCP to /home/unitree/IEA_G1/livox_ws/config/cyclonedds_pc2_wifi.xml and use that persistent path as CYCLONEDDS_URI for domain-42 PC2 sensor nodes.

For this mode, change only the Livox cloud remap to /livox/lidar:=/g1/transport/lidar/points_wifi. Keep the IMU remap unchanged. Do not also publish the canonical cloud directly from PC2.

After building and sourcing the laptop overlay, run ros2 launch g1_navigation_bringup physical_wifi_mapping.launch.py. It selects the laptop Wi-Fi profile and reliable GLIM profile and starts one relay. Do not run an additional manual relay, localization launch, or mapping launch alongside it. Diagnostic and recording terminals still need domain 42 and the Wi-Fi CYCLONEDDS_URI explicitly set.

Keep PointCloud2 disabled during movement. Bounded stationary/manual continuity checks passed; longer route coverage, autonomous control, and physical communications-loss stopping still require validation.
