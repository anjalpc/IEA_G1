# Physical navigation quickstart

## Standard startup — saved-map navigation

Run **one launch on PC2 and one on the laptop**. Start PC2 first.

| Machine | Unified launch file | What it starts |
| --- | --- | --- |
| PC2 | `pc2_navigation_hardware.launch.py` | Hardware State, depth-only D435i, MID-360, and the Sport arbiter |
| Laptop development container | `physical_navigation_standby.launch.py` | Odometry, depth processing, saved-map localization, planning, and controller output |

Both launches default to **Wi-Fi**. For **Ethernet**, append
`transport:=wired` to both launch commands below. The launches select their
own DDS profiles. Source the ROS overlays before launching. Set DDS
explicitly in separate RViz and diagnostic terminals.

For Ethernet, confirm the laptop uses `eno1` at `192.168.123.99` and PC2
uses `eth0` at `192.168.123.164`. Route the cable clear of the robot's feet
and intended corridor. PC2 publishes the LiDAR cloud directly to
`/g1/sensors/lidar/points` in wired mode. Wi-Fi mode uses the laptop relay.

Keep the G1 EDU U7 (`mode_machine` 5) stationary during startup. Stop
existing instances of the same sensor or navigation nodes before
switching modes. Physical forwarding is a separate supervised trial.

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
ros2 launch g1_bringup pc2_navigation_hardware.launch.py
```

The launch uses the installed physical_pc2_deployment.yaml by default.
Set G1_PC2_DEPLOYMENT_PROFILE to the verified local profile path on a PC2
installation that has yet to install the packaged profile.

### 2. Start the laptop — navigation

The laptop launch defaults to **MPPI**. Append `controller:=dwb` to use
the DWB fallback. Either controller starts in standby. Physical motion
requires a separately initiated, guarded trial.

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
ros2 launch g1_navigation_bringup physical_navigation_standby.launch.py
```

### 3. Open RViz and check localization

Use a separate laptop-container terminal for RViz. Set `transport` to
match both launches. Use this same setup in every separate diagnostic or
recording terminal.

**What:** Source the laptop workspace and select the matching DDS profile.<br>
**Why:** Let RViz and diagnostic commands discover the running ROS graph.<br>
**Expected:** The terminal uses application domain 42 and the selected laptop DDS XML.

```bash
cd /workspace/IEA_G1/ros2_ws &&
source /opt/ros/humble/setup.bash &&
source install/setup.bash &&
transport=wifi &&
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp &&
export CYCLONEDDS_URI="file://$PWD/src/navigation/g1_navigation_bringup/config/physical/cyclonedds_laptop_${transport}.xml"
```

For Ethernet, replace `transport=wifi` with `transport=wired` in that block.

**What:** Open RViz with `map` as the fixed frame.<br>
**Why:** View the robot, live scan, and saved map in the same coordinate frame.<br>
**Expected:** RViz opens. The map frame becomes available after AMCL initializes.

```bash
rviz2 -f map
```

In **Displays → Add**, configure these views:

| Display | Topic and settings | What to check |
| --- | --- | --- |
| **Map** | `/map`, Reliable, Transient Local | The intended saved map is loaded. |
| **LaserScan** | `/scan`, Best Effort | Live returns align with fixed mapped walls and corners. |
| **RobotModel** | Description Source **Topic**, `/robot_description` | The model appears at the robot's estimated location. |
| **Map**, named **Navigation local costmap** | `/local_costmap/costmap` | Inspect nearby collision costs if the controller rejects a path. |
| **Path**, optional | `/plan` | Review the route after a plan has been requested. |

Switch **Views → Type** to **TopDownOrtho** to judge alignment. With the
robot stationary, use **2D Pose Estimate** to seed its actual map position
and heading. When the robot is at the saved gantry position and heading,
you can instead append `gantry_start:=true` to the laptop launch command
and check the resulting alignment without clicking a pose.

For stationary depth troubleshooting, add **PointCloud2** on
`/g1/diagnostics/depth/obstacles` with **Best Effort** reliability.
Use `/g1/diagnostics/depth/points` to compare the raw cloud. Disable these
displays before moving. If map localization is unavailable, use the
`pelvis` fixed frame for this depth-only view and restore `map` afterward.

Use **File → Save Config As** if you want to reuse the layout.

### 4. Verify readiness before requesting a path

In another laptop-container terminal, repeat the sourcing and transport
setup from step 3. Keep the robot stationary.

**What:** Check navigation lifecycle, depth status, and the raw velocity topic.<br>
**Why:** Confirm that navigation is ready and physical forwarding is idle.<br>
**Expected:** All three nodes are active, depth reports DIAGNOSTIC_CANDIDATE, and raw velocity has zero publishers.

```bash
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

For **wired mapping**, start PC2 with `transport:=wired` first. In the
laptop terminal, run the terminal setup in the RViz step from standard
startup with `transport=wired`, then use the wired mapping launch:

**What:** Start the wired laptop mapping launch.<br>
**Why:** Build the map from the direct LiDAR topic published by PC2.<br>
**Expected:** RViz shows the map as SLAM Toolbox receives scans.

```bash
ros2 launch g1_navigation_bringup physical_mapping.launch.py
```

In RViz on the selected transport, set **Fixed Frame** to `map`. Add **Map** on
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

If the panel or **Save Map** is unavailable, use **Save a map from the
terminal** below. The RViz button requires the running SLAM Toolbox
map-saver service.

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
ros2 launch g1_navigation_bringup physical_navigation_standby.launch.py map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/new_lab.yaml
```

Replace `new_lab.yaml` with the file you saved. Leave `gantry_start` off for
a new environment. Use **2D Pose Estimate** at the actual position and
heading, then check scan alignment against fixed walls. Complete the
stationary readiness check above before requesting a candidate-only goal.

## Save a map from the terminal

Use this fallback while SLAM Toolbox is still running if the RViz
**Save Map** button is unavailable. Use the standard RViz step's terminal
setup with the transport you are running. Keep the robot stationary.

**What:** Save the current occupancy map to a unique name.<br>
**Why:** Produce a YAML and PGM pair for saved-map navigation.<br>
**Expected:** Both files are listed in the localization package's physical-map directory.

```bash
map_dir=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical
map_path="$map_dir/site_$(date +%Y%m%d_%H%M%S)"
ros2 run nav2_map_server map_saver_cli -f "$map_path" --ros-args -p save_map_timeout:=10.0 -p free_thresh_default:=0.25 -p occupied_thresh_default:=0.65 &&
ls -lh "$map_path.yaml" "$map_path.pgm"
```

## Check LiDAR if the map or scan stays blank

Use a diagnostic terminal with the same transport setup as RViz.

**What:** Measure the canonical LiDAR stream and inspect one cloud's width.<br>
**Why:** Check that full scans are reaching the laptop.<br>
**Expected:** About 10 Hz and roughly 20,000 points per cloud.

```bash
timeout --signal=INT --kill-after=2s 8s ros2 topic hz /g1/sensors/lidar/points
timeout 8s ros2 topic echo /g1/sensors/lidar/points sensor_msgs/msg/PointCloud2 --qos-reliability best_effort --field width --once
```

Clouds of about 96 points at packet rate indicate invalid scan aggregation.
Verify PC2 clock synchronization before restarting the hardware launch.
