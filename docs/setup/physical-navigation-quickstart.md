# Physical navigation quick start

This is the terminal-by-terminal operator runbook for physical MID-360
mapping and saved-map localization.

Mapping and localization are read-only at this stage. Neither launch starts
Nav2 control, locomotion, `cmd_vel`, or robot commands.

## Safety rules

- Use G1 EDU U7 `mode_machine` 5 only.
- Never run duplicate Livox or `hardware_state` processes.
- Never run mapping and localization simultaneously.
- Keep RViz PointCloud2 disabled while the robot moves.
- Stop if LiDAR clouds are not about 10 Hz and roughly 20,000 points.
- Use `Ctrl+C` in the foreground terminal to stop a stack.
- Every new terminal needs its own setup and exports.

## PC2 Terminal 1: measured robot state

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

## PC2 Terminal 2: MID-360 driver

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

## Laptop setup: run in every new ROS terminal

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

## Laptop preflight

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

## Laptop Terminal 1: physical mapping

Run the laptop setup block, then:

```bash
ros2 launch   g1_navigation_bringup   physical_mapping.launch.py
```

Keep the robot stationary until GLIM completes initial IMU-state estimation.

## Laptop Terminal 2: RViz

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

## Save the physical map

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

## Laptop Terminal 1: saved-map localization

Run the laptop setup block, then:

```bash
ros2 launch   g1_navigation_bringup   physical_localization.launch.py   map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map.yaml
```

In RViz, use **2D Pose Estimate** to provide the robot's actual starting pose.
Do not start localization and mapping together.

## Laptop Terminal 1: planning-only navigation

Stop the localization-only launch before starting this stack.
Run the laptop setup block, then:

```bash
ros2 launch g1_navigation_bringup physical_planning.launch.py map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map.yaml autostart:=true
```

Use **2D Pose Estimate** in RViz before requesting a path.
This launch starts localization and global planning only.
It does not start a controller, `cmd_vel`, or robot commands.

If obstacles enclose the robot in the inflated costmap, planning
correctly aborts instead of crossing them.

## Laptop Terminal 1: recording-only control dry run

Stop the planning-only launch, run the laptop setup block, then start the dry run:

    ros2 launch g1_navigation_bringup physical_control_dry_run.launch.py map:=/workspace/IEA_G1/ros2_ws/src/navigation/g1_navigation_localization/maps/physical/building_map.yaml

This stack uses recording backend 1 and cannot move the physical robot. Set the initial pose with RViz 2D Pose Estimate. The planner server, controller server, and velocity smoother must each report active [3]. After every proof, locomotion state must return to owner 0, backend 1, and stopped true. Set the RViz LaserScan Reliability Policy to Best Effort. A Navigation command stream timed out warning after cancellation is the expected watchdog release.

## Clean shutdown

- Stop RViz with `Ctrl+C`.
- Stop the foreground laptop mapping or localization launch with `Ctrl+C`.
- Stop the foreground PC2 Livox process with `Ctrl+C`.
- Stop the foreground PC2 hardware-state process with `Ctrl+C`.

Confirm no duplicates remain on PC2:

```bash
pgrep -af   'g1_hardware_bridge.low_state.helper|hardware_state|livox_ros_driver2_node'   || echo "PASS: physical PC2 publishers stopped"
```
