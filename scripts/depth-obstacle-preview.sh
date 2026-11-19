#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requested="${1:-auto}"
case "${requested}" in
    auto|cpu|cuda) ;;
    *) echo "Usage: $0 [auto|cpu|cuda]" >&2; exit 2 ;;
esac

source /opt/ros/humble/setup.bash
source "${project_root}/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${project_root}/ros2_ws/src/navigation/g1_navigation_bringup/config/physical/cyclonedds_laptop_wifi.xml"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

depth_python=python3
backend=cpu
if [[ "${requested}" != cpu && -x /opt/iea-depth-gpu/bin/python ]]; then
    depth_python=/opt/iea-depth-gpu/bin/python
    backend=cuda
elif [[ "${requested}" == cuda ]]; then
    echo "CUDA environment absent; using CPU diagnostic preview." >&2
fi

exec "${depth_python}" -m g1_navigation_perception.depth_obstacle_preview \
    --ros-args -p "compute_backend:=${backend}"
