# Source the project overlay after it has been built.
project_overlay="/workspace/IEA_G1/ros2_ws/install/setup.bash"
if [ -f "${project_overlay}" ]; then
    source "${project_overlay}"
fi
unset project_overlay

# Make the supported development container unmistakable.
if [ "${IEA_G1_CONTAINER:-}" = "1" ]; then
    PS1='\[\033[1;33m\][IEA_G1 container]\[\033[0m\] \[\033[1;32m\]\u@\h\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ '
fi
