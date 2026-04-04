#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace if built
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# CoppeliaSim path
if [ -d "${COPPELIASIM_ROOT}" ]; then
    export LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}
    export PATH=${COPPELIASIM_ROOT}:${PATH}
fi

exec "$@"
