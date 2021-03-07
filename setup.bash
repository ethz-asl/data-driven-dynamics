#!/bin/bash
#
# Setup environment to make this plugin visible to PX4
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:${SCRIPT_DIR}/build
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:${SCRIPT_DIR}/models
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SCRIPT_DIR}/build

export PX4_ROOT=~/src/PX4-Autopilot

export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:${PX4_ROOT}/build/px4_sitl_gazebo_default/build_gazebo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PX4_ROOT}/build/px4_sitl_gazebo_default/build_gazebo

# Create symbolic link of logging directory if doesn't exist
ln -snf ${PX4_ROOT}/build/px4_sitl_default/logs $SCRIPT_DIR/logs
