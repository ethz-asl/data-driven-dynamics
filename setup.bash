#!/bin/bash
#
# Setup environment to make this plugin visible to PX4
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:${SCRIPT_DIR}/build
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:${SCRIPT_DIR}/models
