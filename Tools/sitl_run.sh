#!/bin/bash
#
# run multiple instances of the 'px4' binary, with the gazebo SITL simulation
# It assumes px4 is already built, with 'make px4_sitl_default gazebo'

# The simulator is expected to send to TCP port 4560+i for i in [0, N-1]
# For example gazebo can be run like this:
#./Tools/gazebo_sitl_multiple_run.sh -n 10 -m iris

function cleanup() {
	pkill -x px4
	pkill gzclient
	pkill gzserver
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
src_path="$SCRIPT_DIR/.."

echo ${SCRIPT}
source ${src_path}/setup.bash

if [ "$1" == "-h" ] || [ "$1" == "--help" ]
then
	echo "Usage: $0 [-m <vehicle_model>] [-w <world>]"
	exit 1
fi

while getopts m:w:s: option
do
	case "${option}"
	in
		m) VEHICLE_MODEL=${OPTARG};;
		w) WORLD=${OPTARG};;
		s) SDF_MODEL=${OPTARG};;
	esac
done

world=${WORLD:=empty}
export PX4_SIM_MODEL=${VEHICLE_MODEL:=iris}
GAZEBO_MODEL=${SDF_MODEL};

build_path=${PX4_ROOT}/build/px4_sitl_default
mavlink_udp_port=14560
mavlink_tcp_port=4560

echo "killing running instances"
pkill -x px4 || true

sleep 1

source ${PX4_ROOT}/Tools/setup_gazebo.bash ${PX4_ROOT} ${PX4_ROOT}/build/px4_sitl_default

echo "Starting gazebo"
gzserver ${PX4_ROOT}/Tools/sitl_gazebo/worlds/${world}.world --verbose &
sleep 5

echo "Spawning ${GAZEBO_MODEL}"

gz model --spawn-file=${src_path}/models/${GAZEBO_MODEL}/${GAZEBO_MODEL}.sdf --model-name=${GAZEBO_MODEL} -x 0.0 -y 3.0 -z 0.83

trap "cleanup" SIGINT SIGTERM EXIT

echo "Starting gazebo client"
gzclient &

# spawn_model ${PX4_SIM_MODEL}

working_dir="$build_path/tmp/rootfs"
[ ! -d "$working_dir" ] && mkdir -p "$working_dir"

pushd "$working_dir" &>/dev/null
echo "starting instance $N in $(pwd)"
../../bin/px4 "$build_path/etc" -w sitl_${MODEL} -s etc/init.d-posix/rcS

popd &>/dev/null
