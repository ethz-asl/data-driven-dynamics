#!/bin/bash
#
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
	echo "Usage: $0 [-m <vehicle_model>] [-w <world>] [-s <sdf model>]"
	echo "	-s sdf model: override spawned sdf model"
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
export PX4_SIM_MODEL=${VEHICLE_MODEL:=techpod}
model=${SDF_MODEL:=none};

if [ "$model" == "none" ]; then
	# If the sdf model override is not specified with the -s flag, use the same as the model
	model=${VEHICLE_MODEL}
fi

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


# Check all paths in ${GAZEBO_MODEL_PATH} for specified model
IFS_bak=$IFS
IFS=":"
for possible_model_path in ${GAZEBO_MODEL_PATH}; do
	if [ -z $possible_model_path ]; then
		continue
	fi
	# trim \r from path
	possible_model_path=$(echo $possible_model_path | tr -d '\r')
	if test -f "${possible_model_path}/${model}/${model}.sdf" ; then
		modelpath=$possible_model_path
		break
	fi
done
IFS=$IFS_bak

echo "Spawning ${model} in ${modelpath}"

gz model --spawn-file=${modelpath}/${model}/${model}.sdf --model-name=${model} -x 0.0 -y 3.0 -z 0.83

trap "cleanup" SIGINT SIGTERM EXIT

echo "Starting gazebo client"
gzclient &

# spawn_model ${PX4_SIM_MODEL}

working_dir="$build_path/tmp/rootfs"
[ ! -d "$working_dir" ] && mkdir -p "$working_dir"

pushd "$working_dir" &>/dev/null
echo "starting instance $N in $(pwd)"
../../bin/px4 "$build_path/etc" -s etc/init.d-posix/rcS

popd &>/dev/null
