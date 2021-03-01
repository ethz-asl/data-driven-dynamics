# data-driven-dynamics
[![Build Tests](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml/badge.svg)](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml)

This repository allows a data-driven dynamics model for PX4 SITL(Software-In-The-Loop) simulations.

## Setting up the environment
This plugin depends on [PX4 SITL](https://github.com/PX4/PX4-SITL_gazebo). Therefore custom messages of PX4 SITL needs to be linked. Therefore, prior to building this package, PX4 and corresponding SITL package needs to be built.

In case you have not cloned the firmware repository
```
git clone --recursive https://github.com/ethz-asl/ethzasl_fw_px4.git ~/dev/PX4-Autopilot ~/src/PX4-Autopilot
```
```
cd <Firmware Dir>
DONT_RUN=1 make px4_sitl gazebo
```

The build directory of PX4 can be linked by setting the environment variable `PX4_ROOT` to the root of the firmware directory. Set the environment variable `export PX4_ROOT=~/src/PX4-Autopilot` or add it to your `bashrc`
```
echo "export PX4_ROOT=~/src/PX4-Autopilot" >> ~/.bashrc
source ~/.bashrc
```

## Build
After the environment has been setup as described in the previous section, build the package as the following.
```
mkdir build
cd build
cmake ..
make
```

## Running the Simulation
To run the simulation,
```
source setup.bash
Tools/sitl_run.sh -m techpod_aerodynamic
```
