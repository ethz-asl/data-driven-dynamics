# data-driven-dynamics
[![Build Tests](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml/badge.svg)](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml)

This repository allows a data-driven dynamics model for PX4 SITL(Software-In-The-Loop) simulations.

## Build
This plugin depends on [PX4 SITL](https://github.com/PX4/PX4-SITL_gazebo). Therefore custom messages of PX4 SITL needs to be linked.

This can be done by setting the environment variable `PX4_ROOT` to the root of the PX4 Autopilot firmware directory.
```
export PX4_ROOT=~/src/PX4-Autopilot
mkdir build
cd build
cmake ..
make
```
