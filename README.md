# data-driven-dynamics

[![Build Tests](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml/badge.svg)](https://github.com/Jaeyoung-Lim/data-driven-dynamics/actions/workflows/build_test.yml)

This repository allows a data-driven dynamics model for PX4 SITL(Software-In-The-Loop) simulations.

## Setting up the environment

This plugin depends on [PX4 SITL](https://github.com/PX4/PX4-SITL_gazebo). Therefore custom messages of PX4 SITL needs to be linked. Therefore, prior to building this package, PX4 and corresponding SITL package needs to be built.

In case you have not cloned the firmware repository

```
git clone --recursive https://github.com/ethz-asl/ethzasl_fw_px4.git ~/src/PX4-Autopilot
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

The use the parametric model structure you need to install python 3.8 and the needed python libraries. It is strongly advised to install the pip packages in a [virtual enviroment](https://docs.python.org/3/tutorial/venv.html) setup for this project.

Install the dependencies:

```
cd Tools/parametric_model
pip3 install -r requirements.txt
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
Tools/sitl_run.sh -m techpod -s techpod_aerodynamics
```

## Generating a Parametric Model from Log File

Link the latest log files to your local logs folder using:

```
source setup.bash
```

Generate the parametric model using:

```
cd Tools/parametric_model
python3 generate_parametric_model model log_file
```

Hereby the arguments model and log_file can be used to specify the model and the log files respectively. As an example you could use:

```
python3 generate_parametric_model simple_quadrotor_model logs/2021-04-12/14_28_28.ulg
```

Current model choices are:

- simple_quadrotor_model

- quad_plane_model
