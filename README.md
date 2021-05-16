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

Update submodules:

```
make submodulesupdate
```

Install the dependencies from the project folder:

```
make install-dependencies
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

Generate the parametric model using a log file (ulog or csv):

```
python3 Tools/parametric_model/generate_parametric_model model log_file
```

Or more simply from the root of the repository

```
make estimate-model [model=<modeltype>] [log=<log path>]
```

Hereby the arguments model and log_file can be used to specify the model and the log files respectively. As an example you could use the reference log_files:

```
python Tools/parametric_model/generate_parametric_model.py --model quad_plane_model resources/simple_quadplane_model.ulg
```

```
python3 Tools/parametric_model/generate_parametric_model.py --model quad_plane_model resources/simple_quadplane_model.csv
```

Current model choices are:

- quadrotor_model

- quad_plane_model

- delta_quad_plane_model

- tilt_wing_model

The results of the model estimation will be saved into the Tools/parametric_model/results folder as a yaml file.

## Testing the functionality of Parametric model

To ensure that the parametric model works as expected you can perform a set of pytests, which are stored in `Tools/parametric_model/tests`. To start the tests you have to run the shell script:

`Tools/parametric_model/test_parametric_model.sh`

Currently only the transformation from body to intertial frame and vise versa are checked. This should be expanded in the future.
