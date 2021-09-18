<img align="right" height="60" src="https://user-images.githubusercontent.com/5248102/126074528-004a32b9-7911-486a-9e79-8b78e6e66fdc.png">

# data-driven-dynamics

[![Build Tests](https://github.com/ethz-asl/data-driven-dynamics/actions/workflows/build_test.yml/badge.svg)](https://github.com/ethz-asl/data-driven-dynamics/actions/workflows/build_test.yml)

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
Tools/sitl_run.sh -m iris -s iris_aerodynamics
```

## Generating a Parametric Model from Log File

Link the latest log files to your local logs folder using:

```
source setup.bash
```

Generate the parametric model using a log file (ulog or csv):

```
make estimate-model [model=<modeltype>] [config=<config_file_path>] [data_selection=<True/False>] log=<log_file_path>
```

### Pipeline Arguments

#### Model Choice

The chosen vehicle model class determines what physical effects are modelled and what parameterts need to be regressed in the system identification process.
Current vehicle model choices are:

- quadrotor_model (default config for quadrotor)

#### Config File

The config file allows to configure the intra class vehicle variations, used log file topics, data processing and other aspects of the pipeline. The default location is in `Tools/parametric_model/configs`. The path can be passed in the make target through the `config=<config_file_path>` argument. If no config is specified the default model config is used.

#### Log File

The Log file contains all data needed for the system identification of the specified model as defined in its config file. Next to the [ULog](https://docs.px4.io/master/en/dev_log/ulog_file_format.html) file format it is also possible to provide the data as a csv file. An example of the required formating can be seen in the `resources` folder.

#### Data Selection

The data_selection argument is optional (per default False) and can be used to visually select subportions of the data, using the [Visual Dataframe Selector](https://github.com/manumerous/visual_dataframe_selector), before running the model estimation. It is also possible to save the selected subportion of data to a csv file in order to use this exact dataset multiple times.

### Results

The resulting parameters of the model estimation together with additional report information will be saved into the `model_results` folder as a yaml file.

### Getting Started

As an example to get started you estimate the parameters of a quadrotor model with the reference log_files:

```
make estimate-model model=quadrotor_model log=resources/quadrotor_model.ulg
```

## Testing the functionality of Parametric model

To ensure that the parametric model works as expected you can perform a set of pytests, which are stored in `Tools/parametric_model/tests`. To start the tests you have to run the shell script:

`Tools/parametric_model/test_parametric_model.sh`

Currently only the transformation from body to intertial frame and vise versa are checked. This should be expanded in the future.
