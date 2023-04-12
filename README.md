<img align="right" height="60" src="https://user-images.githubusercontent.com/5248102/126074528-004a32b9-7911-486a-9e79-8b78e6e66fdc.png">

# data-driven-dynamics

[![SysID Pipeline Test](https://github.com/ethz-asl/data-driven-dynamics/actions/workflows/sysid_test.yml/badge.svg)](https://github.com/ethz-asl/data-driven-dynamics/actions/workflows/sysid_test.yml)

This repository allows a data-driven dynamics model for PX4 SITL(Software-In-The-Loop) simulations.

You can get an overview of the pipeline and its functionalities in the following video presentation:

[![Watch the video](https://img.youtube.com/vi/kAsfptZU4uk/maxresdefault.jpg)](https://www.youtube.com/watch?v=kAsfptZU4uk)

More detailed information can be found in the student paper [Data-Driven Dynamics Modelling Using Flight Logs](https://www.research-collection.ethz.ch/handle/20.500.11850/507495).

## Setting up the environment

This plugin depends on [PX4 SITL](https://github.com/PX4/PX4-SITL_gazebo). Therefore custom messages of PX4 SITL needs to be linked. Therefore, prior to building this package, PX4 and corresponding SITL package needs to be built.

In case you have not cloned the firmware repository

```
git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/src/PX4-Autopilot
```

For internal use, our internal firmware repository is as the following.

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

Install the dependencies including submodule dependencies using:

```
install-full-depdencies
```

## Build

After the environment has been setup as described in the previous section, build the package as the following.

```
mkdir build
cd build
cmake ..
make
```

## Generating a Parametric Model from Log File

Link the latest log files to your local logs folder using:

```
source setup.bash
```

Generate the parametric model using a log file (ulog or csv):

```
make estimate-model [model=<modeltype>] [config=<config_file_path>] [data_selection=<none|interactive|setpoint|auto>] [plot=<True/False>] [log=<log_file_path>]
```

### Pipeline Arguments

#### Model Choice

The chosen vehicle model class determines what physical effects are modelled and what parameterts need to be regressed in the system identification process.
Current vehicle model choices are:

- quadrotor_model (default config for quadrotor)
- fixedwing_model (default config for cruise flight of fixed-wings and VTOLs, documented [here](https://github.com/ethz-asl/data-driven-dynamics/blob/update-documentation/doc/fixed_wing_model.md))

#### Config File

The config file allows to configure the intra class vehicle variations, used log file topics, data processing and other aspects of the pipeline. The default location is in `Tools/parametric_model/configs`. The path can be passed in the make target through the `config=<config_file_path>` argument. If no config is specified the default model config is used.

#### Log File

The Log file contains all data needed for the system identification of the specified model as defined in its config file. Next to the [ULog](https://docs.px4.io/master/en/dev_log/ulog_file_format.html) file format it is also possible to provide the data as a csv file. An example of the required formating can be seen in the `resources` folder.

#### Data Selection

The data_selection argument is optional (per default none) and can be used to visually select subportions of the data.

- none(default): Data selection is disabled, and the whole section of the log is used
- interactive: Data is selected interactively using the [Visual Dataframe Selector](https://github.com/manumerous/visual_dataframe_selector), before running the model estimation. It is also possible to save the selected subportion of data to a csv file in order to use this exact dataset multiple times.
- setpoint: Data is selected based on a certain `manual_control_setpoint` of the `ulog` or `csv` file. The `manual_control_setpoint` topic that should be used for selection, can be specified in the configuration file of the model as `selection_variable` (possible values are `aux1` to `aux6`). Finally, specific `activations` of the `selection_variable` can also be specified as a list in the configuration file (default: all activations will be used).
- auto: Data is selected automatically (Beta)

### Results

The resulting parameters of the model estimation together with additional report information will be saved into the `model_results` folder as a yaml file.

### Getting Started

As an example to get started you estimate the parameters of a quadrotor model with the reference log_files:

```
make estimate-model model=quadrotor_model log=resources/quadrotor_model.ulg
```


## Generating a Model Prediction for Given Parameters and Log

It is also possible to test the obtained parameters for a certain model on a different log using:

```
make predict-model [model=<modeltype>] [config=<config_file_path>] [data_selection=<none|interactive|auto>] [log=<log_file_path>] [model_results=<model_results_path>]
```

## Testing the functionality of Parametric model

To ensure that the parametric model works as expected you can perform a set of pytests, which are stored in `Tools/parametric_model/tests`. To start the tests you have to run the shell script:

`Tools/parametric_model/test_parametric_model.sh`

Currently only the transformation from body to intertial frame and vise versa are checked. This should be expanded in the future.

## Running the Simulation

To run the simulation,

```
source setup.bash
Tools/sitl_run.sh -m iris -s iris_aerodynamics
```

The custom Gazebo quadrotor model will always read the model parameters from the file `model_results/quadrotor_model.yaml`. You can simply rename your desired model results file to fly your estimated model in Gazebo.

## Credits

This project was done in collaboration between the [Autonomous Systems Lab, ETH Zurich](https://asl.ethz.ch/) and [Auterion AG](https://auterion.com/)

To cite this work in a academic context:

```
@article{galliker2021data,
  title={Data-Driven Dynamics Modelling Using Flight Logs},
  author={Galliker, Manuel Yves},
  year={2021}
}
```
