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

Update and initialize submodules:

```
make submodulesupdate
```

Install the dependencies from the project folder:

```
make install-dependencies
```

Or install the dependencies including submodule dependencies:

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

#### Config File

The config file allows to configure the intra class vehicle variations, used log file topics, data processing and other aspects of the pipeline. The default location is in `Tools/parametric_model/configs`. The path can be passed in the make target through the `config=<config_file_path>` argument. If no config is specified the default model config is used.

#### Log File

The Log file contains all data needed for the system identification of the specified model as defined in its config file. Next to the [ULog](https://docs.px4.io/master/en/dev_log/ulog_file_format.html) file format it is also possible to provide the data as a csv file. An example of the required formating can be seen in the `resources` folder.

#### Data Selection

The data_selection argument is optional (per default none) and can be used to visually select subportions of the data.

- none(default): Data selection is disabled, and the whole section of the log is used
- interactive: Data is selected interactively using the [Visual Dataframe Selector](https://github.com/manumerous/visual_dataframe_selector), before running the model estimation. It is also possible to save the selected subportion of data to a csv file in order to use this exact dataset multiple times.
- setpoint: Data is selected based on a certain `manual_control_setpoint` of the `ulog` or `csv` file. The parameter can be specified in the configuration file of the model.
- auto: Data is selected automatically (Beta)

### Results

The resulting parameters of the model estimation together with additional report information will be saved into the `model_results` folder as a yaml file.

### Getting Started

As an example to get started you estimate the parameters of a quadrotor model with the reference log_files:

```
make estimate-model model=quadrotor_model log=resources/quadrotor_model.ulg
```

## Extract PX4 Parameters from Aerodynamic Model (currently only supported for Fixed Wing Vehicles)

In order to extract important longitudinal trimming parameters directly from the identified aerodynamic model, just add the flag `--extraction True` to the function call. The extracted parameters are then saved in the same folder as the model results. The configuration has to include the extractor model that is to be used (e.g. the FixedWingExtractorModel) and an extractor configuration, containing user-determined minimum and maximum flight speeds (and optionally a cruise flight speed).

In preparation for a potentially more model-based control approach in the PX4-Autopilot, more parameters then currently required as estimated. Currently not supported parameters are marked correspondingly:

| Parameter name                    | Description                                                | Estimated? | Comment                                                                                                                    |
| --------------------------------- | ---------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| TRIM_PITCH_MAX_RANGE (additional) | Pitch trim at the speed for maximum range                  | Yes        | (no official PX4 parameter)                                                                                                |
| FW_AIRSPD_MAX_RANGE (additional)  | airspeed for maximum range (no sideslip)                   | Yes        | (no official PX4 parameter)                                                                                                |
| TRIM_PITCH_MIN_SINK (additional)  | Pitch trim at the speed for minimum sinkrate               | Yes        | (no official PX4 parameter)                                                                                                |
| FW_AIRSPD_MIN_SINK (additional)   | Airspeed for minimum sinkrate (no sideslip)                | Yes        | (no official PX4 parameter)                                                                                                |
| FW_DTRIM_P_VMIN                   | Trim differential at Vmin in level flight                  | Yes        | Determined as a function of the user-provided minimum speed. Value is added to TRIM_PITCH to obtain the actual trim value. |
| FW_DTRIM_P_VMAX                   | Trim differential at Vmax in level flight                  | Yes        | Determined as a function of the user-provided maximum speed. Value is added to TRIM_PITCH to obtain the actual trim value. |
| FW_PSP_OFF                        | Pitch offset at level flight (at FW_AIRSPD_TRIM)           | Yes        |                                                                                                                            |
| FW_AIRSPD_MIN                     | Minimum sustainable flight speed                           | No         | The minimum speed is provided as a user input as its experimental identification might be too dangerous                    |
| FW_THR_VMIN (additional)          | Thrust setting for level flight at FW_AIRSPD_MIN           | Yes        | (no official PX4 parameter)                                                                                                |
| FW_AIRSPD_MAX                     | Maximum sustainable flight speed                           | No         | The maximum speed is provided as a user input as especially structural limits cannot be identified from flight logs.       |
| FW_THR_VMAX (additional)          | Thrust setting for level flight at FW_AIRSPD_MAX           | Yes        | (no official PX4 parameter)                                                                                                |
| FW_AIRSPD_TRIM                    | Cruise airspeed                                            | (No)       | This speed can either be specified by the user through the optional --vcruise argument or is set to the max range speed    |
| TRIM_PITCH                        | Trim value for straight and level flight at FW_AIRSPD_TRIM | Yes        |                                                                                                                            |
| FW_THR_TRIM                       | Cruise throttle setting at FW_AIRSPD_TRIM                  | Yes        |                                                                                                                            |
| FW_THR_MIN                        | Minimum throttle setting                                   | No         | Can probably be fixed to zero (at least for electrically powered drones)                                                   |
| FW_THR_MAX                        | Maximum throttle setting                                   | No         | Can probably be fixed to one (at least for electrically powered drones)                                                    |
| FW_T_CLIMB_MAX                    | Maximum climb rate                                         | Yes        |                                                                                                                            |
| TRIM_PITCH_MAX_CLIMB (additional) | Pitch trim at the speed for maximum climb rate             | Yes        | (no official PX4 parameter)                                                                                                |
| FW_T_SINK_MAX                     | Maximum sink rate                                          | Yes        | Determined as a function of the user-provided maximum speed.                                                               |
| TRIM_PITCH_MAX_SINK (additional)  | Pitch trim for max sink rate (at Vmax and zero thrust)     | Yes        | (no official PX4 parameter)                                                                                                |
| FW_T_SINK_MIN                     | Minimum sink rate                                          | Yes        |                                                                                                                            |

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
