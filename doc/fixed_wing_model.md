# Fixed Wing Model

## Getting Started

### Angle of Attack dependent lift/drag model
As an example to get started with a fixed-wing you can estimate the parameters with the default provided log using:

```
make estimate-model model=fixedwing_model log=resources/fixedwing_model.csv
```

### Singularity Free Model
Since the angle of attack (AoA) is not defined for zero airspeed an AoA free model can be estimated using:

```
make estimate-model model=fixedwing_singularityfree_model log=resources/fixedwing_model.csv 
```

The PhiAerodynamics model, implemented in [phiaerodynamics_model.py](https://github.com/ethz-asl/data-driven-dynamics/blob/master/Tools/parametric_model/src/models/aerodynamic_models/phiaerodynamics_model.py) is a global singularity free aerodynamics model implemeted as described in [1]. 

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


## References

[1] Lustosa, Leandro R., François Defaÿ, and Jean-Marc Moschetta. "Global singularity-free aerodynamic model for
 algorithmic flight control of tail sitters." Journal of Guidance, Control, and Dynamics 42.2 (2019): 303-316.