__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """


import numpy as np
import math

from .dynamics_model import DynamicsModel
from .rotor_models import RotorModel
from sklearn.linear_model import LinearRegression
from .model_config import ModelConfig
from .aerodynamic_models import StandardWingModel, ControlSurfaceModel


"""This model estimates forces and moments for quad plane as for example the standard vtol in gazebo."""


class StandardPlaneModel(DynamicsModel):
    def __init__(self, config_file, normalization=True, model_name="standardplane_model"):
        self.config = ModelConfig(config_file)
        super(StandardPlaneModel, self).__init__(
            config_dict=self.config.dynamics_model_config, normalization=normalization)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"],
                                         self.config.model_config["moment_of_inertia"]["Iyy"], self.config.model_config["moment_of_inertia"]["Izz"]])

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.aero_config_dict = self.config.model_config["actuators"]["control_surfaces"]
        self.aerodynamics_dict = self.config.model_config["aerodynamics"]

    def prepare_force_regression_matrices(self):
        # Accelerations
        accel_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        force_mat = accel_mat * self.mass
        self.y_forces = (force_mat).flatten()
        self.data_df[["measured_force_x", "measured_force_y",
                     "measured_force_z"]] = force_mat

        # Aerodynamics features
        airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        aoa_mat = self.data_df[["angle_of_attack"]].to_numpy()
        aero_model = StandardWingModel(self.aerodynamics_dict)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_force_features(
            airspeed_mat, aoa_mat)
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)
        self.y_dict.update({"lin":{"x":"measured_force_x","y":"measured_force_y","z":"measured_force_z"}})

    def prepare_moment_regression_matrices(self):
        # Angular acceleration
        moment_mat = np.matmul(self.data_df[[
            "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy(), self.moment_of_inertia)
        self.y_moments = moment_mat.flatten()
        self.data_df[["measured_moment_x", "measured_moment_y",
                     "measured_moment_z"]] = moment_mat

        # Aerodynamics features
        airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        aoa_mat = self.data_df[["angle_of_attack"]].to_numpy()
        sideslip_mat = self.data_df[["angle_of_sideslip"]].to_numpy()

        aero_model = StandardWingModel(self.aerodynamics_dict)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_moment_features(
            airspeed_mat, aoa_mat, sideslip_mat)
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)

        aero_config_dict = self.aero_config_dict
        for aero_group in aero_config_dict.keys():
            aero_group_list = self.aero_config_dict[aero_group]

            for config_dict in aero_group_list:
                controlsurface_input_name = config_dict["dataframe_name"]
                u_vec = self.data_df[controlsurface_input_name].to_numpy()
                control_surface_model = ControlSurfaceModel(config_dict, self.aerodynamics_dict, u_vec)
                X_controls, coef_dict_controls, col_names_controls = control_surface_model.compute_actuator_moment_matrix(
                    airspeed_mat, aoa_mat)
                self.data_df[col_names_controls] = X_controls
                self.coef_dict.update(coef_dict_controls)

        self.y_dict.update({"rot":{"x":"measured_moment_x","y":"measured_moment_y","z":"measured_moment_z"}})
