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
from scipy.linalg import block_diag
from .model_plots import model_plots, quad_plane_model_plots
from .model_config import ModelConfig
from .aerodynamic_models import AeroModelAAE
import matplotlib.pyplot as plt


"""This model estimates forces and moments for quad plane as for example the standard vtol in gazebo."""


class QuadPlaneModel(DynamicsModel):
    def __init__(self, config_file, model_name="quadplane_model"):
        self.config = ModelConfig(config_file)
        super(QuadPlaneModel, self).__init__(
            config_dict=self.config.dynamics_model_config)

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]

    def prepare_force_regression_matrices(self):
            # Aerodynamics features
            airspeed_mat = self.data_df[["V_air_body_x",
                                        "V_air_body_y", "V_air_body_z"]].to_numpy()
            flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
            aoa_mat = self.data_df[["AoA"]].to_numpy()
            aero_model = AeroModelAAE(
                stall_angle=self.stall_angle, sig_scale_fac=self.sig_scale_fac)
            X_aero_forces, aero_coef_list = aero_model.compute_aero_features(
                airspeed_mat, aoa_mat, flap_commands)
            self.X_forces = np.hstack((self.X_rotor_forces, X_aero_forces))
            self.X = self.X_forces

            # Accelerations
            accel_body_mat = self.data_df[[
                "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
            self.y_forces = accel_body_mat.flatten()
            self.y = self.y_forces

            # Set coefficients
            self.coef_name_list.extend(
                self.rotor_forces_coef_list + aero_coef_list)

    def prepare_moment_regression_matrices(self):
            # features due to rotation of body frame
            X_body_rot_moment, X_body_rot_moment_coef_list = self.compute_body_rotation_features(
                ["ang_vel_x", "ang_vel_y", "ang_vel_z"])
            self.X_moments = np.hstack(
                (self.X_rotor_moments, X_body_rot_moment))
            self.X = self.X_moments

            # Angular acceleration
            angular_accel_body_mat = self.data_df[[
                "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy()
            self.y_moments = angular_accel_body_mat.flatten()
            self.y = self.y_moments

            # Set coefficients
            self.coef_name_list.extend(
                self.rotor_moments_coef_list + X_body_rot_moment_coef_list)

    def plot_model_predicitons(self):

        y_pred = self.reg.predict(self.X)

        if (self.estimate_forces and self.estimate_moments):
            y_forces_pred = y_pred[0:self.y_forces.shape[0]]
            y_moments_pred = y_pred[self.y_forces.shape[0]:]
            model_plots.plot_accel_predeictions(
                self.y_forces, y_forces_pred, self.data_df["timestamp"])
            model_plots.plot_angular_accel_predeictions(
                self.y_moments, y_moments_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])

        elif (self.estimate_forces):
            y_forces_pred = y_pred
            model_plots.plot_accel_predeictions(
                self.y_forces, y_forces_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])

        elif (self.estimate_moments):
            y_moments_pred = y_pred
            model_plots.plot_angular_accel_predeictions(
                self.y_moments, y_moments_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])

        plt.show()

        return
