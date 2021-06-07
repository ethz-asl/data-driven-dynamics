__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """


import numpy as np
import math

from .dynamics_model import DynamicsModel
from sklearn.linear_model import LinearRegression
from scipy.linalg import block_diag
from .model_plots import model_plots, quad_plane_model_plots
from .model_config import ModelConfig
from .aerodynamic_models import AeroModelDelta
import matplotlib.pyplot as plt

"""Currently this model estimates only Forces but no moments."""


class DeltaQuadPlaneModel(DynamicsModel):
    def __init__(self, config_file="qpm_delta_vtol_config.yaml"):
        self.config = ModelConfig(config_file)
        super(DeltaQuadPlaneModel, self).__init__(
            config_dict=self.config.dynamics_model_config)

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]

        assert (self.estimate_moments ==
                False), "Estimation of moments is not yet implemented in DeltaQuadPlaneModel. Disable in config file to estimate forces."

    def prepare_regression_matrices(self):

        if "V_air_body_x" not in self.data_df:
            self.normalize_actuators()
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        # Rotor features
        self.compute_rotor_features(self.rotor_config_dict)

        # Aerodynamics features
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        flap_commands = self.data_df[["u5", "u6"]].to_numpy()
        aoa_mat = self.data_df[["AoA"]].to_numpy()
        aero_model = AeroModelDelta(
            stall_angle=20.0, sig_scale_fac=self.sig_scale_fac)
        X_aero_forces, aero_coef_list = aero_model.compute_aero_features(
            airspeed_mat, aoa_mat, flap_commands)

        self.X_forces = np.hstack((self.X_rotor_forces, X_aero_forces))
        self.coef_name_list.extend(
            self.rotor_forces_coef_list + aero_coef_list)

        # prepare linear accelerations as regressand for forces
        accel_body_mat = self.data_df[[
            "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()
        self.y_forces = accel_body_mat.flatten()

        return self.X_forces, self.y_forces

    def plot_model_predicitons(self):

        y_forces_pred = self.reg.predict(self.X_forces)

        model_plots.plot_accel_predeictions(
            self.y_forces, y_forces_pred, self.data_df["timestamp"])
        model_plots.plot_az_and_collective_input(
            self.y_forces, y_forces_pred, self.data_df[["u0", "u1", "u2", "u3"]],  self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_z_direction(
            self.y_forces, y_forces_pred, self.data_df["V_air_body_z"], self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_y_direction(
            self.y_forces, y_forces_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])
        plt.show()
        return
