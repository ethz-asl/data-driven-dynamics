__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model in this file estimates a simple force motor model for a multirotor.

Model Parameters:
u                    : normalized actuator output scaled between 0 and 1
angular_vel_const    : angular velocity constant
angular_vel_offset   : angular velocity offset
mot_const            : motor constant
m                    : mass of UAV
accel_const          : combined acceleration constant k_2/m

Model:
angular_vel [rad/s] = angular_vel_const*u + angular_vel_offset
F_thrust = - mot_const * angular_vel^2
F_thrust_tot = - mot_const * \
    (angular_vel_1^2 + angular_vel_2^2 + angular_vel_3^2 + angular_vel_4^2)

Note that the forces are calculated in the NED body frame and are therefore negative.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from .dynamics_model import DynamicsModel
from .rotor_models import RotorModel
from .aerodynamic_models import FuselageDragModel
from .model_config import ModelConfig


class MultiRotorModel(DynamicsModel):
    def __init__(self, config_file, model_name="multirotor_model"):
        self.config = ModelConfig(config_file)
        super(MultiRotorModel, self).__init__(
            config_dict=self.config.dynamics_model_config)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"], self.config.model_config["moment_of_inertia"]["Iyy"], self.config.model_config["moment_of_inertia"]["Izz"]])
        
        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]

        self.model_name = model_name

    def prepare_force_regression_matrices(self):

        accel_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        self.y_forces = (accel_mat).flatten() * self.mass

        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        aero_model = FuselageDragModel()
        X_aero, aero_coef_list = aero_model.compute_fuselage_features(
            airspeed_mat)
        self.coef_name_list.extend(
            self.rotor_forces_coef_list + aero_coef_list)
        self.X_forces = np.hstack((self.X_rotor_forces, X_aero))
        print("datapoints for self.regression: ", self.data_df.shape[0])

    def prepare_moment_regression_matrices(self):
        moment_mat = np.matmul(self.data_df[[
            "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy(), self.moment_of_inertia)
        self.y_moments = moment_mat.flatten()
        self.X_moments = self.X_rotor_moments
        self.coef_name_list.extend(self.rotor_moments_coef_list)
