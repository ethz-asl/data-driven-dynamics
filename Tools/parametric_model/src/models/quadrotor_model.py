__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model in this file estimates a simple force motor model for the iris quadrocopter of PX4 sitl gazebo.

Start the model identification:
Call "estimate_model(rel_ulog_path)"
with rel_ulog_path specifying the path of the log file relative to the project directory (e.g. "logs/2021-03-16/21_45_40.ulg")

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

The script estimates [k_1, c, b]
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from .dynamics_model import DynamicsModel
from .rotor_models import RotorModel
from .model_plots import model_plots
from .aerodynamic_models import SimpleDragModel
from .model_config import ModelConfig


class QuadRotorModel(DynamicsModel):
    def __init__(self, rel_data_path, config_file="sqrm_gazebo_standart_config.yaml"):
        self.config = ModelConfig(config_file)
        super(QuadRotorModel, self).__init__(
            config_dict=self.config.dynamics_model_config, rel_data_path=rel_data_path)
        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]

        assert (self.estimate_moments ==
                False), "Estimation of moments is not yet implemented in DeltaQuadPlaneModel. Disable in config file to estimate forces."

    def prepare_regression_mat(self):

        if "V_air_body_x" not in self.data_df:
            self.normalize_actuators()
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        accel_mat = self.data_df[[
            "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()
        self.data_df[["ax_body", "ay_body", "az_body"]] = accel_mat
        y = (accel_mat).flatten()

        # Rotor features
        self.compute_rotor_features(self.rotor_config_dict)

        aoa_mat = self.data_df[["AoA"]].to_numpy()
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        aero_model = SimpleDragModel(35.0)
        X_aero, aero_coef_list = aero_model.compute_aero_features(
            airspeed_mat, aoa_mat)
        self.coef_name_list.extend(
            self.rotor_forces_coef_list + aero_coef_list)
        X = np.hstack((self.X_rotor_forces, X_aero))
        print("datapoints for self.regression: ", self.data_df.shape[0])
        return X, y

    def estimate_model(self):
        print("Estimating quad plane model using the following data:")
        print(self.data_df.columns)
        self.X, self.y = self.prepare_regression_mat()
        self.reg = LinearRegression(fit_intercept=False)
        self.reg.fit(self.X, self.y)
        print("regression complete")
        metrics_dict = {"R2": float(self.reg.score(self.X, self.y))}
        self.coef_name_list.extend(["intercept"])
        coef_list = list(self.reg.coef_) + [self.reg.intercept_]
        self.generate_model_dict(coef_list, metrics_dict)
        self.save_result_dict_to_yaml(file_name="quadrotor_model")
        return

    def plot_model_predicitons(self):

        y_pred = self.reg.predict(self.X)

        model_plots.plot_accel_predeictions(
            self.y, y_pred, self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        return

    def plot_motor_model_prediction(self):
        # plot model prediction
        u = np.linspace(0.0, 1, num=101, endpoint=True)
        u_coll_pred = self.rotor_count*u
        u_squared_coll_pred = self.rotor_count * u**2
        y_pred = np.zeros(u.size)
        coef_dict = self.result_dict["coefficients"]
        for i in range(u.size):
            y_pred[i] = coef_dict["vert_rot_thrust_quad"]*u_squared_coll_pred[i] + \
                coef_dict["vert_rot_thrust_lin"]*u_coll_pred[i] + \
                coef_dict["vert_rot_thrust_offset"]
        plt.plot(u_coll_pred, y_pred, label='prediction')
        # plot underlying data
        y_data = self.data_df["az_body"].to_numpy()
        u_coll_data, u_squared_coll_data = self.compute_collective_input_features()
        plt.plot(u_coll_data, y_data, 'o', label='data')
        plt.ylabel('acceleration in z direction [m/s^2]')
        plt.xlabel('collective input (between [0, 1] per input)')
        plt.legend()
        plt.show()

    def compute_collective_input_features(self):
        u_collective = np.ones(self.data_df.shape[0])
        u_squared_collective = np.ones(self.data_df.shape[0])
        for r in range(self.data_df.shape[0]):
            u0 = self.data_df["u0"].iloc[r]
            u1 = self.data_df["u1"].iloc[r]
            u2 = self.data_df["u2"].iloc[r]
            u3 = self.data_df["u3"].iloc[r]
            u_collective[r] = u0 + u1 + u2 + u3
            u_squared_collective[r] = u0**2 + u1**2 + u2**2 + u3**2
        return u_collective, u_squared_collective
