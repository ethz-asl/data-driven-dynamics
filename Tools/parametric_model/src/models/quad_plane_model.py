__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import yaml
import argparse

from .dynamics_model import DynamicsModel
from ..tools import quaternion_to_rotation_matrix, symmetric_logistic_sigmoid


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": {"ulog_name": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]", "output[4]"],
                                 "dataframe_name":  ["timestamp", "u0", "u1", "u2", "u3", "u4"]},
            "vehicle_local_position": {"ulog_name": ["timestamp", "ax", "ay", "az", "vx", "vy", "vz"]},
            "vehicle_attitude": {"ulog_name": ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"],
                                 "dataframe_name":  ["timestamp", "q0", "q1", "q2", "q3"]},
            "vehicle_angular_velocity":  {"ulog_name": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
                                          "dataframe_name":  ["timestamp", "ang_vel_x", "ang_vel_y", "ang_vel_z"]},
            "vehicle_angular_acceleration": {"ulog_name": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
                                             "dataframe_name":  ["timestamp", "ang_acc_x", "ang_acc_y", "ang_acc_z"]},
        }
        super(QuadPlaneModel, self).__init__(rel_ulog_path, req_topic_dict)
        self.stall_angle = 20 * math.pi/180
        self.actuator_directions = np.array([[0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0],
                                             [-1, -1, -1, -1, 0]]
                                            )

    def normalize_actuators(self):
        # u : normalize actuator output from pwm to be scaled between 0 and 1
        # To be adjusted using parameters:
        self.min_pwm = 1000
        self.max_pwm = 2000
        self.data_df["u0"] = (self.data_df["u0"] -
                              self.min_pwm)/(self.max_pwm - self.min_pwm)
        self.data_df["u1"] = (self.data_df["u1"] -
                              self.min_pwm)/(self.max_pwm - self.min_pwm)
        self.data_df["u2"] = (self.data_df["u2"] -
                              self.min_pwm)/(self.max_pwm - self.min_pwm)
        self.data_df["u3"] = (self.data_df["u3"] -
                              self.min_pwm)/(self.max_pwm - self.min_pwm)
        self.data_df["u4"] = (self.data_df["u4"] -
                              self.min_pwm)/(self.max_pwm - self.min_pwm)

    def compute_airspeed(self):
        groundspeed_ned_mat = np.transpose(
            ((self.data_df[["vx", "vy", "vz"]]).to_numpy()))
        attitude_quat_mat = np.transpose(
            ((self.data_df[["q0", "q1", "q2", "q3"]]).to_numpy()))
        airspeed_body_mat = np.zeros((5, self.data_df_len))
        for col in range(self.data_df_len):
            attitude_quat = attitude_quat_mat[:, col]
            groundspeed_ned = groundspeed_ned_mat[:, col]
            # double check whether inverse needed!
            R_world_to_body = np.linalg.inv(
                quaternion_to_rotation_matrix(attitude_quat))
            airspeed_body_mat[0:3, col] = R_world_to_body @ groundspeed_ned
            airspeed_body_mat[3, col] = (
                airspeed_body_mat[0, col])**2 + (airspeed_body_mat[2, col])**2
            airspeed_body_mat[4, col] = - math.atan(
                airspeed_body_mat[2, col]/airspeed_body_mat[0, col])
        airspeed_body_df = pd.DataFrame(np.transpose(airspeed_body_mat), columns=[
            "V_air_body_x", "V_air_body_y", "V_air_body_z", "V_air_body_xz_mag", "AoA"])
        self.data_df = pd.concat(
            [self.data_df, airspeed_body_df], axis=1, join="inner")

    def compute_actuator_thrust_features(self, u_vec):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs
        F_thrust = X_thrust @ (c_5, c_4, c_3, c_2, c_1, c_0)^T

        Vertical Rotor forces (u_0 to u_3):
        F_z_i = c_2 * u_i^2 + c_1 * u_i + c_0
        Forward Rotor force:
        F_x_i = c_5 * u_4^4 + c_4 * u_4 + c_3

        Input: u_vec = [u_0, .... u_4]"""

        X_thrust_z = np.zeros((3, 3))
        for i in range(4):
            u_i_features = self.actuator_directions[:, i].reshape(
                (3, 1)) @ np.array([[u_vec[i]**2, u_vec[i], 1]])
            X_thrust_z = X_thrust_z + u_i_features
            print(X_thrust_z)

        X_thrust_x = self.actuator_directions[:, 4].reshape(
            (3, 1)) @ np.array([[u_vec[4]**2, u_vec[4], 1]])
        X_thrust = np.hstack((X_thrust_x, X_thrust_z))
        print(X_thrust)
        return X_thrust

    def prepare_regression_mat(self):

        self.normalize_actuators()
        self.compute_airspeed()
        u_mat = self.data_df[["u0", "u1", "u2", "u3", "u4"]].to_numpy()

        y_lin = np.transpose(self.data_df[["ax", "ay", "az"]].to_numpy())
        X_lin_thrust = self.compute_actuator_thrust_features(u_mat[0, :])
        for i in range(1, self.data_df.shape[0]):
            u_curr = u_mat[i, :]
            X_thrust_curr = self.compute_actuator_thrust_features(u_curr)
            X_lin_thrust = np.vstack((X_lin_thrust, X_thrust_curr))

        return X_lin_thrust, y_lin

    def estimate_model(self, des_freq=10.0):
        print("estimating quad plane model...")
        self.data_df = self.compute_resampled_dataframe(des_freq)
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_mat()
        print(X)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    args = parser.parse_args()
    rel_ulog_path = args.log_path
    # estimate simple multirotor drag model
    quadPlaneModel = QuadPlaneModel(rel_ulog_path)
    quadPlaneModel.estimate_model()
