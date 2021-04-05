__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""state vector x = [pos_wb, quat_wb, vel_wb, angular_vel_b]"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import yaml
import argparse

from .dynamics_model import DynamicsModel
from ..tools import quaternion_to_rotation_matrix


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": {"ulog_name": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]"],
                                 "dataframe_name":  ["timestamp", "u0", "u1", "u2", "u3"]},
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
            airspeed_body_mat[4, col] = math.atan(
                airspeed_body_mat[2, col]/airspeed_body_mat[0, col])

        airspeed_body_df = pd.DataFrame(np.transpose(airspeed_body_mat), columns=[
            "V_air_body_x", "V_air_body_y", "V_air_body_z", "V_air_body_xz_mag", "AoA"])
        self.data_df = self.data_df.append(airspeed_body_df)

    def prepare_regression_mat(self):
        y = self.data_df[["ax", "ay", "az"]].to_numpy()
        X = np.ones((self.data_df.shape[0], 2))
        self.normalize_actuators()
        self.compute_airspeed()
        return X, y

    def estimate_model(self, des_freq=10.0):
        print("estimating quad plane model...")
        self.data_df = self.compute_resampled_dataframe(des_freq)
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_mat()
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
