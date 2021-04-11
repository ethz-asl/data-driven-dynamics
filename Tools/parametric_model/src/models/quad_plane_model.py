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

from scipy.spatial.transform import Rotation

from .dynamics_model import DynamicsModel
from .aerodynamics import LinearPlateAeroModel
from .rotor_models import GazeboRotorModel

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": {"ulog_name": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]", "output[4]", "output[5]", "output[6]", "output[7]"],
                                 "dataframe_name":  ["timestamp", "u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"]},
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

    def compute_airspeed(self):
        groundspeed_ned_mat = (self.data_df[["vx", "vy", "vz"]]).to_numpy()
        airspeed_body_mat = self.rot_to_body_frame(groundspeed_ned_mat)
        aoa_vec = np.zeros((airspeed_body_mat.shape[0], 1))
        for i in range(airspeed_body_mat.shape[0]):
            aoa_vec[i, :] = math.atan2(
                airspeed_body_mat[i, 2], airspeed_body_mat[i, 0])
        airspeed_body_mat = np.hstack((airspeed_body_mat, aoa_vec))
        airspeed_body_df = pd.DataFrame(airspeed_body_mat, columns=[
            "V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"])
        self.data_df = pd.concat(
            [self.data_df, airspeed_body_df], axis=1, join="inner")

    def compute_actuator_features(self):
        u_mat = self.data_df[["u0", "u1", "u2", "u3", "u4"]].to_numpy()
        v_airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()

        # Vertical Rotor Features
        # all vertical rotors have the same parameters, therefore their feature matrices are added.
        X_forces_vertical_rotors = np.zeros((3*self.data_df.shape[0], 5))
        for i in range(0, (u_mat.shape[1]-1)):
            currActuator = GazeboRotorModel(self.actuator_directions[:, i])
            X_forces = currActuator.compute_actuator_feature_matrix(
                u_mat[:, i], v_airspeed_mat)
            X_forces_vertical_rotors = X_forces_vertical_rotors + X_forces

        # Forward Rotor Feature
        forwardActuator = GazeboRotorModel(self.actuator_directions[:, 4])
        X_forces_forward_rotor = forwardActuator.compute_actuator_feature_matrix(
            u_mat[:, 4], v_airspeed_mat)

        X_rotor_features = np.hstack(
            (X_forces_vertical_rotors, X_forces_forward_rotor))
        return X_rotor_features

    def prepare_regression_mat(self):
        self.normalize_actuators()
        self.compute_airspeed()
        X_lin_thrust = self.compute_actuator_features()
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
        aoa_mat = self.data_df[["AoA"]].to_numpy()
        aero_model = LinearPlateAeroModel(20.0)
        X_lin_aero = aero_model.compute_aero_features(
            airspeed_mat, aoa_mat, flap_commands)
        accel_mat = self.data_df[["ax", "ay", "az"]].to_numpy()
        y_lin = (self.rot_to_body_frame(accel_mat)).flatten()
        X_lin = np.hstack((X_lin_thrust, X_lin_aero))
        return X_lin, y_lin

    def estimate_model(self):
        print("estimating quad plane model...")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_mat()
        reg = LinearRegression().fit(X, y)
        print("regression complete")
        print("R2 score: ", reg.score(X, y))
        print(reg.coef_, reg.intercept_)
        y_pred = reg.predict(X)
        self.plot_accel_predeictions(y, y_pred)
        self.plot_y_predicitons(y, y_pred)
        self.plot_airspeed()

        return

    def plot_airspeed(self):
        y_mat = self.data_df[["V_air_body_x",
                              "V_air_body_y", "V_air_body_z", "AoA"]].to_numpy()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        ax1.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 0], label='measurement')

        ax2.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 1], label='measurement')

        ax3.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 2], label='measurement')
        ax4.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 3], label='measurement')
        ax1.set_title('airspeed in x direction of body frame [m/s^2]')
        ax2.set_title('airspeed in y direction of body frame [m/s^2]')
        ax3.set_title('airspeed in z direction of body frame [m/s^2]')
        ax4.set_title("Aoa in body frame [radiants]")
        plt.legend()
        plt.show()

    def plot_y_predicitons(self, y, y_pred):
        y_pred_mat = y_pred.reshape((-1, 3))
        y_mat = y.reshape((-1, 3))
        y = self.data_df[["V_air_body_y"]].to_numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Vertically stacked subplots')
        ax1.plot((self.data_df["timestamp"]).to_numpy(),
                 y, label='measurement')
        ax2.plot((self.data_df["timestamp"]).to_numpy(),
                 y**2, label='measurement')

        ax3.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 1], label='measurement')
        ax3.plot((self.data_df["timestamp"]).to_numpy(),
                 y_pred_mat[:, 1], label='prediction')
        ax1.set_title('airspeed in y direction of body frame [m/s^2]')
        ax2.set_title('airspeed in y direction squared of body frame [m/s^2]')
        ax3.set_title('airspeed in y direction of body frame [m/s^2]')
        plt.legend()
        plt.show()

    def plot_accel_predeictions(self, y, y_pred):
        y_pred_mat = y_pred.reshape((-1, 3))
        y_mat = y.reshape((-1, 3))

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Vertically stacked subplots')
        ax1.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 0], label='measurement')
        ax1.plot((self.data_df["timestamp"]).to_numpy(),
                 y_pred_mat[:, 0], label='prediction')
        ax2.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 1], label='measurement')
        ax2.plot((self.data_df["timestamp"]).to_numpy(),
                 y_pred_mat[:, 1], label='prediction')
        ax3.plot((self.data_df["timestamp"]).to_numpy(),
                 y_mat[:, 2], label='measurement')
        ax3.plot((self.data_df["timestamp"]).to_numpy(),
                 y_pred_mat[:, 2], label='prediction')

        ax1.set_title('acceleration in x direction of body frame [m/s^2]')
        ax2.set_title('acceleration in y direction of body frame [m/s^2]')
        ax3.set_title('acceleration in z direction of body frame [m/s^2]')
        plt.legend()
        plt.show()


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
