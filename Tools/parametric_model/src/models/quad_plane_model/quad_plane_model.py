__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """

import numpy as np
import pandas as pd
import math
import argparse

from ..dynamics_model import DynamicsModel
from ..aerodynamic_models import LinearPlateAeroModel
from ..rotor_models import GazeboRotorModel
from sklearn.linear_model import LinearRegression
from ..model_plots import model_plots, quad_plane_model_plots


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

    def compute_rotor_features(self):
        u_mat = self.data_df[["u0", "u1", "u2", "u3", "u4"]].to_numpy()
        v_airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()

        # Vertical Rotor Features
        # all vertical rotors are assumed to have the same rotor parameters, therefore their feature matrices are added.
        X_vertical_rotors = np.zeros((3*self.data_df.shape[0], 5))
        for i in range(0, (u_mat.shape[1]-1)):
            currActuator = GazeboRotorModel(self.actuator_directions[:, i])
            X_curr_rotor, vert_rotors_coef_list = currActuator.compute_actuator_feature_matrix(
                u_mat[:, i], v_airspeed_mat)
            X_vertical_rotors = X_vertical_rotors + X_curr_rotor
        for i in range(len(vert_rotors_coef_list)):
            vert_rotors_coef_list[i] = "vert_" + vert_rotors_coef_list[i]
        # Forward Rotor Feature
        forwardActuator = GazeboRotorModel(self.actuator_directions[:, 4])
        X_forward_rotor, forward_rotors_coef_list = forwardActuator.compute_actuator_feature_matrix(
            u_mat[:, 4], v_airspeed_mat)
        for i in range(len(forward_rotors_coef_list)):
            forward_rotors_coef_list[i] = "forward_" + \
                forward_rotors_coef_list[i]

        # Combine all rotor feature matrices
        X_rotor_features = np.hstack(
            (X_vertical_rotors, X_forward_rotor))
        self.coef_name_list.extend(
            (vert_rotors_coef_list + forward_rotors_coef_list))
        return X_rotor_features

    def prepare_regression_matrices(self):
        self.normalize_actuators()
        self.compute_airspeed()
        X_lin_thrust = self.compute_rotor_features()
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
        aoa_mat = self.data_df[["AoA"]].to_numpy()
        aero_model = LinearPlateAeroModel(20.0)
        X_lin_aero, aero_coef_list = aero_model.compute_aero_features(
            airspeed_mat, aoa_mat, flap_commands)
        self.coef_name_list.extend(aero_coef_list)
        accel_mat = self.data_df[["ax", "ay", "az"]].to_numpy()
        y_lin = (self.rot_to_body_frame(accel_mat)).flatten()
        X_lin = np.hstack((X_lin_thrust, X_lin_aero))
        return X_lin, y_lin

    def estimate_model(self):
        print("estimating quad plane model...")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_matrices()
        reg = LinearRegression().fit(X, y)

        print("regression complete")
        metrics_dict = {"R2": float(reg.score(X, y))}
        self.coef_name_list.extend(["intercept"])
        coef_list = list(reg.coef_) + [reg.intercept_]
        self.generate_model_dict(coef_list, metrics_dict)
        self.save_result_dict_to_yaml()

        y_pred = reg.predict(X)

        model_plots.plot_accel_predeictions(
            y, y_pred, self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_y_direction(
            y, y_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])
        quad_plane_model_plots.plot_accel_predeictions_with_flap_outputs(
            y, y_pred, self.data_df[["u5", "u6", "u7"]], self.data_df["timestamp"])

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
