__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """

import numpy as np
import math

from .dynamics_model import DynamicsModel
from .aerodynamic_models import LinearPlateAeroModel
from .rotor_models import GazeboRotorModel
from sklearn.linear_model import LinearRegression
from scipy.linalg import block_diag
from .model_plots import model_plots, quad_plane_model_plots


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": {"ulog_name": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]", "output[4]", "output[5]", "output[6]", "output[7]"],
                                 "dataframe_name":  ["timestamp", "u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"],
                                 "actuator_type":  ["timestamp", "motor", "motor", "motor", "motor", "motor", "flap", "flap", "flap"]},
            "vehicle_local_position": {"ulog_name": ["timestamp", "vx", "vy", "vz"]},
            "vehicle_attitude": {"ulog_name": ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"],
                                 "dataframe_name":  ["timestamp", "q0", "q1", "q2", "q3"]},
            "vehicle_angular_velocity":  {"ulog_name": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
                                          "dataframe_name":  ["timestamp", "ang_vel_x", "ang_vel_y", "ang_vel_z"]},
            "sensor_combined": {"ulog_name": ["timestamp", "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]},
            "vehicle_angular_acceleration": {"ulog_name": ["timestamp", "xyz[0]", "xyz[1]", "xyz[2]"],
                                             "dataframe_name":  ["timestamp", "ang_acc_x", "ang_acc_y", "ang_acc_z"]},
        }
        super(QuadPlaneModel, self).__init__(rel_ulog_path, req_topic_dict)
        self.stall_angle = 20 * math.pi/180

        # direction of generated force in FRD body frame
        self.actuator_directions = np.array([[0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0],
                                             [-1, -1, -1, -1, 0]])

        # turning direction or rotor: 1 for same direction as force, -1 for opposite direction
        self.actuator_turning_directions = [-1, -1, 1, 1, -1]

        # rotor location in FRD body frame
        self.actuator_positions = np.array([[0.35, -0.35, 0.35, -0.35, 0.22],
                                            [0.35, -0.35, -0.35, 0.35, 0],
                                            [-0.07, -0.07, -0.07, -0.07, 0]]
                                           )

    def compute_rotor_features(self):
        u_mat = self.data_df[["u0", "u1", "u2", "u3", "u4"]].to_numpy()
        v_airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()

        # Vertical Rotor Features
        # all vertical rotors are assumed to have the same rotor parameters, therefore their feature matrices are added.
        for i in range(0, (u_mat.shape[1]-1)):
            currActuator = GazeboRotorModel(
                self.actuator_directions[:, i], self.actuator_positions[:, i], self.actuator_turning_directions[i])
            X_force_curr, X_moment_curr, vert_rotor_forces_coef_list, vert_rotor_moments_coef_list = currActuator.compute_actuator_feature_matrix(
                u_mat[:, i], v_airspeed_mat)
            if 'X_vert_rot_forces' in vars():
                X_vert_rot_forces += X_force_curr
                X_vert_rot_moments += X_moment_curr
            else:
                X_vert_rot_forces = X_force_curr
                X_vert_rot_moments = X_moment_curr
        for i in range(len(vert_rotor_forces_coef_list)):
            vert_rotor_forces_coef_list[i] = "vert_" + \
                vert_rotor_forces_coef_list[i]
        for i in range(len(vert_rotor_moments_coef_list)):
            vert_rotor_moments_coef_list[i] = "vert_" + \
                vert_rotor_moments_coef_list[i]

        # Horizontal Rotor Features
        forwardActuator = GazeboRotorModel(
            self.actuator_directions[:, 4], self.actuator_positions[:, 4], self.actuator_turning_directions[4])
        X_hor_rot_forces, X_hor_rot_moments, hor_rotor_forces_coef_list, hor_rotor_moments_coef_list = forwardActuator.compute_actuator_feature_matrix(
            u_mat[:, 4], v_airspeed_mat)
        for i in range(len(hor_rotor_forces_coef_list)):
            hor_rotor_forces_coef_list[i] = "horizontal_" + \
                hor_rotor_forces_coef_list[i]
        for i in range(len(hor_rotor_moments_coef_list)):
            hor_rotor_moments_coef_list[i] = "horizontal_" + \
                hor_rotor_moments_coef_list[i]

        # Combine all rotor feature matrices
        X_rotor_forces = np.hstack(
            (X_vert_rot_forces, X_hor_rot_forces))
        X_rotor_moments = np.hstack(
            (X_vert_rot_moments, X_hor_rot_moments))
        rotor_forces_coef_list = vert_rotor_forces_coef_list + hor_rotor_forces_coef_list
        rotor_moments_coef_list = vert_rotor_moments_coef_list + hor_rotor_moments_coef_list
        return X_rotor_forces, X_rotor_moments, rotor_forces_coef_list, rotor_moments_coef_list

    def prepare_regression_matrices(self):

        # Prepare data
        self.normalize_actuators()
        self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        # Rotor features
        X_rotor_forces, X_rotor_moments, rotor_forces_coef_list, rotor_moments_coef_list = self.compute_rotor_features()

        # Aerodynamics features
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
        aoa_mat = self.data_df[["AoA"]].to_numpy()
        aero_model = LinearPlateAeroModel(20.0)
        X_aero_forces, aero_coef_list = aero_model.compute_aero_features(
            airspeed_mat, aoa_mat, flap_commands)

        # features due to rotation of body frame
        X_body_rot_moment, X_body_rot_moment_coef_list = self.compute_body_rotation_features(
            ["ang_vel_x", "ang_vel_y", "ang_vel_z"])

        # Concat features
        X_forces = np.hstack((X_rotor_forces, X_aero_forces))
        X_moments = np.hstack((X_rotor_moments, X_body_rot_moment))
        X = block_diag(X_forces, X_moments)
        self.coef_name_list.extend(rotor_forces_coef_list + aero_coef_list +
                                   rotor_moments_coef_list + X_body_rot_moment_coef_list)
        # define separate features for plotting
        self.X_forces = X[0:X_forces.shape[0], :]
        self.X_moments = X[X_forces.shape[0]:X.shape[0], :]

        # prepare linear and angular accelerations as regressand for forces
        accel_body_mat = self.data_df[[
            "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()
        angular_accel_body_mat = self.data_df[[
            "ang_acc_x", "ang_acc_y", "ang_acc_z"]].to_numpy()
        # define separate features for plotting
        self.y_forces = accel_body_mat.flatten()
        self.y_moments = angular_accel_body_mat.flatten()
        y = np.hstack((self.y_forces, self.y_moments))

        return X, y

    def estimate_model(self):
        print("estimating quad plane model...")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_matrices()
        self.reg = LinearRegression().fit(X, y)

        print("regression complete")
        metrics_dict = {"R2": float(self.reg.score(X, y))}
        self.coef_name_list.extend(["intercept"])
        coef_list = list(self.reg.coef_) + [self.reg.intercept_]
        self.generate_model_dict(coef_list, metrics_dict)
        self.save_result_dict_to_yaml(file_name="quad_plane_model")

        return

    def plot_model_predicitons(self):

        y_forces_pred = self.reg.predict(self.X_forces)
        y_moments_pred = self.reg.predict(self.X_moments)

        model_plots.plot_accel_predeictions(
            self.y_forces, y_forces_pred, self.data_df["timestamp"])
        model_plots.plot_angular_accel_predeictions(
            self.y_moments, y_moments_pred, self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        quad_plane_model_plots.plot_accel_predeictions_with_flap_outputs(
            self.y_forces, y_forces_pred, self.data_df[["u5", "u6", "u7"]], self.data_df["timestamp"])
        return
