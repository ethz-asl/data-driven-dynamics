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


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_data_path, config_file="qpm_gazebo_standart_vtol_config.yaml"):
        self.config = ModelConfig(config_file)
        super(QuadPlaneModel, self).__init__(
            config_dict=self.config.dynamics_model_config, rel_data_path=rel_data_path)

        self.rotor_config_list = self.config.model_config["actuators"]["rotors"]
        self.rotor_count = len(self.rotor_config_list)
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]

    def compute_rotor_features(self):
        u_mat = self.data_df[["u0", "u1", "u2", "u3", "u4"]].to_numpy()
        v_airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()

        # Vertical Rotor Features
        # all vertical rotors are assumed to have the same rotor parameters, therefore their feature matrices are added.
        for i in range((self.rotor_count-1)):
            rotor_dict = self.rotor_config_list[i]
            rotor_axis = np.array(rotor_dict["rotor_axis"])
            rotor_position = np.array(rotor_dict["position"])
            currActuator = RotorModel(
                rotor_axis, rotor_position, rotor_dict["turning_direction"])
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
        rotor_dict = self.rotor_config_list[4]
        rotor_axis = np.array(rotor_dict["rotor_axis"])
        rotor_position = np.array(rotor_dict["position"])
        forwardActuator = RotorModel(
            rotor_axis, rotor_position, rotor_dict["turning_direction"])
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

        if "V_air_body_x" not in self.data_df:
            self.normalize_actuators()
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        # Rotor features
        X_rotor_forces, X_rotor_moments, rotor_forces_coef_list, rotor_moments_coef_list = self.compute_rotor_features()

        # Aerodynamics features
        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
        aoa_mat = self.data_df[["AoA"]].to_numpy()
        aero_model = AeroModelAAE(
            stall_angle=20.0, sig_scale_fac=self.sig_scale_fac)
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
        print("Estimating quad plane model using the following data:")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_matrices()

        self.reg = LinearRegression(fit_intercept=False)
        self.reg.fit(X, y)

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
        model_plots.plot_accel_and_airspeed_in_y_direction(
            self.y_forces, y_forces_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])
        quad_plane_model_plots.plot_accel_predeictions_with_flap_outputs(
            self.y_forces, y_forces_pred, self.data_df[["u5", "u6", "u7"]], self.data_df["timestamp"])
        return
