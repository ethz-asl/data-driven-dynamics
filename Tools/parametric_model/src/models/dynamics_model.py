__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model class contains properties shared between all models and shgall simplyfy automated checks and the later
export to a sitl gazebo model by providing a unified interface for all models. """

from progress.bar import Bar
import pandas as pd
import math
import time
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from .rotor_models import RotorModel, BiDirectionalRotorModel, TiltingRotorModel, ChangingAxisRotorModel
from .model_plots import model_plots
from .model_plots import aerodynamics_plots
from src.tools.ulog_tools import load_ulog, pandas_from_topic
from src.tools.dataframe_tools import compute_flight_time, resample_dataframe_list
from src.tools.quat_utils import quaternion_to_rotation_matrix
from src.tools.math_tools import cropped_sym_sigmoid


class DynamicsModel():
    def __init__(self, config_dict):

        assert type(
            config_dict) is dict, 'req_topics_dict input must be a dict'
        assert bool(config_dict), 'req_topics_dict can not be empty'
        self.model_name = "unknown_model"
        self.config_dict = config_dict
        self.resample_freq = config_dict["resample_freq"]
        print("Resample frequency: ", self.resample_freq, "Hz")
        self.req_topics_dict = config_dict["data"]["required_ulog_topics"]
        self.req_dataframe_topic_list = config_dict["data"]["req_dataframe_topic_list"]

        self.visual_dataframe_selector_config_dict = {
            "x_axis_col": "timestamp",
            "sub_plt1_data": ["q0", "q1", "q2", "q3"],
            "sub_plt2_data": ["u0", "u1", "u2", "u3"]}

        self.estimate_forces = config_dict["estimate_forces"]
        self.estimate_moments = config_dict["estimate_moments"]

        # used to generate a dict with the resulting coefficients later on.
        self.coef_name_list = []
        self.result_dict = {}

    def prepare_regression_matrices(self):
        if "V_air_body_x" not in self.data_df:
            self.normalize_actuators()
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        # Rotor features
        angular_vel_mat = self.data_df[[
            "ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy()
        self.compute_rotor_features(self.rotor_config_dict, angular_vel_mat)

        if (self.estimate_forces and self.estimate_moments):
            self.prepare_force_regression_matrices()
            self.prepare_moment_regression_matrices()

            self.X = block_diag(self.X_forces, self.X_moments)
            self.y = np.hstack((self.y_forces, self.y_moments))

        elif (self.estimate_forces):
            self.prepare_force_regression_matrices()

            self.X = self.X_forces
            self.y = self.y_forces

        elif (self.estimate_moments):
            self.prepare_moment_regression_matrices()

            self.X = self.X_moments
            self.y = self.y_moments

        else:
            raise ValueError("Neither Forces nor Moments estimation activated")
        
        return self.X, self.y

    def prepare_force_regression_matrices(self):
        raise NotImplementedError()

    def prepare_moment_regression_matrices(self):
        raise NotImplementedError()

    def get_topic_list_from_topic_type(self, topic_type):
        topic_type_name_dict = self.req_topics_dict[topic_type]
        if "dataframe_name" in topic_type_name_dict.keys():
            topic_columns = topic_type_name_dict["dataframe_name"].copy()
        else:
            topic_columns = topic_type_name_dict["ulog_name"].copy()
        topic_columns.remove("timestamp")
        return topic_columns

    def compute_airspeed_from_groundspeed(self, airspeed_topic_list):
        groundspeed_ned_mat = (self.data_df[airspeed_topic_list]).to_numpy()
        airspeed_body_mat = self.rot_to_body_frame(groundspeed_ned_mat)
        aoa_vec = np.zeros((airspeed_body_mat.shape[0], 1))
        for i in range(airspeed_body_mat.shape[0]):
            aoa_vec[i, :] = math.atan2(
                airspeed_body_mat[i, 2], airspeed_body_mat[i, 0])
        airspeed_body_mat = np.hstack((airspeed_body_mat, aoa_vec))
        airspeed_body_df = pd.DataFrame(airspeed_body_mat, columns=[
            "V_air_body_x", "V_air_body_y", "V_air_body_z", "angle_of_attack"])
        self.data_df = pd.concat(
            [self.data_df, airspeed_body_df], axis=1, join="inner")

    def compute_body_rotation_features(self, angular_vel_topic_list):
        """Include the moment contribution due to rotation body frame:
        w x Iw = X_body_rot * v
        Where v = (I_y-I_z, I_z-I_x, I_x- I_y)^T
        is comprised of the inertia moments we want to estimate
        """
        angular_vel_mat = (self.data_df[angular_vel_topic_list]).to_numpy()
        X_body_rot = np.zeros((3*angular_vel_mat.shape[0], 3))
        X_body_rot_coef_list = ["I_yy-I_zz", "I_zz-I_xx", "I_xx- I_yy"]
        for i in range(angular_vel_mat.shape[0]):
            X_body_rot[3*i, 0] = angular_vel_mat[i,
                                                 1]*angular_vel_mat[i, 2]
            X_body_rot[3*i + 1, 0] = angular_vel_mat[i, 2] * \
                angular_vel_mat[i, 0]
            X_body_rot[3*i + 2, 0] = angular_vel_mat[i, 0] * \
                angular_vel_mat[i, 1]
        return X_body_rot, X_body_rot_coef_list

    def normalize_actuators(self, actuator_topic_types=["actuator_outputs"], control_outputs_used=False):
        # u : normalize actuator output from pwm to be scaled between 0 and 1
        # To be adjusted using parameters:

        # This should probably be adapted in the future to allow different values for each actuator specified in the config.
        if control_outputs_used:
            self.min_output = -1
            self.max_output = 1.01
            self.trim_output = 0
        else:
            self.min_output = 1000
            self.max_output = 2000
            self.trim_output = 1500

        self.actuator_columns = []
        self.actuator_type = []

        for topic_type in actuator_topic_types:
            self.actuator_columns += self.get_topic_list_from_topic_type(
                topic_type)
            self.actuator_type += self.req_topics_dict[topic_type]["actuator_type"]
            self.actuator_type.remove("timestamp")

        for i in range(len(self.actuator_columns)):
            actuator_data = self.data_df[self.actuator_columns[i]].to_numpy()
            if (self.actuator_type[i] == "motor"):
                for j in range(actuator_data.shape[0]):
                    if (actuator_data[j] < self.min_output):
                        actuator_data[j] = 0
                    else:
                        actuator_data[j] = (
                            actuator_data[j] - self.min_output)/(self.max_output - self.min_output)
            elif ((self.actuator_type[i] == "control_surface" or self.actuator_type[i] == "bi_directional_motor")):
                for j in range(actuator_data.shape[0]):
                    if (actuator_data[j] < self.min_output):
                        actuator_data[j] = 0
                    else:
                        actuator_data[j] = 2*(
                            actuator_data[j] - self.trim_output)/(self.max_output - self.min_output)
            else:
                print("actuator type unknown:", self.actuator_type[i])
                print("normalization failed")
                exit(1)
            self.data_df[self.actuator_columns[i]] = actuator_data

    def initialize_rotor_model(self, rotor_config_dict, angular_vel_mat=None):
        valid_rotor_types = ["RotorModel", "ChangingAxisRotorModel",
                             "BiDirectionalRotorModel", "TiltingRotorModel"]
        rotor_input_name = rotor_config_dict["dataframe_name"]
        u_vec = self.data_df[rotor_input_name].to_numpy()
        if "rotor_type" not in rotor_config_dict.keys():
            # Set default rotor model
            rotor_type = "RotorModel"
            print("no Rotor model specified for ", rotor_input_name)
            print("Selecting default: RotorModel")
        else:
            rotor_type = rotor_config_dict["rotor_type"]

        if rotor_type == "RotorModel":
            rotor = RotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_type == "ChangingAxisRotorModel":
            rotor = ChangingAxisRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_type == "BiDirectionalRotorModel":
            rotor = BiDirectionalRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_type == "TiltingRotorModel":
            tilt_actuator_df_name = rotor_config_dict["tilt_actuator_dataframe_name"]
            tilt_actuator_vec = self.data_df[tilt_actuator_df_name]
            rotor = TiltingRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, tilt_actuator_vec, angular_vel_mat=angular_vel_mat)
        else:
            print(rotor_type, " is not a valid rotor model.")
            print("Valid rotor models are: ", valid_rotor_types)
            print("Adapt your config file to a valid rotor model!")
            exit(1)

        return rotor

    def compute_rotor_features(self, rotors_config_dict, angular_vel_mat=None):

        self.v_airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        self.rotor_dict = {}

        for rotor_group in rotors_config_dict.keys():
            rotor_group_list = rotors_config_dict[rotor_group]
            self.rotor_dict[rotor_group] = {}
            if (self.estimate_forces):
                X_force_collector = np.zeros(
                    (3*self.v_airspeed_mat.shape[0], 3))
            if (self.estimate_moments):
                X_moment_collector = np.zeros(
                    (3*self.v_airspeed_mat.shape[0], 5))
            for rotor_config_dict in rotor_group_list:
                rotor = self.initialize_rotor_model(
                    rotor_config_dict, angular_vel_mat)
                self.rotor_dict[rotor_group][rotor_config_dict["dataframe_name"]] = rotor

                if (self.estimate_forces):
                    X_force_curr, curr_rotor_forces_coef_list = rotor.compute_actuator_force_matrix()
                    X_force_collector = X_force_collector + X_force_curr
                    # Include rotor group name in coefficient names:
                    for i in range(len(curr_rotor_forces_coef_list)):
                        curr_rotor_forces_coef_list[i] = rotor_group + \
                            curr_rotor_forces_coef_list[i]

                if (self.estimate_moments):
                    X_moment_curr, curr_rotor_moments_coef_list = rotor.compute_actuator_moment_matrix()
                    X_moment_collector = X_moment_collector + X_moment_curr
                    # Include rotor group name in coefficient names:
                    for i in range(len(curr_rotor_moments_coef_list)):
                        curr_rotor_moments_coef_list[i] = rotor_group + \
                            curr_rotor_moments_coef_list[i]

            if (self.estimate_forces):
                if 'X_rotor_forces' not in vars():
                    X_rotor_forces = X_force_collector
                    self.rotor_forces_coef_list = curr_rotor_forces_coef_list
                else:
                    X_rotor_forces = np.hstack(
                        (X_rotor_forces, X_force_collector))
                    self.rotor_forces_coef_list += curr_rotor_forces_coef_list
                self.X_rotor_forces = X_rotor_forces

            if (self.estimate_moments):
                if 'X_rotor_moments' not in vars():
                    X_rotor_moments = X_moment_collector
                    self.rotor_moments_coef_list = curr_rotor_moments_coef_list
                else:
                    X_rotor_moments = np.hstack(
                        (X_rotor_moments, X_moment_collector))
                    self.rotor_moments_coef_list += curr_rotor_moments_coef_list
                self.X_rotor_moments = X_rotor_moments

        return

    def rot_to_body_frame(self, vec_mat):
        """
        Rotates horizontally stacked 3D vectors from NED world frame to FRD body frame

        inputs:
        vec_mat: numpy array of dimensions (n,3),
        containing the horizontally stacked 3D vectors [x,y,z] in world frame.
        """
        vec_mat_transformed = np.zeros(vec_mat.shape)
        for i in range(vec_mat.shape[0]):
            R_world_to_body = np.linalg.inv(
                quaternion_to_rotation_matrix(self.q_mat[i, :]))
            vec_mat_transformed[i, :] = np.transpose(
                R_world_to_body @ np.transpose(vec_mat[i, :]))
        return vec_mat_transformed

    def rot_to_world_frame(self, vec_mat):
        """
        Rotates horizontally stacked 3D vectors from FRD body frame to NED world frame

        inputs:
        vec_mat: numpy array of dimensions (n,3),
        containing the horizontally stacked 3D vectors [x,y,z] in body frame.
        """
        vec_mat_transformed = np.zeros(vec_mat.shape)
        for i in range(vec_mat.shape[0]):
            R_body_to_world = quaternion_to_rotation_matrix(self.q_mat[i, :])
            vec_mat_transformed[i, :] = R_body_to_world @ vec_mat[i, :]
        return vec_mat_transformed

    def generate_model_dict(self, coefficient_list, metrics_dict, model_dict):
        assert (len(self.coef_name_list) == len(coefficient_list)), \
            ("Length of coefficient list and coefficient name list does not match: Length of coefficient list:",
             len(coefficient_list), "length of coefficient name list: ", len(self.coef_name_list))
        coefficient_list = [float(coef) for coef in coefficient_list]
        coef_dict = dict(zip(self.coef_name_list, coefficient_list))
        self.result_dict = {"model": model_dict,
                            "coefficients": coef_dict,
                            "metrics": metrics_dict,
                            "numper of samples": self.n_samples}

    def save_result_dict_to_yaml(self, file_name="model_parameters", result_path="model_results/"):

        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = result_path + file_name + "_" + timestr + ".yaml"

        with open(file_path, 'w') as outfile:
            print(yaml.dump(self.result_dict, default_flow_style=False))
            yaml.dump(self.result_dict, outfile, default_flow_style=False)

    def load_dataframes(self, data_frames):
        self.data_df = data_frames

        self.n_samples = self.data_df.shape[0]
        self.quaternion_df = self.data_df[["q0", "q1", "q2", "q3"]]
        self.q_mat = self.quaternion_df.to_numpy()

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
        model_dict = {}
        model_dict.update(self.rotor_config_dict)
        if hasattr(self, 'aero_config_dict'):
            model_dict.update(self.aero_config_dict)
        self.generate_model_dict(coef_list, metrics_dict, model_dict)
        self.save_result_dict_to_yaml(file_name=self.model_name)

        return

    def compute_residuals(self):
        
        y_pred = self.reg.predict(self.X)

        y_forces_pred = y_pred[0:self.y_forces.shape[0]]
        y_moments_pred = y_pred[self.y_forces.shape[0]:]

        error_y_forces = y_forces_pred - self.y_forces
        error_y_moments = y_moments_pred - self.y_moments

        stacked_error_y_forces = np.array(error_y_forces)
        acc_mat = stacked_error_y_forces.reshape((-1, 3))
        residual_force_df = pd.DataFrame(acc_mat, columns=[
            "residual_force_x", "residual_force_y", "residual_force_z"])
        self.data_df = pd.concat(
            [self.data_df, residual_force_df], axis=1, join="inner")
        
        stacked_error_y_moments = np.array(error_y_moments)
        mom_mat = stacked_error_y_moments.reshape((-1, 3))
        residual_moment_df = pd.DataFrame(mom_mat, columns=[
            "residual_moment_x", "residual_moment_y", "residual_moment_z"])
        self.data_df = pd.concat(
            [self.data_df, residual_moment_df], axis=1, join="inner")

    def plot_model_predicitons(self):
        def plot_scatter(ax, title, dataframe_x, dataframe_y, dataframe_z, color='blue'):
            ax.scatter(self.data_df[dataframe_x], self.data_df[dataframe_y], self.data_df[dataframe_z], s=10, facecolor=color, lw=0, alpha=0.1)
            ax.set_title(title)
            ax.set_xlabel(dataframe_x)
            ax.set_ylabel(dataframe_y)
            ax.set_zlabel(dataframe_z)

        y_pred = self.reg.predict(self.X)

        if (self.estimate_forces and self.estimate_moments):
            y_forces_pred = y_pred[0:self.y_forces.shape[0]]
            y_moments_pred = y_pred[self.y_forces.shape[0]:]
            model_plots.plot_accel_predeictions(
                self.y_forces, y_forces_pred, self.data_df["timestamp"])
            model_plots.plot_angular_accel_predeictions(
                self.y_moments, y_moments_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "angle_of_attack"]], self.data_df["timestamp"])

        elif (self.estimate_forces):
            y_forces_pred = y_pred
            model_plots.plot_accel_predeictions(
                self.y_forces, y_forces_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "angle_of_attack"]], self.data_df["timestamp"])

        elif (self.estimate_moments):
            y_moments_pred = y_pred
            model_plots.plot_angular_accel_predeictions(
                self.y_moments, y_moments_pred, self.data_df["timestamp"])
            model_plots.plot_airspeed_and_AoA(
                self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "angle_of_attack"]], self.data_df["timestamp"])

        fig = plt.figure("Residual Visualization")
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        plot_scatter(ax1, "Residual force", "residual_force_x", "residual_force_y", "residual_force_z", 'blue')

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')

        plot_scatter(ax2, "Residual moment", "residual_moment_x", "residual_moment_y", "residual_moment_z", 'blue')

        ax3 = fig.add_subplot(2, 2, 3, projection='3d')

        plot_scatter(ax3, "Measured Acceleration", "acc_b_x", "acc_b_y", "acc_b_z", 'blue')

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')

        plot_scatter(ax4, "Measured Angular Acceleration", "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z", 'blue')

        if hasattr(self, 'aero_config_dict'):
            coef_list = list(self.reg.coef_) + [self.reg.intercept_]
            coef_dict = dict(zip(self.coef_name_list, coef_list))
            aerodynamics_plots.plot_liftdrag_curve(coef_dict, self.aerodynamics_dict)

        plt.show()
        return
