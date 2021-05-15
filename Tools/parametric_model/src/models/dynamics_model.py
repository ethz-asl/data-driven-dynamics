__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model class contains properties shared between all models and shgall simplyfy automated checks and the later
export to a sitl gazebo model by providing a unified interface for all models. """

from ..tools import load_ulog, pandas_from_topic, compute_flight_time, resample_dataframe_list
from ..tools import quaternion_to_rotation_matrix
import numpy as np
import yaml
import time
import math
import pandas as pd

from .rotor_models import RotorModel, BiDirectionalRotorModel, TiltingRotorModel, ChangingAxisRotorModel

from progress.bar import Bar


class DynamicsModel():
    def __init__(self, config_dict, rel_data_path):

        assert type(
            config_dict) is dict, 'req_topics_dict input must be a dict'
        assert bool(config_dict), 'req_topics_dict can not be empty'
        self.config_dict = config_dict
        self.resample_freq = config_dict["resample_freq"]
        print("Resample frequency: ", self.resample_freq, "Hz")
        self.req_topics_dict = config_dict["data"]["required_ulog_topics"]
        self.req_dataframe_topic_list = config_dict["data"]["req_dataframe_topic_list"]
        self.rel_data_path = rel_data_path

        self.estimate_forces = config_dict["estimate_forces"]
        self.estimate_moments = config_dict["estimate_moments"]

        if (rel_data_path[-4:] == ".csv"):
            self.data_df = pd.read_csv(rel_data_path, index_col=0)
            for req_topic in self.req_dataframe_topic_list:
                assert(
                    req_topic in self.data_df), ("missing topic in loaded csv: " + str(req_topic))

        elif (rel_data_path[-4:] == ".ulg"):

            self.ulog = load_ulog(rel_data_path)
            self.check_ulog_for_req_topics()

            self.compute_resampled_dataframe()

        else:
            print("ERROR: file extension needs to be either csv or ulg:")
            print(rel_data_path)
            exit(1)

        self.quaternion_df = self.data_df[["q0", "q1", "q2", "q3"]]
        self.q_mat = self.quaternion_df.to_numpy()

        # used to generate a dict with the resulting coefficients later on.
        self.coef_name_list = []
        self.result_dict = {}

    def check_ulog_for_req_topics(self):
        for topic_type in self.req_topics_dict.keys():
            try:
                topic_type_data = self.ulog.get_dataset(topic_type)

            except:
                print("Missing topic type: ", topic_type)
                exit(1)

            topic_type_data = topic_type_data.data
            ulog_topic_list = self.req_topics_dict[topic_type]["ulog_name"]
            for topic_index in range(len(ulog_topic_list)):
                try:
                    topic = ulog_topic_list[topic_index]
                    topic_data = (topic_type_data[topic])
                except:
                    print("Missing topic: ", topic_type,
                          ulog_topic_list[topic_index])
                    exit(1)

        return

    def compute_resampled_dataframe(self):
        print("Starting data resampling of topic types: ",
              self.req_topics_dict.keys())
        # setup object to crop dataframes for flight data
        fts = compute_flight_time(self.ulog)
        df_list = []
        topic_type_bar = Bar('Resampling', max=len(
            self.req_topics_dict.keys()))

        # getting data
        for topic_type in self.req_topics_dict.keys():
            topic_dict = self.req_topics_dict[topic_type]
            curr_df = pandas_from_topic(self.ulog, [topic_type])
            curr_df = curr_df[topic_dict["ulog_name"]]
            if "dataframe_name" in topic_dict.keys():
                assert (len(topic_dict["dataframe_name"]) == len(topic_dict["ulog_name"])), (
                    'could not rename topics of type', topic_type, "due to rename list not having an entry for every topic.")
                curr_df.columns = topic_dict["dataframe_name"]
            df_list.append(curr_df)
            topic_type_bar.next()
        topic_type_bar.finish()
        resampled_df = resample_dataframe_list(
            df_list, fts["t_start"], fts["t_end"], self.resample_freq)
        self.data_df = resampled_df.dropna()

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
            "V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"])
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
            self.max_output = 1
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
            elif ((self.actuator_type[i] == "control_surcafe" or self.actuator_type[i] == "bi_directional_motor")):
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

    def initialize_rotor_model(self, rotor_config_dict, rotor_group, angular_vel_mat=None):
        valid_rotor_models = ["RotorModel", "ChangingAxisRotorModel",
                              "BiDirectionalRotorModel", "TiltingRotorModel"]
        rotor_input_name = rotor_config_dict["dataframe_name"]
        u_vec = self.data_df[rotor_input_name].to_numpy()
        if "rotor_model" not in rotor_config_dict.keys():
            # Set default rotor model
            rotor_model = "RotorModel"
        else:
            rotor_model = rotor_config_dict["rotor_model"]

        if rotor_model == "RotorModel":
            rotor = RotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_model == "ChangingAxisRotorModel":
            rotor = ChangingAxisRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_model == "BiDirectionalRotorModel":
            rotor = BiDirectionalRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, angular_vel_mat=angular_vel_mat)
        elif rotor_model == "TiltingRotorModel":
            tilt_actuator_df_name = rotor_config_dict["tilt_actuator_dataframe_name"]
            tilt_actuator_vec = self.data_df[tilt_actuator_df_name]
            rotor = TiltingRotorModel(
                rotor_config_dict, u_vec, self.v_airspeed_mat, tilt_actuator_vec, angular_vel_mat=angular_vel_mat)
        else:
            print(rotor_model, " is not a valid rotor model.")
            print("Valid rotor models are: ", valid_rotor_models)
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
                    rotor_config_dict, rotor_group, angular_vel_mat)
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
            R_world_to_body = quaternion_to_rotation_matrix(self.q_mat[i, :])
            vec_mat_transformed[i, :] = R_world_to_body @ vec_mat[i, :]
        return vec_mat_transformed

    def generate_model_dict(self, coefficient_list, metrics_dict):
        assert (len(self.coef_name_list) == len(coefficient_list)), \
            ("Length of coefficient list and coefficient name list does not match: Length of coefficient list:",
             len(coefficient_list), "length of coefficient name list: ", len(self.coef_name_list))
        coefficient_list = [float(coef) for coef in coefficient_list]
        coef_dict = dict(zip(self.coef_name_list, coefficient_list))
        self.result_dict = {"coefficients": coef_dict,
                            "metrics": metrics_dict, "log_file": self.rel_data_path}

    def save_result_dict_to_yaml(self, file_name="model_parameters", result_path="resources/model_results/"):

        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = result_path + file_name + "_" + timestr + ".yaml"

        with open(file_path, 'w') as outfile:
            print(yaml.dump(self.result_dict, default_flow_style=False))
            yaml.dump(self.result_dict, outfile, default_flow_style=False)
