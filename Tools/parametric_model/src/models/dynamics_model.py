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


class DynamicsModel():
    def __init__(self, config_dict, rel_data_path):

        assert type(
            config_dict) is dict, 'req_topics_dict input must be a dict'
        assert bool(config_dict), 'req_topics_dict can not be empty'
        self.config_dict = config_dict
        self.resample_freq = config_dict["resample_freq"]
        self.req_topics_dict = config_dict["data"]["required_ulog_topics"]

        if (rel_data_path[-4:] == ".cvs"):
            print("tbd")

        elif (rel_data_path[-4:] == ".ulg"):
            self.rel_ulog_path = rel_data_path
            self.ulog = load_ulog(rel_data_path)
            assert self.check_ulog_for_req_topics(
            ), 'not all required topics or topic types are contained in the log file'

            self.compute_resampled_dataframe()

        self.quat_columns = self.get_topic_list_from_topic_type(
            "vehicle_attitude")
        self.quaternion_df = self.data_df[self.quat_columns]
        self.q_mat = self.quaternion_df.to_numpy()

        # used to generate a dict with the resulting coefficients later on.
        self.coef_name_list = []
        self.result_dict = {}

    def check_ulog_for_req_topics(self):
        for topic_type in self.req_topics_dict.keys():
            try:
                topic_type_data = self.ulog.get_dataset(topic_type)
                topic_type_data = topic_type_data.data
                ulog_topic_list = self.req_topics_dict[topic_type]["ulog_name"]
                for topic_index in range(len(ulog_topic_list)):
                    topic = ulog_topic_list[topic_index]
                    topic_data = (topic_type_data[topic])
            except:
                print("Missing topic type: ", topic_type)
                return False
        return True

    def compute_resampled_dataframe(self):
        # setup object to crop dataframes for flight data
        fts = compute_flight_time(self.ulog)
        df_list = []
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
        X_body_rot_coef_list = ["I_y-I_z", "I_z-I_x", "I_x- I_y"]
        for i in range(angular_vel_mat.shape[0]):
            X_body_rot[3*i, 0] = angular_vel_mat[i,
                                                 1]*angular_vel_mat[i, 2]
            X_body_rot[3*i + 1, 0] = angular_vel_mat[i, 2] * \
                angular_vel_mat[i, 0]
            X_body_rot[3*i + 2, 0] = angular_vel_mat[i, 0] * \
                angular_vel_mat[i, 1]
        return X_body_rot, X_body_rot_coef_list

    def normalize_actuators(self):
        # u : normalize actuator output from pwm to be scaled between 0 and 1
        # To be adjusted using parameters:
        self.min_pwm = 1000
        self.max_pwm = 2000
        self.trim_pwm = 1500
        self.actuator_columns = self.get_topic_list_from_topic_type(
            "actuator_outputs")
        self.actuator_type = self.req_topics_dict["actuator_outputs"]["actuator_type"]
        self.actuator_type.remove("timestamp")
        for i in range(len(self.actuator_columns)):
            actuator_data = self.data_df[self.actuator_columns[i]].to_numpy()
            if (self.actuator_type[i] == "motor"):
                for j in range(actuator_data.shape[0]):
                    if (actuator_data[j] < self.min_pwm):
                        actuator_data[j] = 0
                    else:
                        actuator_data[j] = (
                            actuator_data[j] - self.min_pwm)/(self.max_pwm - self.min_pwm)
            elif (self.actuator_type[i] == "control_surcafe"):
                for j in range(actuator_data.shape[0]):
                    if (actuator_data[j] < self.min_pwm):
                        actuator_data[j] = 0
                    else:
                        actuator_data[j] = 2*(
                            actuator_data[j] - self.trim_pwm)/(self.max_pwm - self.min_pwm)
            else:
                print("actuator type unknown:", self.actuator_type[i])
                print("normalization failed")
                exit(1)
            self.data_df[self.actuator_columns[i]] = actuator_data

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
            ("Length of coefficient list and coefficient name list does not match: Coefficientlist:",
             len(coefficient_list), "Coefficient name list: ", len(self.coef_name_list))
        coefficient_list = [float(coef) for coef in coefficient_list]
        coef_dict = dict(zip(self.coef_name_list, coefficient_list))
        self.result_dict = {"coefficients": coef_dict,
                            "metrics": metrics_dict, "log_file": self.rel_ulog_path}

    def save_result_dict_to_yaml(self, file_name="model_parameters", result_path="resources/model_results/"):

        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = result_path + file_name + "_" + timestr + ".yaml"

        with open(file_path, 'w') as outfile:
            print(yaml.dump(self.result_dict, default_flow_style=False))
            yaml.dump(self.result_dict, outfile, default_flow_style=False)
