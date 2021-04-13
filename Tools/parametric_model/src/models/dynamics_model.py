__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model class contains properties shared between all models and shgall simplyfy automated checks and the later
export to a sitl gazebo model by providing a unified interface for all models. """

from ..tools import load_ulog, pandas_from_topic, compute_flight_time, resample_dataframes
from ..tools import quaternion_to_rotation_matrix
import numpy as np
import yaml


class DynamicsModel():
    def __init__(self, rel_ulog_path, req_topics_dict, resample_freq=100.0):
        assert type(
            req_topics_dict) is dict, 'req_topics_dict input must be a dict'
        assert bool(req_topics_dict), 'req_topics_dict can not be empty'
        self.ulog = load_ulog(rel_ulog_path)
        self.req_topics_dict = req_topics_dict
        self.resample_freq = resample_freq
        print(self.req_topics_dict.keys())
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
        resampled_df = resample_dataframes(
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

    def normalize_actuators(self):
        # u : normalize actuator output from pwm to be scaled between 0 and 1
        # To be adjusted using parameters:
        self.min_pwm = 1000
        self.max_pwm = 2000
        self.actuator_columns = self.get_topic_list_from_topic_type(
            "actuator_outputs")
        for actuator in self.actuator_columns:
            actuator_data = self.data_df[actuator].to_numpy()
            for i in range(actuator_data.shape[0]):
                if (actuator_data[i] < self.min_pwm):
                    actuator_data[i] = 0
                else:
                    actuator_data[i] = (
                        actuator_data[i] - self.min_pwm)/(self.max_pwm - self.min_pwm)
            self.data_df[actuator] = actuator_data

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
        self.result_dict = {"coefficients": coef_dict, "metrics": metrics_dict}

    def save_result_dict_to_yaml(self, file_name="model_results.yml"):
        with open(file_name, 'w') as outfile:
            print(yaml.dump(self.result_dict, default_flow_style=False))
            yaml.dump(self.result_dict, outfile, default_flow_style=False)
