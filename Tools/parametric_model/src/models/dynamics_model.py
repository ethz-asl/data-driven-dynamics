__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model class contains properties shared between all models and shgall simplyfy automated checks and the later 
export to a sitl gazebo model by providing a unified interface for all models. """

from ..tools import load_ulog, pandas_from_topic, compute_flight_time, resample_dataframes
from ..tools import quaternion_to_rotation_matrix
from pyulog import core
import pandas as pd
import numpy as np


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
        self.compute_attitude_df()

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

    def compute_attitude_df(self):
        quat_name_dict = self.req_topics_dict["vehicle_attitude"]
        if "dataframe_name" in quat_name_dict.keys():
            quat_name_list = quat_name_dict["dataframe_name"]
        else:
            quat_name_list = quat_name_dict["ulog_name"]
        quat_name_list.remove("timestamp")
        self.quaternion_df = self.data_df[quat_name_list]
        self.q_mat = self.quaternion_df.to_numpy()

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
