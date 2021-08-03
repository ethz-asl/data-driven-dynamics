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
import matplotlib.pyplot as plt
import os

from src.models.model_config import ModelConfig
from src.tools.ulog_tools import load_ulog, pandas_from_topic
from src.tools.dataframe_tools import compute_flight_time, resample_dataframe_list
from src.tools.quat_utils import quaternion_to_rotation_matrix


class DataHandler():
    def __init__(self, config_file="qpm_gazebo_standard_vtol_config.yaml"):
        self.config = ModelConfig(config_file)
        config_dict=self.config.dynamics_model_config

        assert type(
            config_dict) is dict, 'req_topics_dict input must be a dict'
        assert bool(config_dict), 'req_topics_dict can not be empty'
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
    
    def loadLog(self, rel_data_path):
        self.rel_data_path = rel_data_path
        if (os.path.isdir(rel_data_path)):
            self.data_df = pd.DataFrame()
            for filename in os.listdir(rel_data_path):
                if filename.endswith(".ulg"):
                    print(os.path.join(rel_data_path, filename))
                    ulog = load_ulog(os.path.join(rel_data_path, filename))
                    self.check_ulog_for_req_topics(ulog)
                    self.data_df = self.data_df.append(self.compute_resampled_dataframe(ulog))
                    self.data_df.reset_index(drop=True, inplace=True)

                elif filename.endswith(".csv"):
                    raise Exception("Parsing CSV files from a directory is not supported")
                    
                else:
                    continue

        else:
            if (rel_data_path.endswith(".csv")):
                self.data_df = pd.read_csv(rel_data_path, index_col=0)
                for req_topic in self.req_dataframe_topic_list:
                    assert(
                        req_topic in self.data_df), ("missing topic in loaded csv: " + str(req_topic))

            elif (rel_data_path.endswith(".ulg")):
                ulog = load_ulog(rel_data_path)
                self.check_ulog_for_req_topics(ulog)

                self.data_df = self.compute_resampled_dataframe(ulog)

            else:
                raise TypeError("File extension needs to be either csv or ulg")


    def check_ulog_for_req_topics(self, ulog):
        for topic_type in self.req_topics_dict.keys():
            try:
                topic_type_data = ulog.get_dataset(topic_type)

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

    def compute_resampled_dataframe(self, ulog):
        print("Starting data resampling of topic types: ",
              self.req_topics_dict.keys())
        # setup object to crop dataframes for flight data
        fts = compute_flight_time(ulog)
        df_list = []
        topic_type_bar = Bar('Resampling', max=len(
            self.req_topics_dict.keys()))

        # getting data
        for topic_type in self.req_topics_dict.keys():
            topic_dict = self.req_topics_dict[topic_type]
            curr_df = pandas_from_topic(ulog, [topic_type])
            curr_df = curr_df[topic_dict["ulog_name"]]
            if "dataframe_name" in topic_dict.keys():
                assert (len(topic_dict["dataframe_name"]) == len(topic_dict["ulog_name"])), (
                    'could not rename topics of type', topic_type, "due to rename list not having an entry for every topic.")
                curr_df.columns = topic_dict["dataframe_name"]
            df_list.append(curr_df)
            topic_type_bar.next()
        topic_type_bar.finish()
        resampled_df = resample_dataframe_list(
            df_list, fts, self.resample_freq)
        return resampled_df.dropna()

    def visually_select_data(self, plot_config_dict=None):
        from visual_dataframe_selector.data_selector import select_visual_data
        print("Number of data samples before cropping: ",
              self.data_df.shape[0])
        self.data_df = select_visual_data(
            self.data_df, self.visual_dataframe_selector_config_dict)

    def get_dataframes(self):
        return self.data_df

    def visualize_data(self):
        def plot_scatter(ax, title, dataframe_x, dataframe_y, dataframe_z, color='blue'):
            ax.scatter(self.data_df[dataframe_x], self.data_df[dataframe_y], self.data_df[dataframe_z], s=10, facecolor=color, lw=0, alpha=0.1)
            ax.set_title(title)
            ax.set_xlabel(dataframe_x)
            ax.set_ylabel(dataframe_y)
            ax.set_zlabel(dataframe_z)
        
        num_plots = 2
        fig = plt.figure("Data Visualization")
        ax1 = fig.add_subplot(num_plots, 1, 1, projection='3d')
        plot_scatter(ax1, 'Local Velocity', 'vx', 'vy', 'vz')

        ax2 = fig.add_subplot(num_plots, 1, 2, projection='3d')
        plot_scatter(ax2, 'Body Acceleration', 'acc_b_x', 'acc_b_y', 'acc_b_z', 'red')
        plt.show(block=False)


