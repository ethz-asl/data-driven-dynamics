"""
 *
 * Copyright (c) 2021 Manuel Yves Galliker
 *               2021 Autonomous Systems Lab ETH Zurich
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name Data Driven Dynamics nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *

The model class contains properties shared between all models and shgall simplyfy automated checks and the later
export to a sitl gazebo model by providing a unified interface for all models. """

__author__ = "Manuel Yves Galliker, Julius Schlapbach"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

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


class DataHandler(object):
    visual_dataframe_selector_config_dict = {
        "x_axis_col": "timestamp",
        "sub_plt1_data": ["q0", "q1", "q2", "q3"],
        "sub_plt2_data": ["u0", "u1", "u2", "u3"],
    }

    def __init__(self, config_file):
        print(
            "==============================================================================="
        )
        print(
            "                              Data Processing                                  "
        )
        print(
            "==============================================================================="
        )
        self.config = ModelConfig(config_file)
        config_dict = self.config.dynamics_model_config

        assert type(config_dict) is dict, "req_topics_dict input must be a dict"
        assert bool(config_dict), "req_topics_dict can not be empty"
        self.config_dict = config_dict
        self.resample_freq = config_dict["resample_freq"]
        self.estimate_angular_acceleration = config_dict[
            "estimate_angular_acceleration"
        ]
        print("Resample frequency: ", self.resample_freq, "Hz")
        self.req_topics_dict = config_dict["data"]["required_ulog_topics"]

        self.req_dataframe_topic_list = config_dict["data"]["req_dataframe_topic_list"]

        self.estimate_forces = config_dict["estimate_forces"]
        self.estimate_moments = config_dict["estimate_moments"]

        # used to generate a dict with the resulting coefficients later on.
        self.coef_name_list = []
        self.result_dict = {}

    def loadLogs(self, rel_data_path):
        self.rel_data_path = rel_data_path
        if os.path.isdir(rel_data_path):
            self.data_df = pd.DataFrame()
            for filename in os.listdir(rel_data_path):
                self.loadLogFile(os.path.join(rel_data_path, filename))

        else:
            if not self.loadLogFile(rel_data_path):
                raise TypeError("File extension needs to be either csv or ulg")

    def loadLogFile(self, rel_data_path):
        if rel_data_path.endswith(".csv"):
            print("Loading CSV file: ", rel_data_path)
            self.data_df = pd.read_csv(rel_data_path, index_col=0)
            print("Loading topics: ", self.req_dataframe_topic_list)
            for req_topic in self.req_dataframe_topic_list:
                assert req_topic in self.data_df, "missing topic in loaded csv: " + str(
                    req_topic
                )
            return True

        elif rel_data_path.endswith(".ulg"):
            print("Loading uLog file: ", rel_data_path)
            ulog = load_ulog(rel_data_path)
            print("Loading topics:")
            for req_topic in self.req_topics_dict:
                print(req_topic)
            self.check_ulog_for_req_topics(ulog)

            # compute flight time based on the landed topic
            landed_df = pandas_from_topic(ulog, ["vehicle_land_detected"])
            fts = compute_flight_time(landed_df)

            if len(fts) == 1:
                self.data_df = self.compute_resampled_dataframe(ulog, fts[0])
            else:
                for ft in fts:
                    # check if the dataframe already exists and if so, append to it
                    if getattr(self, "data_df", None) is None:
                        self.data_df = self.compute_resampled_dataframe(ulog, ft)
                    else:
                        self.data_df.append(self.compute_resampled_dataframe(ulog, ft))

            return True

        else:
            return False

    def check_ulog_for_req_topics(self, ulog):
        for topic_type in self.req_topics_dict.keys():
            try:
                topic_dict = self.req_topics_dict[topic_type]
                if "id" in topic_dict.keys():
                    id = topic_dict["id"]
                    topic_type_data = ulog.get_dataset(topic_type, id)
                else:
                    topic_type_data = ulog.get_dataset(topic_type)
            except:
                print("Missing topic type: ", topic_type)
                exit(1)
            topic_type_data = topic_type_data.data
            ulog_topic_list = self.req_topics_dict[topic_type]["ulog_name"]
            for topic_index in range(len(ulog_topic_list)):
                try:
                    topic = ulog_topic_list[topic_index]
                    topic_data = topic_type_data[topic]
                except:
                    print("Missing topic: ", topic_type, ulog_topic_list[topic_index])
                    exit(1)
        return

    def compute_resampled_dataframe(self, ulog, fts):
        print("Starting data resampling of topic types: ", self.req_topics_dict.keys())
        # setup object to crop dataframes for flight data
        df_list = []
        topic_type_bar = Bar("Resampling", max=len(self.req_topics_dict.keys()))

        # getting data
        for topic_type in self.req_topics_dict.keys():
            topic_dict = self.req_topics_dict[topic_type]

            if "id" in topic_dict.keys():
                id = topic_dict["id"]
                curr_df = pandas_from_topic(ulog, [topic_type], id)
            else:
                curr_df = pandas_from_topic(ulog, [topic_type])

            curr_df = curr_df[topic_dict["ulog_name"]]
            if "dataframe_name" in topic_dict.keys():
                assert len(topic_dict["dataframe_name"]) == len(
                    topic_dict["ulog_name"]
                ), (
                    "could not rename topics of type",
                    topic_type,
                    "due to rename list not having an entry for every topic.",
                )
                curr_df.columns = topic_dict["dataframe_name"]
            topic_type_bar.next()
            if (
                topic_type == "vehicle_angular_velocity"
                and self.estimate_angular_acceleration
            ):
                ang_vel_mat = curr_df[
                    ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
                ].to_numpy()
                time_in_secods_np = curr_df[["timestamp"]].to_numpy() / 1000000
                time_in_secods_np = time_in_secods_np.flatten()
                ang_acc_np = np.gradient(ang_vel_mat, time_in_secods_np, axis=0)
                topic_type_bar.next()
                curr_df[["ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]] = ang_acc_np

            df_list.append(curr_df)

        topic_type_bar.finish()

        # Check if actuator topics are empty
        if not fts:
            print("could not select flight time due to missing actuator topic")
            exit(1)

        if isinstance(fts, list):
            resampled_df = []
            for ft in fts:
                new_resampled_df = resample_dataframe_list(
                    df_list, ft, self.resample_freq
                )
                resampled_df.append(new_resampled_df)
            resampled_df = pd.concat(resampled_df, ignore_index=True)
        else:
            resampled_df = resample_dataframe_list(df_list, fts, self.resample_freq)

        if self.estimate_angular_acceleration:
            ang_vel_mat = resampled_df[
                ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
            ].to_numpy()
            for i in range(3):
                ang_vel_mat[:, i] = (
                    np.convolve(ang_vel_mat[:, i], np.ones(33), mode="same") / 33
                )

            # Alternate forward differentiation version
            # ang_vel_mat_1 = np.roll(ang_vel_mat, -1, axis=0)
            # diff_angular_acc_mat = (
            #     ang_vel_mat_1 - ang_vel_mat) * self.resample_freq
            # resampled_df[["ang_acc_b_x", "ang_acc_b_y",
            #               "ang_acc_b_z"]] = diff_angular_acc_mat

            time_in_secods_np = resampled_df[["timestamp"]].to_numpy() / 1000000
            time_in_secods_np = time_in_secods_np.flatten()
            ang_acc_np = np.gradient(ang_vel_mat, time_in_secods_np, axis=0)
            topic_type_bar.next()
            resampled_df[["ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]] = ang_acc_np

        return resampled_df.dropna()

    def visually_select_data(self, plot_config_dict=None):
        print(
            "==============================================================================="
        )
        print(
            "                           Data Selection Enabled                              "
        )
        print(
            "==============================================================================="
        )
        from visual_dataframe_selector.data_selector import select_visual_data

        print("Number of data samples before cropping: ", self.data_df.shape[0])
        self.data_df = select_visual_data(
            self.data_df, self.visual_dataframe_selector_config_dict
        )

    def get_dataframes(self):
        return self.data_df

    def visualize_data(self):
        def plot_scatter(
            ax, title, dataframe_x, dataframe_y, dataframe_z, color="blue"
        ):
            ax.scatter(
                self.data_df[dataframe_x],
                self.data_df[dataframe_y],
                self.data_df[dataframe_z],
                s=10,
                facecolor=color,
                lw=0,
                alpha=0.1,
            )
            ax.set_title(title)
            ax.set_xlabel(dataframe_x)
            ax.set_ylabel(dataframe_y)
            ax.set_zlabel(dataframe_z)

        num_plots = 2
        fig = plt.figure("Data Visualization")
        ax1 = fig.add_subplot(num_plots, 2, 1, projection="3d")
        plot_scatter(ax1, "Local Velocity", "vx", "vy", "vz")

        ax2 = fig.add_subplot(num_plots, 2, 2, projection="3d")
        plot_scatter(ax2, "Body Acceleration", "acc_b_x", "acc_b_y", "acc_b_z", "red")

        ax3 = fig.add_subplot(num_plots, 2, 3, projection="3d")
        plot_scatter(
            ax3, "Body Angular Velocity", "ang_vel_x", "ang_vel_y", "ang_vel_z", "red"
        )

        ax4 = fig.add_subplot(num_plots, 2, 4, projection="3d")
        plot_scatter(
            ax4,
            "Body Angular Acceleration",
            "ang_acc_b_x",
            "ang_acc_b_y",
            "ang_acc_b_z",
            "red",
        )
        plt.show(block=False)
