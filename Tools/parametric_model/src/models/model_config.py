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
"""

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

import os
import yaml
from pathlib import Path


class ModelConfig():
    def __init__(self, config_file_name):
        """
        Input: config_file as found in Tools/parametric_model/src/models/model_config
        """
        log_file_path = config_file_name

        try:
            with open(log_file_path) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                config_dict = yaml.load(file, Loader=yaml.FullLoader)
                assert (type(config_dict) is dict)

        except:
            print("Could not load yaml config file. Does the specified file exist?")
            print(log_file_path)
            exit(1)

        self.model_name = config_dict["model_name"]
        self.model_type = config_dict["model_type"]
        self.model_class = config_dict["model_class"]

        self.check_dynamics_model_config(config_dict)

        self.dynamics_model_config = config_dict["dynamics_model_config"]
        self.model_config = config_dict["model_config"]

        self.generate_req_topic_list()
        print("Initializing of configuration successful. ")

        return

    def generate_req_topic_list(self):
        req_dataframe_topic_list = []
        req_ulog_topics_dict = self.dynamics_model_config["data"]["required_ulog_topics"]
        for req_ulog_topic in req_ulog_topics_dict.keys():
            topic_dict = req_ulog_topics_dict[req_ulog_topic]
            if "dataframe_name" in topic_dict.keys():
                req_dataframe_topic_list.extend(topic_dict["dataframe_name"])
            else:
                req_dataframe_topic_list.extend(topic_dict["ulog_name"])
        # drop duplicates
        seen = set()
        result = []
        for topic in req_dataframe_topic_list:
            if topic not in seen:
                seen.add(topic)
                result.append(topic)
        req_dataframe_topic_list = result
        self.dynamics_model_config["data"]["req_dataframe_topic_list"] = req_dataframe_topic_list
        return

    def check_dynamics_model_config(self, config_dict):
        assert ("dynamics_model_config" in config_dict), \
            "No entry for dynamics_model_config detected in config yaml."
        dynamics_model_config = config_dict["dynamics_model_config"]

        assert ("resample_freq" in dynamics_model_config), \
            "No entry for resample frequency detected in config yaml."
        assert ("data" in dynamics_model_config), \
            "No entry for data detected in config yaml."

        data_dict = dynamics_model_config["data"]

        assert ("required_ulog_topics" in data_dict), \
            "No entry for required_ulog_topics in config yaml."
        assert (type(data_dict["required_ulog_topics"]) == dict), \
            "required_ulog_topics does not contain a dict of topic types"
        for topic_type in data_dict["required_ulog_topics"]:
            topic_type_dict = data_dict["required_ulog_topics"][topic_type]
            assert("ulog_name" in topic_type_dict), \
                print(topic_type, " does not contain an entry for ulog_name")
        return

    def check_estimation_bools(self):
        # check that at least estimating forces or moments is enabled.
        assert (self.dynamics_model_config["estimate_forces"] or self.dynamics_model_config["estimate_moments"]), \
            "Neither estimation of forces or moments is activated in config file."
