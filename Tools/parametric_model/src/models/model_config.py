__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import yaml
from pathlib import Path


class ModelConfig():
    def __init__(self, config_file_name):
        """
        Input: config_file as found in Tools/parametric_model/src/models/model_config
        """
        rel_config_file_path = "Tools/parametric_model/src/models/model_config/" + config_file_name
        proj_path = Path(os.getcwd())
        log_file_path = os.path.join(proj_path, rel_config_file_path)

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

        self.check_dynamics_model_config(config_dict)

        self.dynamics_model_config = config_dict["dynamics_model_config"]
        self.model_config = config_dict["model_config"]

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
        print(data_dict)

        assert ("required_ulog_topics" in data_dict), \
            "No entry for required_ulog_topics in config yaml."
        assert (type(data_dict["required_ulog_topics"]) == dict), \
            "required_ulog_topics does not contain a dict of topic types"
        for topic_type in data_dict["required_ulog_topics"]:
            topic_type_dict = data_dict["required_ulog_topics"][topic_type]
            assert("ulog_name" in topic_type_dict), \
                print(topic_type, " does not contain an entry for ulog_name")
        return
