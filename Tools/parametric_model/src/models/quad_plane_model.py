__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""state vector x = [pos_wb, quat_wb, vel_wb, angular_vel_b]"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import yaml
import argparse

from .dynamics_model import DynamicsModel


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]"],
            "vehicle_local_position": ["timestamp", "ax", "ay", "az"]
        }
        super(QuadPlaneModel, self).__init__(rel_ulog_path, req_topic_dict)

    def estimate_model(self, des_freq=10.0):
        print("estimating quad plane model...")
        self.data_df = self.compute_resampled_dataframe(des_freq)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    args = parser.parse_args()
    rel_ulog_path = args.log_path
    # estimate simple multirotor drag model
    quadPlaneModel = QuadPlaneModel(rel_ulog_path)
    quadPlaneModel.estimate_model()
