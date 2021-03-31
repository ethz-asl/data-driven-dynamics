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


def estimate_model(rel_ulog_path):
    print("estimating quad plane model...")
    print("loading ulog: ", rel_ulog_path)
    topic_dict = {
        "actuator_outputs": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]"],
        "vehicle_local_position": ["timestamp", "ax", "ay", "az"]
    }

    model = DynamicsModel(rel_ulog_path, topic_dict)
    data_df = model.compute_resampled_dataframe(10.0)
    des_freq = 10  # in Hz
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    args = parser.parse_args()
    rel_ulog_path = args.log_path
    # estimate simple multirotor drag model
    estimate_model(rel_ulog_path)
