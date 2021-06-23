__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import sys
import inspect
from src.models import QuadRotorModel, QuadPlaneModel, DeltaQuadPlaneModel, TiltWingModel
from src.tools import DataHandler
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def start_model_estimation(arg_list):
    model_name = arg_list.model
    data_selection_enabled = arg_list.data_selection
    print("Visual Data selection enabled: ", data_selection_enabled)
    config_file = arg_list.config

    data_handler = DataHandler(config_file)
    data_handler.loadLog(arg_list.log_path)
    if data_selection_enabled:
        data_handler.visually_select_data()

    data_df = data_handler.get_dataframes()

    if (model_name == "quadrotor_model"):
        model = QuadRotorModel(config_file)

    elif (model_name == "quad_plane_model"):
        model = QuadPlaneModel(config_file)

    elif (model_name == "delta_quad_plane_model"):
        model = DeltaQuadPlaneModel(config_file)

    elif (model_name == "tilt_wing_model"):
        model = TiltWingModel(config_file)
    else:
        print("no valid model selected")
    model.load_dataframes(data_df)
    model.estimate_model()
    model.plot_model_predicitons()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('--model', metavar='model', type=str,
                        default='quadrotor_model',
                        help='Parametric Model Type [quadrotor_model, quad_plane_model, delta_quad_plane_model, tilt_wing_model]')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='The path of the log to process relative to the project directory.')
    parser.add_argument('--data_selection', metavar='data_selection', type=str2bool, default=False,
                        help='the path of the log to process relative to the project directory.')
    parser.add_argument('--config', metavar='config', type=str, default='quadrotor',
                        help='Configuration file path for pipeline configurations')
    arg_list = parser.parse_args()
    start_model_estimation(arg_list)
