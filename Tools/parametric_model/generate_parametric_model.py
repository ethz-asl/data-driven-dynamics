__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import sys
import inspect
from src.models import MultiRotorModel, StandardPlaneModel, QuadPlaneModel
import src.models as models
from src.tools import DataHandler
from visual_dataframe_selector.data_selector import select_visual_data
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        os.environ['data_selection'] = "True"
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        os.environ['data_selection'] = "False"
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def start_model_estimation(config, log_path, data_selection=False, plot=True):
    data_selection_enabled = data_selection
    print("Visual Data selection enabled: ", data_selection_enabled)

    data_handler = DataHandler(config)
    data_handler.loadLogs(log_path)

    if False:
        data_handler.visually_select_data()
    #data_handler.visualize_data()

    data_df = data_handler.get_dataframes()

    model_class = data_handler.config.model_class
    try:
        # This will call the model constructor directly from the model_class
        # in the yaml config (self-describing)
        # i.e if model_name is MultiRotorModel then it will call that __init__()
        model = getattr(models, model_class)(config)
    except AttributeError:
        error_str = "Model '{0}' not found, is it added to models "\
                    "directory and models/__init__.py?".format(model_class)
        raise AttributeError(error_str)

    visual_dataframe_selector_config_dict = {
        "x_axis_col": "timestamp",
        "sub_plt1_data": ["q0", "q1", "q2", "q3"],
        "sub_plt2_data": ["u0", "u1", "u2", "u3"],
        "sub_plt3_data": []}

    if data_handler.estimate_forces == True:
        visual_dataframe_selector_config_dict["sub_plt3_data"].append("fisher_information_force")

    if data_handler.estimate_moments == True:
        visual_dataframe_selector_config_dict["sub_plt3_data"].append("fisher_information_rot")

    model.load_dataframes(data_df)
    model.prepare_regression_matrices()
    model.compute_fisher_information()
    if data_selection_enabled:
        model.data_df = select_visual_data(model.data_df,visual_dataframe_selector_config_dict)
        model.clear_features()
        model.prepare_regression_matrices()
        model.compute_fisher_information()
    model.estimate_model()
    if plot:
        model.compute_residuals()
        model.plot_model_predicitons()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='The path of the log to process relative to the project directory.')
    parser.add_argument('--data_selection', metavar='data_selection', type=str2bool, default=False,
                        help='the path of the log to process relative to the project directory.')
    parser.add_argument('--config', metavar='config', type=str, default='configs/quadrotor_model.yaml',
                        help='Configuration file path for pipeline configurations')
    parser.add_argument('--plot', metavar='plot', type=str2bool, default='True',
                        help='Show plots after fit.')
    arg_list = parser.parse_args()
    start_model_estimation(**vars(arg_list))
