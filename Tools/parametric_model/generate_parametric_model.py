__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import sys
import inspect
import src.models as models
from src.tools import DataHandler
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


def start_model_estimation(config, log_path, data_selection=False):
    data_selection_enabled = data_selection
    print("Visual Data selection enabled: ", data_selection_enabled)

    data_handler = DataHandler(config)
    data_handler.loadLogs(log_path)

    if data_selection_enabled:
        data_handler.visually_select_data()
    data_handler.visualize_data()

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

    model.load_dataframes(data_df)
    model.estimate_model()
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
    arg_list = parser.parse_args()
    start_model_estimation(**vars(arg_list))
