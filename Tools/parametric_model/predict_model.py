__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import sys
import inspect
from src.models import MultiRotorModel, SimpleFixedWingModel, QuadPlaneModel
from src.models.model_config import ModelConfig
import src.models as models
from src.tools import DataHandler
import argparse
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        os.environ["data_selection"] = "True"
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        os.environ["data_selection"] = "False"
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def start_model_prediction(config, model_results, log_path, data_selection=False):
    data_selection_enabled = data_selection
    print("Visual Data selection enabled: ", data_selection_enabled)

    try:
        with open(model_results) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            model_results_dict = yaml.load(file, Loader=yaml.FullLoader)
            assert type(model_results_dict) is dict
            opt_coefs_dict = model_results_dict["coefficients"]

    except:
        print("Could not load yaml model results file. Does the specified file exist?")
        print(model_results)
        exit(1)

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
        error_str = (
            "Model '{0}' not found, is it added to models "
            "directory and models/__init__.py?".format(model_class)
        )
        raise AttributeError(error_str)

    model.load_dataframes(data_df)
    model.predict_model(opt_coefs_dict)
    model.compute_residuals()
    model.plot_model_predicitons()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate dynamics model from flight log."
    )
    parser.add_argument(
        "log_path",
        metavar="log_path",
        type=str,
        help="The path of the log to process relative to the project directory.",
    )
    parser.add_argument(
        "--data_selection",
        metavar="data_selection",
        type=str2bool,
        default=False,
        help="the path of the log to process relative to the project directory.",
    )
    parser.add_argument(
        "--config",
        metavar="config",
        type=str,
        default="configs/quadrotor_model.yaml",
        help="Configuration file path for pipeline configurations",
    )
    parser.add_argument(
        "--model_results",
        metavar="model_results",
        type=str,
        help="Model results file path for optimal parameters",
    )
    arg_list = parser.parse_args()
    start_model_prediction(**vars(arg_list))
