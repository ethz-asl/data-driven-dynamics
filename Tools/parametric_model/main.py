
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


from src.models import RotorModel, QuadPlaneModel
import argparse
import sys


def start_model_estimation(arg_list):
    rel_ulog_path = arg_list.log_path
    model = arg_list.model

    if (model == "rotor_model"):
        rotorModel = RotorModel(rel_ulog_path)
        rotorModel.estimate_model()
        rotorModel.plot_model_prediction()

    elif (model == "quad_plane_model"):
        quadPlaneModel = QuadPlaneModel(rel_ulog_path)
        quadPlaneModel.estimate_model(100.0)

    else:
        print("no valid model selected")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('model', metavar='model', type=str,
                        help='select an implemented model to estimate.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    arg_list = parser.parse_args()
    start_model_estimation(arg_list)
