__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import os
import sys
import inspect
from src.models import QuadRotorModel, QuadPlaneModel, DeltaQuadPlaneModel, TiltWingModel
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
    rel_ulog_path = arg_list.log_path
    model = arg_list.model
    data_selection_enabled = arg_list.data_selection
    print("Visual Data selection enabled: ", data_selection_enabled)

    if (model == "quadrotor_model"):
        model = QuadRotorModel(rel_ulog_path)

    elif (model == "quad_plane_model"):
        model = QuadPlaneModel(rel_ulog_path)

    elif (model == "delta_quad_plane_model"):
        model = DeltaQuadPlaneModel(rel_ulog_path)

    elif (model == "tilt_wing_model"):
        model = TiltWingModel(rel_ulog_path)

    else:
        print("no valid model selected")

    if data_selection_enabled:
        model.visually_select_data()

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
    arg_list = parser.parse_args()
    start_model_estimation(arg_list)
