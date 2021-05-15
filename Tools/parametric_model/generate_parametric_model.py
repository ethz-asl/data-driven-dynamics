
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


from src.models import QuadRotorModel, QuadPlaneModel, DeltaQuadPlaneModel, TiltWingModel
import argparse


def start_model_estimation(arg_list):
    rel_ulog_path = arg_list.log_path
    model = arg_list.model

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

    model.estimate_model()
    model.plot_model_predicitons()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('--model', metavar='model', type=str,
                        default='quadrotor_model',
                        help='Parametric Model Type [quadrotor_model, quad_plane_model]')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    arg_list = parser.parse_args()
    start_model_estimation(arg_list)
