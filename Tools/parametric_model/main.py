
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


from src.models import simple_multirotor
import argparse
import sys


def start_model_estimation(arg_list):
    rel_ulog_path = arg_list.log_path
    simple_multirotor.estimate_model(rel_ulog_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    arg_list = parser.parse_args()
    start_model_estimation(arg_list)
