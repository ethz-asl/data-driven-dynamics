__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


def estimate_model(rel_ulog_path):
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
