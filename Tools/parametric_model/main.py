
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


from src.models import simple_multirotor
import sys


def main(argv):
    print('Argument List:', argv)
    if argv:
        # e.g. logs/2021-03-16/21_45_40.ulg
        rel_ulog_path = argv[0]
        # estimate simple multirotor drag model
        simple_multirotor.estimate_model(rel_ulog_path)
    else:
        print("Missing Argument: No path to ulog file provided.")
    return


if __name__ == "__main__":
    main(sys.argv[1:])
