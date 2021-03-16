
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


# import libraries


# import models
from src.models import simple_multirotor


def main():

    rel_ulog_path = "logs/2021-03-16/21_45_40.ulg"

    # estimate simple multirotor drag model
    simple_multirotor.estimate_model(rel_ulog_path)

    return


if __name__ == "__main__":
    main()
