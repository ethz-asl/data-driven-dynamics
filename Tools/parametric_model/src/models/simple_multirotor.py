
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import matplotlib

from sklearn.linear_model import LinearRegression

from src.tools import load_ulog, pandas_from_topic, FlightTimeSelector, interpolate_dataframes


def prepare_data(ulog):
    # setup object to crop dataframes for flight data
    fts = FlightTimeSelector(ulog)

    # getting data
    actuator_df = pandas_from_topic(ulog, ["actuator_outputs"])

    accel_df = pandas_from_topic(ulog, ["sensor_accel"])

    interpolate_dataframes([actuator_df, accel_df], fts.t_start, fts.t_end)


def estimate_model(rel_ulog_path):
    print("estimating simple multirotor model...")
    print("loading ulog: ", rel_ulog_path)
    ulog = load_ulog(rel_ulog_path)

    prepare_data(ulog)

    return


if __name__ == "__main__":
    main()
