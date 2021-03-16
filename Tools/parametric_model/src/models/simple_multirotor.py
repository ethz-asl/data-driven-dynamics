__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import matplotlib

from sklearn.linear_model import LinearRegression
from src.tools import load_ulog, pandas_from_topic, FlightTimeSelector, resample_dataframes


def prepare_regression_matrices(data_df):
    y = data_df["z"].to_numpy()
    y = y + 9.81
    X = np.ones((data_df.shape[0], 2))
    for r in range(data_df.shape[0]):
        u0 = data_df["output[0]"].iloc[r]/1000.0 - 1
        u1 = data_df["output[1]"].iloc[r]/1000.0 - 1
        u2 = data_df["output[2]"].iloc[r]/1000.0 - 1
        u3 = data_df["output[3]"].iloc[r]/1000.0 - 1
        X[r, 0] = u0**2 + u1**2 + u2**2 + u3**2
        X[r, 1] = u0 + u1 + u2 + u3
    print(data_df.shape[0])
    # print(X)
    # print(y)
    return X, y


def prepare_data(ulog):
    # setup object to crop dataframes for flight data
    fts = FlightTimeSelector(ulog)

    # getting data
    actuator_df = pandas_from_topic(ulog, ["actuator_outputs"])
    actuator_df = actuator_df[["timestamp", "output[0]",
                               "output[1]", "output[2]", "output[3]"]]
    accel_df = pandas_from_topic(ulog, ["sensor_accel"])
    accel_df = accel_df[["timestamp", "z"]]
    print(accel_df.shape[0])

    df_list = [actuator_df, accel_df]
    resampled_df = resample_dataframes(df_list, fts.t_start, fts.t_end, 1.0)
    return resampled_df


def estimate_model(rel_ulog_path):
    print("estimating simple multirotor model...")
    print("loading ulog: ", rel_ulog_path)
    ulog = load_ulog(rel_ulog_path)

    data_df = prepare_data(ulog)
    X, y = prepare_regression_matrices(data_df)

    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    print(reg.coef_)
    print(reg.intercept_)

    return


if __name__ == "__main__":
    main()
