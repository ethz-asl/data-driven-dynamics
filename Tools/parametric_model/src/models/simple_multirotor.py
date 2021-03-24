__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model in this file estimates a simple motor model for the iris quadrocopter of PXç sitl gazebo. 

Start the model identification:
Call "estimate_model(rel_ulog_path)"
with rel_ulog_path specifying the path of the log file relative to the project directory (e.g. "logs/2021-03-16/21_45_40.ulg")

Model Parameters: 
u                    : normalized actuator output scaled between 0 and 1
angular_vel_const    : angular velocity constant
angular_vel_offset   : angular velocity offset
mot_const            : motor constant
m                    : mass of UAV
accel_const          : combined acceleration constant k_2/m 

Model: 
angular_vel [rad/s] = angular_vel_const*u + angular_vel_offset
F_thrust = - mot_const * angular_vel^2
F_thrust_tot = - mot_const * (angular_vel_1^2 + angular_vel_2^2 + angular_vel_3^2 + angular_vel_4^2)

Note that the forces are calculated in the NED body frame and are therefore negative.

The script estimates [k_1, c, b]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import yaml
import argparse

from sklearn.linear_model import LinearRegression
from ..tools import load_ulog, pandas_from_topic, compute_flight_time, resample_dataframes


def plot_model_prediction(coefficients, intercept, data_df):
    # plot model prediction
    u = np.linspace(0.0, 1, num=101, endpoint=True)
    u_coll_pred = 4*u
    u_squared_coll_pred = 4 * u**2
    y_pred = np.zeros(u.size)
    for i in range(u.size):
        y_pred[i] = coefficients[0]*u_squared_coll_pred[i] + \
            coefficients[1]*u_coll_pred[i] + intercept
    plt.plot(u_coll_pred, y_pred, label='prediction')
    # plot underlying data
    y_data = data_df["az"].to_numpy()
    u_coll_data, u_squared_coll_data = compute_collective_input_features(
        data_df)
    plt.plot(u_coll_data, y_data, 'o', label='data')
    plt.ylabel('acceleration in z direction [m/s^2]')
    plt.xlabel('collective input (between [0, 1] per input)')
    plt.legend()
    plt.show()


def compute_collective_input_features(data_df):
    # u : normalized actuator output scaled between 0 and 1
    u_collective = np.ones(data_df.shape[0])
    u_squared_collective = np.ones(data_df.shape[0])
    for r in range(data_df.shape[0]):
        u0 = data_df["output[0]"].iloc[r]/1000.0 - 1
        u1 = data_df["output[1]"].iloc[r]/1000.0 - 1
        u2 = data_df["output[2]"].iloc[r]/1000.0 - 1
        u3 = data_df["output[3]"].iloc[r]/1000.0 - 1
        u_collective[r] = u0 + u1 + u2 + u3
        u_squared_collective[r] = u0**2 + u1**2 + u2**2 + u3**2
    return u_collective, u_squared_collective


def prepare_regression_matrices(data_df):
    y = data_df["az"].to_numpy()
    X = np.ones((data_df.shape[0], 2))
    # u : normalized actuator output scaled between 0 and 1
    u_coll, u_squared_coll = compute_collective_input_features(data_df)
    X[:, 0] = u_squared_coll
    X[:, 1] = u_coll
    print("datapoints for regression: ", data_df.shape[0])
    return X, y


def prepare_data(ulog):
    # setup object to crop dataframes for flight data
    fts = compute_flight_time(ulog)

    # getting data
    actuator_df = pandas_from_topic(ulog, ["actuator_outputs"])
    actuator_df = actuator_df[["timestamp", "output[0]",
                               "output[1]", "output[2]", "output[3]"]]
    accel_df = pandas_from_topic(ulog, ["vehicle_local_position"])
    accel_df = accel_df[["timestamp", "az"]]

    df_list = [actuator_df, accel_df]
    resampled_df = resample_dataframes(
        df_list, fts["t_start"], fts["t_end"], 10.0)
    return resampled_df


def compute_model_params(coefficients, intercept):
    accel_const = float(-coefficients[1]**2/(2*coefficients[0]))
    angular_vel_const = float(2*coefficients[0]/coefficients[1])
    angular_vel_offset = float(
        math.sqrt(-intercept*coefficients[0])/-coefficients[1])
    model_params = {
        "accel_const": accel_const,
        "angular_vel_const": angular_vel_const,
        "angular_vel_offset": angular_vel_offset}
    return model_params


def estimate_model(rel_ulog_path):
    print("estimating simple multirotor model...")
    print("loading ulog: ", rel_ulog_path)
    ulog = load_ulog(rel_ulog_path)

    data_df = prepare_data(ulog)
    X, y = prepare_regression_matrices(data_df)

    reg = LinearRegression().fit(X, y)
    print("regression complete")
    print("R2 score: ", reg.score(X, y))
    print("coefficients: ", reg.coef_)
    print("intercept: ", reg.intercept_)

    model_params = compute_model_params(reg.coef_, reg.intercept_)
    print("model parameters", model_params)
    with open('model_params.yml', 'w') as outfile:
        yaml.dump(model_params, outfile, default_flow_style=False)

    plot_model_prediction(reg.coef_, reg.intercept_, data_df)

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
