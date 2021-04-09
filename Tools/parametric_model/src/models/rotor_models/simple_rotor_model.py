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
F_thrust_tot = - mot_const * \
    (angular_vel_1^2 + angular_vel_2^2 + angular_vel_3^2 + angular_vel_4^2)

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
from ..dynamics_model import DynamicsModel


class SimpleRotorModel(DynamicsModel):
    def __init__(self, rel_ulog_path):
        req_topic_dict = {
            "actuator_outputs": {"ulog_name": ["timestamp", "output[0]", "output[1]", "output[2]", "output[3]"]},
            "vehicle_local_position": {"ulog_name": ["timestamp", "az"]}
        }
        super(SimpleRotorModel, self).__init__(rel_ulog_path, req_topic_dict)

    def predict_rotor_forces(self, actuator_outputs):
        accel_vec = self.params["c_quadratic"]*actuator_outputs ^ 2 + \
            self.params["c_linear"]*actuator_outputs + self.params["c_offset"]
        return accel_vec

    def plot_model_prediction(self):
        # plot model prediction
        u = np.linspace(0.0, 1, num=101, endpoint=True)
        u_coll_pred = 4*u
        u_squared_coll_pred = 4 * u**2
        y_pred = np.zeros(u.size)
        for i in range(u.size):
            y_pred[i] = self.params["c_quadratic"]*u_squared_coll_pred[i] + \
                self.params["c_linear"]*u_coll_pred[i] + \
                self.params["c_offset"]
        plt.plot(u_coll_pred, y_pred, label='prediction')
        # plot underlying data
        y_data = self.data_df["az"].to_numpy()
        u_coll_data, u_squared_coll_data = self.compute_collective_input_features()
        plt.plot(u_coll_data, y_data, 'o', label='data')
        plt.ylabel('acceleration in z direction [m/s^2]')
        plt.xlabel('collective input (between [0, 1] per input)')
        plt.legend()
        plt.show()

    def compute_collective_input_features(self):
        # u : normalized actuator output scaled between 0 and 1
        u_collective = np.ones(self.data_df.shape[0])
        u_squared_collective = np.ones(self.data_df.shape[0])
        for r in range(self.data_df.shape[0]):
            u0 = self.data_df["output[0]"].iloc[r]/1000.0 - 1
            u1 = self.data_df["output[1]"].iloc[r]/1000.0 - 1
            u2 = self.data_df["output[2]"].iloc[r]/1000.0 - 1
            u3 = self.data_df["output[3]"].iloc[r]/1000.0 - 1
            u_collective[r] = u0 + u1 + u2 + u3
            u_squared_collective[r] = u0**2 + u1**2 + u2**2 + u3**2
        return u_collective, u_squared_collective

    def prepare_regression_mat(self):
        y = self.data_df["az"].to_numpy()
        X = np.ones((self.data_df.shape[0], 2))
        # u : normalized actuator output scaled between 0 and 1
        u_coll, u_squared_coll = self.compute_collective_input_features()
        X[:, 0] = u_squared_coll
        X[:, 1] = u_coll
        print("datapoints for regression: ", self.data_df.shape[0])
        return X, y

    def estimate_model(self):
        print("estimating simple multirotor model...")
        X, y = self.prepare_regression_mat()
        reg = LinearRegression().fit(X, y)
        print("regression complete")
        print("R2 score: ", reg.score(X, y))
        self.params = {
            "c_quadratic": reg.coef_[0],
            "c_linear": reg.coef_[1],
            "c_offset": reg.intercept_}
        print(self.params)

        with open('rotor_model_params.yml', 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='the path of the log to process relative to the project directory.')
    args = parser.parse_args()
    rel_ulog_path = args.log_path
    # estimate simple multirotor drag model
    rotorModel = RotorModel(rel_ulog_path)
    rotorModel.estimate_model()
    rotorModel.plot_model_prediction()
