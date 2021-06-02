__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np
import math


def plot_thrust_prediction_and_underlying_data(rotor_coef_dict, rotor, force_proj_data):
    fig, ax = plt.subplots()
    u_vec = np.linspace(
        0, 1, num=(101))
    thrust_vec = np.zeros(101)
    for i in range(u_vec.shape[0]):
        thrust_vec[i] = rotor.air_density * rotor.prop_diameter**4 * \
            u_vec[i]**2 * rotor_coef_dict["rot_thrust_quad"]

    force_proj_data_coll = []
    u_data_coll = []
    for i in range(force_proj_data.shape[0]):
        if abs(rotor.v_air_parallel_abs[i] <= 0.5):
            force_proj_data_coll.append(force_proj_data[i])
            u_data_coll.append(rotor.actuator_input_vec[i])

    ax.plot(rotor.actuator_input_vec, force_proj_data, '.',
            label="underlying data", color='grey')
    ax.plot(u_vec, thrust_vec, label="prediction")

    ax.set_title("Tail Rotor Force over actuator input")
    plt.legend()
