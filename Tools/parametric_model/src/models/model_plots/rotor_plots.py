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

    ax.plot(rotor.actuator_input_vec, force_proj_data, 'o',
            label="underlying data", color='grey', alpha=0.25)
    ax.plot(u_vec, thrust_vec, label="prediction")

    ax.set_title("Tail Rotor Force over actuator input")
    plt.legend()


def plot_rotor_trust_3d(rotor_coef_dict, rotor):
    fig, ax = plt.subplots(1)
    u_vec = np.arange(0, 1, .01)
    v_air_par_vec = np.arange(0, 10, .1)
    u_vec, v_air_par_vec = np.meshgrid(u_vec, v_air_par_vec)
    f_thrust_mat = rotor.air_density * rotor.prop_diameter**4 * \
        (u_vec**2 * rotor_coef_dict["rot_thrust_quad"] + u_vec *
         v_air_par_vec * rotor_coef_dict["rot_thrust_lin"] / rotor.prop_diameter)
    # for i in range(u_vec.shape[0]):
    #     f_trust_zero_airspeed = rotor.air_density * rotor.prop_diameter**4 * \
    #         u_vec[i]**2 * rotor_coef_dict["rot_thrust_quad"]
    #     f_thrust_mat[i, :] = np.ones(
    #         (1, v_air_par_vec.shape[0])) * f_trust_zero_airspeed - v_air_par_vec * rotor.air_density * rotor.prop_diameter**3 * \
    #         u_vec[i]**2 * rotor_coef_dict["rot_thrust_lin"]

    print(f_thrust_mat.shape)
    ax = plt.axes(projection='3d')
    ax.plot_surface(u_vec, v_air_par_vec, f_thrust_mat, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title("Rotor Thrust Force")
    ax.set_xlabel("Normalized Rotor Input")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_zlabel("Rotor Thrust [N]")
