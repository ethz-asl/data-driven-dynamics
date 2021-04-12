__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np

"""The functions in this file can be used to plot data of any kind of model"""


def plot_accel_predeictions(stacked_acc_vec, stacked_acc_vec_pred, timestamp_array):
    """
    Input:
    acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """
    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(timestamp_array, acc_mat[:, 0], label='measurement')
    ax1.plot(timestamp_array, acc_mat_pred[:, 0], label='prediction')
    ax2.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax3.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')

    ax1.set_title('acceleration in x direction of body frame [m/s^2]')
    ax2.set_title('acceleration in y direction of body frame [m/s^2]')
    ax3.set_title('acceleration in z direction of body frame [m/s^2]')
    plt.legend()
    plt.show()
    return


def plot_airspeed_and_AoA(airspeed_mat, timestamp_array):
    """
    Input:
    airspeed_mat: numpy array Matrix of shape (n,4) containing
    the columns [V_a_x, V_a_y, V_a_z, AoA].
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """
    airspeed_mat = np.array(airspeed_mat)
    timestamp_array = np.array(timestamp_array)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(timestamp_array, airspeed_mat[:, 0], label='measurement')
    ax2.plot(timestamp_array, airspeed_mat[:, 1], label='measurement')
    ax3.plot(timestamp_array, airspeed_mat[:, 2], label='measurement')
    ax4.plot(timestamp_array, airspeed_mat[:, 3], label='measurement')
    ax1.set_title('airspeed in x direction of body frame [m/s^2]')
    ax2.set_title('airspeed in y direction of body frame [m/s^2]')
    ax3.set_title('airspeed in z direction of body frame [m/s^2]')
    ax4.set_title("Aoa in body frame [radiants]")
    plt.legend()
    plt.show()
    return


def plot_accel_and_airspeed_in_y_direction(stacked_acc_vec, stacked_acc_vec_pred, v_a_y, timestamp_array):
    """
    Input:
    acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    v_a_y: numpy array of shape (n,1) containing the airspeed in y direction
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    v_a_y = np.array(v_a_y)
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(timestamp_array, v_a_y, label='measurement')
    ax2.plot(timestamp_array, v_a_y**2, label='measurement')
    ax3.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax1.set_title('airspeed in y direction of body frame [m/s^2]')
    ax2.set_title('airspeed in y direction squared of body frame [m/s^2]')
    ax3.set_title('airspeed in y direction of body frame [m/s^2]')
    plt.legend()
    plt.show()
    return
