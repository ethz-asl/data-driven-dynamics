__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np

"""The functions in this file can be used to plot data specific to the quad plane model"""


def plot_accel_predeictions_with_flap_outputs(stacked_acc_vec, stacked_acc_vec_pred, u_mat, timestamp_array):
    """
    Input:
    acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    u_mat: numpy array of shape (n,3) containing the actuator outputs 5, 6, and 7 corresponding to roll and collective pitch
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    u_mat = np.array(u_mat)
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(timestamp_array, acc_mat[:, 0], label='measurement')
    ax1.plot(timestamp_array, acc_mat_pred[:, 0], label='prediction')
    ax2.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax3.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')
    ax4.plot(timestamp_array, u_mat[:, 0], label='u5')
    ax4.plot(timestamp_array, u_mat[:, 1], label='u6')
    ax4.plot(timestamp_array, u_mat[:, 2], label='u7')

    ax1.set_title('acceleration in x direction of body frame [m/s^2]')
    ax2.set_title('acceleration in y direction of body frame [m/s^2]')
    ax3.set_title('acceleration in z direction of body frame [m/s^2]')
    plt.legend()
    plt.show()
    return
