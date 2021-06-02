__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np
import math

"""The functions in this file can be used to plot data of any kind of model"""


def plot_accel_predeictions_and_tilt(stacked_acc_vec, stacked_acc_vec_pred, timestamp_array, u_vec):
    """
    Input:
    stacked_acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    stacked_acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """
    print(u_vec)

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    timestamp_array = np.array(timestamp_array)/1000000

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('Predictions of linear accelerations')
    ax1.plot(timestamp_array, u_vec, label='tilt angle')
    ax2.plot(timestamp_array, acc_mat[:, 0], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 0], label='prediction')
    ax3.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax4.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax4.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')

    # ax1.set_title('acceleration in x direction of body frame [m/s^2]')
    # ax2.set_title('acceleration in y direction of body frame [m/s^2]')
    # ax3.set_title('acceleration in z direction of body frame [m/s^2]')

    ax1.set_ylabel('tilt angle [deg]')
    ax2.set_ylabel('x [m/s^2]')
    ax3.set_ylabel('y [m/s^2]')
    ax4.set_ylabel('z [m/s^2]')
    ax4.set_xlabel('time [s]')
    plt.legend()
    return


def plot_accel_predeictions(stacked_acc_vec, stacked_acc_vec_pred, timestamp_array):
    """
    Input:
    stacked_acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    stacked_acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    timestamp_array = np.array(timestamp_array)/1000000

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Predictions of linear accelerations')
    ax1.plot(timestamp_array, acc_mat[:, 0], label='measurement')
    ax1.plot(timestamp_array, acc_mat_pred[:, 0], label='prediction')
    ax2.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax3.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')

    # ax1.set_title('acceleration in x direction of body frame [m/s^2]')
    # ax2.set_title('acceleration in y direction of body frame [m/s^2]')
    # ax3.set_title('acceleration in z direction of body frame [m/s^2]')

    ax1.set_ylabel('x [m/s^2]')
    ax2.set_ylabel('y [m/s^2]')
    ax3.set_ylabel('z [m/s^2]')
    ax3.set_xlabel('time [s]')
    plt.legend()
    return


def plot_angular_accel_predeictions(stacked_acc_vec, stacked_acc_vec_pred, timestamp_array):
    """
    Input:
    stacked_acc_vec: numpy array of shape (3*n,1) containing stacked angular accelerations [w_x_1, w_y_1, w_z_1, w_x_2, ...]^T in body frame
    stacked_acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted angular accelerations [w_x_1, w_y_1, w_z_1, w_x_2, ...]^T in body frame
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Prediction of angular accelerations')
    ax1.plot(timestamp_array, acc_mat[:, 0], label='measurement')
    ax1.plot(timestamp_array, acc_mat_pred[:, 0], label='prediction')
    ax2.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax3.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')

    ax1.set_title(
        'angular acceleration in x direction of body frame [rad/s^2]')
    ax2.set_title(
        'angular acceleration in y direction of body frame [rad/s^2]')
    ax3.set_title(
        'angular acceleration in z direction of body frame [rad/s^2]')
    plt.legend()
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
    fig.suptitle('Airspeed and Angle of Attack')
    ax1.plot(timestamp_array, airspeed_mat[:, 0], label='measurement')
    ax2.plot(timestamp_array, airspeed_mat[:, 1], label='measurement')
    ax3.plot(timestamp_array, airspeed_mat[:, 2], label='measurement')
    ax4.plot(timestamp_array, airspeed_mat[:, 3], label='measurement')
    ax1.set_title('airspeed in x direction of body frame [m/s^2]')
    ax2.set_title('airspeed in y direction of body frame [m/s^2]')
    ax3.set_title('airspeed in z direction of body frame [m/s^2]')
    ax4.set_title("Aoa in body frame [radiants]")
    plt.legend()
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
    x_drag_y = np.zeros(v_a_y.shape[0])
    for i in range(x_drag_y.shape[0]):
        x_drag_y[i] = -math.copysign(1, v_a_y[i]) * v_a_y[i]**2
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Acceleration and Airspeed in y direction')
    ax1.plot(timestamp_array, v_a_y, label='measurement')
    ax2.plot(timestamp_array, x_drag_y, label='measurement')
    ax3.plot(timestamp_array, acc_mat[:, 1], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 1], label='prediction')
    ax1.set_title('airspeed in y direction of body frame [m/s^2]')
    ax2.set_title(
        'features corresponding to drag in y direction')
    ax3.set_title('acceleration y direction of body frame [m/s^2]')
    plt.legend()
    return


def plot_accel_and_airspeed_in_z_direction(stacked_acc_vec, stacked_acc_vec_pred, v_a_z, timestamp_array):
    """
    Input:
    acc_vec: numpy array of shape (3*n,1) containing stacked accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    acc_vec_pred: numpy array of shape (3*n,1) containing stacked predicted accelerations [a_x_1, a_y_1, a_z_1, a_x_2, ...]^T in body frame
    v_a_z: numpy array of shape (n,1) containing the airspeed in y direction
    timestamp_array: numpy array with n entries of corresponding timestamps.
    """

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    v_a_z = np.array(v_a_z)
    x_drag_z = np.zeros(v_a_z.shape[0])
    for i in range(x_drag_z.shape[0]):
        x_drag_z[i] = -math.copysign(1, v_a_z[i]) * v_a_z[i]**2
    timestamp_array = np.array(timestamp_array)

    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Acceleration and Airspeed in z direction')
    ax1.plot(timestamp_array, v_a_z, label='measurement')
    ax2.plot(timestamp_array, x_drag_z, label='measurement')
    ax3.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax3.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')
    ax1.set_title('airspeed in z direction of body frame [m/s^2]')
    ax2.set_title(
        '- sign(v_a)*v_a^2 in body frame [m/s^2]')
    ax3.set_title('acceleration z direction of body frame [m/s^2]')
    plt.legend()
    return


def plot_az_and_collective_input(stacked_acc_vec, stacked_acc_vec_pred, u_mat, timestamp_array):

    u_mat = np.array(u_mat)
    u_collective = np.zeros(u_mat.shape[0])
    for i in range(u_mat.shape[1]):
        u_collective = u_collective + u_mat[:, i]

    stacked_acc_vec = np.array(stacked_acc_vec)
    stacked_acc_vec_pred = np.array(stacked_acc_vec_pred)
    acc_mat = stacked_acc_vec.reshape((-1, 3))
    acc_mat_pred = stacked_acc_vec_pred.reshape((-1, 3))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Acceleration and Collective Input in z direction')
    ax1.plot(timestamp_array, u_collective, label='measurement')
    ax2.plot(timestamp_array, acc_mat[:, 2], label='measurement')
    ax2.plot(timestamp_array, acc_mat_pred[:, 2], label='prediction')
    ax1.set_title('collective input')
    ax2.set_title(
        'acceleration in z direction of body frame [m/s^2]')
    plt.legend()


def plot(data, timestamp, plt_title="No title"):
    plt.plot(timestamp, data)
    plt.title(plt_title)
