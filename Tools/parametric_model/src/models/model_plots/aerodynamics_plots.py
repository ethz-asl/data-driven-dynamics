__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np
import math
from src.tools.math_tools import cropped_sym_sigmoid


def plot_lift_curve(c_l_dict, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_rad = aoa_deg * math.pi/180
    c_l_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15*math.pi/180), scale_fac=30)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec[i] = flow_attached_region * c_l_dict["c_l_0"] + flow_attached_region * \
            c_l_dict["c_l_lin"] * aoa_rad[i] + \
            stall_region * math.sin(2*aoa_rad[i]) * c_l_dict["c_l_stall"]

    plt.plot(aoa_deg, c_l_vec)
    plt.title("Lift coefficient over angle of attack [deg]")


def plot_lift_curve2(c_l_dict1, c_l_dict2, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_rad = aoa_deg * math.pi/180
    c_l_vec1 = np.zeros(aoa_deg.shape[0])
    c_l_vec2 = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15*math.pi/180), scale_fac=30)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec1[i] = flow_attached_region * c_l_dict1["c_l_offset"] + flow_attached_region * \
            c_l_dict1["c_l_lin"] * aoa_rad[i] + \
            stall_region * math.sin(2*aoa_rad[i]) * c_l_dict1["c_l_stall"]
        c_l_vec2[i] = flow_attached_region * c_l_dict2["c_l_offset"] + flow_attached_region * \
            c_l_dict2["c_l_lin"] * aoa_rad[i] + \
            stall_region * math.sin(2*aoa_rad[i]) * c_l_dict2["c_l_stall"]

    plt.plot(aoa_deg, c_l_vec1, label="prediction")
    plt.plot(aoa_deg, c_l_vec2, label="expectation")
    plt.title("Lift coefficient over angle of attack [deg]")
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Lift Coefficient')


def plot_aoa_hist(aoa_vec):
    aoa_vec = aoa_vec*180/math.pi
    plt.hist(aoa_vec, bins=100)
    plt.title("Data distribution over angle of attack")
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('n samples')


def plot_drag_curve(c_d_dict, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_rad = aoa_deg * math.pi/180
    c_d_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15*math.pi/180), scale_fac=30)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_d_vec[i] = flow_attached_region * c_d_dict["c_d_0"] + flow_attached_region * \
            c_d_dict["c_d_lin"] * aoa_rad[i] + flow_attached_region * \
            c_d_dict["c_d_quad"] * aoa_rad[i] ^ 2 + \
            stall_region * (1 - math.sin(aoa_rad[i])**2) * c_d_dict["c_d_stall_min"] + \
            stall_region * math.sin(aoa_rad[i])**2 * c_d_dict["c_d_stall_max"]

    plt.plot(aoa_deg, c_d_vec)
    plt.title("Drag coefficient over angle of attack [deg]")


def plot_lift_prediction_and_underlying_data(c_l_pred_dict, c_l_data, aoa_data,  plot_range_deg=[-30, 50]):
    fig, ax = plt.subplots()
    aoa_data_deg = aoa_data*180/math.pi
    aoa_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_rad = aoa_deg * math.pi/180
    c_l_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15*math.pi/180), scale_fac=30)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec[i] = flow_attached_region * c_l_pred_dict["c_l_offset"] + flow_attached_region * \
            c_l_pred_dict["c_l_lin"] * aoa_rad[i] + \
            stall_region * math.sin(2*aoa_rad[i]) * c_l_pred_dict["c_l_stall"]

    ax.plot(aoa_data_deg, c_l_data, 'o',
            label="underlying data", color='grey', alpha=0.25)
    ax.plot(aoa_deg, c_l_vec, label="prediction")

    ax.set_title("Lift coefficient over angle of attack [deg]")
    plt.legend()


def plot_drag_prediction_and_underlying_data(c_d_pred_dict, c_d_data, aoa_data,  plot_range_deg=[-90, 90]):
    fig, ax = plt.subplots()
    aoa_data_deg = aoa_data*180/math.pi
    aoa_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_rad = aoa_deg * math.pi/180
    c_d_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15*math.pi/180), scale_fac=30)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_d_vec[i] = flow_attached_region * c_d_pred_dict["c_d_offset"] + flow_attached_region * \
            c_d_pred_dict["c_d_lin"] * aoa_rad[i] + flow_attached_region * \
            c_d_pred_dict["c_d_quad"] * aoa_rad[i]**2 + \
            stall_region * \
            (1 - math.sin(aoa_rad[i])**2) * c_d_pred_dict["c_d_stall_min"] + stall_region * \
            math.sin(aoa_rad[i])**2 * c_d_pred_dict["c_d_stall_max"]

    ax.plot(aoa_data_deg, c_d_data, 'o',
            label="underlying data", color='grey', alpha=0.25)
    ax.plot(aoa_deg, c_d_vec, label="prediction")

    ax.set_title("Drag coefficient over angle of attack [deg]")
    plt.legend()


def plot_example_plate_model(plot_range_deg=[-100, 100]):
    aoa_range_deg = np.linspace(
        plot_range_deg[0], plot_range_deg[1], num=(plot_range_deg[1] - plot_range_deg[0] + 1))
    aoa_range_rad = aoa_range_deg*math.pi/180
    c_l = np.zeros(len(aoa_range_rad))
    c_d = np.zeros(len(aoa_range_rad))
    for i in range(len(aoa_range_rad)):
        c_l[i] = 0.6*math.sin(2*aoa_range_rad[i])
        c_d[i] = 1.3*math.sin(aoa_range_rad[i])**2 + 0.1 * \
            (1-math.sin(aoa_range_rad[i])**2)

    fig, ax = plt.subplots()
    ax.plot(aoa_range_deg, c_d, label="drag coefficient")
    ax.plot(aoa_range_deg, c_l, label="lift coefficient")
    ax.set_xlabel('Angle of Attack [deg]')
    # ax.set_ylabel('c_l / c_d')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_example_plate_model()
