"""
 *
 * Copyright (c) 2021 Manuel Yves Galliker
 *               2021 Autonomous Systems Lab ETH Zurich
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name Data Driven Dynamics nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
"""

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np
import math
from src.tools.math_tools import cropped_sym_sigmoid


def plot_lift_curve(c_l_dict, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180
    c_l_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15 * math.pi / 180), scale_fac=30
        )
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec[i] = (
            flow_attached_region * c_l_dict["c_l_0"]
            + flow_attached_region * c_l_dict["c_l_lin"] * aoa_rad[i]
            + stall_region * math.sin(2 * aoa_rad[i]) * c_l_dict["c_l_stall"]
        )

    plt.plot(aoa_deg, c_l_vec)
    plt.title("Lift coefficient over angle of attack [deg]")


def plot_lift_curve2(c_l_dict1, c_l_dict2, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180
    c_l_vec1 = np.zeros(aoa_deg.shape[0])
    c_l_vec2 = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15 * math.pi / 180), scale_fac=30
        )
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec1[i] = (
            flow_attached_region * c_l_dict1["c_l_offset"]
            + flow_attached_region * c_l_dict1["c_l_lin"] * aoa_rad[i]
            + stall_region * math.sin(2 * aoa_rad[i]) * c_l_dict1["c_l_stall"]
        )
        c_l_vec2[i] = (
            flow_attached_region * c_l_dict2["c_l_offset"]
            + flow_attached_region * c_l_dict2["c_l_lin"] * aoa_rad[i]
            + stall_region * math.sin(2 * aoa_rad[i]) * c_l_dict2["c_l_stall"]
        )

    plt.plot(aoa_deg, c_l_vec1, label="prediction")
    plt.plot(aoa_deg, c_l_vec2, label="expectation")
    plt.title("Lift coefficient over angle of attack [deg]")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift Coefficient")


def plot_aoa_hist(aoa_vec):
    aoa_vec = aoa_vec * 180 / math.pi
    plt.hist(aoa_vec, bins=100)
    plt.title("Data distribution over angle of attack")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("n samples")


def plot_drag_curve(c_d_dict, plot_range_deg=[-100, 100]):
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180
    c_d_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15 * math.pi / 180), scale_fac=30
        )
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_d_vec[i] = (
            flow_attached_region * c_d_dict["c_d_0"]
            + flow_attached_region * c_d_dict["c_d_lin"] * aoa_rad[i]
            + flow_attached_region * c_d_dict["c_d_quad"] * aoa_rad[i]
            ^ 2
            + stall_region * (1 - math.sin(aoa_rad[i]) ** 2) * c_d_dict["c_d_stall_min"]
            + stall_region * math.sin(aoa_rad[i]) ** 2 * c_d_dict["c_d_stall_max"]
        )

    plt.plot(aoa_deg, c_d_vec)
    plt.title("Drag coefficient over angle of attack [deg]")


def plot_lift_prediction_and_underlying_data(
    c_l_pred_dict, c_l_data, aoa_data, plot_range_deg=[-30, 50]
):
    fig, ax = plt.subplots()
    aoa_data_deg = aoa_data * 180 / math.pi
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180
    c_l_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15 * math.pi / 180), scale_fac=30
        )
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_l_vec[i] = (
            flow_attached_region * c_l_pred_dict["c_l_offset"]
            + flow_attached_region * c_l_pred_dict["c_l_lin"] * aoa_rad[i]
            + stall_region * math.sin(2 * aoa_rad[i]) * c_l_pred_dict["c_l_stall"]
        )

    ax.plot(
        aoa_data_deg, c_l_data, "o", label="underlying data", color="grey", alpha=0.25
    )
    ax.plot(aoa_deg, c_l_vec, label="prediction")

    ax.set_title("Lift coefficient over angle of attack [deg]")
    plt.legend()


def plot_drag_prediction_and_underlying_data(
    c_d_pred_dict, c_d_data, aoa_data, plot_range_deg=[-90, 90]
):
    fig, ax = plt.subplots()
    aoa_data_deg = aoa_data * 180 / math.pi
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180
    c_d_vec = np.zeros(aoa_deg.shape[0])
    for i in range(aoa_deg.shape[0]):
        stall_region = cropped_sym_sigmoid(
            aoa_rad[i], x_offset=(15 * math.pi / 180), scale_fac=30
        )
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region
        c_d_vec[i] = (
            flow_attached_region * c_d_pred_dict["c_d_offset"]
            + flow_attached_region * c_d_pred_dict["c_d_lin"] * aoa_rad[i]
            + flow_attached_region * c_d_pred_dict["c_d_quad"] * aoa_rad[i] ** 2
            + stall_region
            * (1 - math.sin(aoa_rad[i]) ** 2)
            * c_d_pred_dict["c_d_stall_min"]
            + stall_region * math.sin(aoa_rad[i]) ** 2 * c_d_pred_dict["c_d_stall_max"]
        )

    ax.plot(
        aoa_data_deg, c_d_data, "o", label="underlying data", color="grey", alpha=0.25
    )
    ax.plot(aoa_deg, c_d_vec, label="prediction")

    ax.set_title("Drag coefficient over angle of attack [deg]")
    plt.legend()


def plot_example_plate_model(plot_range_deg=[-100, 100]):
    aoa_range_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_range_rad = aoa_range_deg * math.pi / 180
    c_l = np.zeros(len(aoa_range_rad))
    c_d = np.zeros(len(aoa_range_rad))
    for i in range(len(aoa_range_rad)):
        c_l[i] = 0.6 * math.sin(2 * aoa_range_rad[i])
        c_d[i] = 1.3 * math.sin(aoa_range_rad[i]) ** 2 + 0.1 * (
            1 - math.sin(aoa_range_rad[i]) ** 2
        )

    fig, ax = plt.subplots()
    ax.plot(aoa_range_deg, c_d, label="drag coefficient")
    ax.plot(aoa_range_deg, c_l, label="lift coefficient")
    ax.set_xlabel("Angle of Attack [deg]")
    # ax.set_ylabel('c_l / c_d')
    plt.legend()
    plt.grid()
    plt.show()


def plot_liftdrag_curve(data_df, coef_dict, aerodynamics_dict, metric):
    plot_range_deg = [-180, 180]
    aoa_deg = np.linspace(
        plot_range_deg[0],
        plot_range_deg[1],
        num=(plot_range_deg[1] - plot_range_deg[0] + 1),
    )
    aoa_rad = aoa_deg * math.pi / 180

    fig, (ax1, ax2) = plt.subplots(2)

    c_l_vec = np.zeros(aoa_deg.shape[0])
    c_d_vec = np.zeros(aoa_deg.shape[0])

    if aerodynamics_dict["type"] == "LinearWingModel":
        cl_0 = coef_dict["cl0"]
        cl_alpha = coef_dict["clalpha"]
        cd_0 = coef_dict["cd0"]
        cd_alpha = coef_dict["cdalpha"]
        cd_alpha2 = coef_dict["cdalphasq"]

        for i in range(aoa_deg.shape[0]):
            c_l_vec[i] = cl_0 + cl_alpha * aoa_rad[i]
            c_d_vec[i] = (
                cd_0 + cd_alpha * aoa_rad[i] + cd_alpha2 * aoa_rad[i] * aoa_rad[i]
            )

        ax1.plot(aoa_deg, c_l_vec, label="prediction")
        ax2.plot(aoa_deg, c_d_vec, label="prediction")

        # Visualize uncertainty
        c_l_minimum = np.zeros(aoa_deg.shape[0])
        c_l_maximum = np.zeros(aoa_deg.shape[0])
        c_d_minimum = np.zeros(aoa_deg.shape[0])
        c_d_maximum = np.zeros(aoa_deg.shape[0])

        for i in range(aoa_deg.shape[0]):
            c_l_minimum[i] = (
                cl_0
                - metric["Cramer"]["cl0"]
                + (cl_alpha - metric["Cramer"]["clalpha"]) * aoa_rad[i]
            )
            c_l_maximum[i] = (
                cl_0
                + metric["Cramer"]["cl0"]
                + (cl_alpha + metric["Cramer"]["clalpha"]) * aoa_rad[i]
            )
            c_d_minimum[i] = (
                cd_0
                - metric["Cramer"]["cd0"]
                + (cd_alpha - metric["Cramer"]["cdalpha"]) * aoa_rad[i]
                + (cd_alpha2 - metric["Cramer"]["cdalphasq"]) * aoa_rad[i] * aoa_rad[i]
            )
            c_d_maximum[i] = (
                cd_0
                + metric["Cramer"]["cd0"]
                + (cd_alpha + metric["Cramer"]["cdalpha"]) * aoa_rad[i]
                + (cd_alpha2 + metric["Cramer"]["cdalphasq"]) * aoa_rad[i] * aoa_rad[i]
            )
        ax1.fill_between(aoa_deg, c_l_minimum, c_l_maximum, alpha=0.3)
        ax2.fill_between(aoa_deg, c_d_minimum, c_d_maximum, alpha=0.3)

        aoa_measured = data_df["angle_of_attack"].to_numpy()
        c_l_measured = np.zeros(aoa_measured.shape)
        c_d_measured = np.zeros(aoa_measured.shape)

        for j in range(aoa_measured.shape[0]):
            c_l_measured[j] = cl_0 + cl_alpha * aoa_measured[j]
            c_d_measured[j] = (
                cd_0
                + cd_alpha * aoa_measured[j]
                + cd_alpha2 * aoa_measured[j] * aoa_measured[j]
            )
        ax1.scatter(
            aoa_measured * 180.0 / math.pi,
            c_l_measured,
            facecolor="red",
            s=20,
            alpha=0.1,
        )
        ax2.scatter(
            aoa_measured * 180.0 / math.pi,
            c_d_measured,
            facecolor="red",
            s=20,
            alpha=0.1,
        )

    elif aerodynamics_dict["type"] == "PhiAerodynamicsModel":
        phifv_11 = coef_dict["phifv_11"]
        phifv_12 = coef_dict["phifv_12"]
        phifv_13 = coef_dict["phifv_13"]
        phifv_21 = coef_dict["phifv_21"]
        phifv_22 = coef_dict["phifv_22"]
        phifv_23 = coef_dict["phifv_23"]
        phifv_31 = coef_dict["phifv_31"]
        phifv_32 = coef_dict["phifv_32"]
        phifv_33 = coef_dict["phifv_33"]

        phifv = np.zeros((3, 3))
        phifv[0, 0] = phifv_11
        phifv[0, 1] = phifv_12
        phifv[0, 2] = phifv_13
        phifv[1, 0] = phifv_21
        phifv[1, 1] = phifv_22
        phifv[1, 2] = phifv_23
        phifv[2, 0] = phifv_31
        phifv[2, 1] = phifv_32
        phifv[2, 2] = phifv_33

        for i in range(aoa_deg.shape[0]):
            c_l_vec[i] = (
                2.0 * phifv[0, 2] * np.cos(aoa_rad[i]) ** 2
                + (phifv[2, 2] - phifv[0, 0]) * np.sin(aoa_rad[i]) * np.cos(aoa_rad[i])
                - phifv[0, 2]
            )
            c_d_vec[i] = (
                phifv[0, 0]
                - phifv[2, 2] * np.cos(aoa_rad[i]) ** 2
                + 2.0 * phifv[0, 2] * np.sin(aoa_rad[i]) * np.cos(aoa_rad[i])
                + phifv[2, 2]
            )
        ax1.plot(aoa_deg, c_l_vec, label="prediction")
        ax2.plot(aoa_deg, c_d_vec, label="prediction")

        # Visualize uncertainty
        c_l_minimum = np.zeros(aoa_deg.shape[0])
        c_l_maximum = np.zeros(aoa_deg.shape[0])
        c_d_minimum = np.zeros(aoa_deg.shape[0])
        c_d_maximum = np.zeros(aoa_deg.shape[0])

        for i in range(aoa_deg.shape[0]):
            c_l_minimum[i] = (
                2.0
                * (phifv[0, 2] - metric["Cramer"]["phifv_13"])
                * np.cos(aoa_rad[i]) ** 2
                + (
                    (phifv[2, 2] - metric["Cramer"]["phifv_11"])
                    - (phifv[0, 0] - metric["Cramer"]["phifv_11"])
                )
                * np.sin(aoa_rad[i])
                * np.cos(aoa_rad[i])
                - (phifv[0, 2] - metric["Cramer"]["phifv_13"])
            )
            c_l_maximum[i] = (
                2.0
                * (phifv[0, 2] + metric["Cramer"]["phifv_13"])
                * np.cos(aoa_rad[i]) ** 2
                + (
                    (phifv[2, 2] + metric["Cramer"]["phifv_11"])
                    - (phifv[0, 0] + metric["Cramer"]["phifv_11"])
                )
                * np.sin(aoa_rad[i])
                * np.cos(aoa_rad[i])
                - (phifv[0, 2] + metric["Cramer"]["phifv_13"])
            )
            c_d_minimum[i] = (
                (phifv[0, 0] - metric["Cramer"]["phifv_11"])
                - (phifv[2, 2] - metric["Cramer"]["phifv_33"]) * np.cos(aoa_rad[i]) ** 2
                + 2.0
                * (phifv[0, 2] - metric["Cramer"]["phifv_13"])
                * np.sin(aoa_rad[i])
                * np.cos(aoa_rad[i])
                + (phifv[2, 2] - metric["Cramer"]["phifv_33"])
            )
            c_d_maximum[i] = (
                (phifv[0, 0] + metric["Cramer"]["phifv_11"])
                - (phifv[2, 2] + metric["Cramer"]["phifv_33"]) * np.cos(aoa_rad[i]) ** 2
                + 2.0
                * (phifv[0, 2] + metric["Cramer"]["phifv_13"])
                * np.sin(aoa_rad[i])
                * np.cos(aoa_rad[i])
                + (phifv[2, 2] + metric["Cramer"]["phifv_33"])
            )

        ax1.fill_between(aoa_deg, c_l_minimum, c_l_maximum, alpha=0.3)
        ax2.fill_between(aoa_deg, c_d_minimum, c_d_maximum, alpha=0.3)

        aoa_measured = data_df["angle_of_attack"].to_numpy()
        c_l_measured = np.zeros(aoa_measured.shape)
        c_d_measured = np.zeros(aoa_measured.shape)

        for j in range(aoa_measured.shape[0]):
            c_l_measured[j] = (
                2.0 * phifv[0, 2] * np.cos(aoa_measured[j]) ** 2
                + (phifv[2, 2] - phifv[0, 0])
                * np.sin(aoa_measured[j])
                * np.cos(aoa_measured[j])
                - phifv[0, 2]
            )
            c_d_measured[j] = (
                phifv[0, 0]
                - phifv[2, 2] * np.cos(aoa_measured[j]) ** 2
                + 2.0 * phifv[0, 2] * np.sin(aoa_measured[j]) * np.cos(aoa_measured[j])
                + phifv[2, 2]
            )
        ax1.scatter(
            aoa_measured * 180.0 / math.pi,
            c_l_measured,
            facecolor="red",
            s=20,
            alpha=0.05,
        )
        ax2.scatter(
            aoa_measured * 180.0 / math.pi,
            c_d_measured,
            facecolor="red",
            s=20,
            alpha=0.05,
        )

    ax1.set_title("Lift coefficient over angle of attack [deg]")
    ax1.set_xlabel("Angle of Attack [deg]")
    ax1.set_ylabel("Lift Coefficient")
    ax1.grid(True)

    ax2.set_title("Lift coefficient over angle of attack [deg]")
    ax2.set_xlabel("Angle of Attack [deg]")
    ax2.set_ylabel("Drag Coefficient")
    ax2.grid(True)

    return


if __name__ == "__main__":
    plot_example_plate_model()
