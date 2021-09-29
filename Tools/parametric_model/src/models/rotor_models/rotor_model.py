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

import numpy as np
import pandas as pd
import math
from progress.bar import Bar
import copy


class RotorModel():
    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, air_density=1.225, angular_vel_mat=None, rotor_axis_mat=None):
        """
        Inputs:
        actuator_input_vec: vector of actuator inputs (normalized between 0 and 1), numpy array of shape (n, 1)
        v_airspeed_mat: matrix of vertically stacked airspeed vectors, numpy array of shape (n, 3)
        rotor_axis_mat: matrices of rotor axis corresponding to different timestamps. only needed for models with tilting rotors.
        """

        # no more thrust produced at this airspeed inflow velocity
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.rotor_position = np.array(
            rotor_config_dict["position"]).reshape(3, 1)
        self.turning_direction = rotor_config_dict["turning_direction"]
        self.rotor_name = rotor_config_dict["description"]
        self.actuator_input_vec = np.array(actuator_input_vec)
        self.n_timestamps = actuator_input_vec.shape[0]

        # prop diameter in meters
        if "diameter" in rotor_config_dict.keys():
            self.prop_diameter = rotor_config_dict["diameter"]
        else:
            self.prop_diameter = 1
        self.prop_area = math.pi*self.prop_diameter**2 / 4
        # air density in kg/m^3
        self.air_density = air_density

        v_airspeed_mat_copy = copy.deepcopy(v_airspeed_mat)
        self.compute_local_airspeed(
            v_airspeed_mat_copy, angular_vel_mat, rotor_axis_mat)

    def compute_local_airspeed(self, v_airspeed_mat, angular_vel_mat, rotor_axis_mat=None):

        # adjust airspeed with angular acceleration is angular_vel_mat is passed as argument

        if angular_vel_mat is not None:
            self.local_airspeed_mat = np.zeros(v_airspeed_mat.shape)
            assert (v_airspeed_mat.shape ==
                    angular_vel_mat.shape), "RotorModel: v_airspeed_mat and angular_vel_mat differ in size."
            for i in range(self.n_timestamps):
                self.local_airspeed_mat[i, :] = v_airspeed_mat[i, :] + \
                    np.cross(angular_vel_mat[i, :],
                             self.rotor_position.flatten())

        else:
            self.local_airspeed_mat = v_airspeed_mat

        self.v_airspeed_parallel_to_rotor_axis = np.zeros(
            v_airspeed_mat.shape)
        self.v_air_parallel_abs = np.zeros(v_airspeed_mat.shape[0])
        self.v_airspeed_perpendicular_to_rotor_axis = np.zeros(
            v_airspeed_mat.shape)

        # if the rotor axis changes direction and rotor_axis_mat is specified
        if rotor_axis_mat is not None:
            for i in range(self.n_timestamps):
                v_local_airspeed = self.local_airspeed_mat[i, :]
                self.v_airspeed_parallel_to_rotor_axis[i, :] = (np.vdot(
                    rotor_axis_mat[i, :], v_local_airspeed) * rotor_axis_mat[i, :]).flatten()
                self.v_air_parallel_abs[i] = np.linalg.norm(
                    self.v_airspeed_parallel_to_rotor_axis[i, :])
                self.v_airspeed_perpendicular_to_rotor_axis[i, :] = v_local_airspeed - \
                    self.v_airspeed_parallel_to_rotor_axis[i, :]
        else:
            for i in range(self.n_timestamps):
                v_local_airspeed = self.local_airspeed_mat[i, :]
                self.v_airspeed_parallel_to_rotor_axis[i, :] = (np.vdot(
                    self.rotor_axis, v_local_airspeed) * self.rotor_axis).flatten()
                self.v_air_parallel_abs[i] = np.linalg.norm(
                    self.v_airspeed_parallel_to_rotor_axis[i, :])
                self.v_airspeed_perpendicular_to_rotor_axis[i, :] = v_local_airspeed - \
                    self.v_airspeed_parallel_to_rotor_axis[i, :]

    def compute_actuator_force_features(self, index, rotor_axis=None):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs

        Inputs:
        actuator_input: actuator input between 0 and 1
        v_airspeed: airspeed velocity in body frame, numpoy array of shape (3,1)

        For the model explanation have a look at the PDF.
        """

        actuator_input = self.actuator_input_vec[index]
        v_air_parallel_abs = self.v_air_parallel_abs[index]
        v_airspeed_perpendicular_to_rotor_axis = \
            self.v_airspeed_perpendicular_to_rotor_axis[index, :].reshape(
                (3, 1))

        if rotor_axis is None:
            rotor_axis = self.rotor_axis

        # Thrust force computation
        X_thrust = rotor_axis @ np.array(
            [[(actuator_input*v_air_parallel_abs), actuator_input**2 * self.prop_diameter]]) * self.air_density * self.prop_diameter**3
        # Drag force computation
        if (np.linalg.norm(v_airspeed_perpendicular_to_rotor_axis) >= 0.05):
            X_drag = - v_airspeed_perpendicular_to_rotor_axis @ np.array(
                [[actuator_input]])
        else:
            X_drag = np.zeros((3, 1))

        X_forces = np.hstack((X_drag, X_thrust))

        return X_forces

    def compute_actuator_moment_features(self, index, rotor_axis=None):

        actuator_input = self.actuator_input_vec[index]
        v_air_parallel_abs = self.v_air_parallel_abs[index]
        v_airspeed_perpendicular_to_rotor_axis = self.v_airspeed_perpendicular_to_rotor_axis[index, :].reshape(
            (3, 1))

        if rotor_axis is None:
            rotor_axis = self.rotor_axis

        X_moments = np.zeros((3, 5))
        leaver_moment_vec = np.cross(
            self.rotor_position.flatten(), rotor_axis.flatten())
        # Thrust leaver moment
        X_moments[:, 0] = leaver_moment_vec * actuator_input**2 * \
            self.air_density * self.prop_diameter**4
        X_moments[:, 1] = leaver_moment_vec * \
            actuator_input*v_air_parallel_abs * self.air_density * self.prop_diameter**3

        # Rotor drag moment
        X_moments[2, 2] = - self.turning_direction * \
            actuator_input**2 * self.air_density * self.prop_diameter**5
        X_moments[2, 3] = - self.turning_direction * \
            actuator_input*v_air_parallel_abs * self.air_density * self.prop_diameter**5

        # Rotor Rolling Moment
        X_moments[:, 4] = -1 * v_airspeed_perpendicular_to_rotor_axis.flatten() * \
            self.turning_direction * actuator_input

        return X_moments

    def compute_actuator_force_matrix(self):
        print("Computing force features for rotor:", self.rotor_name)
        X_forces = self.compute_actuator_force_features(0)
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_force_curr = self.compute_actuator_force_features(index)
            X_forces = np.vstack((X_forces, X_force_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_forces = ["rot_drag_lin", "rot_thrust_lin", "rot_thrust_quad"
                            ]
        self.X_forces = X_forces
        self.X_thrust = X_forces[:, 1:]
        return X_forces, coef_list_forces

    def compute_actuator_moment_matrix(self):
        print("Computing moment features for rotor:", self.rotor_name)
        X_moments = self.compute_actuator_moment_features(0)
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_moment_curr = self.compute_actuator_moment_features(index)
            X_moments = np.vstack((X_moments, X_moment_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_moments = ["c_m_leaver_quad", "c_m_leaver_lin",
                             "c_m_drag_z_quad", "c_m_drag_z_lin", "c_m_rolling"]
        return X_moments, coef_list_moments

    def predict_thrust_force(self, thrust_coef_list):
        """
        Inputs: thrust_coef_list = ["rot_thrust_lin", "rot_thrust_quad"]
        """
        thrust_coef = np.array(thrust_coef_list).reshape(2, 1)
        stacked_force_vec = self.X_thrust @ thrust_coef
        force_mat = stacked_force_vec.reshape((self.n_timestamps, 3))
        return force_mat
