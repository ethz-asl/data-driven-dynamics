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
        c = self.air_density * self.prop_diameter**3
        X_drag = -(np.array(self.v_airspeed_perpendicular_to_rotor_axis).T * 
            np.array(self.actuator_input_vec) *
            (np.linalg.norm(self.v_airspeed_perpendicular_to_rotor_axis,axis=1) >= 0.05) # Masking 
            ).T
        if hasattr(self, "rotor_axis_mat"):
            self.X_thrust_lin = (self.rotor_axis_mat.T * self.v_air_parallel_abs * self.actuator_input_vec).T*c
            self.X_thrust_quad = (self.rotor_axis_mat.T * self.actuator_input_vec**2 ).T*(c*self.prop_diameter)
        else:
            self.X_thrust_lin = np.outer(self.v_air_parallel_abs * self.actuator_input_vec,self.rotor_axis)*c
            self.X_thrust_quad = np.outer(self.actuator_input_vec**2 * self.prop_diameter,self.rotor_axis)*c

        coef_dict = {
            "rot_drag_lin": {"lin":{ "x": "rot_drag_lin_x","y": "rot_drag_lin_y","z":"rot_drag_lin_z"}},
            "rot_thrust_lin": {"lin":{ "x": "rot_thrust_lin_x","y": "rot_thrust_lin_y","z":"rot_thrust_lin_z"}},
            "rot_thrust_quad": {"lin":{ "x": "rot_thrust_quad_x","y": "rot_thrust_quad_y","z":"rot_thrust_quad_z"}},
            }
        col_names = [
            "rot_drag_lin_x","rot_drag_lin_y", "rot_drag_lin_z",
            "rot_thrust_lin_x","rot_thrust_lin_y","rot_thrust_lin_z",
            "rot_thrust_quad_x","rot_thrust_quad_y", "rot_thrust_quad_z" 
            ]
        self.X_forces = np.hstack((X_drag,self.X_thrust_lin,self.X_thrust_quad))
        return self.X_forces, coef_dict, col_names

    def compute_actuator_moment_matrix(self):
        print("Computing moment features for rotor:", self.rotor_name)
        if hasattr(self, "rotor_axis_mat"):
            leaver_moment_vec = 0
        else:
            leaver_moment_vec = np.cross(
                self.rotor_position.flatten(),self.rotor_axis.flatten())
            leaver_moment_vec = np.outer(leaver_moment_vec,np.ones(self.n_timestamps))

        X_leaver_quad = (leaver_moment_vec * self.actuator_input_vec**2 * \
            self.air_density * self.prop_diameter**4).T

        X_leaver_lin = (leaver_moment_vec * self.actuator_input_vec * \
            self.v_air_parallel_abs * self.air_density * self.prop_diameter**3).T

        X_drag_quad = - np.outer(self.turning_direction * \
            self.actuator_input_vec**2 * self.air_density * self.prop_diameter**5,[0,0,1])

        X_drag_lin = - np.outer(self.turning_direction * \
            self.actuator_input_vec * self.v_air_parallel_abs * 
            self.air_density * self.prop_diameter**5 ,[0,0,1])

        X_rolling = - (self.v_airspeed_perpendicular_to_rotor_axis.T * \
            self.turning_direction * self.actuator_input_vec).T


        coef_list_moments = ["c_m_leaver_quad", "c_m_leaver_lin",
                             "c_m_drag_z_quad", "c_m_drag_z_lin", "c_m_rolling"]

        coef_dict = {
            "c_m_leaver_quad": {"rot":{ "x": "c_m_leaver_quad_x","y": "c_m_leaver_quad_y","z":"c_m_leaver_quad_z"}},
            "c_m_leaver_lin": {"rot":{ "x": "c_m_leaver_lin_x","y": "c_m_leaver_lin_y","z":"c_m_leaver_lin_z"}},
            "c_m_drag_z_quad": {"rot":{ "x": "c_m_drag_z_quad_x","y": "c_m_drag_z_quad_y","z":"c_m_drag_z_quad_z"}},
            "c_m_drag_z_lin": {"rot":{ "x": "c_m_drag_z_lin_x","y": "c_m_drag_z_lin_y","z":"c_m_drag_z_lin_z"}},
            "c_m_rolling": {"rot":{ "x": "c_m_rolling_x","y": "c_m_rolling_y","z":"c_m_rolling_z"}},
            }
        col_names = [
            "c_m_leaver_quad_x","c_m_leaver_quad_y", "c_m_leaver_quad_z",
            "c_m_leaver_lin_x","c_m_leaver_lin_y","c_m_leaver_lin_z",
            "c_m_drag_z_quad_x","c_m_drag_z_quad_y", "c_m_drag_z_quad_z",
            "c_m_drag_z_lin_x","c_m_drag_z_lin_y", "c_m_drag_z_lin_z",
            "c_m_rolling_x","c_m_rolling_y", "c_m_rolling_z",
            ]
        return np.hstack((X_leaver_quad,X_leaver_lin,X_drag_quad,X_drag_lin,X_rolling)), coef_dict, col_names

    def predict_thrust_force(self, thrust_coef_list):
        """
        Inputs: thrust_coef_list = ["rot_thrust_lin", "rot_thrust_quad"]
        """
        force_mat = thrust_coef_list[0]*self.X_thrust_lin + thrust_coef_list[1]*self.X_thrust_quad
        return force_mat
