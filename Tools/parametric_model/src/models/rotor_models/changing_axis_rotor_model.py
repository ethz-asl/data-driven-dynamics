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

from . import RotorModel
import numpy as np
import pandas as pd
import math
from progress.bar import Bar
from . import RotorModel

"""
The Changing Axis Rotor Model provides functionalities for rotors that change the direction of the thrust vector in one way or the other. 
This can be used among other as a basis for:

- Tilting rotors 
- Rotors mounted on a tilting wing
- Rotors tht can reverse thrust. 

It accounts for this computing an matrix where each row represents the rotor axis for the corresponding timestamp.
It then overrides the methods compute_actuator_force_matrix and compute_actuator_moment_matrix from the rotor model
to pass the actuator axis for each timestep. 

This model contains only a dummy computation for the rotor axis matrix. 
The compute_rotor_axis_mat method can be overridden for the desired usecase.
"""


class ChangingAxisRotorModel(RotorModel):

    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, air_density=1.225, angular_vel_mat=None):
        self.n_timestamps = actuator_input_vec.shape[0]
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.compute_rotor_axis_mat()
        super(ChangingAxisRotorModel, self).__init__(
            rotor_config_dict, actuator_input_vec, v_airspeed_mat, air_density, angular_vel_mat=angular_vel_mat, rotor_axis_mat=self.rotor_axis_mat)

    def compute_rotor_axis_mat(self):
        self.rotor_axis_mat = np.zeros((self.n_timestamps, 3))
        for i in range(self.n_timestamps):
            # Active vector rotation around tilt axis:
            self.rotor_axis_mat[i, :] = self.rotor_axis.flatten

    def compute_actuator_force_matrix(self):
        print("Computing force features for rotor:", self.rotor_name)
        X_forces = self.compute_actuator_force_features(
            0, self.rotor_axis_mat[0, :].reshape((3, 1)))
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_force_curr = self.compute_actuator_force_features(
                index, self.rotor_axis_mat[index, :].reshape((3, 1)))
            X_forces = np.vstack((X_forces, X_force_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_forces = ["rot_drag_lin", "rot_thrust_lin", "rot_thrust_quad",
                            ]
        self.X_forces = X_forces
        self.X_thrust = X_forces[:, 1:]
        return X_forces, coef_list_forces

    def compute_actuator_moment_matrix(self):
        print("Computing moment features for rotor:", self.rotor_name)
        X_moments = self.compute_actuator_moment_features(
            0, self.rotor_axis_mat[0, :].reshape((3, 1)))
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_moment_curr = self.compute_actuator_moment_features(
                index, self.rotor_axis_mat[index, :].reshape((3, 1)))
            X_moments = np.vstack((X_moments, X_moment_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_moments = ["c_m_leaver_quad", "c_m_leaver_lin",
                             "c_m_drag_z_quad", "c_m_drag_z_lin", "c_m_rolling"]
        return X_moments, coef_list_moments
