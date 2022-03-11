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

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar

"""
The control surface model is conform to PX4's standard plane
 """


class ControlSurfaceModel():
    def __init__(self, config_dict, aerodynamics_dict, actuator_input_vec):
        self.name = config_dict["description"]
        self.actuator_input_vec = np.array(actuator_input_vec)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.air_density = 1.225
        self.area = aerodynamics_dict["area"]

    def compute_actuator_force_features(self, index, v_airspeed, angle_of_attack):
        """
        Model description:
        """
        actuator_input = self.actuator_input_vec[index]
        q_xz = 0.5 * self.air_density * \
            (v_airspeed[0]**2 + v_airspeed[2]**2)  # TODO Take dynamic pressure
        # TODO Compute lift axis and drag axis
        lift_axis = np.array([v_airspeed[0], 0.0, v_airspeed[2]])
        lift_axis = (lift_axis / np.linalg.norm(lift_axis)).reshape((3, 1))
        drag_axis = (-1.0 * v_airspeed /
                     np.linalg.norm(v_airspeed)).reshape((3, 1))
        X_lift = lift_axis @ np.array([[actuator_input]]) * q_xz * self.area
        X_drag = drag_axis @ np.array([[actuator_input]]) * q_xz * self.area
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_lift_body = R_aero_to_body @ X_lift
        X_drag_body = R_aero_to_body @ X_drag

        return np.hstack((X_lift_body, X_drag_body))

    def compute_actuator_moment_features(self, index, v_airspeed, angle_of_attack):
        """
        Model description:
        """
        actuator_input = self.actuator_input_vec[index]
        q_xz = 0.5 * self.air_density * \
            (v_airspeed[0]**2 + v_airspeed[2]**2)  # TODO Take dynamic pressure
        lift_axis = np.array([v_airspeed[0], 0.0, v_airspeed[2]])
        yaw_axis = (lift_axis / np.linalg.norm(lift_axis)).reshape((3, 1))
        roll_axis = (v_airspeed / np.linalg.norm(v_airspeed)).reshape((3, 1))
        pitch_axis = np.cross(np.transpose(yaw_axis),
                              np.transpose(roll_axis)).reshape((3, 1))
        X_roll_moment = roll_axis @ np.array(
            [[actuator_input]]) * q_xz * self.area
        X_pitch_moment = pitch_axis @ np.array(
            [[actuator_input]]) * q_xz * self.area
        X_yaw_moment = yaw_axis @ np.array([[actuator_input]]
                                           ) * q_xz * self.area
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_roll_moment_body = R_aero_to_body @ X_roll_moment
        X_pitch_moment_body = R_aero_to_body @ X_pitch_moment
        X_yaw_moment_body = R_aero_to_body @ X_yaw_moment
        X_moments = np.hstack((X_roll_moment_body, X_pitch_moment_body, X_yaw_moment_body))
        X_moments = X_moments.flatten()
        return X_moments

    def compute_actuator_force_matrix(self, v_airspeed_mat, angle_of_attack_vec):
        print("Computing force features for control surface:", self.name)

        X_forces = self.compute_actuator_force_features(
            0, v_airspeed_mat[0, :], angle_of_attack_vec[0, :])
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_force_curr = self.compute_actuator_force_features(
                index, v_airspeed_mat[index, :], angle_of_attack_vec[index, :])
            X_forces = np.vstack((X_forces, X_force_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_forces = ["c_l_delta", "c_d_delta"]
        self.X_forces = X_forces
        self.X_thrust = X_forces[:, 1:]
        return X_forces, coef_list_forces

    def compute_actuator_moment_matrix(self, v_airspeed_mat, angle_of_attack_vec):
        print("Computing moment features for control surface:", self.name)

        X_aero = self.compute_actuator_moment_features(
            0, v_airspeed_mat[0, :], angle_of_attack_vec[0, :])
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_moment_curr = self.compute_actuator_moment_features(
                index, v_airspeed_mat[index, :], angle_of_attack_vec[index, :])
            X_aero = np.vstack((X_aero, X_moment_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_dict = {
            self.name + "c_m_x_delta": {"rot":{ "x": self.name+"c_m_x_delta_x","y": self.name+"c_m_x_delta_y","z":self.name+"c_m_x_delta_z"}},
            self.name + "c_m_y_delta": {"rot":{ "x": self.name+"c_m_y_delta_x","y": self.name+"c_m_y_delta_y","z":self.name+"c_m_y_delta_z"}},
            self.name + "c_m_z_delta": {"rot":{ "x": self.name+"c_m_z_delta_x","y": self.name+"c_m_z_delta_y","z":self.name+"c_m_z_delta_z"}},
        }        
        col_names = [self.name+"c_m_x_delta_x", self.name+"c_m_x_delta_y", self.name+"c_m_x_delta_z",
                    self.name+"c_m_y_delta_x", self.name+"c_m_y_delta_y", self.name+"c_m_y_delta_z",
                    self.name+"c_m_z_delta_x", self.name+"c_m_z_delta_y", self.name+"c_m_z_delta_z"]
        return X_aero, coef_dict, col_names
