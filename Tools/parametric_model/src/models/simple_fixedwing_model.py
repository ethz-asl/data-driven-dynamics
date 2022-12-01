"""
 *
 * Copyright (c) 2021 Julius Schlapbach
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

__author__ = "Julius Schlapbach"
__maintainer__ = "Julius Schlapbach"
__license__ = "BSD 3"


import numpy as np

from .dynamics_model import DynamicsModel
from .model_config import ModelConfig
from .aerodynamic_models import LinearWingModel, ControlSurfaceModel


class SimpleFixedWingModel(DynamicsModel):
    def __init__(self, config_file, model_name="simple_fixedwing_model"):
        self.config = ModelConfig(config_file)
        super(SimpleFixedWingModel, self).__init__(
            config_dict=self.config.dynamics_model_config)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"],
                                         self.config.model_config["moment_of_inertia"]["Iyy"], self.config.model_config["moment_of_inertia"]["Izz"]])

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.aero_config_dict = self.config.model_config["actuators"]["control_surfaces"]
        self.aerodynamics_dict = self.config.model_config["aerodynamics"]

    def prepare_force_regression_matrices(self):
        # Accelerations
        accel_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        force_mat = accel_mat * self.mass
        xcorrection_vel_frame = self.mass * \
            self.data_df['ang_vel_y'].to_numpy(
            ) * self.data_df['vz'].to_numpy()
        zcorrection_vel_frame = self.mass * \
            self.data_df['ang_vel_y'].to_numpy(
            ) * self.data_df['vx'].to_numpy()
        angle_of_attack = self.data_df['angle_of_attack'].to_numpy()
        xcorrection_b_frame = xcorrection_vel_frame * \
            np.cos(angle_of_attack) - zcorrection_vel_frame * \
            np.sin(angle_of_attack)
        zcorrection_b_frame = xcorrection_vel_frame * \
            np.sin(angle_of_attack) + zcorrection_vel_frame * \
            np.cos(angle_of_attack)

        # for i in range(len(force_mat)):
        #     force_mat[i][0] -= xcorrection_b_frame[i]
        #     force_mat[i][2] += zcorrection_b_frame[i]

        self.data_df[["measured_force_x", "measured_force_y", "measured_force_z"]] = force_mat

        # Aerodynamics features
        airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        aoa_mat = self.data_df[["angle_of_attack"]].to_numpy()
        elevator_vec = self.data_df["u7"].to_numpy()
        ang_vel_mat = self.data_df[["ang_vel_x",
                                    "ang_vel_y", "ang_vel_z"]].to_numpy()
        gamma_vec = - np.arctan2(self.data_df['vz'], self.data_df['vx'])
        aero_model = LinearWingModel(self.aerodynamics_dict, self.mass)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_force_features(
            airspeed_mat, aoa_mat, elevator_vec, gamma_vec, ang_vel_mat)
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)
        self.y_dict.update({"lin": {"x": "measured_force_x",
                           "y": "measured_force_y", "z": "measured_force_z"}})

    def prepare_moment_regression_matrices(self):
        # Angular acceleration
        moment_mat = np.matmul(self.data_df[[
            "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy(), self.moment_of_inertia)
        self.y_moments = moment_mat.flatten()
        self.data_df[["measured_moment_x", "measured_moment_y",
                     "measured_moment_z"]] = moment_mat

        # Aerodynamics features
        airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        aoa_mat = self.data_df[["angle_of_attack"]].to_numpy()
        sideslip_mat = self.data_df[["angle_of_sideslip"]].to_numpy()

        aero_model = LinearWingModel(self.aerodynamics_dict, self.mass)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_moment_features(
            airspeed_mat, aoa_mat, sideslip_mat)
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)

        aero_config_dict = self.aero_config_dict
        for aero_group in aero_config_dict.keys():
            aero_group_list = self.aero_config_dict[aero_group]

            for config_dict in aero_group_list:
                controlsurface_input_name = config_dict["dataframe_name"]
                u_vec = self.data_df[controlsurface_input_name].to_numpy()
                control_surface_model = ControlSurfaceModel(
                    config_dict, self.aerodynamics_dict, u_vec)
                X_controls, coef_dict_controls, col_names_controls = control_surface_model.compute_actuator_moment_matrix(
                    airspeed_mat, aoa_mat)
                self.data_df[col_names_controls] = X_controls
                self.coef_dict.update(coef_dict_controls)

        self.y_dict.update({"rot": {"x": "measured_moment_x",
                           "y": "measured_moment_y", "z": "measured_moment_z"}})