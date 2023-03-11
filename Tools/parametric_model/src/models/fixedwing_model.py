"""
 *
 * Copyright (c) 2023 Julius Schlapbach
 *               2023 Autonomous Systems Lab ETH Zurich
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

from . import aerodynamic_models
from .dynamics_model import DynamicsModel
from .model_config import ModelConfig
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class FixedWingModel(DynamicsModel):
    def __init__(self, config_file, normalization=True, model_name="simple_fixedwing_model"):
        self.config = ModelConfig(config_file)
        super(FixedWingModel, self).__init__(
            config_dict=self.config.dynamics_model_config, normalization=normalization)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"],
                                         self.config.model_config["moment_of_inertia"]["Iyy"],
                                         self.config.model_config["moment_of_inertia"]["Izz"]])

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.aerodynamics_dict = self.config.model_config["aerodynamics"]

        try:
            self.aero_model = getattr(aerodynamic_models, self.aerodynamics_dict["type"])(self.aerodynamics_dict)
        except AttributeError:
            error_str = "Aerodynamics Model '{0}' not found, is it added to models "\
                        "directory and models/__init__.py?".format(self.aerodynamics_dict.type)
            raise AttributeError(error_str)

    def prepare_force_regression_matrices(self):
        accel_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        force_mat = accel_mat * self.mass
        self.y_forces = (force_mat).flatten()
        self.data_df[["measured_force_x", "measured_force_y",
                     "measured_force_z"]] = force_mat

        # Aerodynamics features
        airspeed_mat = self.data_df[[
            "V_air_body_x", "V_air_body_y", "V_air_body_z"]].to_numpy()
        aoa_mat = self.data_df[["angle_of_attack"]].to_numpy()
        elevator_inputs = self.data_df["elevator"].to_numpy()

        X_aero, coef_dict_aero, col_names_aero = self.aero_model.compute_aero_force_features(
            airspeed_mat, aoa_mat[:, 0], elevator_inputs)
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
        angular_vel_mat = self.data_df[[
            "ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy()
        elevator_inputs = self.data_df["elevator"].to_numpy()

        X_aero, coef_dict_aero, col_names_aero = self.aero_model.compute_aero_moment_features(
            airspeed_mat, aoa_mat[:, 0], elevator_inputs, angular_vel_mat, sideslip_mat)

        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)

        self.y_dict.update({"rot": {"x": "measured_moment_x",
                           "y": "measured_moment_y", "z": "measured_moment_z"}})
