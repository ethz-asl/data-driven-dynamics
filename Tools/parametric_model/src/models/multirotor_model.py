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

The model in this file estimates a simple force motor model for a multirotor.

Model Parameters:
u                    : normalized actuator output scaled between 0 and 1
angular_vel_const    : angular velocity constant
angular_vel_offset   : angular velocity offset
mot_const            : motor constant
m                    : mass of UAV
accel_const          : combined acceleration constant k_2/m

Model:
angular_vel [rad/s] = angular_vel_const*u + angular_vel_offset
F_thrust = - mot_const * angular_vel^2
F_thrust_tot = - mot_const * \
    (angular_vel_1^2 + angular_vel_2^2 + angular_vel_3^2 + angular_vel_4^2)

Note that the forces are calculated in the NED body frame and are therefore negative.
"""

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

from sklearn.linear_model import LinearRegression
from .dynamics_model import DynamicsModel
from .rotor_models import RotorModel
from .aerodynamic_models import FuselageDragModel
from .model_config import ModelConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class MultiRotorModel(DynamicsModel):
    def __init__(self, config_file, model_name="multirotor_model"):
        self.config = ModelConfig(config_file)
        super(MultiRotorModel, self).__init__(
            config_dict=self.config.dynamics_model_config)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"],
                                         self.config.model_config["moment_of_inertia"]["Iyy"], self.config.model_config["moment_of_inertia"]["Izz"]])

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]

        self.model_name = model_name

    def prepare_force_regression_matrices(self):

        accel_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        force_mat = accel_mat * self.mass
        #self.y_forces = (force_mat).flatten()
        self.data_df[["measured_force_x", "measured_force_y",
                     "measured_force_z"]] = force_mat

        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        aero_model = FuselageDragModel()
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_fuselage_features(
            airspeed_mat)
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)
        self.y_dict.update({"lin":{"x":"measured_force_x","y":"measured_force_y","z":"measured_force_z"}})

    def prepare_moment_regression_matrices(self):
        moment_mat = np.matmul(self.data_df[[
            "ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy(), self.moment_of_inertia)
        #self.y_moments = (moment_mat).flatten()
        self.data_df[["measured_moment_x", "measured_moment_y",
                     "measured_moment_z"]] = moment_mat
        
        self.y_dict.update({"rot":{"x":"measured_moment_x","y":"measured_moment_y","z":"measured_moment_z"}})
