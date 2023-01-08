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
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class SimpleFixedWingModel(DynamicsModel):
    def __init__(self, config_file, normalization=True, model_name="simple_fixedwing_model"):
        self.config = ModelConfig(config_file)
        super(SimpleFixedWingModel, self).__init__(
            config_dict=self.config.dynamics_model_config, normalization=normalization)
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag([self.config.model_config["moment_of_inertia"]["Ixx"],
                                         self.config.model_config["moment_of_inertia"]["Iyy"],
                                         self.config.model_config["moment_of_inertia"]["Izz"]])

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.aero_config_dict = self.config.model_config["actuators"]["control_surfaces"]
        self.aerodynamics_dict = self.config.model_config["aerodynamics"]

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

        # TODO: move to separate plotting functions
        self.air_density = 1.225
        self.gravity = 9.81
        self.area = self.aerodynamics_dict["area"]
        self.chord = self.aerodynamics_dict["chord"]
        const = 0.5 * self.air_density * self.area * \
            (airspeed_mat[:, 0]**2 + airspeed_mat[:, 2]**2)
        angles_of_attack = aoa_mat[:, 0]
        throttle = self.data_df["throttle"].to_numpy()

        # vector with cl0, clalpha, cldelta, cd0, cdalpha, cdalpha_sq, ct
        coeff_vec = np.array([1.0911507971564856, 6.47680410045709, 2.2670575265057487,
                             0.11373832931995426, 0.7433772294557139, 2.8798589221670072, 85.7567143535296])

        xyz_b_forces_predicted = np.zeros((3, self.data_df.shape[0]))

        for k in range(angles_of_attack.shape[0]):
            X_wing_aero_frame = np.zeros((3, 7))

            # Compute Drag force coeffiecients:
            X_wing_aero_frame[0, 3] = - const[k]
            X_wing_aero_frame[0, 4] = - const[k] * angles_of_attack[k]
            X_wing_aero_frame[0, 5] = - const[k] * (angles_of_attack[k] ** 2)
            X_wing_aero_frame[0, 6] = throttle[k]
            # Compute Lift force coefficients:
            X_wing_aero_frame[2, 0] = - const[k]
            X_wing_aero_frame[2, 1] = - const[k] * angles_of_attack[k]
            X_wing_aero_frame[2, 2] = - const[k] * elevator_inputs[k]

            # Transorm from stability axis frame to body FRD frame
            R_aero_to_body = Rotation.from_rotvec(
                [0, -angles_of_attack[k], 0]).as_matrix()
            X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame

            temp_xyz_b_forces = X_wing_body_frame @ coeff_vec
            xyz_b_forces_predicted[:, k] = temp_xyz_b_forces

        plt.figure('forces z-direction (body frame)')
        plt.plot(self.data_df.index, force_mat[:, 2], label='measured')
        plt.plot(self.data_df.index,
                 xyz_b_forces_predicted[2, :], label='identified')
        plt.xlabel('timestamp')
        plt.ylabel('body force z-direction')
        plt.legend()
        plt.show()

        plt.figure('forces x-direction (body frame)')
        plt.plot(self.data_df.index, force_mat[:, 0], label='measured')
        plt.plot(self.data_df.index,
                 xyz_b_forces_predicted[0, :], label='identified')
        plt.xlabel('timestamp')
        plt.ylabel('body force x-direction')
        plt.legend()
        plt.show()
        # TODO: move to separate plotting functions

        aero_model = LinearWingModel(self.aerodynamics_dict)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_force_features(
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

        # TODO: move to separate plotting functions
        angles_of_attack = aoa_mat[:, 0]
        const = 0.5 * self.air_density * self.area * self.chord * \
            (airspeed_mat[:, 0]**2 + airspeed_mat[:, 2]**2)
        vel_xz = np.sqrt(airspeed_mat[:, 0]**2 + airspeed_mat[:, 2]**2)
        damping_feature = (angular_vel_mat[:, 1] * self.chord) / (2 * vel_xz)
        throttle = self.data_df["throttle"].to_numpy()

        # moment coefficient vector cm0, cmalpha, cmdelta, cmq, ct_m
        coeff_vec = np.array(
            [0.00021965680447913602, 0.0003774541187492259, -0.0037941448161590324, 8.878397639309345, 0.1415432962319521])

        xyz_b_moments_predicted = np.zeros((3, self.data_df.shape[0]))

        for k in range(angles_of_attack.shape[0]):
            X_wing_aero_frame = np.zeros((3, 5))

            # Compute Pitching moment coefficients:
            X_wing_aero_frame[1, 0] = const[k]
            X_wing_aero_frame[1, 1] = const[k] * angles_of_attack[k]
            X_wing_aero_frame[1, 2] = const[k] * elevator_inputs[k]
            X_wing_aero_frame[1, 3] = const[k] * damping_feature[k]
            X_wing_aero_frame[1, 4] = throttle[k]

            R_aero_to_body = Rotation.from_rotvec(
                [0, - angles_of_attack[k], 0]).as_matrix()
            X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame

            temp_xyz_b_moments = X_wing_body_frame @ coeff_vec
            xyz_b_moments_predicted[:, k] = temp_xyz_b_moments

        plt.figure('moment around y-axis (pitch moment)')
        plt.plot(self.data_df.index, moment_mat[:, 1], label='measured')
        plt.plot(self.data_df.index,
                 xyz_b_moments_predicted[1, :], label='identified')
        plt.xlabel('timestamp')
        plt.ylabel('angular moment around y-axis')
        plt.legend()
        plt.show()

        # df = self.data_df[(self.data_df['timestamp'] >= 709795260) & (
        #     self.data_df['timestamp'] <= 860763711)]
        # first = self.data_df.index.get_loc(df.index[0])
        # last = self.data_df.index.get_loc(df.index[-1])
        # print(first, last)

        # print('timestamps', self.data_df['timestamp'][first:last], self.data_df['timestamp'][first:last].shape, 'measured moment', moment_mat[first:last, 1], moment_mat[first:last, 1].shape, '\nidentified moment', const[first:last] * (0.0025575334672471826 + 0.021617695732657614 * aoa_mat[first:last, 0] + 0.0018050097273585603 * elevator_inputs[first:last] + 7.747755231160494 * damping_feature[first:last]))
        # print('angle of attack', aoa_mat[first:last, 0], aoa_mat[first:last, 0].shape, 'elevator inputs', elevator_inputs[first:last], elevator_inputs[first:last].shape, 'damping feature', damping_feature[first:last], damping_feature[first:last].shape, 'const', const[first:last], const[first:last].shape)
        # print('angular velocity', angular_vel_mat[first:last, 1], angular_vel_mat[first:last, 1].shape, 'velocity in x-z-plane', vel_xz[first:last], vel_xz[first:last].shape)
        # print('timestamps' , self.data_df['timestamp'], self.data_df['timestamp'].shape, 'measured moment', moment_mat[:, 1], moment_mat[:, 1].shape, '\nidentified moment', const * (0.0025575334672471826 + 0.021617695732657614 * aoa_mat[:, 0] + 0.0018050097273585603 * elevator_inputs + 7.747755231160494 * damping_feature))
        # print('angle of attack', aoa_mat, aoa_mat.shape, 'elevator inputs', elevator_inputs, elevator_inputs.shape, 'damping feature', damping_feature, damping_feature.shape, 'const', const, const.shape)
        # print('angular velocity', angular_vel_mat[:, 1], angular_vel_mat[:, 1].shape, 'velocity in x-z-plane', vel_xz, vel_xz.shape)

        # print('timestamp', self.data_df['timestamp'], '\nangle of attacks:', aoa_mat, aoa_mat.shape, '\nelevator inputs:', elevator_inputs, elevator_inputs.shape, '\nconst. values', const, const.shape)
        # TODO: move to separate plotting functions

        aero_model = LinearWingModel(self.aerodynamics_dict)
        X_aero, coef_dict_aero, col_names_aero = aero_model.compute_aero_moment_features(
            airspeed_mat, aoa_mat[:, 0], elevator_inputs, angular_vel_mat, sideslip_mat)

        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)

        # ! READD THIS IF REQUIRED
        # aero_config_dict = self.aero_config_dict
        # for aero_group in aero_config_dict.keys():
        #     aero_group_list = self.aero_config_dict[aero_group]

        #     for config_dict in aero_group_list:
        #         controlsurface_input_name = config_dict["dataframe_name"]
        #         u_vec = self.data_df[controlsurface_input_name].to_numpy()
        #         control_surface_model = ControlSurfaceModel(
        #             config_dict, self.aerodynamics_dict, u_vec)
        #         X_controls, coef_dict_controls, col_names_controls = control_surface_model.compute_actuator_moment_matrix(
        #             airspeed_mat, aoa_mat)
        #         self.data_df[col_names_controls] = X_controls
        #         self.coef_dict.update(coef_dict_controls)

        self.y_dict.update({"rot": {"x": "measured_moment_x",
                           "y": "measured_moment_y", "z": "measured_moment_z"}})
