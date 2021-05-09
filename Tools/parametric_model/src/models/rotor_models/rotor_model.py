__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import math
from progress.bar import Bar


class RotorModel():
    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, density=1.225):
        """
        Inputs:
        actuator_input_vec: vector of actuator inputs (normalized between 0 and 1), numpy array of shape (n, 1)
        v_airspeed_mat: matrix of vertically stacked airspeed vectors, numpy array of shape (n, 3)
        """

        # no more thrust produced at this airspeed inflow velocity
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.rotor_position = np.array(
            rotor_config_dict["position"]).reshape(3, 1)
        self.turning_direction = rotor_config_dict["turning_direction"]
        self.rotor_name = rotor_config_dict["description"]
        self.actuator_input_vec = actuator_input_vec

        # prop diameter in meters
        if "diameter" in rotor_config_dict.keys():
            self.prop_diameter = rotor_config_dict["diameter"]
        else:
            self.prop_diameter = 1
        self.prop_area = math.pi*self.prop_diameter**2 / 4
        # air density in kg/m^3
        self.density = density

        self.initialize_actuator_airspeed(v_airspeed_mat)

    def initialize_actuator_airspeed(self, v_airspeed_mat):

        self.v_airspeed_parallel_to_rotor_axis = np.zeros(
            v_airspeed_mat.shape)
        self.v_air_parallel_abs = np.zeros(v_airspeed_mat.shape[0])
        self.v_airspeed_perpendicular_to_rotor_axis = np.zeros(
            v_airspeed_mat.shape)

        for i in range(v_airspeed_mat.shape[0]):
            v_airspeed = v_airspeed_mat[i, :]
            self.v_airspeed_parallel_to_rotor_axis[i, :] = (np.vdot(
                self.rotor_axis, v_airspeed) * self.rotor_axis).flatten()
            self.v_air_parallel_abs[i] = np.linalg.norm(
                self.v_airspeed_parallel_to_rotor_axis)
            self.v_airspeed_perpendicular_to_rotor_axis[i, :] = v_airspeed - \
                self.v_airspeed_parallel_to_rotor_axis[i, :]

        self.airspeed_initialized = True

    def compute_actuator_force_features(self, actuator_input, v_air_parallel_abs, v_airspeed_perpendicular_to_rotor_axis):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs

        Inputs:
        actuator_input: actuator input between 0 and 1
        v_airspeed: airspeed velocity in body frame, numpoy array of shape (3,1)

        For the model explanation have a look at the PDF.
        """

        # Thrust force computation

        X_thrust = self.rotor_axis @ np.array(
            [[actuator_input**2, (actuator_input*v_air_parallel_abs)]]) * self.density * self.prop_diameter**4
        # Drag force computation
        if (np.linalg.norm(v_airspeed_perpendicular_to_rotor_axis) >= 0.05):
            X_drag = - v_airspeed_perpendicular_to_rotor_axis @ np.array(
                [[actuator_input]])
        else:
            X_drag = np.zeros((3, 1))

        X_forces = np.hstack((X_drag, X_thrust))

        return X_forces

    def compute_actuator_moment_features(self, actuator_input, v_air_parallel_abs, v_airspeed_perpendicular_to_rotor_axis):

        X_moments = np.zeros((3, 5))

        leaver_moment_vec = np.cross(
            self.rotor_position.flatten(), self.rotor_axis.flatten())
        # Thrust leaver moment
        X_moments[:, 0] = leaver_moment_vec * actuator_input**2 * \
            self.density * self.prop_diameter**4
        X_moments[:, 1] = leaver_moment_vec * \
            actuator_input*v_air_parallel_abs * self.density * self.prop_diameter**4

        # Rotor drag moment
        X_moments[2, 2] = - self.turning_direction * \
            actuator_input**2 * self.density * self.prop_diameter**5
        X_moments[2, 3] = - self.turning_direction * \
            actuator_input*v_air_parallel_abs * self.density * self.prop_diameter**5

        # Rotor Rolling Moment
        X_moments[:, 4] = -1 * v_airspeed_perpendicular_to_rotor_axis.flatten() * \
            self.turning_direction * actuator_input

        return X_moments

    def compute_actuator_force_matrix(self):

        assert (self.airspeed_initialized), "Airspeed is not initialized. Call initialize_actuator_airspeed(v_airspeed_mat) first."

        print("Computing force features for rotor:", self.rotor_name)

        X_forces = self.compute_actuator_force_features(
            self.actuator_input_vec[0], self.v_air_parallel_abs[0], self.v_airspeed_perpendicular_to_rotor_axis[0, :].reshape((3, 1)))
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for i in range(1, self.actuator_input_vec.shape[0]):
            X_force_curr = self.compute_actuator_force_features(
                self.actuator_input_vec[i], self.v_air_parallel_abs[i], self.v_airspeed_perpendicular_to_rotor_axis[i, :].reshape((3, 1)))
            X_forces = np.vstack((X_forces, X_force_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_forces = ["rot_drag_lin", "rot_thrust_quad",
                            "rot_thrust_lin"]
        return X_forces, coef_list_forces

    def compute_actuator_moment_matrix(self):

        assert (self.airspeed_initialized), "Airspeed is not initialized. Call initialize_actuator_airspeed(v_airspeed_mat) first."

        print("Computing moment features for rotor:", self.rotor_name)

        X_moments = self.compute_actuator_moment_features(
            self.actuator_input_vec[0], self.v_air_parallel_abs[0], self.v_airspeed_perpendicular_to_rotor_axis[0, :].reshape((3, 1)))
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for i in range(1, self.actuator_input_vec.shape[0]):
            X_moment_curr = self.compute_actuator_moment_features(
                self.actuator_input_vec[i], self.v_air_parallel_abs[i], self.v_airspeed_perpendicular_to_rotor_axis[i, :].reshape((3, 1)))
            X_moments = np.vstack((X_moments, X_moment_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_moments = ["c_m_leaver_quad", "c_m_leaver_lin",
                             "c_m_drag_z_quad", "c_m_drag_z_lin", "c_m_rolling"]
        return X_moments, coef_list_moments
