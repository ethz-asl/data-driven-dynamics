__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import math


class GazeboRotorModel():
    def __init__(self, rotor_axis=np.array([[0], [0], [-1]]), max_rotor_inflow_vel=25.0, rotor_direction=1):
        # no more thrust produced at this airspeed inflow velocity
        self.max_rotor_inflow_vel = max_rotor_inflow_vel
        self.rotor_axis = rotor_axis.reshape((3, 1))
        self.rotrotor_direction = rotor_direction

    def compute_actuator_force_features(self, actuator_input, v_airspeed):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs

        Inputs: 
        actuator_input: actuator input between 0 and 1
        v_airspeed: airspeed velocity in body frame, numpoy array of shape (3,1)

        For the model explanation have a look at the PDF.
        """

        # Thrust force computation
        v_airspeed_parallel_to_rotor_axis = np.vdot(
            self.rotor_axis, v_airspeed) * self.rotor_axis
        v_air_parallel = np.linalg.norm(v_airspeed_parallel_to_rotor_axis)

        X_thrust = self.rotor_axis @ np.array(
            [[actuator_input**2, (actuator_input*v_air_parallel)]])
        # Drag force computation
        v_airspeed_perpendicular_to_rotor_axis = v_airspeed - \
            v_airspeed_parallel_to_rotor_axis
        if (np.linalg.norm(v_airspeed_perpendicular_to_rotor_axis) >= 0.1):
            X_drag = v_airspeed_perpendicular_to_rotor_axis @ np.array(
                [[actuator_input]])
        else:
            X_drag = np.zeros((3, 1))

        X_forces = np.hstack((X_drag, X_thrust))

        # Thrust leaver moment

        # Rotor drag moment

        return X_forces

    def compute_actuator_feature_matrix(self, actuator_input_vec, v_airspeed_mat):
        """
        Inputs: 
        actuator_input_vec: vector of actuator inputs (normalized between 0 and 1), numpy array of shape (n, 1)
        v_airspeed_mat: matrix of vertically stacked airspeed vectors, numpy array of shape (n, 3)
        """
        X_actuator_forces = self.compute_actuator_force_features(
            actuator_input_vec[0], v_airspeed_mat[0, :].reshape((3, 1)))
        for i in range(1, actuator_input_vec.shape[0]):
            X_curr = self.compute_actuator_force_features(
                actuator_input_vec[i], v_airspeed_mat[i, :].reshape((3, 1)))
            X_actuator_forces = np.vstack((X_actuator_forces, X_curr))
        coef_list = ["rot_drag_lin", "rot_thrust_quad",
                     "rot_thrust_lin"]
        return X_actuator_forces, coef_list
