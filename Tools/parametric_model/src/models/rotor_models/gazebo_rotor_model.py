__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import math


class GazeboRotorModel():
    def __init__(self, max_rotor_inflow_vel=25.0, rotor_axis=np.array([[0], [0], [1]]), rotor_direction=1):
        # no more thrust produced at this airspeed inflow velocity
        self.max_rotor_inflow = max_rotor_inflow_vel
        self.rotor_axis = rotor_axis.reshape((3, 1))
        self.rotrotor_direction = rotor_direction

    def compute_actuator_force_features(self, actuator_input, v_airspeed):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs

        Inputs: 
        actuator_input: actuator input between 0 and 1
        v_airspeed: airspeed velocity in body frame, numpoy array of shape (3,1)

        Model:
        angular_vel [rad/s] = angular_vel_const*actuator_input + angular_vel_offset
        inflow_scaler = 1 - v_airspeed_parallel_to_rotor_axis/self.max_rotor_inflow_vel
        F_thrust = inflow_scaler * mot_const * angular_vel^2 * rotor_axis_vec
        F_drag = - angular_vel * drag_coef * v_airspeed_perpendicular_to_rotor_axis_vec

        Rotor Thrust Features:
        F_thrust/m = (c_2 * actuator_input^2 + c_1 * actuator_input + c_0)* rotor_axis_vec
        F_thrust/m = X_thrust @ (c_2, c_1, c_0)^T

        Rotor Drag Features: 
        F_drag/m = X_drag @ (c_4, c_3)^T = v_airspeed_perpendicular_to_rotor_axis_vec @ (u, 1) @ (c_4, c_3)^T
        """

        # Thrust computation
        v_airspeed_parallel_to_rotor_axis = np.inner(
            self.rotor_axis, v_airspeed) * self.rotor_axis
        vel = np.linalg.norm(vel_parallel_to_rotor_axis)
        inflow_scaler = 1 - vel/self.max_rotor_inflow_vel
        X_thrust = inflow_scaler * self.rotor_axis @ np.array(
            [[actuator_input**2, actuator_input, 1]])
        # Drag computation
        v_airspeed_perpendicular_to_rotor_axis = v_airspeed - vel_parallel_to_rotor_axis
        X_drag = vel_perpendicular_to_rotor_axis @ np.array(
            [[actuator_input, 1]])
        X_forces = np.hstack((X_drag, X_thrust))

        return X_forces

    def compute_thrust_features(self, actuator_input_vec, v_airspeed_mat):
        """
        Inputs: 
        actuator_input_vec: vector of actuator inputs (normalized between 0 and 1), numpy array of shape (n, 1)
        v_airspeed_mat: matrix of vertically stacked airspeed vectors, numpy array of shape (n, 3)
        """
        X_lin_thrust = self.compute_actuator_force_features(
            actuator_input_vec[0], v_airspeed_mat[0, :].reshape((3, 1)))
        for i in range(1, self.data_df.shape[0]):
            u_curr = actuator_input_vec[i, :]
            X_thrust_curr = self.compute_actuator_thrust_feature(
                actuator_input_vec[i], v_airspeed_mat[i, :].reshape((3, 1)))
            X_lin_thrust = np.vstack((X_lin_thrust, X_thrust_curr))
        return X_lin_thrust
