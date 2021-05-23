__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.models.rotor_models import RotorModel
import os
import numpy as np
import math


def test_local_airspeed_computation():
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "RotorModel",
                         "rotor_axis": [1, 0, 0],
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0, 0])
    v_airspeed_mat = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])

    rotor = RotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat)

    assert np.array_equal(rotor.v_air_parallel_abs, np.array([5, 0, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[0, :], np.array([0, 0, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[1, :], np.array([0, 5, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[2, :], np.array([0, 0, 5]))


def test_rotor_thrust_prediction():
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "RotorModel",
                         "rotor_axis": [0, 0, -1],
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0.5, 1, 0, 0.5, 1])
    v_airspeed_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [
                              0, 0, -5], [0, 0, -5], [0, 0, -5]])
    thrust_coef_list = [10, -1]

    correct_force_prediction = np.array([[0,       0,       0.],
                                         [0,       0,  -3.0625],
                                         [0,       0,   -12.25],
                                         [0,       0,       0.],
                                         [0,       0,       0.],
                                         [0,       0,  -6.125]])

    rotor = RotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat)
    X_moments, coef_list_moments = rotor.compute_actuator_force_matrix()
    print(rotor.predict_thrust_force(
        thrust_coef_list))
    assert (np.array_equal(rotor.predict_thrust_force(
        thrust_coef_list), correct_force_prediction))


if __name__ == "__main__":
    # set cwd to project directory when run as module
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.pardir)
    des_cwd = os.path.join(parent, os.pardir)
    os.chdir(des_cwd)

    test_local_airspeed_computation()
    test_rotor_thrust_prediction()
