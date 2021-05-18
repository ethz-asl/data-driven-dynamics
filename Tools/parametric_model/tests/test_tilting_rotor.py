__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.models.rotor_models import TiltingRotorModel
import os
import numpy as np
import math


def test_tilting_x_axis():
    rotor_config_dict = {"description": "rotor wing right, right",
                         "dataframe_name": "u0",
                         "rotor_type": "TiltingRotorModel",
                         "tilt_actuator_dataframe_name": "u_tilt",
                         "rotor_axis": [0, 1, 0],
                         "tilt_axis": [1, 0, 0],
                         "max_tilt_angle_deg": 180,
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0, 0])
    tilt_actuator_vec = np.array([0, 0.5, 1])
    v_airspeed_mat = np.zeros((3, 3))
    rotor = TiltingRotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat,
                              tilt_actuator_vec)
    print(rotor.rotor_axis_mat)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[0, :] - np.array([0, 1, 0])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[1, :] - np.array([0, 0, 1])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[2, :] - np.array([0, -1, 0])) < 10e-10)


def test_tilting_y_axis():
    rotor_config_dict = {"description": "rotor wing right, right",
                         "dataframe_name": "u0",
                         "rotor_type": "TiltingRotorModel",
                         "tilt_actuator_dataframe_name": "u_tilt",
                         "rotor_axis": [1, 0, 0],
                         "tilt_axis": [0, 1, 0],
                         "max_tilt_angle_deg": 90,
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0, 0])
    tilt_actuator_vec = np.array([0, 0.5, 1])
    v_airspeed_mat = np.zeros((3, 3))
    rotor = TiltingRotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat,
                              tilt_actuator_vec)
    print(rotor.rotor_axis_mat)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[0, :] - np.array([1, 0, 0])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[1, :] - np.array([math.sqrt(0.5), 0, -math.sqrt(0.5)])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[2, :] - np.array([0, 0, -1])) < 10e-10)


def test_tilting_z_axis():
    rotor_config_dict = {"description": "rotor wing right, right",
                         "dataframe_name": "u0",
                         "rotor_type": "TiltingRotorModel",
                         "tilt_actuator_dataframe_name": "u_tilt",
                         "rotor_axis": [0, 1, 0],
                         "tilt_axis": [0, 0, 1],
                         "max_tilt_angle_deg": 360,
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0, 0])
    tilt_actuator_vec = np.array([0.25, 0.5, 0.75])
    v_airspeed_mat = np.zeros((3, 3))
    rotor = TiltingRotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat,
                              tilt_actuator_vec)
    print(rotor.rotor_axis_mat)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[0, :] - np.array([-1, 0, 0])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[1, :] - np.array([0, -1, 0])) < 10e-10)
    assert (np.linalg.norm(
        rotor.rotor_axis_mat[2, :] - np.array([1, 0, 0])) < 10e-10)


# Run uas module using 'python3 -m tests/test_dynamics_model' for development and testing of the test.
if __name__ == "__main__":
    # set cwd to project directory when run as module
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.pardir)
    des_cwd = os.path.join(parent, os.pardir)
    os.chdir(des_cwd)

    test_tilting_x_axis()
    test_tilting_y_axis()
    test_tilting_z_axis()
