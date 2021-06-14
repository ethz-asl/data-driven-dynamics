__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar
import copy

# AAE stands for aileron, aileron, elevator


class TiltWingSection():
    def __init__(self, wing_section_config, v_airspeed_mat, control_surface_output, air_density=1.225, angular_vel_mat=None, rotor=None):
        self.stall_angle = wing_section_config["stall_angle_deg"]*math.pi/180.0
        self.sig_scale_fac = wing_section_config["sig_scale_factor"]
        self.air_density = air_density
        self.n_timestamps = v_airspeed_mat.shape[0]
        self.control_surface_output = control_surface_output
        self.cp_position = np.array(
            wing_section_config["cp_position"]).reshape(3, 1)
        self.wing_surface = wing_section_config["wing_surface"]
        self.description = wing_section_config["description"]
        self.aero_coef_list = ['c_d_wing_xz_cs', 'c_d_wing_xz_lin', 'c_d_wing_xz_offset', 'c_d_wing_xz_quad', 'c_d_wing_xz_stall_90_deg',
                               'c_d_wing_xz_stall_min', 'c_l_wing_xz_cs', 'c_l_wing_xz_lin', 'c_l_wing_xz_offset', 'c_l_wing_xz_stall']

        self.qs_factor = 0.5 * self.air_density*self.wing_surface

        # upstream rotor influencing the downwash over wing section.
        if rotor == None:
            self.in_downstream = False
        else:
            self.in_downstream = True
            self.rotor = rotor

        v_airspeed_mat_copy = copy.deepcopy(v_airspeed_mat)

        # angle around which the wing frame is tilted. The wing is always tilted around the y axis.
        self.wing_angle = np.zeros(self.n_timestamps)
        for i in range(self.n_timestamps):
            self.wing_angle[i] = math.atan2(
                self.rotor.rotor_axis_mat[i, 2],   self.rotor.rotor_axis_mat[i, 0])
        self.compute_static_local_airspeed(
            v_airspeed_mat_copy, angular_vel_mat)
        self.local_airspeed_mat = np.zeros(
            self.static_local_airspeed_mat.shape)
        self.local_aoa_vec = np.zeros(self.static_local_airspeed_mat.shape[0])

    def compute_static_local_airspeed(self, v_airspeed_mat, angular_vel_mat):
        """
            The only component of the airspeed chaning during optimization is the part due to the actuator downwash.
            The other "static" components (airspeed due to groundspeed, wind and angular velocity can be precomputed in this function"""

        if angular_vel_mat is not None:
            self.static_local_airspeed_mat = np.zeros(v_airspeed_mat.shape)
            self.static_aoa_vec = np.zeros(v_airspeed_mat.shape[0])

            assert (v_airspeed_mat.shape ==
                    angular_vel_mat.shape), "RotorModel: v_airspeed_mat and angular_vel_mat differ in size."
            for i in range(self.n_timestamps):
                self.static_local_airspeed_mat[i, :] = v_airspeed_mat[i, :] + \
                    np.cross(angular_vel_mat[i, :],
                             self.cp_position.flatten())
                self.static_aoa_vec[i] = - self.wing_angle[i] + math.atan2(
                    self.static_local_airspeed_mat[i, 2],   self.static_local_airspeed_mat[i, 0])

        else:
            self.static_local_airspeed_mat = v_airspeed_mat
            self.static_aoa_vec = np.zeros(v_airspeed_mat.shape[0])
            for i in range(self.n_timestamps):
                self.static_aoa_vec[i] = - self.wing_angle[i] + math.atan2(
                    self.static_local_airspeed_mat[i, 2],   self.static_local_airspeed_mat[i, 0])

    def update_local_airspeed_and_aoa(self, rotor_thrust_coef):

        rotor_thrust_mat = self.rotor.predict_thrust_force(rotor_thrust_coef)
        if self.in_downstream:
            for i in range(self.n_timestamps):
                abs_thrust = np.linalg.norm(rotor_thrust_mat[i, :])
                v_air_parr = self.rotor.v_air_parallel_abs[i]
                v_downwash = self.rotor.rotor_axis_mat[i, :] * 0.5 * (-v_air_parr + math.sqrt(v_air_parr**2 + (
                    2*abs_thrust/(self.rotor.air_density*self.rotor.prop_area))))
                self.local_airspeed_mat[i,
                                        :] = self.static_local_airspeed_mat[i, :] + v_downwash.flatten()
                # Angle of attack: angle around -y axis
                self.local_aoa_vec[i] = - self.wing_angle[i] + math.atan2(
                    self.local_airspeed_mat[i, 2],   self.local_airspeed_mat[i, 0])
        else:
            print("Not implemented yet.")
            raise NotImplementedError

        self.compute_dynamic_pressure()

    def compute_dynamic_pressure(self):
        self.qs_vec = np.zeros(self.n_timestamps)
        for i in range(self.n_timestamps):
            self.qs_vec[i] = self.qs_factor * \
                (self.local_airspeed_mat[i, 0]**2 +
                 self.local_airspeed_mat[i, 2]**2)

    def compute_aero_frame(self):
        self.aero_frame_angle = self.wing_angle + self.local_aoa_vec
        self.aero_frame_x_mat = np.zeros((self.n_timestamps, 3))
        self.aero_frame_z_mat = np.zeros((self.n_timestamps, 3))
        print(Rotation.from_rotvec(
            [0, (-0.2), 0]).as_matrix())
        for i in range(self.n_timestamps):
            R_aero_to_body = Rotation.from_rotvec(
                [0, (-self.aero_frame_angle[i]), 0]).as_matrix()
            self.aero_frame_x_mat = R_aero_to_body[:, 0].flatten()
            self.aero_frame_z_mat = R_aero_to_body[:, 2].flatten()

    def predict_single_wing_segment_feature_mat(self, qs, control_surface_input, angle_of_attack, wing_angle):
        """
        Model description:

        Compute lift and drag forces in stability axis frame.

        This is done by interpolating two models:
        1. More suffisticated Model for abs(AoA) < stall_angle

            - Lift force coefficient as linear function of AoA:
                F_Lift = 0.5 * density * area * \
                    V_air_xz^2 * (c_l_0 + c_l_lin*AoA)

            - Drag force coefficient as quadratic function of AoA
                F_Drag = 0.5 * density * area * V_air_xz^2 * \
                    (c_d_0 + c_d_lin * AoA + c_d_quad * AoA^2)

        2. Simple plate model for abs(AoA) > stall_angle
                F_Lift = density * area * V_air_xz^2 * \
                    cos(AoA) * sin(AoA) * c_l_stall
                F_Drag = 0.5 * density * area * \
                    V_air_xz^2 * sin(AoA) * c_d_stall

        The two models are interpolated with a symmetric sigmoid function obtained by multiplying two logistic functions:
            if abs(AoA) < stall_angle: cropped_sym_sigmoid(AoA) = 0
            if abs(AoA) > stall_angle: cropped_sym_sigmoid(AoA) = 1
        """
        X_xz_aero_frame = np.zeros((3, 10))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region
        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Drag force coeffiecients:
        X_xz_aero_frame[0, 2] = -flow_attached_region*qs
        X_xz_aero_frame[0, 1] = -flow_attached_region*angle_of_attack*qs
        X_xz_aero_frame[0, 3] = -flow_attached_region*angle_of_attack**2*qs
        X_xz_aero_frame[0, 0] = -flow_attached_region * \
            abs(control_surface_input)*qs
        X_xz_aero_frame[0, 5] = -stall_region * \
            (1 - math.sin(angle_of_attack)**2)*qs
        X_xz_aero_frame[0, 4] = -stall_region*(math.sin(angle_of_attack)**2)*qs
        # Compute Lift force coefficients:
        X_xz_aero_frame[2, 7] = -flow_attached_region*angle_of_attack*qs
        X_xz_aero_frame[2, 8] = -flow_attached_region*qs
        X_xz_aero_frame[2, 6] = flow_attached_region * \
            control_surface_input*qs*0
        X_xz_aero_frame[2, 9] = -stall_region * math.sin(2*angle_of_attack)*qs

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, (-wing_angle-angle_of_attack), 0]).as_matrix()
        X_xz_body_frame = R_aero_to_body @ X_xz_aero_frame
        return X_xz_body_frame

    def predict_wing_segment_forces(self, rotor_thrust_coef, aero_force_coef):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        flap_commands = numpy array of dimension (n,3) with columns for [u_aileron_right, u_aileron_left, u_elevator]
        """
        aero_force_coef = aero_force_coef.reshape(
            (aero_force_coef.shape[0], 1))
        self.update_local_airspeed_and_aoa(rotor_thrust_coef)
        self.X_aero = self.predict_single_wing_segment_feature_mat(
            self.qs_vec[0], self.control_surface_output[0], self.local_aoa_vec[0], self.wing_angle[0])
        for i in range(1, self.n_timestamps):
            X_curr = self.predict_single_wing_segment_feature_mat(
                self.qs_vec[i], self.control_surface_output[i], self.local_aoa_vec[i], self.wing_angle[i])
            self.X_aero = np.vstack((self.X_aero, X_curr))
        F_aero_force_pred = self.X_aero @ aero_force_coef
        return F_aero_force_pred

    def predict_wing_segment_drag_forces(self, aero_force_coef):
        F_drag_force_pred = self.X_aero[:, 0:6] @ aero_force_coef[0:6]
        return F_drag_force_pred

    def predict_wing_segment_lift_forces(self, aero_force_coef):
        F_lift_force_pred = self.X_aero[:, 6:10] @ aero_force_coef[6:10]
        return F_lift_force_pred
