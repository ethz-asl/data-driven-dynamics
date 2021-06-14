__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """


import numpy as np
import pandas as pd
import math
import copy

from .dynamics_model import DynamicsModel
from scipy.optimize import minimize
from scipy.linalg import block_diag
from .model_plots import model_plots, aerodynamics_plots, rotor_plots, tilt_wing_plots
from .model_config import ModelConfig
from .aerodynamic_models import TiltWingSection, FuselageDragModel, ElevatorModel
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from src.tools.math_tools import cropped_sym_sigmoid


"""This model estimates forces and moments for quad plane as for example the standard vtol in gazebo."""


class TiltWingModel(DynamicsModel):
    def __init__(self, rel_data_path, config_file="tilt_wing_config.yaml"):
        self.config = ModelConfig(config_file)
        super(TiltWingModel, self).__init__(
            config_dict=self.config.dynamics_model_config, rel_data_path=rel_data_path)

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]
        self.mass = self.config.model_config["mass"]
        self.u_tilt_vec = self.data_df["u_tilt"].to_numpy()

        self.visual_dataframe_selector_config_dict = {
            "x_axis_col": "timestamp",
            "sub_plt1_data": ["q0", "q1", "q2", "q3"],
            "sub_plt2_data": ["u0", "u1", "u2", "u3", "u4"],
            "sub_plt3_data": ["u5", "u6", "u7", "u8", "u9", "u_tilt"]}

        if "V_air_body_x" not in self.data_df:
            print("computing airspeed")
            self.normalize_actuators(
                ["actuator_controls_0", "actuator_controls_1"], control_outputs_used=True)
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        assert (self.estimate_moments ==
                False), "Estimation of moments is not yet implemented in TiltWingModel. Disable in config file to estimate forces."

    def prepare_opimization_matrices(self):

        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        # Rotor features as self.X_rotor_forces
        self.compute_rotor_features(self.rotor_config_dict)

        # Initialize fuselage drag model
        self.fuselage_model = FuselageDragModel()
        self.X_fuselage, self.fuselage_coef_list = self.fuselage_model.compute_fuselage_features(
            airspeed_mat)

        # Initialize elevator model
        self.elevator_model = ElevatorModel()
        self.X_elevator, self.elevator_coef_list = self.elevator_model.compute_elevator_features(
            airspeed_mat, self.data_df["u9"].to_numpy(), self.data_df["AoA"].to_numpy())
        # Initialize wing sections
        self.wing_sections = []
        wing_sections_config_list = self.config.model_config["aerodynamics"]["main_wing_sections"]
        for i in range(len(wing_sections_config_list)):
            curr_wing_section_config = wing_sections_config_list[i]
            curr_rotor = self.rotor_dict["wing_"][(
                curr_wing_section_config["rotor"])]
            curr_control_surface_output = self.data_df[curr_wing_section_config["control_surface_dataframe_name"]]
            curr_wing_section = TiltWingSection(
                curr_wing_section_config, airspeed_mat, curr_control_surface_output, rotor=curr_rotor)
            self.wing_sections.append(curr_wing_section)
            self.aero_coef_list = curr_wing_section.aero_coef_list

        # Accelerations
        accel_body_mat = self.data_df[[
            "accel_x", "accel_y", "accel_z"]].to_numpy()
        self.y_accel = accel_body_mat.flatten()
        self.y_forces = self.mass * accel_body_mat.flatten()

    def predict_separated_forces(self, x):
        fuselage_coef = np.array(x[0:3]).reshape(3, 1)
        wing_coef = np.array(x[3:13]).reshape(10, 1)
        elevator_coef = np.array(x[13]).reshape(1, 1)
        main_wing_rotor_thrust_coef = np.array(x[[18, 19]]).reshape(2, 1)
        main_wing_rotor_drag_coef = np.array(x[[17]]).reshape(1, 1)
        tail_rotor_thrust_coef = np.array(x[[15, 16]]).reshape(2, 1)
        tail_rotor_drag_coef = np.array(x[[14]]).reshape(1, 1)
        rotor_coef = np.array(x[14:20]).reshape(6, 1)
        self.F_aero_lift_pred = np.zeros((self.y_forces.shape[0], 1))
        self.F_aero_drag_pred = np.zeros((self.y_forces.shape[0], 1))
        self.wing_airframe_x_mat = np.zeros((int(self.y_forces.shape[0]/3), 3))
        self.wing_airframe_z_mat = np.zeros((int(self.y_forces.shape[0]/3), 3))

        self.average_wing_qs = np.zeros(self.data_df.shape[0])
        self.average_wing_aoa = np.zeros(self.data_df.shape[0])
        for wing_section in self.wing_sections:
            F_aero_segment_lift_pred = wing_section.predict_wing_segment_lift_forces(
                wing_coef)
            self.F_aero_lift_pred = np.add(
                self.F_aero_lift_pred, F_aero_segment_lift_pred)
            F_aero_segment_drag_pred = wing_section.predict_wing_segment_lift_forces(
                wing_coef)
            self.F_aero_drag_pred = np.add(
                self.F_aero_drag_pred, F_aero_segment_drag_pred)
            wing_section.compute_aero_frame()
            self.wing_airframe_x_mat = self.wing_airframe_x_mat + wing_section.aero_frame_x_mat
            self.wing_airframe_z_mat = self.wing_airframe_z_mat + wing_section.aero_frame_z_mat
            self.average_wing_qs = self.average_wing_qs + wing_section.qs_vec
            self.average_wing_aoa = self.average_wing_aoa + wing_section.local_aoa_vec
            self.tilt_mat = wing_section.rotor.rotor_axis_mat
        self.average_wing_qs = self.average_wing_qs / 4
        self.average_wing_aoa = self.average_wing_aoa / 4
        self.wing_airframe_x_mat = self.wing_airframe_x_mat / 4
        self.wing_airframe_z_mat = self.wing_airframe_z_mat / 4

        # collective rotor thrust and drag
        self.F_main_rotor_thrust_pred = np.zeros(self.y_forces.shape[0])
        self.F_main_rotor_drag_pred = np.zeros(self.y_forces.shape[0])
        rotor_group_dict = self.rotor_dict["tail_"]
        for rotor_name in rotor_group_dict.keys():
            rotor = rotor_group_dict[rotor_name]
            self.F_tail_rotor_thrust_pred = rotor.predict_thrust_force(
                tail_rotor_thrust_coef).flatten()
            self.F_tail_rotor_drag_pred = rotor.predict_drag_force(
                tail_rotor_drag_coef).flatten()
        rotor_group_list = self.rotor_dict["wing_"]
        for rotor_name in rotor_group_dict.keys():
            rotor = rotor_group_dict[rotor_name]
            self.F_main_rotor_thrust_pred = np.add(
                self.F_main_rotor_thrust_pred, rotor.predict_thrust_force(main_wing_rotor_thrust_coef).flatten())
            self.F_main_rotor_drag_pred = np.add(
                self.F_main_rotor_drag_pred, rotor.predict_drag_force(main_wing_rotor_drag_coef).flatten())
        print(self.F_main_rotor_thrust_pred)

    def predict_forces(self, x):

        fuselage_coef = np.array(x[0:3]).reshape(3, 1)
        wing_coef = np.array(x[3:13]).reshape(10, 1)
        elevator_coef = np.array(x[13]).reshape(1, 1)
        main_wing_rotor_thrust_coef = np.array(x[[18, 19]]).reshape(2, 1)
        rotor_coef = np.array(x[14:20]).reshape(6, 1)
        self.F_aero_pred = np.zeros((self.y_forces.shape[0], 1))
        for wing_section in self.wing_sections:
            F_aero_segment_pred = wing_section.predict_wing_segment_forces(
                main_wing_rotor_thrust_coef, wing_coef)
            self.F_aero_pred = np.add(self.F_aero_pred, F_aero_segment_pred)

        self.F_rotor_pred = self.X_rotor_forces @ rotor_coef
        self.F_fuselage_pred = self.X_fuselage @ fuselage_coef
        self.F_elevator_pred = self.X_elevator @ elevator_coef
        self.F_pred = np.add(self.F_aero_pred, self.F_rotor_pred)
        self.F_pred = np.add(self.F_pred, self.F_fuselage_pred)
        self.F_pred = np.add(self.F_pred, self.F_elevator_pred)
        return self.F_pred

    def objective(self, x):
        print("Evaluating objective function.")
        F_pred = self.predict_forces(x)
        a_pred = F_pred.flatten() / self.mass
        # pred_error = np.linalg.norm((a_pred - self.y_accel))
        pred_error = np.sqrt(((a_pred - self.y_accel) ** 2).mean())
        print("acceleration prediction RMSE: ", pred_error)
        return pred_error

    def objective_influence(self):
        param_change = []
        cost_opt = self.objective(self.x_opt)
        for i in range(len(self.x_opt)):
            x = copy.deepcopy(self.x_opt)
            x[i] = 1.1*self.x_opt[i]
            cost_change_up = self.objective(x) - cost_opt
            x[i] = 0.9*self.x_opt[i]
            cost_change_down = self.objective(x) - cost_opt
            cost_change = (cost_change_down, cost_change_up)
            param_change.append(str(cost_change))

        param_change_dict = dict(zip(self.coef_name_list, param_change))
        import yaml
        print(yaml.dump(self.x_opt_dict, default_flow_style=False))
        print(yaml.dump(param_change_dict, default_flow_style=False))

    def estimate_model(self):
        print("Estimating quad plane model using the following data:")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        self.prepare_opimization_matrices()

        # optimization_variables:
        optimization_parameters = self.config.model_config["optimzation_parameters"]
        self.coef_list = self.fuselage_coef_list + self.aero_coef_list + \
            self.elevator_coef_list + self.rotor_forces_coef_list
        config_coef_list = (
            optimization_parameters["initial_coefficients"]).keys()
        print("estimating coefficients: ", self.coef_list)
        assert (self.coef_list == list(config_coef_list)), (
            "keys on config file differ from coefficient list: ",  self.coef_list, config_coef_list)

        # initial guesses
        x0_dict = optimization_parameters["initial_coefficients"]
        x0 = np.array(
            list(optimization_parameters["initial_coefficients"].values()))
        self.coef_name_list = list(
            optimization_parameters["initial_coefficients"].keys())
        print("Initial coefficients: ", x0)
        coef_bounds = list(
            optimization_parameters["coefficient_bounds"].values())
        print("Coefficients bounds: ", coef_bounds)

        print(self.objective(x0))
        self.x_opt = x0
        self.x_opt_dict = x0_dict

        # solution = minimize(self.objective, x0, method='Powell',
        #                     bounds=coef_bounds)
        # self.x_opt = solution.x

        # initial_conditions_dict = dict(zip(self.coef_name_list, x0.tolist()))

        # metrics_dict = {"R2": float(r2_score(self.y_forces, self.predict_forces(self.x_opt))),
        #                 "RMSE": float(self.objective(self.x_opt))}
        # model_dict = {"metrics": metrics_dict,
        #               "initial_conditions": initial_conditions_dict}

        # self.y_accel_pred = self.predict_forces(self.x_opt)/self.mass
        # self.y_accel_pred_mat = self.y_accel_pred.reshape((-1, 3))
        # self.data_df[["ax_pred", "ay_pred", "az_pred"]] = self.y_accel_pred_mat
        # self.data_df[["V_air_wing_x", "V_air_wing_y", "V_air_wing_z", "AoA_wing"]] = np.hstack(
        #     (self.wing_sections[0].local_airspeed_mat, self.wing_sections[0].local_aoa_vec.reshape(self.data_df.shape[0], 1)))

        # self.generate_model_dict(self.x_opt, model_dict)
        # self.save_result_dict_to_yaml(file_name="tiltwing_model")

        return

    def plot_model_predicitons(self):

        self.y_accel_pred = self.predict_forces(self.x_opt)/self.mass

        wing_local_airspeed = self.wing_sections[0].local_airspeed_mat
        # self.plot_parameter_analysis_curves()

        fig, ax = plt.subplots()
        ax.plot(self.data_df["timestamp"].to_numpy(),
                180/math.pi*self.data_df[["AoA"]].to_numpy(), label="aircraft aoa")
        ax.plot(self.data_df["timestamp"].to_numpy(),
                180/math.pi*self.wing_sections[0].static_aoa_vec.reshape(wing_local_airspeed.shape[0], 1), label="wing aoa")
        ax.plot(self.data_df["timestamp"].to_numpy(),
                180/math.pi*self.wing_sections[0].local_aoa_vec.reshape(wing_local_airspeed.shape[0], 1), label="wing aoa with propwash")
        plt.legend()

        # model_plots.plot_airspeed_and_AoA(
        #     np.hstack((self.wing_sections[0].static_local_airspeed_mat, self.wing_sections[0].local_aoa_vec.reshape(wing_local_airspeed.shape[0], 1))), self.data_df["timestamp"])
        # model_plots.plot_airspeed_and_AoA(
        #     np.hstack((self.wing_sections[0].local_airspeed_mat, self.wing_sections[0].local_aoa_vec.reshape(wing_local_airspeed.shape[0], 1))), self.data_df["timestamp"])
        # model_plots.plot_airspeed_and_AoA(
        #     self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        # u_tilt_vec = self.data_df["u_tilt"].to_numpy()*90
        # tilt_wing_plots.plot_accel_predeictions_and_tilt(
        #     self.y_accel, self.y_accel_pred, self.data_df["timestamp"], u_tilt_vec)
        # model_plots.plot_accel_and_airspeed_in_z_direction(
        #     self.y_accel, self.y_accel_pred, self.data_df["V_air_body_z"], self.data_df["timestamp"])

        # model_plots.plot_accel_and_airspeed_in_y_direction(
        #     self.y_accel, self.y_accel_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])

        plt.show()
        return

    def compute_coefficient_data(self):
        # lift coefficient data
        lift_force_data = (self.y_forces.flatten() - self.F_rotor_pred.flatten() -
                           self.F_fuselage_pred.flatten() - self.F_elevator_pred.flatten() - self.F_aero_drag_pred.flatten()).reshape(
            int(self.y_forces.shape[0]/3), 3)
        self.lift_coef_data = np.zeros(lift_force_data.shape[0])
        for i in range(lift_force_data.shape[0]):
            self.lift_coef_data[i] = np.vdot(
                lift_force_data[i, :], -self.wing_airframe_z_mat[i, :]) / (4*self.average_wing_qs[i])

        # drag coefficient data
        drag_force_data = (self.y_forces.flatten() - self.F_rotor_pred.flatten() -
                           self.F_fuselage_pred.flatten() - self.F_elevator_pred.flatten() - self.F_aero_lift_pred.flatten()).reshape(
            int(self.y_forces.shape[0]/3), 3)
        self.drag_coef_data = np.zeros(drag_force_data.shape[0])
        for i in range(drag_force_data.shape[0]):
            self.drag_coef_data[i] = np.vdot(
                drag_force_data[i, :], -self.wing_airframe_x_mat[i, :]) / (4*self.average_wing_qs[i])

        # Tail Rotor Thrust Data
        tail_thrust_force_data = (self.y_forces.flatten() - self.F_fuselage_pred.flatten() - self.F_elevator_pred.flatten() -
                                  self.F_aero_pred.flatten() - self.F_main_rotor_thrust_pred.flatten() - self.F_main_rotor_drag_pred.flatten() -
                                  self.F_tail_rotor_drag_pred.flatten()).reshape(int(self.y_forces.shape[0]/3), 3)
        self.tail_thrust_force_data_proj = self.project_data(
            tail_thrust_force_data, np.array([0, 0, -1]))

    def plot_parameter_analysis_curves(self):
        self.predict_separated_forces(self.x_opt)
        self.compute_coefficient_data()

        c_l_pred_dict = {"c_l_offset": self.x_opt_dict["c_l_wing_xz_offset"],
                         "c_l_lin": self.x_opt_dict["c_l_wing_xz_lin"],
                         "c_l_stall": self.x_opt_dict["c_l_wing_xz_stall"]}

        c_l_exp_dict = {"c_l_offset": 0,
                        "c_l_lin": 5.7,
                        "c_l_stall": 0.6}

        c_d_pred_dict = {"c_d_offset": self.x_opt_dict["c_d_wing_xz_offset"],
                         "c_d_lin": self.x_opt_dict["c_d_wing_xz_lin"],
                         "c_d_quad": self.x_opt_dict["c_d_wing_xz_quad"],
                         "c_d_stall_min": self.x_opt_dict["c_d_wing_xz_stall_min"],
                         "c_d_stall_max": self.x_opt_dict["c_d_wing_xz_stall_90_deg"]}

        # aerodynamics_plots.plot_aoa_hist(self.average_wing_aoa)
        aerodynamics_plots.plot_lift_curve2(c_l_pred_dict, c_l_exp_dict)
        aerodynamics_plots.plot_lift_prediction_and_underlying_data(
            c_l_pred_dict, self.lift_coef_data, self.average_wing_aoa)
        aerodynamics_plots.plot_drag_prediction_and_underlying_data(
            c_d_pred_dict, self.drag_coef_data, self.average_wing_aoa)

        tail_rot_coef_dict = {"rot_thrust_quad": self.x_opt_dict["tail_rot_thrust_quad"],
                              "rot_thrust_lin": self.x_opt_dict["tail_rot_thrust_lin"],
                              "rot_drag_lin": self.x_opt_dict["tail_rot_drag_lin"], }

        wing_rot_coef_dict = {"rot_thrust_quad": self.x_opt_dict["wing_rot_thrust_quad"],
                              "rot_thrust_lin": self.x_opt_dict["wing_rot_thrust_lin"],
                              "rot_drag_lin": self.x_opt_dict["wing_rot_drag_lin"], }

        rotor_plots.plot_thrust_prediction_and_underlying_data(tail_rot_coef_dict, self.rotor_dict["tail_"]["u4"],
                                                               self.tail_thrust_force_data_proj)

        rotor_plots.plot_rotor_trust_3d(
            tail_rot_coef_dict, self.rotor_dict["tail_"]["u4"])

        rotor_plots.plot_rotor_trust_3d(
            wing_rot_coef_dict, self.rotor_dict["wing_"]["u1"])
