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

from src.models.model_config import ModelConfig

import numpy as np
import time
import yaml
from scipy.optimize import fsolve


class SingularityFreeExtractorModel():
    def __init__(self, config, model_config_file, coefficients):
        """
        Initialize the global singularity-free extractor model.
        The configuration dictionary should at least contain specifications for minimum and maximum airspeed for the aircraft (strutural and stall limits).
        Optionally, a criuse flight speed can be specified. Otherwise, the maximum range airspeed is used as cruise speed.

        :param config: configuration dictionary
        :param model_config_file: path to model configuration file
        :param coefficients: dictionary with identified aerodynamic coefficients
        """
        print("\n\n===============================================================================")
        print("                        PX4 Parameter Extraction                               ")
        print("===============================================================================")

        self.model_name = "singularityfree_extractor"
        self.config = config
        self.model_config = ModelConfig(model_config_file)
        self.aero_params = coefficients

        self.gravity = 9.81
        self.rho = 1.225

        self.mass = self.model_config.model_config["mass"]
        self.area = self.model_config.model_config['aerodynamics']["area"]
        self.chord = self.model_config.model_config['aerodynamics']["chord"]
        self.span = self.area / self.chord

        self.px4_params = {}

        # check that all required aerodynamic coefficients are given
        aero_coeffs = ["cmdelta", "phifv_11", "phifv_12", "phifv_13", "phifv_21", "phifv_22",
                       "phifv_23", "phifv_31", "phifv_32", "phifv_33", "puller_ct", "puller_cmt"]
        for coeff in aero_coeffs:
            if not coeff in self.aero_params:
                raise ValueError(
                    "Coefficient {} not found in identified coefficient dictionary".format(coeff))

    def get_px4_params(self):
        """
        Getter function for the extarcted px4 parameters

        :return: dictionary with extracted px4 parameters
        """
        return self.px4_params

    def save_px4_params_to_yaml(self, output_file):
        """
        Save the extracted px4 parameters to a yaml file

        :param output_file: path to the output file
        """
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = output_file + "_" + timestr + ".yaml"

        with open(file_path, 'w') as outfile:
            yaml.dump(self.px4_params, outfile, default_flow_style=False)

        return

    def compute_px4_params(self):
        """
        Main function to compute the px4 parameters
        To see which parameters are computed, check the README of the repository
        """
        print('Starting parameter extraction with the following configuration parameters:')
        print('Minimum Airspeed: ', self.config['vmin'])
        print('Maximum Airspeed: ', self.config['vmax'])
        print('Cruise speed: ', self.config['vcruise']
              if 'vcruise' in self.config else 'speed for maximum range')

        self.px4_params['FW_AIRSPD_MIN'] = self.config['vmin']
        self.px4_params['FW_AIRSPD_MAX'] = self.config['vmax']

        self.px4_params['TRIM_PITCH_MAX_RANGE'], self.px4_params['FW_AIRSPD_MAX_RANGE'] = self.get_max_range_params()
        self.px4_params['FW_AIRSPD_TRIM'] = self.config['vcruise'] if 'vcruise' in self.config else self.px4_params['FW_AIRSPD_MAX_RANGE']

        self.px4_params['TRIM_PITCH_MIN_SINK'], self.px4_params['FW_AIRSPD_MIN_SINK'], self.px4_params['FW_T_SINK_MIN'] = self.get_min_sink_params()

        self.px4_params['FW_THR_TRIM'], self.px4_params['FW_PSP_OFF'], self.px4_params['TRIM_PITCH'] = self.get_cruise_params(
            self.px4_params['FW_AIRSPD_TRIM'])

        self.px4_params['FW_DTRIM_P_VMIN'], self.px4_params['FW_THR_VMIN'] = self.get_min_vel_params(
            self.config['vmin'], self.px4_params['TRIM_PITCH'])

        self.px4_params['TRIM_PITCH_MAX_SINK'], self.px4_params['FW_T_SINK_MAX'], self.px4_params['FW_DTRIM_P_VMAX'], self.px4_params['FW_THR_VMAX'] = self.get_max_vel_params(
            self.config['vmax'], self.px4_params['TRIM_PITCH'])

        self.px4_params['FW_T_CLIMB_MAX'], self.px4_params['TRIM_PITCH_MAX_CLIMB'] = self.get_max_climb_params(
            self.px4_params['FW_AIRSPD_MIN_SINK'])

        print("\n===============================================================================")
        print("                      END PX4 Parameter Extraction                             ")
        print("===============================================================================")

        return

    def cL(self, alpha):
        """
        This function computes the lift coefficient.

        :param alpha: angle of attack
        :return: lift coefficient
        """
        return self.aero_params['phifv_13'] * np.cos(2 * alpha) + 0.5 * (self.aero_params['phifv_33'] - self.aero_params['phifv_11']) * np.sin(2 * alpha)

    def cD(self, alpha):
        """
        This function computes the drag coefficient.

        :param alpha: angle of attack
        :return: drag coefficient
        """
        return 0.5 * ((self.aero_params['phifv_11'] - self.aero_params['phifv_33']) * np.cos(2 * alpha) + 2 * self.aero_params['phifv_13'] * np.sin(2 * alpha) + self.aero_params['phifv_33'] + self.aero_params['phifv_11'])

    def lift(self, alpha, delta_e, velocity):
        """
        This function computes the lift force.

        :param alpha: angle of attack
        :param delta_e: elevator deflection
        :return: lift force
        """
        return 0.5 * self.rho * self.area * (velocity ** 2) * self.cL(alpha)

    def drag(self, alpha, velocity):
        """
        This function computes the drag force.

        :param alpha: angle of attack
        :return: drag force
        """
        return 0.5 * self.rho * self.area * (velocity ** 2) * self.cD(alpha)

    def thrust(self, throttle):
        """
        This function computes the thrust force.

        :param throttle: throttle setting
        :return: thrust force
        """
        return self.aero_params['puller_ct'] * throttle

    def zero_moment_trim(self, alpha, throttle, velocity):
        """
        This function computes the elevator trim for zero resulting pitching moment.
        CAUTION: For simplicity, it is assumed that the sideslip velocity V_y is zero in trimmed flight.

        :param alpha: angle of attack
        :return: elevator trim
        """
        gamma = np.arctan2(self.cD(alpha), self.cL(alpha))

        Vx = np.sqrt((velocity ** 2) / (1 + (np.tan(gamma) ** 2)))
        Vz = np.tan(gamma) * Vx

        eta = velocity
        constant = - 0.5 * self.rho * self.area * eta
        vf_x = self.chord**2 * constant * Vx
        vf_z = self.chord**2 * constant * Vz

        return - (vf_x * self.aero_params['phifv_21'] + vf_z * self.aero_params['phifv_23'] + self.aero_params['puller_cmt'] * throttle) / self.aero_params['cmdelta']

    def get_max_range_params(self):
        """
        This function computes the elevator trim and flight speed for maximum range (gliding flight)
        Condition for max range: max cL/cD

        :return: elevator trim and flight speed
        """
        print("Computing max range parameters...")
        alphas = np.linspace(-20 * np.pi / 180, 20 * np.pi / 180, 500)

        airspeeds = np.zeros(len(alphas))
        for i in range(len(alphas)):
            airspeeds[i], _ = self.get_flight_vel_zero_thrust(alphas[i])

        trims = self.zero_moment_trim(alphas, np.zeros(len(alphas)), airspeeds)
        lift_drag_ratio = self.cL(alphas) / self.cD(alphas)

        # find max lift/drag ratio, which corresponds to the max range flight state
        idx = np.argmax(lift_drag_ratio)
        elevator_trim = trims[idx]
        airspeed = airspeeds[idx]

        print("Max range parameters computed successfully:")
        print("Elevator trim: ", elevator_trim,
              "Speed for max range: ", airspeed, '\n')

        return float(elevator_trim), float(airspeed)

    def get_min_sink_params(self):
        """
        This function computes the elevator trim, flight speed and sink rate for the minimum sink rate flight state (gliding flight)
        Condition for min sink rate: max lift^3 / drag^2

        :return: elevator trim, flight speed and sink rate
        """
        print("Computing min sink parameters...")
        alphas = np.linspace(-30 * np.pi / 180, 30 * np.pi / 180, 500)

        airspeeds = np.zeros(len(alphas))
        gammas = np.zeros(len(alphas))
        for i in range(len(alphas)):
            airspeeds[i], gammas[i] = self.get_flight_vel_zero_thrust(
                alphas[i])

        trims = self.zero_moment_trim(alphas, 0.0, airspeeds)

        # find max lift^3 / drag^2 ratio, which corresponds to the minimum sink rate flight state
        ratio = (self.cL(alphas) ** 3) / (self.cD(alphas) ** 2)
        idx = np.argmax(ratio)
        airspeed = airspeeds[idx]
        gamma = gammas[idx]
        elevator_trim = trims[idx]

        min_sink_rate = airspeed / (np.sqrt(1 + 1 / (np.tan(- gamma) ** 2)))

        print("Min sink parameters computed successfully.")
        print("Elevator trim: ", elevator_trim, "Speed for min sink: ",
              airspeed, "Min sink rate: ", min_sink_rate, '\n')
        return float(elevator_trim), float(airspeed), float(min_sink_rate)

    def get_cruise_params(self, airspeed: float):
        """
        This function computes the elevator trim and throttle setting for level flight
        The velocity (airspeed) is given as an input

        :param airspeed: flight speed
        :return: elevator trim, throttle setting and level flight pitch (= angle of attack)
        """
        print("Computing cruise flight parameters...")

        throttle_setting, pitch_level_flight, elevator_trim = self.get_level_flight_params(
            airspeed)

        print("Level flight parameters computed successfully.")
        print("Level flight pitch: ", pitch_level_flight,
              "Throttle setting: ", throttle_setting, "Level flight trim: ", elevator_trim, '\n')
        return float(throttle_setting), float(pitch_level_flight), float(elevator_trim)

    def get_min_vel_params(self, vmin: float, level_trim: float):
        """
        This function computes the elevator trim and flight speed for minimum velocity (user-provided).

        :param vmin: minimum velocity
        :return: differential elevator trim (to cruise trim) and throttle setting at minimum flight speed
        """
        print("Computing min velocity parameters...")

        throttle_setting, _, elevator_trim = self.get_level_flight_params(vmin)
        diff_trim = elevator_trim - level_trim

        print("Min velocity parameters computed successfully.")
        print("Differential elevator trim: ", diff_trim,
              "at user-specified min velocity: ", vmin,
              'Throttle setting', throttle_setting, '\n')
        return float(diff_trim), float(throttle_setting)

    def get_max_vel_params(self, vmax: float, level_trim: float):
        """
        This function computes the elevator trim and maximum sink rate for maximum velocity 

        :param vmax: maximum velocity (user input)
        :return: differential elevator trim (to cruise trim), throttle setting for Vmax at level flight and maximum sink rate
        """
        print("Computing max velocity parameters...")

        throttle_setting, _, elevator_trim = self.get_level_flight_params(vmax)
        diff_trim = elevator_trim - level_trim

        # compute max sink rate with motor off
        def eom_zero_thrust(initial_values):
            gamma, alpha = initial_values
            return (- self.mass * self.gravity * np.sin(gamma) - self.drag(alpha, vmax),
                    self.mass * self.gravity * np.cos(gamma) - self.lift(alpha, self.zero_moment_trim(alpha, 0.0, vmax), vmax))

        gamma, alpha = fsolve(eom_zero_thrust, [- 0.2, 0.1])
        max_sink_rate = vmax / (np.sqrt(1 + 1 / (np.tan(- gamma) ** 2)))
        max_sink_trim = self.zero_moment_trim(alpha, 0.0, vmax)

        print("Max velocity parameters computed successfully.")
        print("Differential elevator trim: ", diff_trim,
              "Max velocity level flight throttle setting: ", throttle_setting,
              "Max sink rate at zero throttle: ", max_sink_rate,
              "at user-specified max velocity: ", vmax,
              "and angle of attack: ", alpha, '\n')

        return float(max_sink_trim), float(max_sink_rate), float(diff_trim), float(throttle_setting)

    def get_max_climb_params(self, airspeed):
        print("Computing max climb rate parameters...")

        def eom(initial_values):
            aoa, gamma = initial_values
            return (self.drag(aoa, airspeed) + self.mass * self.gravity * np.sin(gamma) - self.thrust(1.0) * np.cos(aoa),
                    self.lift(aoa, self.zero_moment_trim(aoa, 1.0, airspeed), airspeed) - self.mass *
                    self.gravity * np.cos(gamma) + self.thrust(1.0) * np.sin(aoa))

        alpha, gamma = fsolve(eom, [0.0, 0.0])
        elevator_trim = self.zero_moment_trim(alpha, 1.0, airspeed)

        max_climb_rate = - airspeed / (np.sqrt(1 + 1 / (np.tan(- gamma) ** 2)))

        print("Max climb rate parameters computed successfully.")
        print("Angle of attack: ", alpha, "Elevator trim: ", elevator_trim,
              "Speed for max climb rate: ", airspeed,
              "Max climb rate: ", max_climb_rate, '\n')
        return float(max_climb_rate), float(elevator_trim)

    # auxilary function to compute the level flight parameters with missing angle of attack (but known velocity)
    def get_level_flight_params(self, airspeed):
        def throttle(alpha):
            return self.drag(alpha, airspeed) / (self.aero_params['puller_ct'] * np.cos(alpha))

        def eom_z(initial_value):
            alpha = initial_value
            return self.lift(alpha, self.zero_moment_trim(alpha, throttle(alpha), airspeed), airspeed) + \
                self.thrust(throttle(alpha)) * np.sin(alpha) - \
                self.gravity * self.mass

        [alpha] = fsolve(eom_z, [0.2])
        throttle_setting = throttle(alpha)
        elevator_trim = self.zero_moment_trim(
            alpha, throttle_setting, airspeed)
        pitch_level_flight = alpha

        return throttle_setting, pitch_level_flight, elevator_trim

    # auxilary function to compute the flight speed from a given angle of attack and elevator trim at zero thrust
    def get_flight_vel_zero_thrust(self, alpha):
        """
        This function computes the flight speed that corresponds to a certain 
        elevator deflection, angle of attack and zero thrust, using the provided
        aerodynamic parameters.

        :param alpha: angle of attack

        :return: flight speed, flight path angle
        """
        def gilde_eom(initial_values):
            airspeed, gamma = initial_values
            return (gamma + np.arctan2(self.cD(alpha), self.cL(alpha)),
                    airspeed - np.sqrt((2 * self.mass * self.gravity) /
                    (self.rho * self.area * (self.cL(alpha) * np.cos(gamma) - self.cD(alpha) * np.sin(gamma)))))

        airspeed, flight_path_angle = fsolve(gilde_eom, [10.0, 0.2])

        return airspeed, flight_path_angle
