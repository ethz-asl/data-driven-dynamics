/*
 * Copyright 2017 Pavel Vechersky, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parametric_dynamics_model.h"
#include <iostream>

namespace gazebo {

ParametricDynamicsModel::ParametricDynamicsModel() {
  aero_params_ = std::make_shared<FWAerodynamicParameters>();
  vehicle_params_ = std::make_shared<FWVehicleParameters>();
}

ParametricDynamicsModel::~ParametricDynamicsModel() {}

void ParametricDynamicsModel::setState(const ignition::math::Vector3d &B_air_speed_W_B,
                                       const ignition::math::Vector3d &B_angular_velocity_W_B,
                                       double delta_aileron_left, double delta_aileron_right, double delta_elevator,
                                       double delta_flap, double delta_rudder, const Eigen::VectorXd actuator_inputs) {
  // Traditionally, fixed-wing aerodynamics use NED (North-East-Down) frame,
  // but since our model's body frame is in North-West-Up frame we rotate the
  // linear and angular velocities by 180 degrees around the X axis.
  double u = B_air_speed_W_B.X();
  double v = -B_air_speed_W_B.Y();
  double w = -B_air_speed_W_B.Z();

  double p = B_angular_velocity_W_B.X();
  double q = -B_angular_velocity_W_B.Y();
  double r = -B_angular_velocity_W_B.Z();

  Eigen::Vector3d force_rotor_B{Eigen::Vector3d::Zero()};
  Eigen::Vector3d moment_rotor_B{Eigen::Vector3d::Zero()};
  computeTotalRotorWrench(ignition2eigen(B_air_speed_W_B), actuator_inputs, force_rotor_B, moment_rotor_B);

  // Transform all the forces and moments into the body frame
  force_ = force_rotor_B;
  moment_ = moment_rotor_B;
}

void ParametricDynamicsModel::computeTotalRotorWrench(const Eigen::Vector3d airspeed,
                                                      const Eigen::VectorXd &actuator_inputs,
                                                      Eigen::Vector3d &rotor_force, Eigen::Vector3d &rotor_moment) {
  rotor_force = Eigen::Vector3d::Zero();
  rotor_moment = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < aero_params_->rotor_parameters_.size(); i++) {
    Eigen::Vector3d single_rotor_force =
        computeRotorForce(airspeed, actuator_inputs[i], aero_params_->rotor_parameters_[i]);
    Eigen::Vector3d single_rotor_moment =
        computeRotorMoment(airspeed, actuator_inputs[i], aero_params_->rotor_parameters_[i], single_rotor_force);
    rotor_force += single_rotor_force;
    rotor_moment += single_rotor_moment;
  }
}

Eigen::Vector3d ParametricDynamicsModel::computeRotorForce(const Eigen::Vector3d airspeed, const double actuator_input,
                                                           const RotorParameters &rotor_params) {
  if (!std::isfinite(actuator_input)) return Eigen::Vector3d::Zero();
  // Thrust force computation
  const double prop_diameter = rotor_params.diameter;
  const double thrust_lin = rotor_params.vertical_rot_thrust_lin;
  const double thrust_quad = rotor_params.vertical_rot_thrust_quad;

  Eigen::Vector3d rotor_axis = (rotor_params.rotor_axis).normalized();

  Eigen::Vector3d v_airspeed_parallel_to_rotor_axis = airspeed.dot(rotor_axis) * rotor_axis;
  Eigen::Vector3d v_airspeed_vertical_to_rotor_axis = airspeed - v_airspeed_parallel_to_rotor_axis;

  /// TODO: Compensate for angular rates
  Eigen::Vector3d rotor_thrust = ((thrust_lin * v_airspeed_parallel_to_rotor_axis.norm() * actuator_input +
                                   thrust_quad * std::pow(actuator_input, 2) * prop_diameter)) *
                                 kAirDensity * std::pow(prop_diameter, 3) * rotor_axis;
  Eigen::Vector3d rotor_drag = Eigen::Vector3d::Zero();

  return rotor_thrust + rotor_drag;
}

Eigen::Vector3d ParametricDynamicsModel::computeRotorMoment(const Eigen::Vector3d airspeed, const double actuator_input,
                                                            const RotorParameters &rotor_params,
                                                            Eigen::Vector3d rotor_force) {
  // Thrust force computation
  // const double prop_diameter = rotor_params.diameter;
  // const double c_m_leaver_quad = rotor_params.c_m_leaver_quad;
  // const double c_m_leaver_lin = rotor_params.c_m_leaver_lin;
  // const double c_m_drag_z_quad = rotor_params.c_m_drag_z_quad;
  // const double c_m_drag_z_lin = rotor_params.c_m_drag_z_lin;
  // const double c_m_rolling = rotor_params.c_m_rolling;

  Eigen::Vector3d rotor_axis = (rotor_params.rotor_axis).normalized();
  Eigen::Vector3d rotor_position = rotor_params.position;

  Eigen::Vector3d v_airspeed_parallel_to_rotor_axis = airspeed.dot(rotor_axis) * rotor_axis;
  Eigen::Vector3d v_airspeed_vertical_to_rotor_axis = airspeed - v_airspeed_parallel_to_rotor_axis;

  // Eigen::Vector3d moment_drag = Eigen::Vector3d::Zero();
  // Eigen::Vector3d moment_rolling = Eigen::Vector3d::Zero();
  // return moment_drag + moment_rolling;

  /// TODO: Using rotor forces to calculate moment generation as a wip patch
  Eigen::Vector3d rotor_moment = rotor_position.cross(rotor_force);

  return rotor_moment;
}
}  // namespace gazebo
