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

namespace gazebo {

ParametricDynamicsModel::ParametricDynamicsModel() {
    aero_params_ = std::make_shared<FWAerodynamicParameters>();
    vehicle_params_ = std::make_shared<FWVehicleParameters>();

}

ParametricDynamicsModel::~ParametricDynamicsModel() {}

void ParametricDynamicsModel::setState(const ignition::math::Vector3d &B_air_speed_W_B, const ignition::math::Vector3d &B_angular_velocity_W_B, 
    double delta_aileron_left, double delta_aileron_right, double delta_elevator, double delta_flap, double delta_rudder) {
  // Traditionally, fixed-wing aerodynamics use NED (North-East-Down) frame,
  // but since our model's body frame is in North-West-Up frame we rotate the
  // linear and angular velocities by 180 degrees around the X axis.
  double u = B_air_speed_W_B.X();
  double v = -B_air_speed_W_B.Y();
  double w = -B_air_speed_W_B.Z();

  double p = B_angular_velocity_W_B.X();
  double q = -B_angular_velocity_W_B.Y();
  double r = -B_angular_velocity_W_B.Z();

  // Compute the angle of attack (alpha) and the sideslip angle (beta). To
  // avoid division by zero, there is a minimum air speed threshold below which
  // alpha and beta are zero.
  double V = B_air_speed_W_B.Length();
  double beta = (V < kMinAirSpeedThresh) ? 0.0 : asin(v / V);
  double alpha = (u < kMinAirSpeedThresh) ? 0.0 : atan(w / u);

  // Bound the angle of attack.
  if (alpha > aero_params_->alpha_max)
    alpha = aero_params_->alpha_max;
  else if (alpha <aero_params_->alpha_min)
    alpha =aero_params_->alpha_min;

  // Pre-compute the common component in the force and moment calculations.
  const double q_bar_S = 0.5 * kAirDensity * V * V * vehicle_params_->wing_surface;

  // Combine some of the control surface deflections.
  double aileron_sum = delta_aileron_left + delta_aileron_right;
  double aileron_diff = delta_aileron_left - delta_aileron_right;
  double flap_sum = 2.0 * delta_flap;
  double flap_diff = 0.0;

  // Compute the forces in the wind frame.
  const double drag = q_bar_S *
      (aero_params_->c_drag_alpha.dot(
           Eigen::Vector3d(1.0, alpha, alpha * alpha)) +
      aero_params_->c_drag_beta.dot(
           Eigen::Vector3d(0.0, beta, beta * beta)) +
      aero_params_->c_drag_delta_ail.dot(
           Eigen::Vector3d(0.0, aileron_sum, aileron_sum * aileron_sum)) +
      aero_params_->c_drag_delta_flp.dot(
           Eigen::Vector3d(0.0, flap_sum, flap_sum * flap_sum)));

  const double side_force = q_bar_S *
      (aero_params_->c_side_force_beta.dot(
           Eigen::Vector2d(0.0, beta)));

  const double lift = q_bar_S *
      (aero_params_->c_lift_alpha.dot(
           Eigen::Vector4d(1.0, alpha, alpha * alpha, alpha * alpha * alpha)) +
      aero_params_->c_lift_delta_ail.dot(
           Eigen::Vector2d(0.0, aileron_sum)) +
      aero_params_->c_lift_delta_flp.dot(
           Eigen::Vector2d(0.0, flap_sum)));

  const Eigen::Vector3d forces_Wind(-drag, side_force, -lift);

  // Non-dimensionalize the angular rates for inclusion in the computation of
  // moments. To avoid division by zero, there is a minimum air speed threshold
  // below which the values are zero.
  const double p_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      p * vehicle_params_->wing_span / (2.0 * V);
  const double q_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      q * vehicle_params_->chord_length / (2.0 * V);
  const double r_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      r * vehicle_params_->wing_span / (2.0 * V);

  // Compute the moments in the wind frame.
  const double rolling_moment = q_bar_S * vehicle_params_->wing_span *
      (aero_params_->c_roll_moment_beta.dot(
           Eigen::Vector2d(0.0, beta)) +
      aero_params_->c_roll_moment_p.dot(
           Eigen::Vector2d(0.0, p_hat)) +
      aero_params_->c_roll_moment_r.dot(
           Eigen::Vector2d(0.0, r_hat)) +
      aero_params_->c_roll_moment_delta_ail.dot(
           Eigen::Vector2d(0.0, aileron_diff)) +
      aero_params_->c_roll_moment_delta_flp.dot(
           Eigen::Vector2d(0.0, flap_diff)));

  const double pitching_moment = q_bar_S * vehicle_params_->chord_length *
      (aero_params_->c_pitch_moment_alpha.dot(
           Eigen::Vector2d(1.0, alpha)) +
      aero_params_->c_pitch_moment_q.dot(
           Eigen::Vector2d(0.0, q_hat)) +
      aero_params_->c_pitch_moment_delta_elv.dot(
           Eigen::Vector2d(0.0, delta_elevator)));

  const double yawing_moment = q_bar_S * vehicle_params_->wing_span *
      (aero_params_->c_yaw_moment_beta.dot(
           Eigen::Vector2d(0.0, beta)) +
      aero_params_->c_yaw_moment_r.dot(
           Eigen::Vector2d(0.0, r_hat)) +
      aero_params_->c_yaw_moment_delta_rud.dot(
           Eigen::Vector2d(0.0, delta_rudder)));

  const Eigen::Vector3d moments_Wind(rolling_moment,
                                     pitching_moment,
                                     yawing_moment);

  // Compute the thrust force in the body frame.
  const double thrust =aero_params_->c_thrust.dot(
      Eigen::Vector3d(1.0, throttle_, throttle_ * throttle_));

  const Eigen::Vector3d force_thrust_B = thrust * Eigen::Vector3d(
      cos(vehicle_params_->thrust_inclination),
      0.0,
      sin(vehicle_params_->thrust_inclination));

  // Compute the transform between the body frame and the wind frame.
  double ca = cos(alpha);
  double sa = sin(alpha);
  double cb = cos(beta);
  double sb = sin(beta);

  Eigen::Matrix3d R_Wind_B;
  R_Wind_B << ca * cb, sb, sa * cb,
              -sb * ca, cb, -sa * sb,
              -sa, 0.0, ca;

  const Eigen::Matrix3d R_Wind_B_t = R_Wind_B.transpose();

  // Transform all the forces and moments into the body frame
  force_ = R_Wind_B_t * forces_Wind + force_thrust_B;
  moment_ = R_Wind_B_t * moments_Wind;
}
}
