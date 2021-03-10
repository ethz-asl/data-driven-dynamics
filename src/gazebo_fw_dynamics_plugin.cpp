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

// MODULE HEADER
#include "gazebo_fw_dynamics_plugin.h"

namespace gazebo {

GazeboFwDynamicsPlugin::GazeboFwDynamicsPlugin()
    : ModelPlugin(),
      node_handle_(0),
      W_wind_speed_W_B_(0, 0, 0),
      delta_aileron_left_(0.0),
      delta_aileron_right_(0.0),
      delta_elevator_(0.0),
      delta_flap_(0.0),
      delta_rudder_(0.0),
      throttle_(0.0),
      pubs_and_subs_created_(false) {
}

GazeboFwDynamicsPlugin::~GazeboFwDynamicsPlugin() {
}

void GazeboFwDynamicsPlugin::Load(physics::ModelPtr _model,
                                  sdf::ElementPtr _sdf) {

  gzdbg << "_model = " << _model->GetName() << std::endl;

  // Store the pointer to the model.
  model_ = _model;
  world_ = model_->GetWorld();

  namespace_.clear();

  gzdbg << "dbg1" << std::endl;

  // Get the robot namespace.
  if (_sdf->HasElement("robotNamespace"))
    namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
  else
    gzerr << "[gazebo_fw_dynamics_plugin] Please specify a robotNamespace.\n";

  // Create the node handle.
  node_handle_ = transport::NodePtr(new transport::Node());

  // Initialise with default namespace (typically /gazebo/default/).
  node_handle_->Init();

  gzdbg << "dbg2" << std::endl;

  // Get the link name.
  std::string link_name;
  if (_sdf->HasElement("linkName"))
    link_name = _sdf->GetElement("linkName")->Get<std::string>();
  else
    gzerr << "[gazebo_fw_dynamics_plugin] Please specify a linkName.\n";
  // Get the pointer to the link.
  link_ = model_->GetLink(link_name);
  if (link_ == NULL) {
    gzthrow("[gazebo_fw_dynamics_plugin] Couldn't find specified link \""
            << link_name << "\".");
  }

  gzdbg << "dbg3" << std::endl;

  // Get the path to fixed-wing aerodynamics parameters YAML file. If not
  // provided, default Techpod parameters are used.
  if (_sdf->HasElement("aeroParamsYAML")) {
    gzdbg << "dbg3.0" << std::endl;
    std::string aero_params_yaml =
        _sdf->GetElement("aeroParamsYAML")->Get<std::string>();

    aero_params_.LoadAeroParamsYAML(aero_params_yaml);
  } else {
    gzwarn << "[gazebo_fw_dynamics_plugin] No aerodynamic paramaters YAML file"
        << " specified, using default Techpod parameters.\n";
  }

  // Get the path to fixed-wing vehicle parameters YAML file. If not provided,
  // default Techpod parameters are used.
  if (_sdf->HasElement("vehicleParamsYAML")) {
    std::string vehicle_params_yaml =
        _sdf->GetElement("vehicleParamsYAML")->Get<std::string>();

    vehicle_params_.LoadVehicleParamsYAML(vehicle_params_yaml);
  } else {
    gzwarn << "[gazebo_fw_dynamics_plugin] No vehicle paramaters YAML file"
        << " specified, using default Techpod parameters.\n";
  }

  std::string actuator_sub_topic_ = "/gazebo/command/motor_speed";
  if (_sdf->HasElement("commandSubTopic")) {
    actuator_sub_topic_ = _sdf->GetElement("commandSubTopic")->Get<std::string>();
  }

  wind_speed_sub_ = node_handle_->Subscribe("~/world_wind", &GazeboFwDynamicsPlugin::WindVelocityCallback, this);
  actuators_sub_ = node_handle_->Subscribe("~/" + model_->GetName() + actuator_sub_topic_, &GazeboFwDynamicsPlugin::ActuatorsCallback, this);

  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  this->updateConnection_ = event::Events::ConnectWorldUpdateBegin(
          boost::bind(&GazeboFwDynamicsPlugin::OnUpdate, this, _1));
}

void GazeboFwDynamicsPlugin::OnUpdate(const common::UpdateInfo& _info) {
  Eigen::Vector3d forces, moments;

  UpdateForcesAndMoments(forces, moments);

  ignition::math::Vector3d forces_msg = ignition::math::Vector3d (forces[0], forces[1], forces[2]);
  ignition::math::Vector3d moments_msg = ignition::math::Vector3d (moments[0], moments[1], moments[2]);

  link_->AddRelativeForce(forces_msg);
  link_->AddRelativeTorque(moments_msg);
}

void GazeboFwDynamicsPlugin::UpdateForcesAndMoments(Eigen::Vector3d &forces, Eigen::Vector3d &moments) {
  // Express the air speed and angular velocity in the body frame.
  // B denotes body frame and W world frame ... e.g., W_rot_W_B denotes
  // rotation of B wrt. W expressed in W.
  ignition::math::Quaterniond W_rot_W_B = link_->WorldPose().Rot();
  ignition::math::Vector3d B_air_speed_W_B = W_rot_W_B.RotateVectorReverse(
      link_->WorldLinearVel() - W_wind_speed_W_B_);
  ignition::math::Vector3d B_angular_velocity_W_B = link_->RelativeAngularVel();

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
  if (alpha > aero_params_.alpha_max)
    alpha = aero_params_.alpha_max;
  else if (alpha < aero_params_.alpha_min)
    alpha = aero_params_.alpha_min;

  // Pre-compute the common component in the force and moment calculations.
  const double q_bar_S = 0.5 * kAirDensity * V * V * vehicle_params_.wing_surface;

  // Combine some of the control surface deflections.
  double aileron_sum = delta_aileron_left_ + delta_aileron_right_;
  double aileron_diff = delta_aileron_left_ - delta_aileron_right_;
  double flap_sum = 2.0 * delta_flap_;
  double flap_diff = 0.0;

  // Compute the forces in the wind frame.
  const double drag = q_bar_S *
      (aero_params_.c_drag_alpha.dot(
           Eigen::Vector3d(1.0, alpha, alpha * alpha)) +
       aero_params_.c_drag_beta.dot(
           Eigen::Vector3d(0.0, beta, beta * beta)) +
       aero_params_.c_drag_delta_ail.dot(
           Eigen::Vector3d(0.0, aileron_sum, aileron_sum * aileron_sum)) +
       aero_params_.c_drag_delta_flp.dot(
           Eigen::Vector3d(0.0, flap_sum, flap_sum * flap_sum)));

  const double side_force = q_bar_S *
      (aero_params_.c_side_force_beta.dot(
           Eigen::Vector2d(0.0, beta)));

  const double lift = q_bar_S *
      (aero_params_.c_lift_alpha.dot(
           Eigen::Vector4d(1.0, alpha, alpha * alpha, alpha * alpha * alpha)) +
       aero_params_.c_lift_delta_ail.dot(
           Eigen::Vector2d(0.0, aileron_sum)) +
       aero_params_.c_lift_delta_flp.dot(
           Eigen::Vector2d(0.0, flap_sum)));

  const Eigen::Vector3d forces_Wind(-drag, side_force, -lift);

  // Non-dimensionalize the angular rates for inclusion in the computation of
  // moments. To avoid division by zero, there is a minimum air speed threshold
  // below which the values are zero.
  const double p_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      p * vehicle_params_.wing_span / (2.0 * V);
  const double q_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      q * vehicle_params_.chord_length / (2.0 * V);
  const double r_hat = (V < kMinAirSpeedThresh) ? 0.0 :
      r * vehicle_params_.wing_span / (2.0 * V);

  // Compute the moments in the wind frame.
  const double rolling_moment = q_bar_S * vehicle_params_.wing_span *
      (aero_params_.c_roll_moment_beta.dot(
           Eigen::Vector2d(0.0, beta)) +
       aero_params_.c_roll_moment_p.dot(
           Eigen::Vector2d(0.0, p_hat)) +
       aero_params_.c_roll_moment_r.dot(
           Eigen::Vector2d(0.0, r_hat)) +
       aero_params_.c_roll_moment_delta_ail.dot(
           Eigen::Vector2d(0.0, aileron_diff)) +
       aero_params_.c_roll_moment_delta_flp.dot(
           Eigen::Vector2d(0.0, flap_diff)));

  const double pitching_moment = q_bar_S * vehicle_params_.chord_length *
      (aero_params_.c_pitch_moment_alpha.dot(
           Eigen::Vector2d(1.0, alpha)) +
       aero_params_.c_pitch_moment_q.dot(
           Eigen::Vector2d(0.0, q_hat)) +
       aero_params_.c_pitch_moment_delta_elv.dot(
           Eigen::Vector2d(0.0, delta_elevator_)));

  const double yawing_moment = q_bar_S * vehicle_params_.wing_span *
      (aero_params_.c_yaw_moment_beta.dot(
           Eigen::Vector2d(0.0, beta)) +
       aero_params_.c_yaw_moment_r.dot(
           Eigen::Vector2d(0.0, r_hat)) +
       aero_params_.c_yaw_moment_delta_rud.dot(
           Eigen::Vector2d(0.0, delta_rudder_)));

  const Eigen::Vector3d moments_Wind(rolling_moment,
                                     pitching_moment,
                                     yawing_moment);

  // Compute the thrust force in the body frame.
  const double thrust = aero_params_.c_thrust.dot(
      Eigen::Vector3d(1.0, throttle_, throttle_ * throttle_));

  const Eigen::Vector3d force_thrust_B = thrust * Eigen::Vector3d(
      cos(vehicle_params_.thrust_inclination),
      0.0,
      sin(vehicle_params_.thrust_inclination));

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
  const Eigen::Vector3d forces_B = R_Wind_B_t * forces_Wind + force_thrust_B;
  const Eigen::Vector3d moments_B = R_Wind_B_t * moments_Wind;

  // Once again account for the difference between our body frame orientation
  // and the traditional aerodynamics frame.
  forces << forces_B[0], -forces_B[1], -forces_B[2];
  moments << moments_B[0], -moments_B[1], -moments_B[2];
}

double GazeboFwDynamicsPlugin::NormalizedInputToAngle(
    const ControlSurface& surface, double input) {
  return (surface.deflection_max + surface.deflection_min) * 0.5 +
      (surface.deflection_max - surface.deflection_min) * 0.5 * input;
}

void GazeboFwDynamicsPlugin::ActuatorsCallback(CommandMotorSpeedPtr &actuators_msg) {

  //TODO: Get channel information from yml file
  delta_aileron_left_ = -NormalizedInputToAngle(vehicle_params_.aileron_left,
      static_cast<double>(actuators_msg->motor_speed(5)));
  delta_aileron_right_ = -NormalizedInputToAngle(vehicle_params_.aileron_right,
      static_cast<double>(actuators_msg->motor_speed(6)));
  delta_elevator_ = -NormalizedInputToAngle(vehicle_params_.elevator,
      static_cast<double>(actuators_msg->motor_speed(7)));
  delta_flap_ = NormalizedInputToAngle(vehicle_params_.flap,
      static_cast<double>(actuators_msg->motor_speed(3)));
  delta_rudder_ = NormalizedInputToAngle(vehicle_params_.rudder,
      static_cast<double>(actuators_msg->motor_speed(2)));
  //TODO: Throttle is set to zero since force is applied outside of this plugin
  // throttle_ = actuators_msg->normalized(vehicle_params_.throttle_channel);
  throttle_ = 0.0;
}

void GazeboFwDynamicsPlugin::WindVelocityCallback(
    WindPtr& msg) {
   ignition::math::Vector3d wind_vel_ = ignition::math::Vector3d(msg->velocity().x(),
             msg->velocity().y(),
             msg->velocity().z());
}

GZ_REGISTER_MODEL_PLUGIN(GazeboFwDynamicsPlugin);

}  // namespace gazebo
