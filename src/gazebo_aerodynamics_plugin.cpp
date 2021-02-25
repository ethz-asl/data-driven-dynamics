/****************************************************************************
 *
 *   Copyright (c) 2020 PX4 Development Team. All rights reserved.
 *
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
 * 3. Neither the name PX4 nor the names of its contributors may be
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
 ****************************************************************************/
/**
 * @brief Aerodynmaics Plugin
 *
 * Model aerodynamic forces 
 *
 * @author Jaeyoung Lim <jalim@ethz.ch>
 */

#include <gazebo_aerodynamics_plugin.h>

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(AerodynamicsPlugin)

AerodynamicsPlugin::AerodynamicsPlugin() : ModelPlugin()
{ }

AerodynamicsPlugin::~AerodynamicsPlugin()
{
    if (updateConnection_)
      updateConnection_->~Connection();
}

void AerodynamicsPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  // Store the pointer to the model.
  model_ = _model;
  world_ = model_->GetWorld();

  if (_sdf->HasElement("robotNamespace")) {
    namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
  } else {
    gzerr << "[gazebo_aerodynamics_plugin] Please specify a robotNamespace.\n";
  }

  node_handle_ = transport::NodePtr(new transport::Node());
  node_handle_->Init(namespace_);

  if (_sdf->HasElement("linkName"))
    link_name_ = _sdf->GetElement("linkName")->Get<std::string>();
  else
    gzerr << "[gazebo_aerodynamics_plugin] Please specify a linkName.\n";
  // Get the pointer to the link
  link_ = model_->GetLink(link_name_);
  if (link_ == NULL)
    gzthrow("[gazebo_aerodynamics_plugin] Couldn't find specified link \"" << link_name_ << "\".");

  if (_sdf->HasElement("surface_area")) {
    surface_area_ = _sdf->GetElement("surface_area")->Get<double>();
  } else {
    gzerr << "[gazebo_aerodynamics_plugin] Please specify the surface area.\n";
  }
  std::string actuator_sub_topic_ = "/gazebo/command/motor_speed";
  if (_sdf->HasElement("commandSubTopic")) {
    actuator_sub_topic_ = _sdf->GetElement("commandSubTopic")->Get<std::string>();
  }

  if (_sdf->HasElement("center_of_pressure")) {
    center_of_pressure_ = _sdf->Get<ignition::math::Vector3d>("cp");
  }


  int n_out_max = 16;
  cl_delta_vector_ = Eigen::VectorXd::Zero(n_out_max);
  cd_delta_vector_ = Eigen::VectorXd::Zero(n_out_max);
  cm_delta_vector_ = Eigen::VectorXd::Zero(n_out_max);

  if (_sdf->HasElement("cl0")) {
    C_lift_.c0 = _sdf->Get<double>("cl0");
  }
  if (_sdf->HasElement("cl_alpha")) {
    C_lift_.c1 = _sdf->Get<double>("cl_alpha");
  }

  if (_sdf->HasElement("cd0")) {
    C_drag_.c0 = _sdf->Get<double>("cd0");
  }
  if (_sdf->HasElement("cd_alpha")) {
    C_drag_.c1 = _sdf->Get<double>("cd_alpha");
  }

  if (_sdf->HasElement("cm0")) {
    C_sideslip_.c0 = _sdf->Get<double>("cm0");
  }
  if (_sdf->HasElement("cm_alpha")) {
    C_drag_.c1 = _sdf->Get<double>("cm_alpha");
  }

  if (_sdf->HasElement("control_channels")) {
    sdf::ElementPtr control_channels = _sdf->GetElement("control_channels");
    sdf::ElementPtr channel = control_channels->GetElement("channel");

    while (channel) {
      if (channel->HasElement("input_index")) {
        int index = channel->Get<int>("input_index");
        if (index < n_out_max) {
          if (channel->HasElement("cl_delta")) {
            cl_delta_vector_[index] = channel->Get<double>("cl_delta");
          }
          if (channel->HasElement("cd_delta")) {
            cd_delta_vector_[index] = channel->Get<double>("cd_delta");
          }
          if (channel->HasElement("cm_delta")) {
            cm_delta_vector_[index] = channel->Get<double>("cm_delta");
          }
        }
      }
      channel = channel->GetNextElement("channel");
    }
  }

  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&AerodynamicsPlugin::OnUpdate, this, _1));

  wind_sub_ = node_handle_->Subscribe("~/world_wind", &AerodynamicsPlugin::WindVelocityCallback, this);
  actuator_sub_ = node_handle_->Subscribe("~/" + model_->GetName() + actuator_sub_topic_, &AerodynamicsPlugin::ActuatorCallback, this);
}

void AerodynamicsPlugin::OnUpdate(const common::UpdateInfo&){
#if GAZEBO_MAJOR_VERSION >= 9
  common::Time current_time = world_->SimTime();
#else
  common::Time current_time = world_->GetSimTime();
#endif

#if GAZEBO_MAJOR_VERSION >= 9
  ignition::math::Pose3d T_W_I = link_->WorldPose();
#else
  ignition::math::Pose3d T_W_I = ignitionFromGazeboMath(link_->GetWorldPose());
#endif
  ignition::math::Quaterniond C_W_I = T_W_I.Rot();

  //Calculate air_relative_velocity vector in body frame
#if GAZEBO_MAJOR_VERSION >= 9
  ignition::math::Vector3d air_relative_velocity = link_->RelativeLinearVel() - C_W_I.RotateVector(wind_vel_);
#else
  ignition::math::Vector3d air_relative_velocity = ignitionFromGazeboMath(link_->GetRelativeLinearVel()) - C_W_I.RotateVector(wind_vel_);
#endif

  static constexpr double air_density = 1.225f;
  //Calculate air relative attitudes
  ignition::math::Vector3d air_relative_velocity_unit_vector = air_relative_velocity.Normalize();
  ignition::math::Vector3d air_relative_velocity_LD_unit_vector = air_relative_velocity.Normalize();
  double alpha = atan2(air_relative_velocity_unit_vector.Z(), air_relative_velocity_unit_vector.X()); //Angle of attack
  double beta = atan2(air_relative_velocity_unit_vector.Y(), air_relative_velocity_unit_vector.X()); //Side slip angle

  const double dynamic_pressure = 0.005f * air_density * air_relative_velocity.Length() * air_relative_velocity.Length();
  const double q_bar_S = dynamic_pressure * surface_area_;
  
  const double lift = q_bar_S * getCoefficient(alpha, C_lift_);
  const double drag = q_bar_S * getCoefficient(alpha, C_drag_);
  const double sideslip = q_bar_S * getCoefficient(beta, C_sideslip_);

  ignition::math::Vector3d force = ignition::math::Vector3d(- drag, -sideslip, -lift) * air_relative_velocity_unit_vector;

  last_time_ = current_time;

  //TODO: Hook this up to a real flight
  // link_->AddForceAtRelativePosition(force, center_of_pressure_);
  // link_->AddTorque(moment);
}

double AerodynamicsPlugin::getCoefficient(double angle, const AeroDynamicCoefficient &coefficient) {
  return coefficient.c0 + coefficient.c1 * angle + cl_delta_vector_.dot(input_reference_);
}

void AerodynamicsPlugin::WindVelocityCallback(WindPtr& msg) {
   wind_vel_ = ignition::math::Vector3d(msg->velocity().x(),
             msg->velocity().y(),
             msg->velocity().z());
}

void AerodynamicsPlugin::ActuatorCallback(CommandMotorSpeedPtr &rot_velocities) {
  //TODO: Subscribe to actuators and filter out what is used and not used
}

} // namespace gazebo
