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
#include "data_driven_dynamics_plugin.h"

namespace gazebo {

DataDrivenDynamicsPlugin::DataDrivenDynamicsPlugin() : ModelPlugin(), node_handle_(0), W_wind_speed_W_B_(0, 0, 0) {
  actuator_inputs_ = Eigen::VectorXd::Zero(num_input_channels);
}

DataDrivenDynamicsPlugin::~DataDrivenDynamicsPlugin() {}

void DataDrivenDynamicsPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
  // Store the pointer to the model.
  model_ = _model;
  world_ = model_->GetWorld();

  namespace_.clear();

  // Get the robot namespace.
  if (_sdf->HasElement("robotNamespace"))
    namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
  else
    gzerr << "[gazebo_fw_dynamics_plugin] Please specify a robotNamespace.\n";

  // Create the node handle.
  node_handle_ = transport::NodePtr(new transport::Node());

  // Initialise with default namespace (typically /gazebo/default/).
  node_handle_->Init();

  // Get the link name.
  std::string link_name;
  if (_sdf->HasElement("linkName"))
    link_name = _sdf->GetElement("linkName")->Get<std::string>();
  else
    gzerr << "[gazebo_fw_dynamics_plugin] Please specify a linkName.\n";
  // Get the pointer to the link.
  link_ = model_->GetLink(link_name);
  if (link_ == NULL) {
    gzthrow("[gazebo_fw_dynamics_plugin] Couldn't find specified link \"" << link_name << "\".");
  }

  parametric_model_ = std::make_shared<ParametricDynamicsModel>();
  // Get the path to fixed-wing aerodynamics parameters YAML file. If not
  // provided, default Techpod parameters are used.
  if (_sdf->HasElement("aeroParamsYAML")) {
    std::string aero_params_yaml = _sdf->GetElement("aeroParamsYAML")->Get<std::string>();
    if (aero_params_yaml.at(0) != '/') {
      const char* root_dir = std::getenv("DATA_DRIVEN_DYNAMICS_ROOT");
      aero_params_yaml = std::string(root_dir) + "/" + aero_params_yaml;
      gzmsg << "Using relative path for model config: " << aero_params_yaml << std::endl;
    } else {
      gzmsg << "Using absolute path for model config: " << aero_params_yaml << std::endl;
    }
    parametric_model_->LoadAeroParams(aero_params_yaml);
  } else {
    gzwarn << "[gazebo_data_driven_dynamics_plugin] No aerodynamic paramaters YAML file"
           << " specified, using default Techpod parameters.\n";
  }

  std::string actuator_sub_topic_ = "/gazebo/command/motor_speed";
  if (_sdf->HasElement("commandSubTopic")) {
    actuator_sub_topic_ = _sdf->GetElement("commandSubTopic")->Get<std::string>();
  }

  wind_speed_sub_ = node_handle_->Subscribe("~/world_wind", &DataDrivenDynamicsPlugin::WindVelocityCallback, this);
  actuators_sub_ = node_handle_->Subscribe("~/" + model_->GetName() + actuator_sub_topic_,
                                           &DataDrivenDynamicsPlugin::ActuatorsCallback, this);

  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  this->updateConnection_ =
      event::Events::ConnectWorldUpdateBegin(boost::bind(&DataDrivenDynamicsPlugin::OnUpdate, this, _1));
}

void DataDrivenDynamicsPlugin::OnUpdate(const common::UpdateInfo& _info) {
  Eigen::Vector3d forces, moments;

  UpdateForcesAndMoments(forces, moments);

  ignition::math::Vector3d forces_msg = ignition::math::Vector3d(forces[0], forces[1], forces[2]);
  ignition::math::Vector3d moments_msg = ignition::math::Vector3d(moments[0], moments[1], moments[2]);

  link_->AddRelativeForce(forces_msg);
  link_->AddRelativeTorque(moments_msg);
}

void DataDrivenDynamicsPlugin::UpdateForcesAndMoments(Eigen::Vector3d& forces, Eigen::Vector3d& moments) {
  // Express the air speed and angular velocity in the body frame.
  // B denotes body frame and W world frame ... e.g., W_rot_W_B denotes
  // rotation of B wrt. W expressed in W.
  ignition::math::Quaterniond W_rot_W_B = link_->WorldPose().Rot();
  ignition::math::Vector3d B_air_speed_W_B = W_rot_W_B.RotateVectorReverse(link_->WorldLinearVel() - W_wind_speed_W_B_);
  ignition::math::Vector3d B_angular_velocity_W_B = link_->RelativeAngularVel();

  parametric_model_->setState(B_air_speed_W_B, B_angular_velocity_W_B, actuator_inputs_);

  const Eigen::Vector3d parametric_force = parametric_model_->getForce();
  const Eigen::Vector3d parametric_moment = parametric_model_->getMoment();

  // TODO: Implement Residual dynamics prediction
  const Eigen::Vector3d residual_force{Eigen::Vector3d::Zero()};
  const Eigen::Vector3d residual_moment{Eigen::Vector3d::Zero()};

  const Eigen::Vector3d forces_B = parametric_force + residual_force;
  const Eigen::Vector3d moments_B = parametric_moment + residual_moment;

  // Once again account for the difference between our body frame orientation
  // and the traditional aerodynamics frame.
  forces << forces_B[0], -forces_B[1], -forces_B[2];
  moments << moments_B[0], -moments_B[1], -moments_B[2];
}

double DataDrivenDynamicsPlugin::NormalizedInputToAngle(const ControlSurface& surface, double input) {
  return (surface.deflection_max + surface.deflection_min) * 0.5 +
         (surface.deflection_max - surface.deflection_min) * 0.5 * input;
}

void DataDrivenDynamicsPlugin::ActuatorsCallback(CommandMotorSpeedPtr& actuators_msg) {
  for (size_t i = 0; i < actuator_inputs_.size(); i++) {
    actuator_inputs_(i) = static_cast<double>(actuators_msg->motor_speed(i));
  }
}

void DataDrivenDynamicsPlugin::WindVelocityCallback(WindPtr& msg) {
  ignition::math::Vector3d wind_vel_ =
      ignition::math::Vector3d(msg->velocity().x(), msg->velocity().y(), msg->velocity().z());
}

GZ_REGISTER_MODEL_PLUGIN(DataDrivenDynamicsPlugin);

}  // namespace gazebo
