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

#ifndef ROTORS_GAZEBO_PLUGINS_FW_DYNAMICS_PLUGIN_H
#define ROTORS_GAZEBO_PLUGINS_FW_DYNAMICS_PLUGIN_H

#include <Eigen/Eigen>

#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/util/system.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>

#include "CommandMotorSpeed.pb.h"
#include "Wind.pb.h"

#include "aero_parameters.h"
#include "parametric_dynamics_model.h"

namespace gazebo {

typedef const boost::shared_ptr<const physics_msgs::msgs::Wind> WindPtr;
typedef const boost::shared_ptr<const mav_msgs::msgs::CommandMotorSpeed> CommandMotorSpeedPtr;

class DataDrivenDynamicsPlugin : public ModelPlugin {
 public:
  /// \brief    Constructor.
  DataDrivenDynamicsPlugin();

  /// \brief    Destructor.
  virtual ~DataDrivenDynamicsPlugin();

 protected:
  /// \brief    Called when the plugin is first created, and after the world
  ///           has been loaded. This function should not be blocking.
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

  /// \brief  	This gets called by the world update start event.
  void OnUpdate(const common::UpdateInfo&);

  /// \brief	Calculates the forces and moments to be applied to the
  ///           fixed-wing body.
  void UpdateForcesAndMoments(Eigen::Vector3d &forces, Eigen::Vector3d &moments);

  double NormalizedInputToAngle(const ControlSurface& surface, double input);

 private:

  /// \brief    Flag to indicate that gazebo_mavlink_interface plugin handles
  ///           routing of actuation data (instead of gazebo_ros_interface_plugin)
  bool use_gazebo_mavlink_interface_;
  /// \brief    Are the input commands coming from a joystick (as opposed to
  ///           a remote control via HIL interface, for example)?
  bool is_input_joystick_;

  /// \brief    Flag that is set to true once CreatePubsAndSubs() is called,
  ///           used to prevent CreatePubsAndSubs() from be called on every
  ///           OnUpdate().
  bool pubs_and_subs_created_{false};

  /// \brief    Transport namespace.
  std::string namespace_;
  /// \brief    Topic name for actuator commands.
  std::string actuators_sub_topic_;
  /// \brief    Topic name for roll_pitch_yawrate_thrust commands.
  std::string roll_pitch_yawrate_thrust_sub_topic_;
  /// \brief    Topic name for wind speed readings.
  std::string wind_speed_sub_topic_;

  /// \brief    Handle for the Gazebo node.
  transport::NodePtr node_handle_;

  /// \brief    Subscriber for receiving actuator commands.
  gazebo::transport::SubscriberPtr actuators_sub_;
  /// \brief    Subscriber for receiving roll_pitch_yawrate_thrust commands.
  gazebo::transport::SubscriberPtr roll_pitch_yawrate_thrust_sub_;
  /// \brief    Subscriber ror receiving wind speed readings.
  gazebo::transport::SubscriberPtr wind_speed_sub_;

  /// \brief    Pointer to the world.
  physics::WorldPtr world_;
  /// \brief    Pointer to the model.
  physics::ModelPtr model_;
  /// \brief    Pointer to the link.
  physics::LinkPtr link_;
  /// \brief    Pointer to the update event connection.
  event::ConnectionPtr updateConnection_;

  /// \brief    Most current wind speed reading [m/s].
  ignition::math::Vector3d W_wind_speed_W_B_;

  /// \brief    The physical properties of the aircraft.
  FWVehicleParameters vehicle_params_;

  /// \brief    The parametric model of the aircraft
  std::shared_ptr<ParametricDynamicsModel> parametric_model_;

  /// \brief    Left aileron deflection [rad].
  double delta_aileron_left_{0.0};
  /// \brief    Right aileron deflection [rad].
  double delta_aileron_right_{0.0};
  /// \brief    Elevator deflection [rad].
  double delta_elevator_{0.0};
  /// \brief    Flap deflection [rad].
  double delta_flap_{0.0};
  /// \brief    Rudder deflection [rad].
  double delta_rudder_{0.0};
  /// \brief    Throttle input, in range from 0 to 1.
  double throttle_{0.0};

  int num_input_channels = 16;
  Eigen::VectorXd actuator_inputs_;

  /// \brief    Processes the actuator commands.
  /// \details  Converts control surface actuator inputs into physical angles
  ///           before storing them and throttle values for later use in
  ///           calculation of forces and moments.
  void ActuatorsCallback(CommandMotorSpeedPtr& actuators_msg);

  /// \brief    Processes the wind speed readings.
  /// \details  Stores the most current wind speed reading for later use in
  ///           calculation of forces and moments.
  void WindVelocityCallback(WindPtr& msg);
};

}  // namespace gazebo

#endif // ROTORS_GAZEBO_PLUGINS_FW_DYNAMICS_PLUGIN_H
