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

#ifndef ROTORS_GAZEBO_PLUGINS_FW_PARAMETERS_H_
#define ROTORS_GAZEBO_PLUGINS_FW_PARAMETERS_H_

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <exception>
#include <gazebo/gazebo.hh>

namespace gazebo {

// Forward declaration.
struct ControlSurface;

// Default vehicle parameters (Techpod model)
static constexpr double kDefaultWingSpan = 2.59;
static constexpr double kDefaultWingSurface = 0.47;
static constexpr double kDefaultChordLength = 0.18;
static constexpr double kDefaultThrustInclination = 0.0;

// Default aerodynamic parameter values (Techpod model)
static constexpr double kDefaultAlphaMax = 0.27;
static constexpr double kDefaultAlphaMin = -0.27;

static const Eigen::Vector3d kDefaultCDragAlpha = Eigen::Vector3d(0.1360, -0.6737, 5.4546);
static const Eigen::Vector3d kDefaultCDragBeta = Eigen::Vector3d(0.0195, 0.0, -0.3842);
static const Eigen::Vector3d kDefaultCDragDeltaAil = Eigen::Vector3d(0.0195, 1.4205e-4, 7.5037e-6);
static const Eigen::Vector3d kDefaultCDragDeltaFlp = Eigen::Vector3d(0.0195, 2.7395e-4, 1.23e-5);

static const Eigen::Vector2d kDefaultCSideForceBeta = Eigen::Vector2d(0.0, -0.3073);

static const Eigen::Vector4d kDefaultCLiftAlpha = Eigen::Vector4d(0.2127, 10.8060, -46.8324, 60.6017);
static const Eigen::Vector2d kDefaultCLiftDeltaAil = Eigen::Vector2d(0.3304, 0.0048);
static const Eigen::Vector2d kDefaultCLiftDeltaFlp = Eigen::Vector2d(0.3304, 0.0073);

static const Eigen::Vector2d kDefaultCRollMomentBeta = Eigen::Vector2d(0.0, -0.0154);
static const Eigen::Vector2d kDefaultCRollMomentP = Eigen::Vector2d(0.0, -0.1647);
static const Eigen::Vector2d kDefaultCRollMomentR = Eigen::Vector2d(0.0, 0.0117);
static const Eigen::Vector2d kDefaultCRollMomentDeltaAil = Eigen::Vector2d(0.0, 0.0570);
static const Eigen::Vector2d kDefaultCRollMomentDeltaFlp = Eigen::Vector2d(0.0, 0.001);

static const Eigen::Vector2d kDefaultCPitchMomentAlpha = Eigen::Vector2d(0.0435, -2.9690);
static const Eigen::Vector2d kDefaultCPitchMomentQ = Eigen::Vector2d(-0.1173, -106.1541);
static const Eigen::Vector2d kDefaultCPitchMomentDeltaElv = Eigen::Vector2d(-0.1173, -6.1308);

static const Eigen::Vector2d kDefaultCYawMomentBeta = Eigen::Vector2d(0.0, 0.0430);
static const Eigen::Vector2d kDefaultCYawMomentR = Eigen::Vector2d(0.0, -0.0827);
static const Eigen::Vector2d kDefaultCYawMomentDeltaRud = Eigen::Vector2d(0.0, 0.06);

static const Eigen::Vector3d kDefaultCThrust = Eigen::Vector3d(0.0, 14.7217, 0.0);

// Default values for fixed-wing controls (Techpod model)
static constexpr double kDefaultControlSurfaceDeflectionMin = -20.0 * M_PI / 180.0;
static constexpr double kDefaultControlSurfaceDeflectionMax = 20.0 * M_PI / 180.0;

static constexpr int kDefaultAileronLeftChannel = 0;
static constexpr int kDefaultAileronRightChannel = 0;
static constexpr int kDefaultElevatorChannel = 1;
static constexpr int kDefaultFlapChannel = 4;
static constexpr int kDefaultRudderChannel = 2;
static constexpr int kDefaultThrottleChannel = 3;

/// \brief  This function reads a vector from a YAML node and converts it into
///         a vector of type Eigen.
template <typename Derived>
inline void YAMLReadEigenVector(const YAML::Node& node, const std::string& name, Eigen::MatrixBase<Derived>& value);

/// \brief  This function reads a parameter from a YAML node.
template <typename T>
inline void YAMLReadParam(const YAML::Node& node, const std::string& name, T& value);

/// \brief  Macros to reduce copies of names.
#define READ_EIGEN_VECTOR(node, item) YAMLReadEigenVector(node, #item, item);
#define READ_PARAM(node, item) YAMLReadParam(node, #item, item);

struct RotorParameters {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rotor_axis{Eigen::Vector3d(0.0, 0.0, 1.0)};
  double diameter{1.0};
  double turning_direction{1.0};
  double vertical_rot_drag_lin{0.07444735702448266};
  double vertical_rot_thrust_lin{-0.0017229667485354344};
  double vertical_rot_thrust_quad{4.0095427586089745};
  double vertical_c_m_leaver_quad{0.0};
  double vertical_c_m_leaver_lin{0.0};
  double vertical_c_m_drag_z_quad{0.0};
  double vertical_c_m_drag_z_lin{0.0};
  double vertical_c_m_rolling{0.0};
};
struct FWAerodynamicParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FWAerodynamicParameters() {}

  std::vector<RotorParameters> rotor_parameters_;
  double vertical_rot_drag_lin{0.07444735702448266};
  double vertical_rot_thrust_lin{-0.0017229667485354344};
  double vertical_rot_thrust_quad{4.0095427586089745};
  double vertical_c_m_drag_z_lin{-10.324851252250626};
  double vertical_c_m_drag_z_quad{6.0213939854338685};
  double vertical_c_m_leaver_lin{-8.33722923229799};
  double vertical_c_m_leaver_quad{32.623014913712176};
  double vertical_c_m_rolling{-1.467193735480539};

  void LoadAeroParamsYAML(const std::string& yaml_path) {
    const YAML::Node node = YAML::LoadFile(yaml_path);

    gzdbg << yaml_path << std::endl;
    gzdbg << "IsDefined" << node.IsDefined() << std::endl;
    gzdbg << "IsMap" << node.IsMap() << std::endl;
    gzdbg << "IsNull" << node.IsNull() << std::endl;
    gzdbg << "IsScalar" << node.IsScalar() << std::endl;
    gzdbg << "IsSequence" << node.IsSequence() << std::endl;

    try {
      const YAML::Node coefficients = node["coefficients"];
      READ_PARAM(coefficients, vertical_rot_drag_lin);
      READ_PARAM(coefficients, vertical_rot_thrust_lin);
      READ_PARAM(coefficients, vertical_rot_thrust_quad);
      READ_PARAM(coefficients, vertical_c_m_leaver_lin);
      READ_PARAM(coefficients, vertical_c_m_drag_z_lin);
      READ_PARAM(coefficients, vertical_c_m_drag_z_quad);
      READ_PARAM(coefficients, vertical_c_m_leaver_lin);
      READ_PARAM(coefficients, vertical_c_m_leaver_quad);
      READ_PARAM(coefficients, vertical_c_m_rolling);

      const YAML::Node configs = node["model"];
      /// TODO: iterate through yaml files and append rotor elements to model
      const YAML::Node rotors = configs["vertical_"];
      for (auto rotor : rotors) {
        gzdbg << rotor["description"] << std::endl;
        RotorParameters rotor_parameter;
        Eigen::Vector3d position, rotor_axis;
        double diameter, turning_direction;
        READ_EIGEN_VECTOR(rotor, position);
        READ_EIGEN_VECTOR(rotor, rotor_axis);
        READ_PARAM(rotor, turning_direction);
        // READ_PARAM(rotor, diameter);
        std::cout << "Reading yaml file: " << std::endl;
        rotor_parameter.position = position;
        rotor_parameter.rotor_axis = rotor_axis;
        rotor_parameter.turning_direction = turning_direction;
        rotor_parameter.vertical_rot_thrust_lin = vertical_rot_thrust_lin;
        rotor_parameter.vertical_rot_thrust_quad = vertical_rot_thrust_quad;
        rotor_parameter.vertical_rot_drag_lin = vertical_rot_drag_lin;
        rotor_parameter.vertical_c_m_leaver_lin = vertical_c_m_leaver_lin;
        rotor_parameter.vertical_c_m_leaver_quad = vertical_c_m_leaver_quad;
        rotor_parameter.vertical_c_m_drag_z_lin = vertical_c_m_drag_z_lin;
        rotor_parameter.vertical_c_m_drag_z_quad = vertical_c_m_drag_z_quad;
        rotor_parameter.vertical_c_m_leaver_lin = vertical_c_m_leaver_lin;
        rotor_parameter.vertical_c_m_rolling = vertical_c_m_rolling;
        // rotor_parameter.diameter = diameter;
        rotor_parameters_.push_back(rotor_parameter);
      }

    } catch (const std::exception& ex) {
      gzerr << ex.what() << std::endl;
    } catch (const std::string& ex) {
      gzerr << ex << std::endl;
    } catch (...) {
      gzerr << "meeep" << std::endl;
    }
  }
};

struct ControlSurface {
  ControlSurface(int cs_channel, double defl_min = kDefaultControlSurfaceDeflectionMin,
                 double defl_max = kDefaultControlSurfaceDeflectionMax)
      : channel(cs_channel), deflection_min(defl_min), deflection_max(defl_max) {}

  int channel;

  double deflection_min;
  double deflection_max;

  void LoadControlSurfaceNode(const YAML::Node& node) {
    READ_PARAM(node, channel);
    READ_PARAM(node, deflection_min);
    READ_PARAM(node, deflection_max);
  }
};

template <typename Derived>
inline void YAMLReadEigenVector(const YAML::Node& node, const std::string& name, Eigen::MatrixBase<Derived>& value) {
  std::vector<typename Derived::RealScalar> vec = node[name].as<std::vector<typename Derived::RealScalar>>();
  assert(vec.size() == Derived::SizeAtCompileTime);
  value = Eigen::Map<Derived>(&vec[0], vec.size());
}

template <typename T>
inline void YAMLReadParam(const YAML::Node& node, const std::string& name, T& value) {
  value = node[name].as<T>();
}

}  // namespace gazebo

#endif /* ROTORS_GAZEBO_PLUGINS_FW_PARAMETERS_H_ */
