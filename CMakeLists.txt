cmake_minimum_required(VERSION 3.5 FATAL_ERROR)


# Find Eigen
find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(Eigen QUIET)
  if(NOT EIGEN_FOUND)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
  else()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
  endif()
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(yaml-cpp REQUIRED)

find_package(Boost 1.58 REQUIRED COMPONENTS system thread filesystem)
# Find Gazebo
find_package(gazebo REQUIRED)

add_compile_options(-Wno-deprecated-declarations -Wno-address-of-packed-member)

link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

set(PROTOBUF_IMPORT_DIRS "")
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(PROTOBUF_IMPORT_DIRS "${ITR}/gazebo/msgs/proto")
  endif()
endforeach()

# PROTOBUF_IMPORT_DIRS has to be set before
# find_package is called
find_package(Protobuf REQUIRED)
pkg_check_modules(PROTOBUF protobuf)

if ("${PROTOBUF_VERSION}" VERSION_LESS "2.5.0")
  message(FATAL_ERROR "protobuf version: ${PROTOBUF_VERSION} not compatible, must be >= 2.5.0")
endif()

set(GAZEBO_MSG_INCLUDE_DIRS)
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(GAZEBO_MSG_INCLUDE_DIRS "${ITR}/gazebo/msgs")
  endif()
endforeach()

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
  ${GAZEBO_MSG_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}/eigen3	# Workaround for Eigen3
  ${YAML_CPP_INCLUDE_DIR}
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo
  )

# Build our plugin
add_library(gazebo_fw_dynamics_plugin
  SHARED 
  src/gazebo_fw_dynamics_plugin.cpp
  src/parametric_dynamics_model.cpp
  )
target_link_libraries(gazebo_fw_dynamics_plugin 
  ${GAZEBO_LIBRARIES} 
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo/libmav_msgs.so
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo/libnav_msgs.so
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo/libphysics_msgs.so
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo/libsensor_msgs.so
  $ENV{PX4_ROOT}/build/px4_sitl_default/build_gazebo/libstd_msgs.so
  )

add_subdirectory(unit_tests)
