cmake_minimum_required(VERSION 3.10)
project(g1_locomotion_interfaces)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocomotionCommand.msg"
  "msg/LocomotionState.msg"
  "srv/AcquireLocomotion.srv"
  "srv/ReleaseLocomotion.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
