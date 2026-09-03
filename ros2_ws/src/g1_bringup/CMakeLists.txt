cmake_minimum_required(VERSION 3.10)
project(g1_bringup)

# ROS 2 Foxy's ament_cmake still uses FindPythonInterp. Newer CMake versions
# require this compatibility policy until Foxy's CMake modules are upgraded.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

ament_package()
