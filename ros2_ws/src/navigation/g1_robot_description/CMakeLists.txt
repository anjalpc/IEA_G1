cmake_minimum_required(VERSION 3.5)
project(g1_robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf vendor launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_description
    test/test_description.py
  )

  ament_add_pytest_test(
    test_install
    test/test_install.py
  )

  ament_add_pytest_test(
    test_runtime_package
    test/test_runtime_package.py
  )

  ament_add_pytest_test(
    test_runtime_launch
    test/test_runtime_launch.py
  )
endif()

ament_package()
