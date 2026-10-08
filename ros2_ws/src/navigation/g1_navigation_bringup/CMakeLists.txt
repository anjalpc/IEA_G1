cmake_minimum_required(VERSION 3.5)
project(g1_navigation_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_package_contract
    test/test_package_contract.py
  )

  ament_add_pytest_test(
    test_navigation_contract
    test/test_navigation_contract.py
  )

  ament_add_pytest_test(
    test_control_integration_contract
    test/test_control_integration_contract.py
  )

  ament_add_pytest_test(
    test_navigation_simulation_launch
    test/test_navigation_simulation_launch.py
  )
endif()

ament_package()
