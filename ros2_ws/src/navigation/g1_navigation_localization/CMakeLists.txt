cmake_minimum_required(VERSION 3.5)
project(g1_navigation_localization)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch maps
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_package_contract
    test/test_package_contract.py
  )

  ament_add_pytest_test(
    test_localization_contract
    test/test_localization_contract.py
  )

  ament_add_pytest_test(
    test_map_asset
    test/test_map_asset.py
  )
endif()

ament_package()
