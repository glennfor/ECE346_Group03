cmake_minimum_required(VERSION 3.8)
project(racecar_ece346)

# Warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
#Packages needed from imports
find_package(builtin_interfaces REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

#Packages that were created
find_package(racecar_msgs REQUIRED)

#there are interfaces in this package
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetFRS.srv"
  DEPENDENCIES racecar_msgs
)

# --- Python package install ---
ament_python_install_package(ece346)

# Optional: Python executables
install(PROGRAMS
  ece346/Lab0/scripts/lab0_node.py
  ece346/lab0_truck/scripts/lab0_truck_node.py
  ece346/Lab1/scripts/traj_planning_node.py
  ece346/lab1_truck/scripts/lab1_truck_planning_node.py
  ece346/Lab3/scripts/Lab3_traj_planning_node.py
  ece346/Lab3/scripts/dyn_obs_node.py
  ece346/Lab5/scripts/data_recorder_node.py
  ece346/Lab5/scripts/bc_eval_node.py
  ece346/Final_Project/scripts/joy_to_servo_node.py
  ece346/Final_Project/scripts/safety_filter_node.py
  ece346/Final_Project/scripts/safety_viz_node.py
  ece346/ilqr_safety_filter/scripts/ilqr_safety_filter_node.py
  ece346/ilqr_safety_filter/scripts/ilqr_safety_viz_node.py
  ece346/cbf_filter/scripts/cbf_backup_planner_node.py
  ece346/cbf_filter/scripts/cbf_safety_monitor_node.py
  ece346/cbf_filter/scripts/cbf_safety_filter_qp_node.py
  DESTINATION lib/${PROJECT_NAME}
)



# FinalProject executables (prefixed with fp_ to avoid collisions)
# safety_filter_node.py is the student-edited file — installed via
# ament_python_install_package (symlinked). The fp_ wrapper imports it
# and is itself installed WITHOUT RENAME so --symlink-install works and
# edits are live without a rebuild.
install(PROGRAMS
  ece346/FinalProject/scripts/fp_safety_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  ece346/FinalProject/scripts/joy_to_ackermann_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fp_joy_to_ackermann_node.py
)
install(PROGRAMS
  ece346/FinalProject/scripts/drive_to_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fp_drive_to_servo_node.py
)
install(PROGRAMS
  ece346/FinalProject/scripts/obstacle_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fp_obstacle_detection_node.py
)
install(PROGRAMS
  ece346/FinalProject/scripts/static_obstacle_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fp_static_obstacle_publisher_node.py
)

# Data files

#config files
install(DIRECTORY
  ece346/Lab0/config
  ece346/lab0_truck/config
  ece346/Lab1/config
  ece346/lab1_truck/config
  ece346/Lab3/config
  ece346/Lab5/config
  ece346/FinalProject/config
  ece346/Final_Project/config
  ece346/ilqr_safety_filter/config
  ece346/cbf_filter/config
  DESTINATION share/${PROJECT_NAME}
)

#launch
install(DIRECTORY
  ece346/Lab0/launch
  ece346/lab0_truck/launch
  ece346/Lab1/launch
  ece346/lab1_truck/launch
  ece346/Lab3/launch
  ece346/Lab5/launch
  ece346/FinalProject/launch
  ece346/Final_Project/launch
  ece346/ilqr_safety_filter/launch
  ece346/cbf_filter/launch
  DESTINATION share/${PROJECT_NAME}
)


ament_export_dependencies(rosidl_default_runtime)
ament_package()
