cmake_minimum_required(VERSION 3.8)
project(racecar_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


#message files
set(MSG_FILES
  "msg/OdometryArray.msg"
  "msg/ServoMsg.msg"
  "msg/SetArray.msg"
  "msg/AprilTagDetection.msg"
  "msg/AprilTagDetectionArray.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  DEPENDENCIES std_msgs nav_msgs geometry_msgs
)



# Export dependencies
# In ROS 2, message packages must export their runtime dependencies.
# This ensures that other packages can find and use the generated messages.
ament_export_dependencies(rosidl_default_runtime)
ament_package() # generate CMake config files.