ros2 service call /simulation/reset racecar_interface/srv/Reset "{x: 2.0, y: 0.4, yaw: 0}"

 ros2 topic echo /safety/value 

 ros2 service call /simulation/reset racecar_interface/srv/Reset "{x: 2.0, y: 0.3, yaw: 0.0}"

 ros2 service call /simulation/reset racecar_interface/srv/Reset "{x: 2.0, y: 0.18, yaw: 0.0}"


 ros2 topic pub /human_control racecar_msgs/msg/ServoMsg "{throttle: 0.2, steer: 0.0, reverse: false}" -r 20

 ros2 service call /simulation/reset_static_obstacles racecar_interface/srv/ResetObstacle "{n: 10}"

 ros2 launch racecar_ece346 safety_filter_sim_launch.py 

 ros2 run joy joy_node