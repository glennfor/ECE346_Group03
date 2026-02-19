#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interface.simulator.simulator import Simulator
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from ece346.Lab1.scripts.traj_planner import TrajectoryPlanner

def main(args = None):
    rclpy.init(args = args)

    planner = TrajectoryPlanner()
    planner.get_logger().info("Started trajectory planning node")
    try:    
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException: #context shutdown from outside, ros2 launch
        pass
    except Exception as e:
        #ignore shutdown races
        if not rclpy.ok():
            pass
        else:
            raise
    finally:
        planner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

