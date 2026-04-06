#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from ece346.Lab5.scripts.data_recorder import DataRecorder


def main(args=None):
    rclpy.init(args=args)

    recorder = DataRecorder()
    recorder.get_logger().info("Started data recorder node")
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    except Exception as e:
        if not rclpy.ok():
            pass
        else:
            raise
    finally:
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
