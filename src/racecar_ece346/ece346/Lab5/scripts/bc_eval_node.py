#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from ece346.Lab5.scripts.bc_eval import BCEval


def main(args=None):
    rclpy.init(args=args)

    evaluator = BCEval()
    evaluator.get_logger().info("Started BC evaluation node")
    try:
        rclpy.spin(evaluator)
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
        evaluator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
