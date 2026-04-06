#!/usr/bin/env python3
"""
Behavioral Cloning Evaluation Node (Lab5).

Loads a trained PyTorch model and uses it to drive the truck autonomously.
Publishes AckermannDriveStamped to /drive so the control_gate forwards it
when R2 is held on the joystick.

Services:
    /learning/start_eval  - Start autonomous driving
    /learning/stop_eval   - Stop autonomous driving (publish zero)
"""

import threading
import os
import pickle
import time

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty

from scipy.spatial.transform import Rotation as R

from .neural_network import BCNetwork
from .utils.realtime_buffer import RealtimeBuffer


class BCEval(Node):
    def __init__(self):
        super().__init__("bc_eval_node")

        self.declare_read_parameters()

        # Load normalization stats
        self.X_mean = np.zeros(self.input_size)
        self.X_std = np.ones(self.input_size)
        self.Y_mean = np.zeros(2)
        self.Y_std = np.ones(2)
        self.feature_indices = list(range(self.input_size))  # default: [0,1,2,3,4,5]

        if self.norm_path and os.path.exists(self.norm_path):
            with open(self.norm_path, "rb") as f:
                norm = pickle.load(f)
            self.X_mean = norm["X_mean"]
            self.X_std = norm["X_std"]
            self.Y_mean = norm["Y_mean"]
            self.Y_std = norm["Y_std"]
            self.feature_indices = norm["feature_indices"]
            self.input_size = norm["input_size"]
            self.hidden_sizes = norm["hidden_sizes"]
            self.get_logger().info(f"Loaded normalization from {self.norm_path}")

        # Load model
        self.model = BCNetwork(
            input_size=self.input_size,
            output_size=2,
            hidden_sizes=list(self.hidden_sizes),
        )
        if os.path.exists(self.model_path):
            self.model.load_model(self.model_path)
            self.get_logger().info(f"Loaded model from {self.model_path}")
        else:
            self.get_logger().error(f"Model not found at {self.model_path}")

        # State
        self.odom_buffer = RealtimeBuffer()
        self.running = False

        self.setup_publisher()
        self.setup_subscriber()
        self.setup_services()

        # Control loop thread
        threading.Thread(target=self.control_thread, daemon=True).start()

        self.get_logger().info(
            f"BC Eval node initialized.\n"
            f"  Odom topic:    {self.odom_topic}\n"
            f"  Drive topic:   {self.drive_topic}\n"
            f"  Model:         {self.model_path}\n"
            f"  Control rate:  {self.control_rate} Hz\n"
            f"  Use: ros2 service call /learning/start_eval std_srvs/srv/Empty\n"
            f"       (then hold R2 on joystick to enable autonomous mode)"
        )

    def declare_read_parameters(self):
        self.declare_parameter("odom_topic", "/SLAM/Pose")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("model_path", "")
        self.declare_parameter("norm_path", "")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("max_speed", 0.5)
        self.declare_parameter("max_steering", 0.34)
        self.declare_parameter("input_size", 6)
        self.declare_parameter("hidden_sizes", [64, 128, 128, 64])

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.norm_path = self.get_parameter("norm_path").get_parameter_value().string_value
        self.control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value
        self.max_steering = self.get_parameter("max_steering").get_parameter_value().double_value
        self.input_size = self.get_parameter("input_size").get_parameter_value().integer_value
        self.hidden_sizes = self.get_parameter("hidden_sizes").get_parameter_value().integer_array_value

    def setup_publisher(self):
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1
        )

    def setup_subscriber(self):
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

    def setup_services(self):
        self.start_eval_srv = self.create_service(
            Empty, "/learning/start_eval", self.start_eval_callback
        )
        self.stop_eval_srv = self.create_service(
            Empty, "/learning/stop_eval", self.stop_eval_callback
        )

    # ---- Callbacks ----

    def odom_callback(self, msg: Odometry):
        state = self.extract_state(msg)
        self.odom_buffer.writeFromNonRT(state)

    def extract_state(self, msg: Odometry) -> np.ndarray:
        """Extract [x, y, yaw, v_long, v_lat, omega] from Odometry."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        quat = [orient.x, orient.y, orient.z, orient.w]
        yaw = R.from_quat(quat).as_euler('xyz')[2]

        v_long = msg.twist.twist.linear.x
        v_lat = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        return np.array([pos.x, pos.y, yaw, v_long, v_lat, omega], dtype=np.float64)

    def state_to_features(self, state: np.ndarray) -> np.ndarray:
        """
        Convert full state to normalized model input features.
        Uses feature_indices and normalization stats from training.
        """
        raw = state[self.feature_indices]
        return (raw - self.X_mean) / self.X_std

    # ---- Service handlers ----

    def start_eval_callback(self, request, response):
        self.running = True
        self.get_logger().info("Evaluation STARTED. Hold R2 to drive autonomously.")
        return response

    def stop_eval_callback(self, request, response):
        self.running = False
        # Publish zero command
        self.publish_zero()
        self.get_logger().info("Evaluation STOPPED.")
        return response

    # ---- Control loop ----

    def control_thread(self):
        dt = 1.0 / self.control_rate

        while rclpy.ok():
            if not self.running:
                time.sleep(dt)
                continue

            state = self.odom_buffer.readFromRT()
            if state is None:
                time.sleep(dt)
                continue

            # Extract features, predict, and denormalize
            features = self.state_to_features(state)
            action_norm = self.model.predict(features)
            action = action_norm * self.Y_std + self.Y_mean

            # Clip to safety limits
            speed = float(np.clip(action[0], -self.max_speed, self.max_speed))
            steering = float(np.clip(action[1], -self.max_steering, self.max_steering))

            # Publish
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.speed = speed
            msg.drive.steering_angle = steering

            self.drive_pub.publish(msg)

            time.sleep(dt)

    def publish_zero(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)
