#!/usr/bin/env python3
"""
Data Recorder for Behavioral Cloning (Lab5).

Subscribes to odometry and teleop (joystick) topics, records synchronized
state-action pairs, and saves them to pickle files for offline training.

Recording is gated by joystick activity: data is only recorded while /teleop
messages are actively arriving (i.e., L2 is held). When L2 is released,
/teleop stops publishing, and recording automatically pauses. This means you
can call start_record once, then just drive — release L2 to pause, hold L2
to resume. Call save_data whenever you're done.

Services:
    /learning/start_record  - Arm recording (records while L2 held)
    /learning/stop_record   - Disarm recording
    /learning/save_data     - Save recorded data to disk
"""

import threading
import time
import pickle
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty

from scipy.spatial.transform import Rotation as R

from .utils.realtime_buffer import RealtimeBuffer


class DataRecorder(Node):
    def __init__(self):
        super().__init__("data_recorder_node")

        self.declare_read_parameters()

        # Buffers for latest messages
        self.odom_buffer = RealtimeBuffer()
        self.control_buffer = RealtimeBuffer()

        # Recording state
        self.armed = False  # set by service call
        self.recording_lock = threading.Lock()
        self.last_control_time = 0.0  # timestamp of last /teleop message
        self.control_timeout = 0.3    # seconds — if no teleop msg, assume L2 released

        # Collected data
        self.recorded_states = []
        self.recorded_actions = []
        self.recorded_timestamps = []

        self.setup_subscribers()
        self.setup_services()

        self.get_logger().info(
            f"Data recorder initialized.\n"
            f"  Odom topic:    {self.odom_topic}\n"
            f"  Control topic: {self.control_topic}\n"
            f"  Save dir:      {self.save_dir}\n"
            f"  Workflow:\n"
            f"    1. ros2 service call /learning/start_record std_srvs/srv/Empty\n"
            f"    2. Drive with L2 held — recording happens automatically\n"
            f"       Release L2 to pause, hold L2 to resume\n"
            f"    3. ros2 service call /learning/save_data std_srvs/srv/Empty"
        )

    def declare_read_parameters(self):
        self.declare_parameter("odom_topic", "/SLAM/Pose")
        self.declare_parameter("control_topic", "/teleop")
        self.declare_parameter("save_dir", "")

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.control_topic = self.get_parameter("control_topic").get_parameter_value().string_value
        save_dir = self.get_parameter("save_dir").get_parameter_value().string_value

        if save_dir == "":
            # Default: save to /ros2_ws/src Lab5/data (source tree, survives rebuilds)
            self.save_dir = "/ros2_ws/src/racecar_ece346/ece346/Lab5/data"
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def setup_subscribers(self):
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.control_sub = self.create_subscription(
            AckermannDriveStamped,
            self.control_topic,
            self.control_callback,
            10,
        )

    def setup_services(self):
        self.start_record_srv = self.create_service(
            Empty, "/learning/start_record", self.start_record_callback
        )
        self.stop_record_srv = self.create_service(
            Empty, "/learning/stop_record", self.stop_record_callback
        )
        self.save_data_srv = self.create_service(
            Empty, "/learning/save_data", self.save_data_callback
        )

    # ---- Callbacks ----

    @property
    def control_active(self):
        """True if /teleop messages are arriving (L2 is held)."""
        return (time.time() - self.last_control_time) < self.control_timeout

    def odom_callback(self, msg: Odometry):
        state = self.extract_state(msg)
        self.odom_buffer.writeFromNonRT(state)

        # Only record when armed AND joystick is actively sending commands
        if self.armed and self.control_active:
            control = self.control_buffer.readFromRT()
            if control is not None:
                with self.recording_lock:
                    self.recorded_states.append(state)
                    self.recorded_actions.append(control)
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    self.recorded_timestamps.append(t)

    def control_callback(self, msg: AckermannDriveStamped):
        self.last_control_time = time.time()
        action = self.extract_action(msg)
        self.control_buffer.writeFromNonRT(action)

    # ---- State/Action extraction ----

    def extract_state(self, msg: Odometry) -> np.ndarray:
        """
        Extract state vector from Odometry message.

        Returns: [x, y, yaw, v_long, v_lat, omega]
        """
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        quat = [orient.x, orient.y, orient.z, orient.w]
        yaw = R.from_quat(quat).as_euler('xyz')[2]

        v_long = msg.twist.twist.linear.x
        v_lat = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        return np.array([pos.x, pos.y, yaw, v_long, v_lat, omega], dtype=np.float64)

    def extract_action(self, msg: AckermannDriveStamped) -> np.ndarray:
        """
        Extract action vector from AckermannDriveStamped message.

        Returns: [speed, steering_angle]
        """
        return np.array(
            [msg.drive.speed, msg.drive.steering_angle], dtype=np.float64
        )

    # ---- Service handlers ----

    def start_record_callback(self, request, response):
        self.armed = True
        self.get_logger().info(
            "Recording ARMED. Hold L2 and drive — data records automatically.\n"
            "Release L2 to pause. Call save_data when done."
        )
        return response

    def stop_record_callback(self, request, response):
        self.armed = False
        with self.recording_lock:
            n = len(self.recorded_states)
        self.get_logger().info(f"Recording DISARMED. {n} samples collected so far.")
        return response

    def save_data_callback(self, request, response):
        self.armed = False

        with self.recording_lock:
            if len(self.recorded_states) == 0:
                self.get_logger().warn("No data to save!")
                return response

            states = np.array(self.recorded_states)
            actions = np.array(self.recorded_actions)
            timestamps = np.array(self.recorded_timestamps)

        data = {
            "states": states,           # (N, 6): [x, y, yaw, v_long, v_lat, omega]
            "actions": actions,          # (N, 2): [speed, steering_angle]
            "timestamps": timestamps,    # (N,)
            "state_labels": ["x", "y", "yaw", "v_long", "v_lat", "omega"],
            "action_labels": ["speed", "steering_angle"],
        }

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bc_data_{timestamp_str}.pkl"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Clear buffer after saving
        with self.recording_lock:
            self.recorded_states.clear()
            self.recorded_actions.clear()
            self.recorded_timestamps.clear()

        self.get_logger().info(
            f"Saved {states.shape[0]} samples to {filepath}\n"
            f"  States shape:  {states.shape}\n"
            f"  Actions shape: {actions.shape}\n"
            f"  Buffer cleared."
        )

        return response
