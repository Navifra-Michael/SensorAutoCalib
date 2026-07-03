#!/usr/bin/env python3

import copy
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_to_euler(q):
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def odom_position(msg):
    position = msg.pose.pose.position
    return np.array(
        [float(position.x), float(position.y), float(position.z)],
        dtype=np.float64,
    )


def declare_vector_parameter(node, name, default):
    node.declare_parameter(name, list(default))
    value = node.get_parameter(name).value
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")
    return np.asarray(value, dtype=np.float64)


class OdomNoisy(Node):
    def __init__(self):
        super().__init__("odom_noisy")

        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/odom_noisy")
        self.declare_parameter("noise_mode", "both")
        self.declare_parameter("seed", -1)
        self.declare_parameter("apply_to_twist", False)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.noise_mode = str(self.get_parameter("noise_mode").value).lower()
        self.apply_to_twist = bool(self.get_parameter("apply_to_twist").value)

        valid_modes = {"none", "jitter", "drift", "both"}
        if self.noise_mode not in valid_modes:
            raise ValueError(f"noise_mode must be one of: {sorted(valid_modes)}")

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.declare_parameter("jitter.frame_interval", 1)
        self.jitter_frame_interval = max(
            int(self.get_parameter("jitter.frame_interval").value),
            1,
        )
        self.jitter_position_std = declare_vector_parameter(
            self,
            "jitter.position_std",
            [0.0, 0.0, 0.0],
        )
        self.jitter_rpy_std = declare_vector_parameter(
            self,
            "jitter.rpy_std",
            [0.0, 0.0, 0.0],
        )

        self.declare_parameter("drift.reference_distance_m", 0.0)
        self.drift_reference_distance_m = float(
            self.get_parameter("drift.reference_distance_m").value
        )
        self.drift_position_error_at_reference = declare_vector_parameter(
            self,
            "drift.position_error_at_reference",
            [0.0, 0.0, 0.0],
        )
        self.drift_rpy_error_at_reference = declare_vector_parameter(
            self,
            "drift.rpy_error_at_reference",
            [0.0, 0.0, 0.0],
        )
        self.drift_position_rate = declare_vector_parameter(
            self,
            "drift.position_rate",
            [0.0, 0.0, 0.0],
        )
        self.drift_rpy_rate = declare_vector_parameter(
            self,
            "drift.rpy_rate",
            [0.0, 0.0, 0.0],
        )
        self.drift_position_random_walk_std = declare_vector_parameter(
            self,
            "drift.position_random_walk_std",
            [0.0, 0.0, 0.0],
        )
        self.drift_rpy_random_walk_std = declare_vector_parameter(
            self,
            "drift.rpy_random_walk_std",
            [0.0, 0.0, 0.0],
        )
        self.drift_max_abs_position = declare_vector_parameter(
            self,
            "drift.max_abs_position",
            [0.0, 0.0, 0.0],
        )
        self.drift_max_abs_rpy = declare_vector_parameter(
            self,
            "drift.max_abs_rpy",
            [0.0, 0.0, 0.0],
        )

        self.frame_count = 0
        self.current_jitter_position = np.zeros(3, dtype=np.float64)
        self.current_jitter_rpy = np.zeros(3, dtype=np.float64)
        self.drift_position = np.zeros(3, dtype=np.float64)
        self.drift_rpy = np.zeros(3, dtype=np.float64)
        self.last_stamp_sec = None
        self.last_odom_position = None

        self.publisher = self.create_publisher(Odometry, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            10,
        )

        self.get_logger().info(
            "odom_noisy: "
            f"{self.input_topic} -> {self.output_topic}, mode={self.noise_mode}"
        )

    def odom_callback(self, msg):
        noisy_msg = copy.deepcopy(msg)
        dt = self.message_dt(msg)
        step_distance = self.message_step_distance(msg)
        self.frame_count += 1

        position_noise = np.zeros(3, dtype=np.float64)
        rpy_noise = np.zeros(3, dtype=np.float64)

        if self.noise_mode in ("jitter", "both"):
            self.update_jitter()
            position_noise += self.current_jitter_position
            rpy_noise += self.current_jitter_rpy

        if self.noise_mode in ("drift", "both"):
            self.update_drift(dt, step_distance)
            position_noise += self.drift_position
            rpy_noise += self.drift_rpy

        self.apply_pose_noise(noisy_msg, position_noise, rpy_noise)
        if self.apply_to_twist:
            self.apply_twist_noise(noisy_msg, position_noise, rpy_noise, dt)

        self.publisher.publish(noisy_msg)

    def message_dt(self, msg):
        stamp_sec = stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            now = self.get_clock().now().nanoseconds * 1e-9
            stamp_sec = float(now)

        if self.last_stamp_sec is None:
            dt = 0.0
        else:
            dt = max(stamp_sec - self.last_stamp_sec, 0.0)

        self.last_stamp_sec = stamp_sec
        return dt

    def message_step_distance(self, msg):
        position = odom_position(msg)
        if self.last_odom_position is None:
            distance = 0.0
        else:
            distance = float(np.linalg.norm(position - self.last_odom_position))

        self.last_odom_position = position
        return distance

    def update_jitter(self):
        should_update = (
            self.frame_count == 1
            or (self.frame_count - 1) % self.jitter_frame_interval == 0
        )
        if not should_update:
            return

        self.current_jitter_position = self.rng.normal(
            0.0,
            self.jitter_position_std,
        )
        self.current_jitter_rpy = self.rng.normal(0.0, self.jitter_rpy_std)

    def update_drift(self, dt, step_distance):
        if self.drift_reference_distance_m > 0.0:
            self.update_distance_based_drift(step_distance)
        else:
            self.update_time_based_drift(dt)

        self.drift_position = clip_by_optional_limit(
            self.drift_position,
            self.drift_max_abs_position,
        )
        self.drift_rpy = clip_by_optional_limit(
            self.drift_rpy,
            self.drift_max_abs_rpy,
        )

    def update_distance_based_drift(self, step_distance):
        if step_distance <= 0.0:
            return

        scale = step_distance / self.drift_reference_distance_m
        self.drift_position += self.drift_position_error_at_reference * scale
        self.drift_rpy += self.drift_rpy_error_at_reference * scale

        self.drift_position += self.rng.normal(
            0.0,
            self.drift_position_random_walk_std * math.sqrt(scale),
        )
        self.drift_rpy += self.rng.normal(
            0.0,
            self.drift_rpy_random_walk_std * math.sqrt(scale),
        )

    def update_time_based_drift(self, dt):
        if dt <= 0.0:
            return

        self.drift_position += self.drift_position_rate * dt
        self.drift_rpy += self.drift_rpy_rate * dt

        self.drift_position += self.rng.normal(
            0.0,
            self.drift_position_random_walk_std * math.sqrt(dt),
        )
        self.drift_rpy += self.rng.normal(
            0.0,
            self.drift_rpy_random_walk_std * math.sqrt(dt),
        )

    def apply_pose_noise(self, msg, position_noise, rpy_noise):
        pose = msg.pose.pose
        pose.position.x += float(position_noise[0])
        pose.position.y += float(position_noise[1])
        pose.position.z += float(position_noise[2])

        roll, pitch, yaw = quaternion_to_euler(pose.orientation)
        roll += float(rpy_noise[0])
        pitch += float(rpy_noise[1])
        yaw += float(rpy_noise[2])

        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

    def apply_twist_noise(self, msg, position_noise, rpy_noise, dt):
        if dt <= 0.0:
            return

        twist = msg.twist.twist
        twist.linear.x += float(position_noise[0] / dt)
        twist.linear.y += float(position_noise[1] / dt)
        twist.linear.z += float(position_noise[2] / dt)
        twist.angular.x += float(rpy_noise[0] / dt)
        twist.angular.y += float(rpy_noise[1] / dt)
        twist.angular.z += float(rpy_noise[2] / dt)


def clip_by_optional_limit(values, limits):
    clipped = values.copy()
    for idx, limit in enumerate(limits):
        limit = float(limit)
        if limit > 0.0:
            clipped[idx] = np.clip(clipped[idx], -limit, limit)
    return clipped


def main():
    rclpy.init()
    node = OdomNoisy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
