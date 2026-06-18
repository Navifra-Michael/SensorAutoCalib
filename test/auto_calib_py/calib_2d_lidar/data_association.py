import os
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass
from bisect import bisect_left

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud, PointCloud2

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None

from ndt_common import (
    apply_runtime_overrides,
    graph_data_path,
    load_merged_calib_config,
    load_yaml,
    resolve_output_path,
    save_graph_data,
    validate_target,
)
from graph_plot import (
    AsyncPlotter,
    CollectionLivePlot,
    NdtLivePlot2D,
    NdtLivePlot3D,
    generate_graphs,
    print_graph_summary,
)


@dataclass
class CalibrationRun:
    target: dict
    paths: dict
    lidar_config: dict
    calib_config: dict
    env: dict
    python_executable: str
    command: list


@dataclass
class CalibrationResult:
    returncode: int
    target: dict
    paths: dict
    result_yaml: dict
    calibrated_config: dict

    @property
    def success(self):
        return self.returncode == 0 and bool(self.result_yaml.get("success", False))


def build_runtime_env(args):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NDT_MAIN_ACTIVE"] = "1"

    if args.backend:
        env["NDT_PLOT_BACKEND"] = args.backend
    if args.live_plot:
        env["NDT_LIVE_PLOT"] = "1"
    elif args.no_live_plot:
        env["NDT_LIVE_PLOT"] = "0"
    if args.keep_live_plot:
        env["NDT_KEEP_LIVE_PLOT"] = "1"
    elif args.close_live_plot:
        env["NDT_KEEP_LIVE_PLOT"] = "0"

    return env


def apply_runtime_env(env):
    for key in ("NDT_PLOT_BACKEND", "NDT_LIVE_PLOT", "NDT_KEEP_LIVE_PLOT"):
        if key in env:
            os.environ[key] = env[key]


def prepare_run(target, args):
    paths = validate_target(target)
    lidar_config = load_yaml(paths["lidar_config_path"])
    calib_config = load_merged_calib_config(paths)
    env = build_runtime_env(args)
    apply_runtime_env(env)
    calib_config = apply_runtime_overrides(
        calib_config,
        mode=target["label"],
    )

    output_dir = resolve_output_path(
        target["workdir"],
        calib_config.get("output_dir", "output"),
    )
    paths["output_dir"] = output_dir
    paths["output_yaml"] = resolve_output_path(
        output_dir,
        calib_config.get("output_yaml", "calibrated_output.yaml"),
    )
    paths["calibrated_config_yaml"] = resolve_output_path(
        output_dir,
        calib_config.get("calibrated_config_yaml", "calibrated_config.yaml"),
    )
    paths["graph_data_npz"] = graph_data_path(target, calib_config)

    command = ["internal", target.get("command_label", target["label"])]

    return CalibrationRun(
        target=target,
        paths=paths,
        lidar_config=lidar_config,
        calib_config=calib_config,
        env=env,
        python_executable=args.python,
        command=command,
    )


def run_data_association(calibration_run, dry_run=False):
    if dry_run:
        return CalibrationResult(
            returncode=0,
            target=calibration_run.target,
            paths=calibration_run.paths,
            result_yaml={},
            calibrated_config={},
        )

    try:
        print("[data_association] collecting topic data...")
        cloud_buffers, collection_elapsed_sec = collect_topic_data(calibration_run)
        print_collection_summary(cloud_buffers, collection_elapsed_sec)
        print("[data_association] collection finished. Running NDT calculation...")
        ndt_start_time = time.perf_counter()
        calibration_output = run_ndt_calculation(
            calibration_run,
            cloud_buffers,
            collection_elapsed_sec,
        )
        print(
            "[data_association] NDT calculation finished: "
            f"{time.perf_counter() - ndt_start_time:.3f}s"
        )
        save_calibration_output(calibration_run, calibration_output)
        result_yaml = calibration_output["result_yaml"]
        calibrated_config = calibration_output["calibrated_config"]
        returncode = 0
    except Exception as exc:
        print(f"[data_association] calibration failed: {exc}", file=sys.stderr)
        result_yaml = {
            "success": False,
            "reason": str(exc),
        }
        calibrated_config = {}
        returncode = 1

    if returncode == 0 and result_yaml:
        graph_outputs = generate_graphs(
            calibration_run.target,
            calibration_run.lidar_config,
            calibration_run.calib_config,
            result_yaml,
        )
        print_graph_summary(graph_outputs)

    return CalibrationResult(
        returncode=returncode,
        target=calibration_run.target,
        paths=calibration_run.paths,
        result_yaml=result_yaml,
        calibrated_config=calibrated_config,
    )


def lidar_entries(lidar_config, required_pose_keys):
    lidar_cfg = lidar_config.get("lidars")
    if not isinstance(lidar_cfg, dict) or len(lidar_cfg) == 0:
        raise ValueError("lidar_config.yaml must contain lidars.")

    topics_cfg = lidar_config.get("topics", {})
    topic_types_cfg = lidar_config.get("topic_types", {})
    entries = []
    for name, pose in lidar_cfg.items():
        missing = [key for key in required_pose_keys if key not in pose]
        if missing:
            raise ValueError(f"lidars.{name} is missing keys: {missing}")

        topic_cfg = pose.get("topic", topics_cfg.get(name))
        if isinstance(topic_cfg, dict):
            topic = topic_cfg.get("name", topic_cfg.get("topic"))
            topic_type = topic_cfg.get("type", "LaserScan")
        else:
            topic = topic_cfg
            topic_type = pose.get("topic_type", topic_types_cfg.get(name, "LaserScan"))
        if topic is None:
            raise ValueError(f"No topic configured for {name}.")

        entries.append({
            "name": name,
            "topic": topic,
            "topic_type": normalize_topic_type(topic_type),
            "pose": pose,
        })

    return entries


def normalize_topic_type(topic_type):
    normalized = str(topic_type).strip().lower()
    aliases = {
        "laserscan": "LaserScan",
        "sensor_msgs/msg/laserscan": "LaserScan",
        "pointcloud": "PointCloud",
        "sensor_msgs/msg/pointcloud": "PointCloud",
        "pointcloud2": "PointCloud2",
        "sensor_msgs/msg/pointcloud2": "PointCloud2",
    }
    if normalized not in aliases:
        raise ValueError(
            "topic type must be one of LaserScan, PointCloud, PointCloud2 "
            f"(got {topic_type})"
        )
    return aliases[normalized]


def message_class_for_topic_type(topic_type):
    if topic_type == "LaserScan":
        return LaserScan
    if topic_type == "PointCloud":
        return PointCloud
    if topic_type == "PointCloud2":
        return PointCloud2
    raise ValueError(f"Unsupported topic type: {topic_type}")


def laserscan_to_xy(msg):
    ranges = np.asarray(msg.ranges, dtype=np.float64)
    angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

    valid = np.isfinite(ranges)
    valid &= ranges >= msg.range_min
    valid &= ranges <= msg.range_max

    ranges = ranges[valid]
    angles = angles[valid]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.stack([x, y], axis=1)


def pointcloud_to_xyz(msg):
    if len(msg.points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(
        [[point.x, point.y, point.z] for point in msg.points],
        dtype=np.float64,
    )


def pointcloud2_to_xyz(msg):
    if point_cloud2 is None:
        raise ImportError(
            "sensor_msgs_py.point_cloud2 is required for PointCloud2 topics."
        )

    points = []
    for point in point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    ):
        points.append([point[0], point[1], point[2]])

    if not points:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def message_to_xy(msg, topic_type):
    if topic_type == "LaserScan":
        return laserscan_to_xy(msg)
    if topic_type == "PointCloud":
        return pointcloud_to_xyz(msg)[:, :2]
    if topic_type == "PointCloud2":
        return pointcloud2_to_xyz(msg)[:, :2]
    raise ValueError(f"Unsupported topic type: {topic_type}")


def laserscan_to_xyz(msg):
    points_xy = laserscan_to_xy(msg)
    z = np.zeros((len(points_xy), 1), dtype=points_xy.dtype)
    return np.hstack([points_xy, z])


def message_to_xyz(msg, topic_type):
    if topic_type == "LaserScan":
        return laserscan_to_xyz(msg)
    if topic_type == "PointCloud":
        return pointcloud_to_xyz(msg)
    if topic_type == "PointCloud2":
        return pointcloud2_to_xyz(msg)
    raise ValueError(f"Unsupported topic type: {topic_type}")


def rotation_matrix_3d(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


def apply_initial_pose_2d(points, pose):
    if len(points) == 0:
        return points

    roll = float(pose.get("roll", 0.0))
    yaw = float(pose["yaw"])
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotated_points = points.copy()
    rotated_points[:, 1] *= np.cos(roll)
    rot = np.array([[c, -s], [s, c]])
    return rotated_points @ rot.T + np.array([float(pose["x"]), float(pose["y"])])


def apply_initial_pose_3d(points, pose):
    if len(points) == 0:
        return points

    rot = rotation_matrix_3d(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    trans = np.array([float(pose["x"]), float(pose["y"]), float(pose["z"])])
    return points @ rot.T + trans


def apply_transform(points, transform):
    if len(points) == 0:
        return points

    return points @ transform[:3, :3].T + transform[:3, 3]


def quaternion_to_matrix(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(rot):
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quaternion_array_to_matrix(quat):
    x, y, z, w = quat
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.eye(3)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp_quaternion(q0, q1, ratio):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        quat = q0 + ratio * (q1 - q0)
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            return q0
        return quat / norm

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * ratio
    sin_theta = np.sin(theta)

    scale_0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale_1 = sin_theta / sin_theta_0
    return scale_0 * q0 + scale_1 * q1


def odom_msg_to_transform(msg):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(msg.pose.pose.orientation)
    transform[:3, 3] = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z,
    ])
    return transform


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def message_stamp_sec(msg):
    header = getattr(msg, "header", None)
    if header is None:
        return None
    return stamp_to_sec(header.stamp)


def invert_transform(transform):
    inv = np.eye(4)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def interpolate_transform(tf0, tf1, ratio):
    ratio = float(np.clip(ratio, 0.0, 1.0))
    trans = tf0[:3, 3] + ratio * (tf1[:3, 3] - tf0[:3, 3])
    q0 = matrix_to_quaternion(tf0[:3, :3])
    q1 = matrix_to_quaternion(tf1[:3, :3])
    quat = slerp_quaternion(q0, q1, ratio)

    transform = np.eye(4)
    transform[:3, :3] = quaternion_array_to_matrix(quat)
    transform[:3, 3] = trans
    return transform


class DataAssociationCollector(Node):
    def __init__(self, calibration_run):
        super().__init__("ndt_data_association")
        self.calibration_run = calibration_run
        self.mode = calibration_run.target["label"]
        self.collect_duration_sec = float(
            calibration_run.calib_config.get("collect_duration_sec", 5.0)
        )
        self.finished = False
        self.wall_start_time = time.perf_counter()
        self.start_time = self.get_clock().now()
        self.subscribers = []
        collection_cfg = calibration_run.calib_config.get("collection", {})
        self.overlap_filter_voxel = float(
            collection_cfg.get("overlap_filter_voxel", 0.0)
        )
        self.sample_mode = str(
            collection_cfg.get("sample_mode", "all")
        ).strip().lower()
        self.sample_interval_sec = float(
            collection_cfg.get("sample_interval_sec", 0.0)
        )
        if self.sample_mode not in ("all", "time_interval"):
            raise ValueError(
                "collection.sample_mode must be one of: all, time_interval"
            )
        if self.sample_interval_sec <= 0.0:
            self.sample_mode = "all"
        stable_cfg = collection_cfg.get("stable_filter", {})
        self.stable_filter_enabled = bool(stable_cfg.get("enabled", False))
        self.stable_filter_voxel = float(
            stable_cfg.get("voxel", self.overlap_filter_voxel)
        )
        self.stable_filter_min_observations = max(
            1,
            int(stable_cfg.get("min_observations", 2)),
        )
        self.stable_filter_min_odom_translation = float(
            stable_cfg.get("min_odom_translation", 0.0)
        )
        self.stable_filter_min_odom_yaw_deg = float(
            stable_cfg.get("min_odom_yaw_deg", 0.0)
        )
        self.odom_lookup_mode = str(
            collection_cfg.get("odom_lookup_mode", "timestamp")
        ).strip().lower()
        self.odom_interpolation = bool(
            collection_cfg.get("odom_interpolation", True)
        )
        self.odom_max_time_diff_sec = float(
            collection_cfg.get("odom_max_time_diff_sec", 0.2)
        )
        self.odom_buffer_sec = float(
            collection_cfg.get("odom_buffer_sec", 10.0)
        )
        plot_cfg = calibration_run.calib_config.get("plot", {})
        live_enabled = bool(plot_cfg.get("show_collection_live_plot", False))
        self.live_plot = AsyncPlotter(
            CollectionLivePlot,
            enabled=live_enabled,
            mode=self.mode,
            title=f"Collecting {self.mode} LiDAR data",
            max_points=int(plot_cfg.get("live_max_points", 5000)),
            update_interval_sec=float(
                plot_cfg.get("live_update_interval_sec", 0.5)
            ),
            keep_open=bool(plot_cfg.get("keep_live_ndt_open", False)),
            lidar_config=calibration_run.lidar_config,
        )

        if self.mode == "6dof":
            self.entries = lidar_entries(
                calibration_run.lidar_config,
                ("x", "y", "z", "roll", "pitch", "yaw"),
            )
            topics_cfg = calibration_run.lidar_config.get("topics", {})
            self.odom_topic = topics_cfg.get(
                "odom",
                calibration_run.calib_config.get("odom_topic", "/odom"),
            )
            self.require_odom = bool(
                calibration_run.calib_config.get("require_odom", True)
            )
            self.latest_odom_tf = None
            self.odom_buffer = []
            self.odom_trajectory = []
            self.odom_origin_inv = None
            self.get_logger().info(f"Subscribe odom: {self.odom_topic}")
            self.odom_subscriber = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                50,
            )
        else:
            self.entries = lidar_entries(
                calibration_run.lidar_config,
                ("x", "y", "yaw"),
            )
            self.require_odom = False
            self.latest_odom_tf = None
            self.odom_buffer = []
            self.odom_trajectory = []
            self.odom_origin_inv = None

        self.cloud_buffers = {entry["name"]: [] for entry in self.entries}
        self.entry_poses = {
            entry["name"]: entry["pose"]
            for entry in self.entries
        }
        self.collection_voxel_keys = {entry["name"]: set() for entry in self.entries}
        self.stable_voxel_counts = {entry["name"]: {} for entry in self.entries}
        self.stable_voxel_last_odom = {entry["name"]: {} for entry in self.entries}
        self.last_sample_time_by_lidar = {
            entry["name"]: None for entry in self.entries
        }
        self.topic_types = {
            entry["name"]: entry["topic_type"]
            for entry in self.entries
        }
        if self.sample_mode == "time_interval":
            self.get_logger().info(
                "Collection sample mode: time_interval "
                f"interval={self.sample_interval_sec:.3f}s"
            )
        if self.overlap_filter_voxel > 0.0:
            self.get_logger().info(
                "Collection overlap filter enabled: "
                f"voxel={self.overlap_filter_voxel:.4f}m"
            )
        if self.stable_filter_enabled:
            self.get_logger().info(
                "Collection stable filter enabled: "
                f"voxel={self.stable_filter_voxel:.4f}m, "
                f"min_observations={self.stable_filter_min_observations}, "
                f"min_odom_translation={self.stable_filter_min_odom_translation:.4f}m, "
                f"min_odom_yaw={self.stable_filter_min_odom_yaw_deg:.3f}deg"
            )
        if self.mode == "6dof":
            self.get_logger().info(
                "Odom lookup: "
                f"mode={self.odom_lookup_mode}, "
                f"interpolation={self.odom_interpolation}, "
                f"max_dt={self.odom_max_time_diff_sec:.3f}s"
            )
        for entry in self.entries:
            self.get_logger().info(
                f"Subscribe {entry['name']}: {entry['topic']} "
                f"({entry['topic_type']})"
            )
            sub = self.create_subscription(
                message_class_for_topic_type(entry["topic_type"]),
                entry["topic"],
                lambda msg, name=entry["name"]: self.sensor_callback(msg, name),
                10,
            )
            self.subscribers.append(sub)

        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info(
            f"Collecting {self.mode} calibration data for "
            f"{self.collect_duration_sec} sec..."
        )

    @property
    def collection_elapsed_sec(self):
        return time.perf_counter() - self.wall_start_time

    def odom_callback(self, msg):
        odom_tf = odom_msg_to_transform(msg)
        odom_stamp = message_stamp_sec(msg)
        if self.odom_origin_inv is None:
            self.odom_origin_inv = invert_transform(odom_tf)

        relative_odom_tf = self.odom_origin_inv @ odom_tf
        self.latest_odom_tf = relative_odom_tf
        self.odom_trajectory.append(relative_odom_tf[:3, 3].copy())

        if odom_stamp is None:
            return

        if self.odom_buffer and odom_stamp < self.odom_buffer[-1][0]:
            insert_idx = bisect_left([item[0] for item in self.odom_buffer], odom_stamp)
            self.odom_buffer.insert(insert_idx, (odom_stamp, relative_odom_tf))
        else:
            self.odom_buffer.append((odom_stamp, relative_odom_tf))

        if self.odom_buffer_sec > 0.0:
            min_stamp = odom_stamp - self.odom_buffer_sec
            while self.odom_buffer and self.odom_buffer[0][0] < min_stamp:
                self.odom_buffer.pop(0)

    def lookup_odom_tf(self, sensor_stamp):
        if self.odom_lookup_mode == "latest" or sensor_stamp is None:
            return None if self.latest_odom_tf is None else self.latest_odom_tf.copy()

        if not self.odom_buffer:
            return None

        stamps = [item[0] for item in self.odom_buffer]
        right_idx = bisect_left(stamps, sensor_stamp)
        candidates = []
        if right_idx < len(self.odom_buffer):
            candidates.append(self.odom_buffer[right_idx])
        if right_idx > 0:
            candidates.append(self.odom_buffer[right_idx - 1])

        if not candidates:
            return None

        nearest_stamp, nearest_tf = min(
            candidates,
            key=lambda item: abs(item[0] - sensor_stamp),
        )
        nearest_dt = abs(nearest_stamp - sensor_stamp)
        if (
            self.odom_max_time_diff_sec > 0.0
            and nearest_dt > self.odom_max_time_diff_sec
        ):
            return None

        can_interpolate = (
            self.odom_interpolation
            and right_idx > 0
            and right_idx < len(self.odom_buffer)
        )
        if not can_interpolate:
            return nearest_tf.copy()

        left_stamp, left_tf = self.odom_buffer[right_idx - 1]
        right_stamp, right_tf = self.odom_buffer[right_idx]
        if right_stamp <= left_stamp:
            return nearest_tf.copy()

        ratio = (sensor_stamp - left_stamp) / (right_stamp - left_stamp)
        return interpolate_transform(left_tf, right_tf, ratio)

    def filter_collection_overlap(self, lidar_name, points, odom_tf=None):
        if self.overlap_filter_voxel <= 0.0 or len(points) == 0:
            return points

        keyed_points = self.collection_key_points(lidar_name, points, odom_tf)
        keys = np.floor(keyed_points / self.overlap_filter_voxel).astype(np.int64)
        seen = self.collection_voxel_keys[lidar_name]
        keep_indices = []
        for point_idx, key in enumerate(map(tuple, keys)):
            if key in seen:
                continue
            seen.add(key)
            keep_indices.append(point_idx)

        if not keep_indices:
            return points[:0]

        return points[np.asarray(keep_indices, dtype=np.int64)]

    def collection_key_points(self, lidar_name, points, odom_tf=None):
        pose = self.entry_poses[lidar_name]
        if self.mode == "6dof":
            keyed_points = apply_initial_pose_3d(points, pose)
            if odom_tf is not None:
                keyed_points = apply_transform(keyed_points, odom_tf)
            return keyed_points

        return apply_initial_pose_2d(points, pose)

    def filter_collection_stable(self, lidar_name, points, odom_tf=None):
        if (
            not self.stable_filter_enabled
            or self.stable_filter_voxel <= 0.0
            or len(points) == 0
        ):
            return points

        keyed_points = self.collection_key_points(lidar_name, points, odom_tf)
        keys = np.floor(keyed_points / self.stable_filter_voxel).astype(np.int64)
        counts = self.stable_voxel_counts[lidar_name]
        last_odom_by_key = self.stable_voxel_last_odom[lidar_name]
        seen_in_scan = set()
        keep_indices = []
        for point_idx, key in enumerate(map(tuple, keys)):
            if key not in seen_in_scan:
                if self.should_count_stable_observation(
                    last_odom_by_key.get(key),
                    odom_tf,
                ):
                    counts[key] = counts.get(key, 0) + 1
                    last_odom_by_key[key] = self.stable_odom_state(odom_tf)
                seen_in_scan.add(key)
            if counts.get(key, 0) >= self.stable_filter_min_observations:
                keep_indices.append(point_idx)

        if not keep_indices:
            return points[:0]

        return points[np.asarray(keep_indices, dtype=np.int64)]

    def stable_odom_state(self, odom_tf):
        if odom_tf is None:
            return None

        yaw = np.arctan2(odom_tf[1, 0], odom_tf[0, 0])
        return {
            "translation": odom_tf[:3, 3].copy(),
            "yaw": float(yaw),
        }

    def should_count_stable_observation(self, last_state, odom_tf):
        if last_state is None:
            return True

        current_state = self.stable_odom_state(odom_tf)
        if current_state is None:
            return True

        delta_translation = np.linalg.norm(
            current_state["translation"] - last_state["translation"]
        )
        delta_yaw = abs(np.arctan2(
            np.sin(current_state["yaw"] - last_state["yaw"]),
            np.cos(current_state["yaw"] - last_state["yaw"]),
        ))

        return (
            delta_translation >= self.stable_filter_min_odom_translation
            or delta_yaw >= np.deg2rad(self.stable_filter_min_odom_yaw_deg)
        )

    def sample_time_sec(self, msg):
        stamp = message_stamp_sec(msg)
        if stamp is not None:
            return stamp
        return self.collection_elapsed_sec

    def should_sample_scan(self, lidar_name, msg):
        if self.sample_mode != "time_interval":
            return True

        current_time = self.sample_time_sec(msg)
        last_time = self.last_sample_time_by_lidar[lidar_name]
        if (
            last_time is not None
            and current_time - last_time < self.sample_interval_sec
        ):
            return False

        return True

    def mark_sampled_scan(self, lidar_name, msg):
        if self.sample_mode == "time_interval":
            self.last_sample_time_by_lidar[lidar_name] = self.sample_time_sec(msg)

    def sensor_callback(self, msg, lidar_name):
        if self.finished:
            return
        if not self.should_sample_scan(lidar_name, msg):
            return
        topic_type = self.topic_types[lidar_name]

        if self.mode == "6dof":
            sensor_stamp = message_stamp_sec(msg)
            odom_tf = self.lookup_odom_tf(sensor_stamp)
            if odom_tf is None:
                if self.require_odom:
                    return
                odom_tf = np.eye(4)

            points = message_to_xyz(msg, topic_type)
            points = self.filter_collection_stable(lidar_name, points, odom_tf)
            points = self.filter_collection_overlap(lidar_name, points, odom_tf)
            if len(points) == 0:
                return

            self.cloud_buffers[lidar_name].append({
                "points": points,
                "odom_tf": odom_tf,
            })
            self.mark_sampled_scan(lidar_name, msg)
            return

        points = message_to_xy(msg, topic_type)
        points = self.filter_collection_stable(lidar_name, points)
        points = self.filter_collection_overlap(lidar_name, points)
        if len(points) == 0:
            return

        self.cloud_buffers[lidar_name].append(points)
        self.mark_sampled_scan(lidar_name, msg)

    def timer_callback(self):
        if self.finished:
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        self.live_plot.update(
            self.cloud_buffers,
            elapsed,
            trajectory=self.odom_trajectory,
        )
        if elapsed < self.collect_duration_sec:
            return

        self.finished = True
        self.get_logger().info("Collection finished.")
        self.live_plot.update(
            self.cloud_buffers,
            elapsed,
            trajectory=self.odom_trajectory,
            force=True,
        )
        self.live_plot.close(force=True)


class PrintLogger:
    def info(self, message):
        print(f"[ndt] {message}")

    def warn(self, message):
        print(f"[ndt][WARN] {message}")

    def error(self, message):
        print(f"[ndt][ERROR] {message}")


def collect_topic_data(calibration_run):
    if not rclpy.ok():
        rclpy.init(args=None)
    node = DataAssociationCollector(calibration_run)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.cloud_buffers, node.collection_elapsed_sec
    finally:
        try:
            node.live_plot.close(force=True)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def print_collection_summary(cloud_buffers, collection_elapsed_sec):
    print(
        "[data_association] collection summary: "
        f"elapsed={collection_elapsed_sec:.3f}s"
    )
    for name, chunks in cloud_buffers.items():
        if not chunks:
            print(f"[data_association] {name}: scans=0, points=0")
            continue

        if isinstance(chunks[0], dict):
            point_count = sum(len(chunk["points"]) for chunk in chunks)
        else:
            point_count = sum(len(points) for points in chunks)

        print(
            f"[data_association] {name}: "
            f"scans={len(chunks)}, points={point_count}"
        )


def load_ndt_module(calibration_run):
    module_name = f"ndt_{calibration_run.target['label']}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        calibration_run.paths["script_path"],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {calibration_run.paths['script_path']}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ndt_calculation(calibration_run, cloud_buffers, collection_elapsed_sec):
    module = load_ndt_module(calibration_run)
    logger = PrintLogger()
    plot_cfg = calibration_run.calib_config.get("plot", {})
    live_enabled = bool(plot_cfg.get("show_live_ndt", False))
    plotter_class = (
        NdtLivePlot2D
        if calibration_run.target["label"] == "3dof"
        else NdtLivePlot3D
    )
    ndt_live_plot = AsyncPlotter(
        plotter_class,
        enabled=live_enabled,
        title="NDT optimization",
        max_points=int(plot_cfg.get("live_max_points", 5000)),
        update_interval_sec=float(plot_cfg.get("live_update_interval_sec", 0.5)),
        keep_open=bool(plot_cfg.get("keep_live_ndt_open", False)),
    )
    separate_initial_plots = bool(
        plot_cfg.get("separate_initial_overlap_plots", False)
    )
    initial_plotters = {}
    initial_plotters_lock = threading.Lock()
    if separate_initial_plots and live_enabled:
        for plot_key in calibration_run.lidar_config.get("lidars", {}).keys():
            initial_plotters[plot_key] = AsyncPlotter(
                plotter_class,
                enabled=live_enabled,
                title=f"Initial overlap: {plot_key}",
                max_points=int(plot_cfg.get("live_max_points", 5000)),
                update_interval_sec=float(
                    plot_cfg.get("live_update_interval_sec", 0.5)
                ),
                keep_open=bool(plot_cfg.get("keep_live_ndt_open", False)),
            )

    def get_initial_plotter(plot_key):
        with initial_plotters_lock:
            if plot_key not in initial_plotters:
                initial_plotters[plot_key] = AsyncPlotter(
                    plotter_class,
                    enabled=live_enabled,
                    title=f"Initial overlap: {plot_key}",
                    max_points=int(plot_cfg.get("live_max_points", 5000)),
                    update_interval_sec=float(
                        plot_cfg.get("live_update_interval_sec", 0.5)
                    ),
                    keep_open=bool(plot_cfg.get("keep_live_ndt_open", False)),
                )
            return initial_plotters[plot_key]

    def progress_callback(event):
        target_plotter = ndt_live_plot
        if (
            separate_initial_plots
            and event.get("plot_group") == "initial_overlap"
        ):
            target_plotter = get_initial_plotter(
                event.get("plot_key", event["lidar_name"]),
            )

        target_plotter.update(
            lidar_name=event["lidar_name"],
            stage_idx=event["stage_idx"],
            case_idx=event["case_idx"],
            cases=event["cases"],
            best=event["best"],
            target_cloud=event["target_cloud"],
            candidate_cloud=event["candidate_cloud"],
            best_target_cloud=event.get("best_target_cloud"),
            best_candidate_cloud=event.get("best_candidate_cloud"),
            current_score=event.get("current_score"),
            current_vector=event.get("current_vector"),
            best_vector=event.get("best_vector"),
            force=event.get("force", False),
        )

    if calibration_run.target["label"] == "3dof":
        try:
            return module.calibrate_3dof(
                calibration_run.lidar_config,
                calibration_run.calib_config,
                cloud_buffers,
                collection_elapsed_sec=collection_elapsed_sec,
                logger=logger,
                progress_callback=progress_callback,
            )
        finally:
            ndt_live_plot.close()
            for plotter in initial_plotters.values():
                plotter.close(force=True)

    try:
        return module.calibrate_6dof(
            calibration_run.lidar_config,
            calibration_run.calib_config,
            cloud_buffers,
            collection_elapsed_sec=collection_elapsed_sec,
            logger=logger,
            progress_callback=progress_callback,
        )
    finally:
        ndt_live_plot.close(force=True)
        for plotter in initial_plotters.values():
            plotter.close(force=True)


def save_calibration_output(calibration_run, calibration_output):
    os.makedirs(calibration_run.paths["output_dir"], exist_ok=True)

    with open(calibration_run.paths["output_yaml"], "w") as f:
        yaml.dump(calibration_output["result_yaml"], f, sort_keys=False)

    with open(calibration_run.paths["calibrated_config_yaml"], "w") as f:
        yaml.dump(calibration_output["calibrated_config"], f, sort_keys=False)

    save_graph_data(
        calibration_run.paths["graph_data_npz"],
        list(calibration_output["before_clouds"].keys()),
        calibration_output["before_clouds"],
        calibration_output["after_clouds"],
        calibration_output["point_dim"],
        extra_graph_clouds=calibration_output.get("extra_graph_clouds"),
    )
    print(f"[data_association] saved result: {calibration_run.paths['output_yaml']}")
    print(
        "[data_association] saved calibrated config: "
        f"{calibration_run.paths['calibrated_config_yaml']}"
    )
    print(f"[data_association] saved graph data: {calibration_run.paths['graph_data_npz']}")
