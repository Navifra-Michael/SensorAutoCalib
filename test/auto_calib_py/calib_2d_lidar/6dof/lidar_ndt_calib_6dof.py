#!/usr/bin/env python3

import copy
import os
import site
import time

import numpy as np
import yaml

MPL_3D_AVAILABLE = False
MPL_3D_ERROR = None
try:
    import mpl_toolkits

    user_mpl_toolkits_path = os.path.join(
        site.getusersitepackages(),
        "mpl_toolkits",
    )
    if os.path.isdir(user_mpl_toolkits_path):
        mpl_toolkits.__path__ = [
            user_mpl_toolkits_path,
            *[
                path
                for path in mpl_toolkits.__path__
                if path != user_mpl_toolkits_path
            ],
        ]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MPL_3D_AVAILABLE = True
except Exception as exc:
    MPL_3D_ERROR = exc
    MPL_3D_AVAILABLE = False

import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


LIDAR_CONFIG_PATH = "lidar_config.yaml"
CALIB_CONFIG_PATH = "calib_config.yaml"
MAX_LIDAR_COUNT = 10


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def laserscan_to_xyz(msg):
    ranges = np.asarray(msg.ranges, dtype=np.float64)
    angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

    valid = np.isfinite(ranges)
    valid &= ranges >= msg.range_min
    valid &= ranges <= msg.range_max

    ranges = ranges[valid]
    angles = angles[valid]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)

    return np.stack([x, y, z], axis=1)


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


def odom_msg_to_transform(msg):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(msg.pose.pose.orientation)
    transform[:3, 3] = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z,
    ])
    return transform


def invert_transform(transform):
    inv = np.eye(4)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def apply_transform(points, transform):
    if len(points) == 0:
        return points

    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_matrix(roll, pitch, yaw):
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


def transform_xyz(points, pose):
    if len(points) == 0:
        return points

    rot = rotation_matrix(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    trans = np.array([
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
    ])

    return points @ rot.T + trans


def pose_to_transform(pose):
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    transform[:3, 3] = np.array([
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
    ])
    return transform


def transform_scan_chunks(scan_chunks, extrinsic_pose):
    if not scan_chunks:
        return np.empty((0, 3))

    extrinsic_tf = pose_to_transform(extrinsic_pose)
    transformed_chunks = []

    for chunk in scan_chunks:
        lidar_in_base = apply_transform(chunk["points"], extrinsic_tf)
        lidar_in_session = apply_transform(lidar_in_base, chunk["odom_tf"])
        transformed_chunks.append(lidar_in_session)

    if not transformed_chunks:
        return np.empty((0, 3))

    return np.vstack(transformed_chunks)


def precompute_scan_data(scan_chunks):
    local_points = []
    odom_rotations = []
    odom_translations = []

    for chunk in scan_chunks:
        points = chunk["points"]
        if len(points) == 0:
            continue

        count = len(points)
        odom_tf = chunk["odom_tf"]
        local_points.append(points)
        odom_rotations.append(np.repeat(odom_tf[:3, :3][None, :, :], count, axis=0))
        odom_translations.append(np.repeat(odom_tf[:3, 3][None, :], count, axis=0))

    if not local_points:
        return {
            "points": np.empty((0, 3)),
            "odom_rotations": np.empty((0, 3, 3)),
            "odom_translations": np.empty((0, 3)),
        }

    return {
        "points": np.vstack(local_points),
        "odom_rotations": np.vstack(odom_rotations),
        "odom_translations": np.vstack(odom_translations),
    }


def transform_precomputed_scan_data(scan_data, extrinsic_pose):
    points = scan_data["points"]
    if len(points) == 0:
        return points

    extrinsic_tf = pose_to_transform(extrinsic_pose)
    lidar_in_base = apply_transform(points, extrinsic_tf)

    return (
        np.einsum("nij,nj->ni", scan_data["odom_rotations"], lidar_in_base)
        + scan_data["odom_translations"]
    )


def pose_delta_from_initial(init_pose, candidate_pose):
    return {
        "x": float(candidate_pose["x"]) - float(init_pose["x"]),
        "y": float(candidate_pose["y"]) - float(init_pose["y"]),
        "z": float(candidate_pose["z"]) - float(init_pose["z"]),
        "roll": float(normalize_angle(
            float(candidate_pose["roll"]) - float(init_pose["roll"])
        )),
        "pitch": float(normalize_angle(
            float(candidate_pose["pitch"]) - float(init_pose["pitch"])
        )),
        "yaw": float(normalize_angle(
            float(candidate_pose["yaw"]) - float(init_pose["yaw"])
        )),
    }


def transform_accumulated_cloud(accumulated_cloud, init_pose, candidate_pose):
    delta_pose = pose_delta_from_initial(init_pose, candidate_pose)
    return transform_xyz(accumulated_cloud, delta_pose)


def downsample_xyz(points, voxel_size):
    if len(points) == 0:
        return points

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)

    return points[unique_indices]


def build_ndt_grid(points, resolution, min_points):
    grid = {}
    indices = np.floor(points / resolution).astype(np.int64)

    for idx, point in zip(map(tuple, indices), points):
        grid.setdefault(idx, []).append(point)

    ndt_grid = {}

    for idx, cell_points in grid.items():
        cell_points = np.asarray(cell_points)

        if len(cell_points) < min_points:
            continue

        mean = np.mean(cell_points, axis=0)
        cov = np.cov(cell_points.T)

        if cov.shape != (3, 3):
            continue

        cov += np.eye(3) * 1e-3

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue

        ndt_grid[idx] = {
            "mean": mean,
            "inv_cov": inv_cov,
        }

    return ndt_grid


def ndt_score(points, ndt_grid, resolution):
    if len(points) == 0:
        return 1e9

    indices = np.floor(points / resolution).astype(np.int64)
    total_score = 0.0
    used_count = 0
    missed_count = 0

    for idx, point in zip(map(tuple, indices), points):
        cell = ndt_grid.get(idx)

        if cell is None:
            missed_count += 1
            continue

        diff = point - cell["mean"]
        d2 = diff.T @ cell["inv_cov"] @ diff
        total_score += d2
        used_count += 1

    if used_count == 0:
        return 1e9

    miss_ratio = missed_count / max(used_count + missed_count, 1)
    return (total_score / used_count) + miss_ratio * 10.0


def load_yaml_file(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data or {}


def load_lidar_entries(config):
    lidar_cfg = config.get("lidars")

    if not isinstance(lidar_cfg, dict) or len(lidar_cfg) == 0:
        raise ValueError(
            "lidar_config.yaml must contain at least one lidar under 'lidars'."
        )

    if len(lidar_cfg) > MAX_LIDAR_COUNT:
        raise ValueError(
            f"Too many lidars configured: {len(lidar_cfg)}. "
            f"Maximum supported count is {MAX_LIDAR_COUNT}."
        )

    topics_cfg = config.get("topics", {})
    lidar_entries = []

    for name, pose in lidar_cfg.items():
        missing_keys = [
            key for key in ("x", "y", "z", "roll", "pitch", "yaw")
            if key not in pose
        ]
        if missing_keys:
            raise ValueError(f"lidars.{name} is missing keys: {missing_keys}")

        topic = pose.get("topic", topics_cfg.get(name))
        if topic is None:
            raise ValueError(
                f"No topic configured for {name}. Add topics.{name} or "
                f"lidars.{name}.topic."
            )

        lidar_entries.append({
            "name": name,
            "topic": topic,
            "pose": pose,
        })

    return lidar_entries


def resolve_output_path(output_dir, path):
    if os.path.isabs(path):
        return path

    return os.path.join(output_dir, path)


def make_initial_result_yaml(reference_lidar, lidars, lidar_poses):
    result_yaml = {
        "reference_lidar": reference_lidar,
        "lidar_count": len(lidars),
        "lidars": {},
    }

    for lidar in lidars:
        name = lidar["name"]
        pose = lidar_poses[name]
        result_yaml["lidars"][name] = {
            key: float(pose[key])
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        result_yaml["lidars"][name]["optimized"] = False

    return result_yaml


def make_calibrated_config(config, result_yaml):
    calibrated_config = copy.deepcopy(config)

    for name, result_pose in result_yaml["lidars"].items():
        if name not in calibrated_config["lidars"]:
            continue

        calibrated_pose = calibrated_config["lidars"][name]
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            calibrated_pose[key] = float(result_pose[key])

    return calibrated_config


def stage_values(center, value_range, step):
    return np.arange(center - value_range, center + value_range + step, step)


class ProgressPlot3D:
    def __init__(
        self,
        enabled,
        title,
        target_cloud,
        max_points=3000,
        update_interval_sec=1.0,
    ):
        self.enabled = enabled
        self.title = title
        self.target_cloud = sample_points(target_cloud, max_points)
        self.max_points = max_points
        self.update_interval_sec = update_interval_sec
        self.last_update_time = 0.0
        self.last_event_time = 0.0
        self.fig = None
        self.ax = None
        self.target_plot = None
        self.moving_plot = None

        if self.enabled and not MPL_3D_AVAILABLE:
            print(f"3D progress plot disabled: {MPL_3D_ERROR}")
            self.enabled = False
            return

        if self.enabled:
            try:
                plt.ion()
                self.fig = plt.figure(figsize=(9, 7))
                self.ax = self.fig.add_subplot(111, projection="3d")
                self._setup_plot()
            except Exception:
                self.close()
                self.enabled = False

    def _setup_plot(self):
        self.target_plot = self.ax.scatter(
            self.target_cloud[:, 0],
            self.target_cloud[:, 1],
            self.target_cloud[:, 2],
            s=1,
            c="tab:blue",
            alpha=0.35,
            label="target/fused",
        )
        self.moving_plot = self.ax.scatter(
            [],
            [],
            [],
            s=1,
            c="tab:orange",
            alpha=0.55,
            label="current best",
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.legend(loc="upper right")
        set_axes_equal_3d(self.ax, [self.target_cloud])
        self.fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)

    def close(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.target_plot = None
        self.moving_plot = None

    def pump_events(self):
        if not self.enabled or self.fig is None:
            return

        now = time.perf_counter()
        if now - self.last_event_time < 0.2:
            return

        self.last_event_time = now
        try:
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self.enabled = False

    def update(self, stage_idx, case_idx, cases, best, moving_cloud, force=False):
        if (
            not self.enabled
            or self.fig is None
            or self.ax is None
            or self.moving_plot is None
        ):
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return

        self.last_update_time = now
        moving_cloud = sample_points(moving_cloud, self.max_points)

        if len(moving_cloud) > 0:
            self.moving_plot._offsets3d = (
                moving_cloud[:, 0],
                moving_cloud[:, 1],
                moving_cloud[:, 2],
            )
        else:
            self.moving_plot._offsets3d = ([], [], [])

        progress = case_idx / max(cases, 1) * 100.0
        self.ax.set_title(
            f"{self.title}\n"
            f"stage={stage_idx + 1}, {case_idx}/{cases} ({progress:.1f}%), "
            f"score={best['score']:.4f}"
        )
        set_axes_equal_3d(self.ax, [self.target_cloud, moving_cloud])
        self.fig.canvas.draw_idle()
        self.pump_events()


def optimize_lidar_grid_search(
    scan_chunks,
    target_ndt,
    init_pose,
    resolution,
    search_stages,
    logger,
    progress_plot=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    best["score"] = 1e9
    best_points = np.empty((0, 3))

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = stage_values(center["z"], float(stage["range_z"]), float(stage["step_z"]))
        rolls = stage_values(
            center["roll"],
            np.deg2rad(float(stage["range_roll_deg"])),
            np.deg2rad(float(stage["step_roll_deg"])),
        )
        pitches = stage_values(
            center["pitch"],
            np.deg2rad(float(stage["range_pitch_deg"])),
            np.deg2rad(float(stage["step_pitch_deg"])),
        )
        yaws = stage_values(
            center["yaw"],
            np.deg2rad(float(stage["range_yaw_deg"])),
            np.deg2rad(float(stage["step_yaw_deg"])),
        )

        cases = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitches) * len(yaws)
        logger.info(
            f"Stage {stage_idx + 1}: "
            f"x={len(xs)}, y={len(ys)}, z={len(zs)}, "
            f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
            f"cases={cases}"
        )

        case_idx = 0
        progress_interval = max(cases // 20, 1)
        stage_start_time = time.perf_counter()

        for x in xs:
            for y in ys:
                for z in zs:
                    for roll in rolls:
                        for pitch in pitches:
                            for yaw in yaws:
                                case_idx += 1
                                pose = {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "roll": roll,
                                    "pitch": pitch,
                                    "yaw": yaw,
                                }
                                transformed = transform_scan_chunks(
                                    scan_chunks,
                                    pose,
                                )
                                score = ndt_score(transformed, target_ndt, resolution)

                                if score < best["score"]:
                                    best = {
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z),
                                        "roll": float(normalize_angle(roll)),
                                        "pitch": float(normalize_angle(pitch)),
                                        "yaw": float(normalize_angle(yaw)),
                                        "score": float(score),
                                    }
                                    best_points = transformed
                                    logger.info(
                                        f"Stage {stage_idx + 1} new best: "
                                        f"case={case_idx}/{cases}, "
                                        f"score={best['score']:.4f}, "
                                        f"x={best['x']:.4f}, "
                                        f"y={best['y']:.4f}, "
                                        f"z={best['z']:.4f}, "
                                        f"roll={best['roll']:.6f}, "
                                        f"pitch={best['pitch']:.6f}, "
                                        f"yaw={best['yaw']:.6f}"
                                    )

                                if progress_plot is not None:
                                    progress_plot.update(
                                        stage_idx,
                                        case_idx,
                                        cases,
                                        best,
                                        best_points,
                                    )

                                if case_idx % progress_interval == 0 or case_idx == cases:
                                    elapsed = time.perf_counter() - stage_start_time
                                    progress = case_idx / cases * 100.0
                                    rate = case_idx / max(elapsed, 1e-6)
                                    remaining = (cases - case_idx) / max(rate, 1e-6)
                                    logger.info(
                                        f"Stage {stage_idx + 1} progress: "
                                        f"{case_idx}/{cases} "
                                        f"({progress:.1f}%), "
                                        f"best={best['score']:.4f}, "
                                        f"elapsed={elapsed:.1f}s, "
                                        f"eta={remaining:.1f}s"
                                    )
                                    if progress_plot is not None:
                                        progress_plot.update(
                                            stage_idx,
                                            case_idx,
                                            cases,
                                            best,
                                            best_points,
                                            force=case_idx == cases,
                                        )

        center.update(best)
        logger.info(
            f"Stage {stage_idx + 1} best: "
            f"x={best['x']:.4f}, y={best['y']:.4f}, z={best['z']:.4f}, "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
    }


def optimize_accumulated_cloud_grid_search(
    source_cloud,
    target_ndt,
    init_pose,
    resolution,
    search_stages,
    logger,
    progress_plot=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    best["score"] = 1e9
    best_points = np.empty((0, 3))

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = stage_values(center["z"], float(stage["range_z"]), float(stage["step_z"]))
        rolls = stage_values(
            center["roll"],
            np.deg2rad(float(stage["range_roll_deg"])),
            np.deg2rad(float(stage["step_roll_deg"])),
        )
        pitches = stage_values(
            center["pitch"],
            np.deg2rad(float(stage["range_pitch_deg"])),
            np.deg2rad(float(stage["step_pitch_deg"])),
        )
        yaws = stage_values(
            center["yaw"],
            np.deg2rad(float(stage["range_yaw_deg"])),
            np.deg2rad(float(stage["step_yaw_deg"])),
        )

        cases = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitches) * len(yaws)
        logger.info(
            f"Fast accumulated stage {stage_idx + 1}: "
            f"x={len(xs)}, y={len(ys)}, z={len(zs)}, "
            f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
            f"cases={cases}"
        )

        case_idx = 0
        progress_interval = max(cases // 20, 1)
        stage_start_time = time.perf_counter()

        for x in xs:
            for y in ys:
                for z in zs:
                    for roll in rolls:
                        for pitch in pitches:
                            for yaw in yaws:
                                case_idx += 1
                                pose = {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "roll": roll,
                                    "pitch": pitch,
                                    "yaw": yaw,
                                }
                                transformed = transform_accumulated_cloud(
                                    source_cloud,
                                    init_pose,
                                    pose,
                                )
                                score = ndt_score(transformed, target_ndt, resolution)

                                if score < best["score"]:
                                    best = {
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z),
                                        "roll": float(normalize_angle(roll)),
                                        "pitch": float(normalize_angle(pitch)),
                                        "yaw": float(normalize_angle(yaw)),
                                        "score": float(score),
                                    }
                                    best_points = transformed
                                    logger.info(
                                        f"Fast accumulated stage {stage_idx + 1} "
                                        f"new best: case={case_idx}/{cases}, "
                                        f"score={best['score']:.4f}, "
                                        f"x={best['x']:.4f}, "
                                        f"y={best['y']:.4f}, "
                                        f"z={best['z']:.4f}, "
                                        f"roll={best['roll']:.6f}, "
                                        f"pitch={best['pitch']:.6f}, "
                                        f"yaw={best['yaw']:.6f}"
                                    )

                                if progress_plot is not None:
                                    progress_plot.update(
                                        stage_idx,
                                        case_idx,
                                        cases,
                                        best,
                                        best_points,
                                    )

                                if case_idx % progress_interval == 0 or case_idx == cases:
                                    elapsed = time.perf_counter() - stage_start_time
                                    progress = case_idx / cases * 100.0
                                    rate = case_idx / max(elapsed, 1e-6)
                                    remaining = (cases - case_idx) / max(rate, 1e-6)
                                    logger.info(
                                        f"Fast accumulated stage {stage_idx + 1} "
                                        f"progress: {case_idx}/{cases} "
                                        f"({progress:.1f}%), "
                                        f"best={best['score']:.4f}, "
                                        f"elapsed={elapsed:.1f}s, "
                                        f"eta={remaining:.1f}s"
                                    )
                                    if progress_plot is not None:
                                        progress_plot.update(
                                            stage_idx,
                                            case_idx,
                                            cases,
                                            best,
                                            best_points,
                                            force=case_idx == cases,
                                        )

        center.update(best)
        logger.info(
            f"Fast accumulated stage {stage_idx + 1} best: "
            f"x={best['x']:.4f}, y={best['y']:.4f}, z={best['z']:.4f}, "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
    }


def optimize_precomputed_scan_grid_search(
    scan_data,
    target_ndt,
    init_pose,
    resolution,
    search_stages,
    logger,
    progress_plot=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    best["score"] = 1e9
    best_points = np.empty((0, 3))

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = stage_values(center["z"], float(stage["range_z"]), float(stage["step_z"]))
        rolls = stage_values(
            center["roll"],
            np.deg2rad(float(stage["range_roll_deg"])),
            np.deg2rad(float(stage["step_roll_deg"])),
        )
        pitches = stage_values(
            center["pitch"],
            np.deg2rad(float(stage["range_pitch_deg"])),
            np.deg2rad(float(stage["step_pitch_deg"])),
        )
        yaws = stage_values(
            center["yaw"],
            np.deg2rad(float(stage["range_yaw_deg"])),
            np.deg2rad(float(stage["step_yaw_deg"])),
        )

        cases = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitches) * len(yaws)
        logger.info(
            f"Precomputed exact stage {stage_idx + 1}: "
            f"x={len(xs)}, y={len(ys)}, z={len(zs)}, "
            f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
            f"cases={cases}"
        )

        case_idx = 0
        progress_interval = max(cases // 20, 1)
        stage_start_time = time.perf_counter()

        for x in xs:
            for y in ys:
                for z in zs:
                    for roll in rolls:
                        for pitch in pitches:
                            for yaw in yaws:
                                case_idx += 1
                                pose = {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "roll": roll,
                                    "pitch": pitch,
                                    "yaw": yaw,
                                }
                                transformed = transform_precomputed_scan_data(
                                    scan_data,
                                    pose,
                                )
                                score = ndt_score(transformed, target_ndt, resolution)

                                if score < best["score"]:
                                    best = {
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z),
                                        "roll": float(normalize_angle(roll)),
                                        "pitch": float(normalize_angle(pitch)),
                                        "yaw": float(normalize_angle(yaw)),
                                        "score": float(score),
                                    }
                                    best_points = transformed
                                    logger.info(
                                        f"Precomputed exact stage {stage_idx + 1} "
                                        f"new best: case={case_idx}/{cases}, "
                                        f"score={best['score']:.4f}, "
                                        f"x={best['x']:.4f}, "
                                        f"y={best['y']:.4f}, "
                                        f"z={best['z']:.4f}, "
                                        f"roll={best['roll']:.6f}, "
                                        f"pitch={best['pitch']:.6f}, "
                                        f"yaw={best['yaw']:.6f}"
                                    )

                                if progress_plot is not None:
                                    progress_plot.update(
                                        stage_idx,
                                        case_idx,
                                        cases,
                                        best,
                                        best_points,
                                    )

                                if case_idx % progress_interval == 0 or case_idx == cases:
                                    elapsed = time.perf_counter() - stage_start_time
                                    progress = case_idx / cases * 100.0
                                    rate = case_idx / max(elapsed, 1e-6)
                                    remaining = (cases - case_idx) / max(rate, 1e-6)
                                    logger.info(
                                        f"Precomputed exact stage {stage_idx + 1} "
                                        f"progress: {case_idx}/{cases} "
                                        f"({progress:.1f}%), "
                                        f"best={best['score']:.4f}, "
                                        f"elapsed={elapsed:.1f}s, "
                                        f"eta={remaining:.1f}s"
                                    )
                                    if progress_plot is not None:
                                        progress_plot.update(
                                            stage_idx,
                                            case_idx,
                                            cases,
                                            best,
                                            best_points,
                                            force=case_idx == cases,
                                        )

        center.update(best)
        logger.info(
            f"Precomputed exact stage {stage_idx + 1} best: "
            f"x={best['x']:.4f}, y={best['y']:.4f}, z={best['z']:.4f}, "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
    }


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


def set_axes_equal_3d(ax, cloud_sets):
    all_points = [
        points
        for points in cloud_sets
        if points is not None and len(points) > 0
    ]
    if not all_points:
        return

    points = np.vstack(all_points)
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range, 1.0)
    margin = half_range * 0.05
    half_range += margin

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def cloud_axis_limits(cloud_sets, axes):
    all_points = []

    for clouds in cloud_sets:
        for points in clouds.values():
            if len(points) > 0:
                all_points.append(points[:, axes])

    if not all_points:
        return None

    points = np.vstack(all_points)
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range, 1.0)
    margin = half_range * 0.05

    return (
        center[0] - half_range - margin,
        center[0] + half_range + margin,
        center[1] - half_range - margin,
        center[1] + half_range + margin,
    )


def plot_clouds_projection(
    title,
    clouds,
    save_path,
    axes,
    axis_labels,
    axis_limits=None,
    max_points=12000,
    show_default_grid=True,
):
    plt.figure(figsize=(8, 8))

    for name, points in clouds.items():
        if len(points) == 0:
            continue

        points = sample_points(points, max_points)
        plt.scatter(points[:, axes[0]], points[:, axes[1]], s=1, label=name)

    if axis_limits is not None:
        axis_1_min, axis_1_max, axis_2_min, axis_2_max = axis_limits
        plt.xlim(axis_1_min, axis_1_max)
        plt.ylim(axis_2_min, axis_2_max)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel(f"{axis_labels[0]} [m]")
    plt.ylabel(f"{axis_labels[1]} [m]")
    if show_default_grid:
        plt.grid(True, color="0.9", linewidth=0.4)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_clouds_xy(title, clouds, save_path, axis_limits=None, max_points=12000):
    plot_clouds_projection(
        title=title,
        clouds=clouds,
        save_path=save_path,
        axes=(0, 1),
        axis_labels=("x", "y"),
        axis_limits=axis_limits,
        max_points=max_points,
    )


class LidarNdtCalib6Dof(Node):
    def __init__(self):
        super().__init__("lidar_ndt_calib_6dof")

        self.lidar_config = load_yaml_file(LIDAR_CONFIG_PATH)
        self.calib_config = load_yaml_file(CALIB_CONFIG_PATH)

        self.lidars = load_lidar_entries(self.lidar_config)
        self.lidar_names = [lidar["name"] for lidar in self.lidars]
        self.lidar_poses = {
            lidar["name"]: lidar["pose"]
            for lidar in self.lidars
        }

        self.reference_lidar = self.calib_config.get(
            "reference_lidar",
            self.lidar_names[0],
        )
        if self.reference_lidar not in self.lidar_poses:
            raise ValueError(
                f"reference_lidar '{self.reference_lidar}' is not listed "
                "under lidars."
            )

        self.collect_duration_sec = float(
            self.calib_config.get("collect_duration_sec", 5.0)
        )
        topics_cfg = self.lidar_config.get("topics", {})
        self.odom_topic = topics_cfg.get(
            "odom",
            self.calib_config.get("odom_topic", "/odom"),
        )
        self.require_odom = bool(self.calib_config.get("require_odom", True))
        self.cloud_buffers = {
            lidar["name"]: [] for lidar in self.lidars
        }
        self.latest_odom_tf = None
        self.odom_origin_inv = None

        self.start_time = self.get_clock().now()
        self.wall_start_time = time.perf_counter()
        self.finished = False
        self.subscribers = []

        self.get_logger().info(
            f"Loaded {len(self.lidars)} lidar(s): {', '.join(self.lidar_names)}"
        )
        self.get_logger().info(f"Subscribe odom: {self.odom_topic}")
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            50,
        )

        for lidar in self.lidars:
            lidar_name = lidar["name"]
            topic_name = lidar["topic"]
            self.get_logger().info(f"Subscribe {lidar_name}: {topic_name}")
            sub = self.create_subscription(
                LaserScan,
                topic_name,
                lambda msg, name=lidar_name: self.scan_callback(msg, name),
                10,
            )
            self.subscribers.append(sub)

        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info(
            f"Collecting LaserScan + odom data for {self.collect_duration_sec} sec..."
        )

    def odom_callback(self, msg):
        odom_tf = odom_msg_to_transform(msg)
        if self.odom_origin_inv is None:
            self.odom_origin_inv = invert_transform(odom_tf)

        self.latest_odom_tf = self.odom_origin_inv @ odom_tf

    def scan_callback(self, msg, lidar_name):
        if self.finished:
            return

        if self.latest_odom_tf is None:
            if self.require_odom:
                return

            odom_tf = np.eye(4)
        else:
            odom_tf = self.latest_odom_tf.copy()

        points = laserscan_to_xyz(msg)
        if len(points) == 0:
            return

        self.cloud_buffers[lidar_name].append({
            "points": points,
            "odom_tf": odom_tf,
        })

    def timer_callback(self):
        if self.finished:
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if elapsed < self.collect_duration_sec:
            return

        self.finished = True
        self.get_logger().info("Collection finished. Running 2D LiDAR + odom 6DOF calibration...")
        self.run_calibration()
        rclpy.shutdown()

    def run_calibration(self):
        calibration_start_time = time.perf_counter()
        collection_elapsed_sec = calibration_start_time - self.wall_start_time

        ndt_cfg = self.calib_config.get("ndt", {})
        resolution = float(ndt_cfg.get("resolution", 0.5))
        min_points = int(ndt_cfg.get("min_points_per_cell", 5))
        downsample_voxel = float(ndt_cfg.get("downsample_voxel", 0.08))
        plot_cfg = self.calib_config.get("plot", {})
        progress_3d_cfg = plot_cfg.get("progress_3d", {})
        progress_3d_enabled = bool(progress_3d_cfg.get("enabled", False))
        progress_3d_max_points = int(progress_3d_cfg.get("max_points", 3000))
        progress_3d_update_interval = float(
            progress_3d_cfg.get("update_interval_sec", 1.0)
        )
        match_mode = self.calib_config.get("match_mode", "exact_precomputed")
        valid_match_modes = {
            "exact_precomputed",
            "exact_odom_chunks",
            "fast_accumulated_cloud",
        }
        if match_mode not in valid_match_modes:
            raise ValueError(
                "calib_config.yaml match_mode must be one of "
                f"{sorted(valid_match_modes)}"
            )

        search_stages = self.calib_config.get("search", {}).get("stages", [])
        if not search_stages:
            raise ValueError("calib_config.yaml must define search.stages.")

        output_dir = self.calib_config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        output_yaml = resolve_output_path(
            output_dir,
            self.calib_config.get("output_yaml", "calibrated_output.yaml"),
        )
        calibrated_config_yaml = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "calibrated_config_yaml",
                "calibrated_config.yaml",
            ),
        )
        before_plot_path = resolve_output_path(
            output_dir,
            self.calib_config.get("before_plot_png", "before_calibration.png"),
        )
        after_plot_path = resolve_output_path(
            output_dir,
            self.calib_config.get("after_plot_png", "after_calibration.png"),
        )
        before_plot_xz_path = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "before_plot_xz_png",
                "before_calibration_xz.png",
            ),
        )
        after_plot_xz_path = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "after_plot_xz_png",
                "after_calibration_xz.png",
            ),
        )
        before_plot_yz_path = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "before_plot_yz_png",
                "before_calibration_yz.png",
            ),
        )
        after_plot_yz_path = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "after_plot_yz_png",
                "after_calibration_yz.png",
            ),
        )

        scan_chunks_by_lidar = {}
        raw_clouds = {}
        for name, chunks in self.cloud_buffers.items():
            if len(chunks) == 0:
                self.get_logger().warn(f"{name}: no scan received")
                scan_chunks_by_lidar[name] = []
                raw_clouds[name] = np.empty((0, 3))
                continue

            downsampled_chunks = []
            for chunk in chunks:
                points = downsample_xyz(chunk["points"], downsample_voxel)
                downsampled_chunks.append({
                    "points": points,
                    "odom_tf": chunk["odom_tf"],
                })

            non_empty_points = [
                chunk["points"]
                for chunk in downsampled_chunks
                if len(chunk["points"]) > 0
            ]
            if not non_empty_points:
                self.get_logger().warn(f"{name}: no valid scan points")
                raw_clouds[name] = np.empty((0, 3))
                scan_chunks_by_lidar[name] = []
                continue

            points = np.vstack(non_empty_points)
            points = downsample_xyz(points, downsample_voxel)
            raw_clouds[name] = points
            scan_chunks_by_lidar[name] = downsampled_chunks
            self.get_logger().info(
                f"{name}: {len(points)} points after downsample "
                f"from {len(downsampled_chunks)} scans"
            )

        before_clouds = {}
        for name, chunks in scan_chunks_by_lidar.items():
            before_clouds[name] = transform_scan_chunks(
                chunks,
                self.lidar_poses[name],
            )

        reference_cloud = before_clouds[self.reference_lidar]
        target_ndt = build_ndt_grid(reference_cloud, resolution, min_points)
        self.get_logger().info(f"NDT cells: {len(target_ndt)}")

        result_yaml = make_initial_result_yaml(
            self.reference_lidar,
            self.lidars,
            self.lidar_poses,
        )
        result_yaml["calibration_mode"] = (
            f"sequential_fused_ndt_2d_lidar_odom_6dof_{match_mode}"
        )
        result_yaml["match_mode"] = match_mode
        result_yaml["calibration_order"] = [lidar["name"] for lidar in self.lidars]
        result_yaml["odom_topic"] = self.odom_topic
        result_yaml["timing_sec"] = {
            "collection": collection_elapsed_sec,
            "per_lidar_optimization": {},
        }

        if len(target_ndt) == 0:
            result_yaml["success"] = False
            result_yaml["reason"] = "empty_ndt_grid"
            with open(output_yaml, "w") as f:
                yaml.dump(result_yaml, f, sort_keys=False)
            return

        result_yaml["success"] = True
        after_clouds = {self.reference_lidar: reference_cloud}
        fused_cloud = reference_cloud
        precomputed_scan_data_by_lidar = {
            name: precompute_scan_data(chunks)
            for name, chunks in scan_chunks_by_lidar.items()
        }

        for lidar in self.lidars:
            name = lidar["name"]
            chunks = scan_chunks_by_lidar[name]

            if name == self.reference_lidar:
                continue

            if len(chunks) == 0:
                result_yaml["lidars"][name]["success"] = False
                result_yaml["lidars"][name]["reason"] = "no_scan_received"
                continue

            target_ndt = build_ndt_grid(fused_cloud, resolution, min_points)
            if len(target_ndt) == 0:
                result_yaml["success"] = False
                result_yaml["lidars"][name]["success"] = False
                result_yaml["lidars"][name]["reason"] = "empty_fused_ndt_grid"
                continue

            self.get_logger().info(f"Optimizing {name} in 6DOF...")
            start_time = time.perf_counter()
            progress_plot = ProgressPlot3D(
                enabled=progress_3d_enabled,
                title=f"Optimizing {name}",
                target_cloud=fused_cloud,
                max_points=progress_3d_max_points,
                update_interval_sec=progress_3d_update_interval,
            )
            try:
                if match_mode == "fast_accumulated_cloud":
                    result = optimize_accumulated_cloud_grid_search(
                        source_cloud=before_clouds[name],
                        target_ndt=target_ndt,
                        init_pose=self.lidar_poses[name],
                        resolution=resolution,
                        search_stages=search_stages,
                        logger=self.get_logger(),
                        progress_plot=progress_plot,
                    )
                elif match_mode == "exact_precomputed":
                    result = optimize_precomputed_scan_grid_search(
                        scan_data=precomputed_scan_data_by_lidar[name],
                        target_ndt=target_ndt,
                        init_pose=self.lidar_poses[name],
                        resolution=resolution,
                        search_stages=search_stages,
                        logger=self.get_logger(),
                        progress_plot=progress_plot,
                    )
                else:
                    result = optimize_lidar_grid_search(
                        scan_chunks=chunks,
                        target_ndt=target_ndt,
                        init_pose=self.lidar_poses[name],
                        resolution=resolution,
                        search_stages=search_stages,
                        logger=self.get_logger(),
                        progress_plot=progress_plot,
                    )
            finally:
                progress_plot.close()
            elapsed = time.perf_counter() - start_time
            result_yaml["timing_sec"]["per_lidar_optimization"][name] = elapsed

            result_yaml["lidars"][name] = {
                key: result[key]
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            }
            result_yaml["lidars"][name]["score"] = result["score"]
            result_yaml["lidars"][name]["optimization_time_sec"] = elapsed
            result_yaml["lidars"][name]["success"] = result["success"]
            result_yaml["lidars"][name]["optimized"] = True

            if match_mode == "fast_accumulated_cloud":
                after_clouds[name] = transform_accumulated_cloud(
                    before_clouds[name],
                    self.lidar_poses[name],
                    result,
                )
            elif match_mode == "exact_precomputed":
                after_clouds[name] = transform_precomputed_scan_data(
                    precomputed_scan_data_by_lidar[name],
                    result,
                )
            else:
                after_clouds[name] = transform_scan_chunks(chunks, result)
            fused_cloud = np.vstack([fused_cloud, after_clouds[name]])
            fused_cloud = downsample_xyz(fused_cloud, downsample_voxel)

        max_plot_points = int(plot_cfg.get("max_points", 12000))
        show_default_grid = bool(plot_cfg.get("show_default_grid", True))
        plot_limits_xy = cloud_axis_limits([before_clouds, after_clouds], (0, 1))
        plot_limits_xz = cloud_axis_limits([before_clouds, after_clouds], (0, 2))
        plot_limits_yz = cloud_axis_limits([before_clouds, after_clouds], (1, 2))
        plot_clouds_projection(
            "Before 6DOF Calibration - XY Projection",
            before_clouds,
            before_plot_path,
            axes=(0, 1),
            axis_labels=("x", "y"),
            axis_limits=plot_limits_xy,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )
        plot_clouds_projection(
            "After 6DOF Calibration - XY Projection",
            after_clouds,
            after_plot_path,
            axes=(0, 1),
            axis_labels=("x", "y"),
            axis_limits=plot_limits_xy,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )
        plot_clouds_projection(
            "Before 6DOF Calibration - XZ Projection",
            before_clouds,
            before_plot_xz_path,
            axes=(0, 2),
            axis_labels=("x", "z"),
            axis_limits=plot_limits_xz,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )
        plot_clouds_projection(
            "After 6DOF Calibration - XZ Projection",
            after_clouds,
            after_plot_xz_path,
            axes=(0, 2),
            axis_labels=("x", "z"),
            axis_limits=plot_limits_xz,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )
        plot_clouds_projection(
            "Before 6DOF Calibration - YZ Projection",
            before_clouds,
            before_plot_yz_path,
            axes=(1, 2),
            axis_labels=("y", "z"),
            axis_limits=plot_limits_yz,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )
        plot_clouds_projection(
            "After 6DOF Calibration - YZ Projection",
            after_clouds,
            after_plot_yz_path,
            axes=(1, 2),
            axis_labels=("y", "z"),
            axis_limits=plot_limits_yz,
            max_points=max_plot_points,
            show_default_grid=show_default_grid,
        )

        finish_time = time.perf_counter()
        result_yaml["timing_sec"]["calibration"] = (
            finish_time - calibration_start_time
        )
        result_yaml["timing_sec"]["total"] = finish_time - self.wall_start_time

        with open(output_yaml, "w") as f:
            yaml.dump(result_yaml, f, sort_keys=False)

        calibrated_config = make_calibrated_config(self.lidar_config, result_yaml)
        with open(calibrated_config_yaml, "w") as f:
            yaml.dump(calibrated_config, f, sort_keys=False)

        self.get_logger().info(f"Saved result: {output_yaml}")
        self.get_logger().info(f"Saved calibrated config: {calibrated_config_yaml}")
        self.get_logger().info(f"Saved plot: {before_plot_path}")
        self.get_logger().info(f"Saved plot: {after_plot_path}")
        self.get_logger().info(f"Saved plot: {before_plot_xz_path}")
        self.get_logger().info(f"Saved plot: {after_plot_xz_path}")
        self.get_logger().info(f"Saved plot: {before_plot_yz_path}")
        self.get_logger().info(f"Saved plot: {after_plot_yz_path}")


def main():
    rclpy.init()
    node = LidarNdtCalib6Dof()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
