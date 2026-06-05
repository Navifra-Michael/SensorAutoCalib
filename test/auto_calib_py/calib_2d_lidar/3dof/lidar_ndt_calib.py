#!/usr/bin/env python3

import copy
import os
import time
import warnings

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_CONFIG_PATH = os.path.join(SCRIPT_DIR, "lidar_config.yaml")
CALIB_CONFIG_PATH = os.path.join(SCRIPT_DIR, "calib_config.yaml")


def load_plot_backend(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return None

    return (config.get("plot", {}) or {}).get("backend")


import matplotlib
PLOT_BACKEND = load_plot_backend(CALIB_CONFIG_PATH)
if PLOT_BACKEND:
    matplotlib.use(PLOT_BACKEND, force=True)
elif not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


MAX_LIDAR_COUNT = 10


def is_noninteractive_backend():
    return matplotlib.get_backend().lower() == "agg"


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def laserscan_to_xy(msg: LaserScan):
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


def transform_xy(points, x, y, yaw, roll=0.0):
    if len(points) == 0:
        return points

    c = np.cos(yaw)
    s = np.sin(yaw)

    rot = np.array([
        [c, -s],
        [s,  c],
    ])

    rolled_points = points.copy()
    rolled_points[:, 1] *= np.cos(roll)

    return rolled_points @ rot.T + np.array([x, y])


def downsample_xy(points, voxel_size):
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

        if cov.shape != (2, 2):
            continue

        cov += np.eye(2) * 1e-3

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
    unique_indices, inverse_indices, counts = np.unique(
        indices,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(inverse_indices)
    sorted_inverse = inverse_indices[order]
    sorted_points = points[order]
    boundaries = np.concatenate([
        np.array([0]),
        np.flatnonzero(np.diff(sorted_inverse)) + 1,
        np.array([len(sorted_inverse)]),
    ])

    total_score = 0.0
    used_count = 0
    missed_count = 0

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        unique_idx = sorted_inverse[start]
        cell = ndt_grid.get(tuple(unique_indices[unique_idx]))

        if cell is None:
            missed_count += int(counts[unique_idx])
            continue

        cell_points = sorted_points[start:end]
        diff = cell_points - cell["mean"]
        d2 = np.einsum("ni,ij,nj->n", diff, cell["inv_cov"], diff)

        total_score += float(np.sum(d2))
        used_count += len(cell_points)

    if used_count == 0:
        return 1e9

    miss_ratio = missed_count / max(used_count + missed_count, 1)
    return (total_score / used_count) + miss_ratio * 10.0


def default_search_stages():
    return [
        {
            "range_x": 0.30,
            "range_y": 0.30,
            "range_yaw_deg": 15.0,
            "step_x": 0.05,
            "step_y": 0.05,
            "step_yaw_deg": 2.0,
        },
        {
            "range_x": 0.08,
            "range_y": 0.08,
            "range_yaw_deg": 4.0,
            "step_x": 0.01,
            "step_y": 0.01,
            "step_yaw_deg": 0.5,
        },
        {
            "range_x": 0.02,
            "range_y": 0.02,
            "range_yaw_deg": 1.0,
            "step_x": 0.002,
            "step_y": 0.002,
            "step_yaw_deg": 0.1,
        },
    ]


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
        if not isinstance(pose, dict):
            raise ValueError(f"lidars.{name} must be a mapping.")

        missing_pose_keys = [
            key for key in ("x", "y", "yaw")
            if key not in pose
        ]
        if missing_pose_keys:
            raise ValueError(
                f"lidars.{name} is missing pose keys: {missing_pose_keys}"
            )

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


def load_yaml_file(path, default=None):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return default if default is not None else {}

    if data is None:
        return default if default is not None else {}

    return data


def resolve_output_path(output_dir, path):
    if os.path.isabs(path):
        return path

    return os.path.join(output_dir, path)


def resolve_base_path(path):
    if os.path.isabs(path):
        return path

    return os.path.join(SCRIPT_DIR, path)


def pose_roll(pose):
    return float(pose.get("roll", 0.0))


def optimize_lidar_grid_search(
    raw_points,
    target_ndt,
    init_pose,
    resolution,
    search_stages,
    logger,
    live_plotter=None,
    initial_score=None,
):
    roll = pose_roll(init_pose)
    if initial_score is None:
        initial_points = transform_xy(
            raw_points,
            float(init_pose["x"]),
            float(init_pose["y"]),
            float(init_pose["yaw"]),
            roll,
        )
        initial_score = ndt_score(initial_points, target_ndt, resolution)

    best = {
        "x": float(init_pose["x"]),
        "y": float(init_pose["y"]),
        "yaw": float(init_pose["yaw"]),
        "roll": roll,
        "score": float(initial_score),
    }

    center_x = best["x"]
    center_y = best["y"]
    center_yaw = best["yaw"]
    rolled_points = raw_points.copy()
    rolled_points[:, 1] *= np.cos(roll)

    for stage_idx, stage in enumerate(search_stages):
        range_x = float(stage["range_x"])
        range_y = float(stage["range_y"])
        range_yaw = np.deg2rad(float(stage["range_yaw_deg"]))

        step_x = float(stage["step_x"])
        step_y = float(stage["step_y"])
        step_yaw = np.deg2rad(float(stage["step_yaw_deg"]))

        xs = np.arange(center_x - range_x, center_x + range_x + step_x, step_x)
        ys = np.arange(center_y - range_y, center_y + range_y + step_y, step_y)
        yaws = np.arange(
            center_yaw - range_yaw,
            center_yaw + range_yaw + step_yaw,
            step_yaw,
        )

        logger.info(
            f"Stage {stage_idx + 1}: "
            f"x={len(xs)}, y={len(ys)}, yaw={len(yaws)}, "
            f"cases={len(xs) * len(ys) * len(yaws)}, "
            f"start_best={best['score']:.4f}"
        )

        for yaw in yaws:
            c = np.cos(yaw)
            s = np.sin(yaw)
            rot = np.array([
                [c, -s],
                [s, c],
            ])
            rotated = rolled_points @ rot.T

            for x in xs:
                for y in ys:
                    transformed = rotated + np.array([x, y])
                    score = ndt_score(transformed, target_ndt, resolution)

                    if score < best["score"]:
                        best = {
                            "x": float(x),
                            "y": float(y),
                            "yaw": float(normalize_angle(yaw)),
                            "roll": roll,
                            "score": float(score),
                        }
                        if live_plotter is not None:
                            live_plotter.update(
                                stage_idx=stage_idx,
                                candidate_points=transformed,
                                best=best,
                            )

        center_x = best["x"]
        center_y = best["y"]
        center_yaw = best["yaw"]

        logger.info(
            f"Stage {stage_idx + 1} best: "
            f"x={best['x']:.4f}, "
            f"y={best['y']:.4f}, "
            f"yaw={best['yaw']:.6f}, "
            f"score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
    }


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
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "roll": pose_roll(pose),
            "yaw": float(pose["yaw"]),
            "optimized": False,
        }

    return result_yaml


def make_calibrated_config(config, result_yaml):
    calibrated_config = copy.deepcopy(config)

    for name, result_pose in result_yaml["lidars"].items():
        if name not in calibrated_config["lidars"]:
            continue

        calibrated_pose = calibrated_config["lidars"][name]
        calibrated_pose["x"] = float(result_pose["x"])
        calibrated_pose["y"] = float(result_pose["y"])
        calibrated_pose["roll"] = float(result_pose.get("roll", 0.0))
        calibrated_pose["yaw"] = float(result_pose["yaw"])

    return calibrated_config


def make_pose_map_from_result(result_yaml):
    return {
        name: {
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
        }
        for name, pose in result_yaml["lidars"].items()
    }


def cloud_axis_limits(cloud_sets):
    all_points = []

    for clouds in cloud_sets:
        for points in clouds.values():
            if len(points) > 0:
                all_points.append(points)

    if not all_points:
        return None

    points = np.vstack(all_points)
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    center = (min_xy + max_xy) / 2.0
    half_range = np.max(max_xy - min_xy) / 2.0
    half_range = max(half_range, 1.0)
    margin = half_range * 0.05

    return (
        center[0] - half_range - margin,
        center[0] + half_range + margin,
        center[1] - half_range - margin,
        center[1] + half_range + margin,
    )


def plot_clouds(
    title,
    clouds,
    save_path,
    axis_limits=None,
    grid_resolution=None,
    poses=None,
    pose_color="black",
    secondary_poses=None,
    secondary_pose_color="red",
    options=None,
):
    options = options or {}
    plt.figure(figsize=(8, 8))

    for name, points in clouds.items():
        if len(points) == 0:
            continue

        plt.scatter(points[:, 0], points[:, 1], s=1, label=name)

    if axis_limits is not None:
        xmin, xmax, ymin, ymax = axis_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    if grid_resolution is not None:
        draw_ndt_grid(axis_limits, grid_resolution)

    if poses is not None:
        if bool(options.get("show_lidar_arrows", True)):
            draw_pose_arrows(
                poses,
                axis_limits,
                color=pose_color,
                label_font_size=float(options.get("lidar_label_font_size", 6.0)),
            )

    if secondary_poses is not None:
        if bool(options.get("show_lidar_arrows", True)):
            draw_pose_arrows(
                secondary_poses,
                axis_limits,
                color=secondary_pose_color,
                linestyle="dashed",
                show_labels=False,
                label_font_size=float(options.get("lidar_label_font_size", 6.0)),
            )

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if bool(options.get("show_robot_frame", True)):
        draw_robot_frame(axis_limits)
    draw_scale_bar(axis_limits)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if bool(options.get("show_default_grid", True)):
        plt.grid(True, color="0.9", linewidth=0.4)
    else:
        plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_ndt_grid(axis_limits, resolution):
    if axis_limits is None or resolution <= 0:
        return

    xmin, xmax, ymin, ymax = axis_limits
    x_start = np.floor(xmin / resolution) * resolution
    x_end = np.ceil(xmax / resolution) * resolution
    y_start = np.floor(ymin / resolution) * resolution
    y_end = np.ceil(ymax / resolution) * resolution

    ax = plt.gca()
    x_lines = np.arange(x_start, x_end + resolution, resolution)
    y_lines = np.arange(y_start, y_end + resolution, resolution)

    for x in x_lines:
        ax.axvline(x, color="0.82", linewidth=0.4, zorder=0)

    for y in y_lines:
        ax.axhline(y, color="0.82", linewidth=0.4, zorder=0)


def draw_scale_bar(axis_limits):
    if axis_limits is None:
        return

    xmin, xmax, ymin, ymax = axis_limits
    width = xmax - xmin
    height = ymax - ymin
    scale_len = 1.0
    x0 = xmin + width * 0.06
    y0 = ymin + height * 0.06

    ax = plt.gca()
    ax.plot(
        [x0, x0 + scale_len],
        [y0, y0],
        color="black",
        linewidth=2.0,
        zorder=8,
    )
    ax.text(
        x0 + scale_len / 2.0,
        y0 + height * 0.02,
        "1 m",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=8,
    )


def draw_robot_frame(axis_limits):
    if axis_limits is None:
        axis_len = 0.8
    else:
        xmin, xmax, ymin, ymax = axis_limits
        axis_len = max(xmax - xmin, ymax - ymin) * 0.08

    ax = plt.gca()
    alpha = 0.28
    ax.scatter(
        [0.0],
        [0.0],
        s=32,
        c="0.35",
        marker="o",
        alpha=alpha,
        zorder=2,
    )
    ax.arrow(
        0.0,
        0.0,
        axis_len,
        0.0,
        width=axis_len * 0.025,
        head_width=axis_len * 0.12,
        head_length=axis_len * 0.16,
        length_includes_head=True,
        color="0.25",
        alpha=alpha,
        zorder=2,
    )
    ax.arrow(
        0.0,
        0.0,
        0.0,
        axis_len,
        width=axis_len * 0.025,
        head_width=axis_len * 0.12,
        head_length=axis_len * 0.16,
        length_includes_head=True,
        color="0.25",
        alpha=alpha,
        zorder=2,
    )
    ax.text(
        axis_len * 1.15,
        -axis_len * 0.18,
        "robot x",
        fontsize=8,
        color="0.25",
        alpha=alpha,
        zorder=2,
    )
    ax.text(
        axis_len * 0.12,
        axis_len * 1.15,
        "robot y",
        fontsize=8,
        color="0.25",
        alpha=alpha,
        zorder=2,
    )


def draw_pose_arrows(
    poses,
    axis_limits=None,
    color="black",
    linestyle="solid",
    show_labels=True,
    label_font_size=6.0,
):
    if axis_limits is None:
        arrow_length = 0.35
    else:
        xmin, xmax, ymin, ymax = axis_limits
        arrow_length = max(xmax - xmin, ymax - ymin) * 0.06

    ax = plt.gca()

    for name, pose in poses.items():
        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose["yaw"])

        dx = np.cos(yaw) * arrow_length
        dy = np.sin(yaw) * arrow_length

        ax.arrow(
            x,
            y,
            dx,
            dy,
            width=arrow_length * 0.025,
            head_width=arrow_length * 0.12,
            head_length=arrow_length * 0.16,
            length_includes_head=True,
            color=color,
            linestyle=linestyle,
            zorder=5,
        )
        ax.scatter([x], [y], s=36, c=color, marker="x", zorder=6)

        if show_labels:
            label_x = x + dx * 1.08
            label_y = y + dy * 1.08
            ax.text(
                label_x,
                label_y,
                name,
                fontsize=label_font_size,
                color=color,
                ha="center",
                va="center",
                zorder=7,
            )


def sample_points_for_plot(points, max_points):
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


class LiveNdtPlotter:
    def __init__(
        self,
        lidar_name,
        target_points,
        axis_limits,
        grid_resolution,
        update_interval_sec=0.2,
        max_points=4000,
    ):
        self.lidar_name = lidar_name
        self.target_points = sample_points_for_plot(target_points, max_points)
        self.axis_limits = axis_limits
        self.grid_resolution = grid_resolution
        self.update_interval_sec = float(update_interval_sec)
        self.max_points = int(max_points)
        self.last_update_time = 0.0
        self.enabled = True
        self.figure = None
        self.ax = None
        self.disable_reason = ""

        if is_noninteractive_backend():
            self.enabled = False
            self.disable_reason = "matplotlib backend is non-interactive Agg"
            return

        try:
            plt.ion()
            self.figure, self.ax = plt.subplots(figsize=(8, 8))
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as exc:
            self.disable_reason = str(exc)
            print(f"Live NDT plot disabled: {self.disable_reason}")
            self.enabled = False

    def update(self, stage_idx, candidate_points, best, force=False):
        if not self.enabled or self.figure is None or self.ax is None:
            return

        if not plt.fignum_exists(self.figure.number):
            self.enabled = False
            self.disable_reason = "figure window no longer exists"
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return

        try:
            self.last_update_time = now
            candidate_points = sample_points_for_plot(
                candidate_points,
                self.max_points,
            )

            self.ax.clear()
            plt.sca(self.ax)

            if self.axis_limits is not None:
                xmin, xmax, ymin, ymax = self.axis_limits
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)

            if self.grid_resolution is not None:
                draw_ndt_grid(self.axis_limits, self.grid_resolution)

            if len(self.target_points) > 0:
                self.ax.scatter(
                    self.target_points[:, 0],
                    self.target_points[:, 1],
                    s=1,
                    c="0.65",
                    label="fused target",
                )

            if len(candidate_points) > 0:
                self.ax.scatter(
                    candidate_points[:, 0],
                    candidate_points[:, 1],
                    s=1,
                    c="tab:red",
                    label=self.lidar_name,
                )

            draw_pose_arrows(
                {
                    self.lidar_name: {
                        "x": best["x"],
                        "y": best["y"],
                        "yaw": best["yaw"],
                    }
                },
                self.axis_limits,
                color="black",
                label_font_size=7,
            )

            self.ax.set_aspect("equal", adjustable="box")
            self.ax.grid(True, color="0.9", linewidth=0.4)
            self.ax.legend()
            self.ax.set_title(
                f"NDT live: {self.lidar_name} | "
                f"stage {stage_idx + 1} | "
                f"score {best['score']:.4f}"
            )
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:
            self.disable_reason = str(exc)
            print(f"Live NDT plot disabled: {self.disable_reason}")
            self.enabled = False

    def close(self):
        if self.figure is not None and plt.fignum_exists(self.figure.number):
            plt.close(self.figure)
        self.figure = None
        self.ax = None
        self.enabled = False


class CollectionPlotter:
    def __init__(self, lidar_names, collect_duration_sec):
        self.lidar_names = lidar_names
        self.collect_duration_sec = collect_duration_sec
        self.enabled = True
        self.figure = None
        self.ax = None
        self.last_update_time = 0.0

        if is_noninteractive_backend():
            self.enabled = False
            return

        try:
            plt.ion()
            self.figure, self.ax = plt.subplots(figsize=(7, 4))
            plt.show(block=False)
            self.update(0.0, {name: 0 for name in lidar_names}, force=True)
        except Exception as exc:
            print(f"Collection plot disabled: {exc}")
            self.enabled = False

    def update(self, elapsed_sec, scan_counts, force=False):
        if not self.enabled or self.figure is None or self.ax is None:
            return

        if not plt.fignum_exists(self.figure.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < 0.5:
            return

        try:
            self.last_update_time = now
            remaining = max(self.collect_duration_sec - elapsed_sec, 0.0)
            labels = []
            for name in self.lidar_names:
                labels.append(f"{name}: {scan_counts.get(name, 0)} scans")

            self.ax.clear()
            self.ax.axis("off")
            self.ax.set_title("Collecting LaserScan data")
            self.ax.text(
                0.5,
                0.62,
                f"elapsed {elapsed_sec:.1f}s / {self.collect_duration_sec:.1f}s",
                ha="center",
                va="center",
                fontsize=13,
            )
            self.ax.text(
                0.5,
                0.46,
                f"remaining {remaining:.1f}s",
                ha="center",
                va="center",
                fontsize=11,
            )
            self.ax.text(
                0.5,
                0.26,
                "\n".join(labels),
                ha="center",
                va="center",
                fontsize=10,
            )
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:
            print(f"Collection plot disabled: {exc}")
            self.enabled = False

    def close(self):
        if self.figure is not None and plt.fignum_exists(self.figure.number):
            plt.close(self.figure)
        self.figure = None
        self.ax = None
        self.enabled = False


class LidarNdtCalib(Node):
    def __init__(self):
        super().__init__("lidar_ndt_calib")

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
        plot_cfg = self.calib_config.get("plot", {})
        self.show_live_ndt = bool(plot_cfg.get("show_live_ndt", False))

        self.cloud_buffers = {
            lidar["name"]: [] for lidar in self.lidars
        }

        self.start_time = self.get_clock().now()
        self.wall_start_time = time.perf_counter()
        self.finished = False
        self.subscribers = []
        self.collection_plotter = None

        self.get_logger().info(
            f"Loaded {len(self.lidars)} lidar(s): {', '.join(self.lidar_names)}"
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
        if self.show_live_ndt:
            self.collection_plotter = CollectionPlotter(
                self.lidar_names,
                self.collect_duration_sec,
            )

        self.get_logger().info(
            f"Collecting LaserScan data for {self.collect_duration_sec} sec..."
        )

    def scan_callback(self, msg, lidar_name):
        if self.finished:
            return

        points = laserscan_to_xy(msg)

        if len(points) == 0:
            return

        self.cloud_buffers[lidar_name].append(points)

    def timer_callback(self):
        if self.finished:
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        if self.collection_plotter is not None:
            self.collection_plotter.update(
                elapsed,
                {
                    name: len(chunks)
                    for name, chunks in self.cloud_buffers.items()
                },
            )

        if elapsed < self.collect_duration_sec:
            return

        self.finished = True
        self.get_logger().info("Collection finished. Running calibration...")
        if self.collection_plotter is not None:
            self.collection_plotter.close()
            self.collection_plotter = None

        self.run_calibration()
        rclpy.shutdown()

    def run_calibration(self):
        calibration_start_time = time.perf_counter()
        collection_elapsed_sec = calibration_start_time - self.wall_start_time

        ndt_cfg = self.calib_config.get("ndt", {})

        resolution = float(ndt_cfg.get("resolution", 0.2))
        min_points = int(ndt_cfg.get("min_points_per_cell", 3))
        downsample_voxel = float(ndt_cfg.get("downsample_voxel", 0.03))
        plot_grid_resolution = None
        plot_cfg = self.calib_config.get("plot", {})
        show_ndt_grid = bool(
            plot_cfg.get(
                "show_ndt_grid",
                self.calib_config.get("plot_ndt_grid", True),
            )
        )
        if show_ndt_grid:
            plot_grid_resolution = resolution
        show_live_ndt = bool(plot_cfg.get("show_live_ndt", False))
        live_update_interval_sec = float(
            plot_cfg.get("live_update_interval_sec", 1.0)
        )
        live_max_points = int(plot_cfg.get("live_max_points", 2000))
        keep_live_ndt_open = bool(plot_cfg.get("keep_live_ndt_open", False))

        search_cfg = self.calib_config.get("search", {})
        search_stages = search_cfg.get(
            "stages",
            default_search_stages(),
        )
        output_dir = resolve_base_path(
            self.calib_config.get("output_dir", "output")
        )
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
            self.calib_config.get(
                "before_plot_png",
                "before_calibration.png",
            ),
        )
        after_plot_path = resolve_output_path(
            output_dir,
            self.calib_config.get(
                "after_plot_png",
                "after_calibration.png",
            ),
        )
        for plot_path in (before_plot_path, after_plot_path):
            if os.path.exists(plot_path):
                os.remove(plot_path)

        raw_clouds = {}

        for name, chunks in self.cloud_buffers.items():
            if len(chunks) == 0:
                self.get_logger().warn(f"{name}: no scan received")
                raw_clouds[name] = np.empty((0, 2))
                continue

            points = np.vstack(chunks)
            points = downsample_xy(points, downsample_voxel)
            raw_clouds[name] = points

            self.get_logger().info(
                f"{name}: {len(points)} points after downsample"
            )

        before_clouds = {}

        for name, points in raw_clouds.items():
            pose = self.lidar_poses[name]

            before_clouds[name] = transform_xy(
                points,
                float(pose["x"]),
                float(pose["y"]),
                float(pose["yaw"]),
                pose_roll(pose),
            )

        reference_cloud = before_clouds[self.reference_lidar]

        target_ndt = build_ndt_grid(
            reference_cloud,
            resolution=resolution,
            min_points=min_points,
        )

        self.get_logger().info(f"NDT cells: {len(target_ndt)}")

        if len(target_ndt) == 0:
            self.get_logger().error(
                "NDT grid is empty. Increase collect_duration_sec, "
                "increase ndt.resolution, or reduce ndt.min_points_per_cell."
            )
            plot_clouds(
                "Before Calibration",
                before_clouds,
                before_plot_path,
                cloud_axis_limits([before_clouds]),
                plot_grid_resolution,
                self.lidar_poses,
                "red",
                options=plot_cfg,
            )
            result_yaml = make_initial_result_yaml(
                self.reference_lidar,
                self.lidars,
                self.lidar_poses,
            )
            result_yaml["success"] = False
            result_yaml["reason"] = "empty_ndt_grid"
            finish_time = time.perf_counter()
            result_yaml["timing_sec"] = {
                "collection": collection_elapsed_sec,
                "calibration": finish_time - calibration_start_time,
                "total": finish_time - self.wall_start_time,
            }
            with open(output_yaml, "w") as f:
                yaml.dump(result_yaml, f, sort_keys=False)
            calibrated_config = make_calibrated_config(
                self.lidar_config,
                result_yaml,
            )
            with open(calibrated_config_yaml, "w") as f:
                yaml.dump(calibrated_config, f, sort_keys=False)
            self.get_logger().info(f"Saved failed result: {output_yaml}")
            self.get_logger().info(
                f"Saved calibrated config: {calibrated_config_yaml}"
            )
            self.get_logger().info(
                "Timing: "
                f"collection={result_yaml['timing_sec']['collection']:.3f}s, "
                f"calibration={result_yaml['timing_sec']['calibration']:.3f}s, "
                f"total={result_yaml['timing_sec']['total']:.3f}s"
            )
            return

        result_yaml = make_initial_result_yaml(
            self.reference_lidar,
            self.lidars,
            self.lidar_poses,
        )
        result_yaml["success"] = True
        result_yaml["calibration_mode"] = "sequential_fused_ndt"
        result_yaml["calibration_order"] = [
            lidar["name"] for lidar in self.lidars
        ]
        result_yaml["timing_sec"] = {
            "collection": collection_elapsed_sec,
            "per_lidar_optimization": {},
        }

        after_clouds = {
            self.reference_lidar: reference_cloud,
        }
        fused_cloud = reference_cloud
        kept_live_plotters = []

        for lidar in self.lidars:
            name = lidar["name"]
            points = raw_clouds[name]

            if name == self.reference_lidar:
                continue

            if len(points) == 0:
                pose = self.lidar_poses[name]
                result_yaml["lidars"][name] = {
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "roll": pose_roll(pose),
                    "yaw": float(pose["yaw"]),
                    "success": False,
                    "optimized": False,
                    "reason": "no_scan_received",
                }
                continue

            target_ndt = build_ndt_grid(
                fused_cloud,
                resolution=resolution,
                min_points=min_points,
            )

            self.get_logger().info(
                f"Optimizing {name} against fused map "
                f"({len(fused_cloud)} points, {len(target_ndt)} NDT cells)..."
            )

            if len(target_ndt) == 0:
                pose = self.lidar_poses[name]
                result_yaml["success"] = False
                result_yaml["lidars"][name] = {
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "roll": pose_roll(pose),
                    "yaw": float(pose["yaw"]),
                    "success": False,
                    "optimized": False,
                    "reason": "empty_fused_ndt_grid",
                }
                continue

            init_pose = self.lidar_poses[name]
            lidar_optimization_start = time.perf_counter()
            initial_cloud = transform_xy(
                points,
                float(init_pose["x"]),
                float(init_pose["y"]),
                float(init_pose["yaw"]),
                pose_roll(init_pose),
            )
            initial_score = ndt_score(initial_cloud, target_ndt, resolution)
            self.get_logger().info(
                f"{name} initial score: {initial_score:.4f} "
                f"(x={float(init_pose['x']):.4f}, "
                f"y={float(init_pose['y']):.4f}, "
                f"yaw={float(init_pose['yaw']):.6f})"
            )
            live_plotter = None
            if show_live_ndt:
                live_axis_limits = cloud_axis_limits([
                    {
                        "target": fused_cloud,
                        name: initial_cloud,
                    }
                ])
                live_plotter = LiveNdtPlotter(
                    lidar_name=name,
                    target_points=fused_cloud,
                    axis_limits=live_axis_limits,
                    grid_resolution=plot_grid_resolution,
                    update_interval_sec=live_update_interval_sec,
                    max_points=live_max_points,
                )
                live_plotter.update(
                    stage_idx=0,
                    candidate_points=initial_cloud,
                    best={
                        "x": float(init_pose["x"]),
                        "y": float(init_pose["y"]),
                        "yaw": float(init_pose["yaw"]),
                        "score": initial_score,
                    },
                    force=True,
                )
                if not live_plotter.enabled:
                    self.get_logger().warn(
                        "Live NDT plot is disabled. Check matplotlib backend "
                        f"({matplotlib.get_backend()}). "
                        f"Reason: {live_plotter.disable_reason}"
                    )

            result = optimize_lidar_grid_search(
                raw_points=points,
                target_ndt=target_ndt,
                init_pose=init_pose,
                resolution=resolution,
                search_stages=search_stages,
                logger=self.get_logger(),
                live_plotter=live_plotter,
                initial_score=initial_score,
            )
            if live_plotter is not None:
                live_plotter.update(
                    stage_idx=len(search_stages) - 1,
                    candidate_points=transform_xy(
                        points,
                        result["x"],
                        result["y"],
                        result["yaw"],
                        result["roll"],
                    ),
                    best=result,
                    force=True,
                )
                if keep_live_ndt_open and live_plotter.enabled:
                    kept_live_plotters.append(live_plotter)
                else:
                    live_plotter.close()
            lidar_optimization_sec = (
                time.perf_counter() - lidar_optimization_start
            )
            result_yaml["timing_sec"]["per_lidar_optimization"][name] = (
                lidar_optimization_sec
            )

            self.get_logger().info(
                f"{name} result: "
                f"x={result['x']:.4f}, "
                f"y={result['y']:.4f}, "
                f"yaw={result['yaw']:.6f}, "
                f"score={result['score']:.4f}, "
                f"time={lidar_optimization_sec:.3f}s"
            )

            result_yaml["lidars"][name] = {
                "x": result["x"],
                "y": result["y"],
                "roll": result["roll"],
                "yaw": result["yaw"],
                "initial_score": initial_score,
                "score": result["score"],
                "score_improvement": initial_score - result["score"],
                "optimization_time_sec": lidar_optimization_sec,
                "success": result["success"],
                "optimized": True,
            }

            after_clouds[name] = transform_xy(
                points,
                result["x"],
                result["y"],
                result["yaw"],
                result["roll"],
            )
            fused_cloud = np.vstack([fused_cloud, after_clouds[name]])
            fused_cloud = downsample_xy(fused_cloud, downsample_voxel)

        plot_limits = cloud_axis_limits([before_clouds, after_clouds])

        plot_clouds(
            "Before Calibration",
            before_clouds,
            before_plot_path,
            plot_limits,
            plot_grid_resolution,
            self.lidar_poses,
            "red",
            options=plot_cfg,
        )

        plot_clouds(
            "After Calibration",
            after_clouds,
            after_plot_path,
            plot_limits,
            plot_grid_resolution,
            make_pose_map_from_result(result_yaml),
            "black",
            self.lidar_poses,
            "red",
            options=plot_cfg,
        )

        with open(output_yaml, "w") as f:
            finish_time = time.perf_counter()
            result_yaml["timing_sec"]["calibration"] = (
                finish_time - calibration_start_time
            )
            result_yaml["timing_sec"]["total"] = (
                finish_time - self.wall_start_time
            )
            yaml.dump(result_yaml, f, sort_keys=False)

        calibrated_config = make_calibrated_config(self.lidar_config, result_yaml)
        with open(calibrated_config_yaml, "w") as f:
            yaml.dump(calibrated_config, f, sort_keys=False)

        self.get_logger().info(f"Saved result: {output_yaml}")
        self.get_logger().info(f"Saved calibrated config: {calibrated_config_yaml}")
        self.get_logger().info(f"Saved plot: {before_plot_path}")
        self.get_logger().info(f"Saved plot: {after_plot_path}")
        self.get_logger().info(
            "Timing: "
            f"collection={result_yaml['timing_sec']['collection']:.3f}s, "
            f"calibration={result_yaml['timing_sec']['calibration']:.3f}s, "
            f"total={result_yaml['timing_sec']['total']:.3f}s"
        )
        if kept_live_plotters:
            self.get_logger().info(
                "Keeping live NDT plot open. Close the plot window to finish."
            )
            plt.show(block=True)
            for live_plotter in kept_live_plotters:
                live_plotter.close()


def main():
    rclpy.init()
    node = LidarNdtCalib()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
