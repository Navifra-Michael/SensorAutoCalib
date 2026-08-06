#!/usr/bin/env python3
"""Estimate LiDAR roll, pitch and z from circular sections of upright cones.

Pipeline:
  1. Keep LaserScan returns inside a configured distance interval.
  2. Split consecutive returns into object clusters.
  3. Extract one circle from each cluster with RANSAC + least-squares refinement.
  4. Convert circle radius r to cone height h = H * (1 - r / R).
  5. Fit z = a*x + b*y + c through all (circle_x, circle_y, h) centers.
  6. Convert the plane normal to roll/pitch and use c as LiDAR height z.

The method implements the requested circular-section approximation. A tilted
plane intersects an ideal cone in a conic rather than an exact circle, so the
reported circle RMSE and plane RMSE should always be checked.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import signal
import sys
import threading
import time
import webbrowser
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class Circle:
    x: float
    y: float
    radius: float
    rmse: float
    inlier_count: int
    point_count: int


@dataclass
class ConeSection:
    circle: Circle
    height: float


@dataclass
class PlaneResult:
    a: float
    b: float
    c: float
    normal: np.ndarray
    roll: float
    pitch: float
    rmse: float


def ignore_worker_sigint() -> None:
    """Let the parent perform orderly realtime shutdown on Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pose_rotation_matrix(pose: dict) -> np.ndarray:
    """Return robot-frame Rz(yaw) Ry(pitch) Rx(roll)."""
    roll = float(pose.get("roll", 0.0))
    pitch = float(pose.get("pitch", 0.0))
    yaw = float(pose.get("yaw", 0.0))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def transform_to_robot(points: np.ndarray, pose: dict) -> np.ndarray:
    """Transform Nx3 points from a LiDAR frame into the robot frame."""
    translation = np.array((float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                            float(pose.get("z", 0.0))))
    return points @ pose_rotation_matrix(pose).T + translation


def robot_scan_plane(pose: dict) -> Tuple[float, float, float]:
    """Return robot-frame scan plane z = a*x + b*y + c."""
    normal = pose_rotation_matrix(pose)[:, 2]
    if abs(normal[2]) < 1e-8:
        raise ValueError("LiDAR scan plane is vertical and cannot be expressed as z(x,y)")
    tx, ty, tz = (float(pose.get(key, 0.0)) for key in ("x", "y", "z"))
    a = -normal[0] / normal[2]
    b = -normal[1] / normal[2]
    c = tz - a * tx - b * ty
    return float(a), float(b), float(c)


def load_offline(path: Path, polar: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        data = archive[key]
    else:
        data = np.genfromtxt(path, delimiter=",", comments="#")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("input must contain at least two columns")
    data = data[:, :2]
    if polar:
        angles, ranges = data[:, 0], data[:, 1]
        data = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
    return data


def collect_ros_scan(topic: str, duration: float) -> np.ndarray:
    """Collect stationary scans and return one angle-ordered median scan."""
    try:
        import rclpy
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        raise RuntimeError("ROS mode requires rclpy and sensor_msgs") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("cone_circle_calibrator")
    frames: List[Tuple[np.ndarray, float, float]] = []

    def callback(message: LaserScan) -> None:
        frames.append(
            (np.asarray(message.ranges, dtype=float), message.angle_min, message.angle_increment)
        )

    subscription = node.create_subscription(LaserScan, topic, callback, 10)
    deadline = time.monotonic() + duration
    print(f"collecting stationary scans from {topic} for {duration:.1f} s ...")
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    del subscription
    node.destroy_node()
    rclpy.shutdown()
    if not frames:
        raise RuntimeError(f"no LaserScan received from {topic}")

    length = min(len(frame[0]) for frame in frames)
    ranges = np.vstack([frame[0][:length] for frame in frames])
    ranges[~np.isfinite(ranges)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(ranges, axis=0)
    angles = frames[-1][1] + np.arange(length) * frames[-1][2]
    return np.column_stack((median * np.cos(angles), median * np.sin(angles)))


def collect_ros_scans(topics: dict, duration: float) -> dict:
    """Collect all configured LaserScan topics during one stationary interval."""
    try:
        import rclpy
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        raise RuntimeError("ROS mode requires rclpy and sensor_msgs") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("cone_circle_calibrator")
    frames = {name: [] for name in topics}
    subscriptions = []

    def make_callback(name):
        def callback(message: LaserScan) -> None:
            frames[name].append(
                (np.asarray(message.ranges, dtype=float), message.angle_min, message.angle_increment)
            )
        return callback

    for name, topic in topics.items():
        subscriptions.append(node.create_subscription(LaserScan, topic, make_callback(name), 10))
    print(f"collecting {len(topics)} stationary LiDAR(s) for {duration:.1f} s ...")
    deadline = time.monotonic() + duration
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    del subscriptions

    result = {}
    for name, lidar_frames in frames.items():
        if not lidar_frames:
            print(f"warning: no LaserScan received for {name} ({topics[name]})", file=sys.stderr)
            continue
        length = min(len(frame[0]) for frame in lidar_frames)
        ranges = np.vstack([frame[0][:length] for frame in lidar_frames])
        ranges[~np.isfinite(ranges)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(ranges, axis=0)
        angles = lidar_frames[-1][1] + np.arange(length) * lidar_frames[-1][2]
        result[name] = np.column_stack((median * np.cos(angles), median * np.sin(angles)))
        print(f"  {name}: {len(lidar_frames)} scans")
    return result


def filter_and_cluster(
    points: np.ndarray,
    min_range: float,
    max_range: float,
    cluster_gap: float,
    min_points: int,
    max_width: float,
) -> List[np.ndarray]:
    """Apply range limits, then split angle-ordered points at discontinuities."""
    ranges = np.linalg.norm(points, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & (ranges >= min_range)
        & (ranges <= max_range)
    )
    clusters: List[np.ndarray] = []
    current: List[np.ndarray] = []
    previous = None

    def finish() -> None:
        nonlocal current
        if len(current) >= min_points:
            cluster = np.asarray(current)
            width = float(np.linalg.norm(np.ptp(cluster, axis=0)))
            if width <= max_width:
                clusters.append(cluster)
        current = []

    for point, keep in zip(points, valid):
        if not keep:
            finish()
            previous = None
            continue
        if previous is not None and np.linalg.norm(point - previous) > cluster_gap:
            finish()
        current.append(point)
        previous = point
    finish()
    return clusters


def circle_from_three(points: np.ndarray) -> Tuple[np.ndarray, float] | None:
    p1, p2, p3 = points
    matrix = 2.0 * np.array((p2 - p1, p3 - p1))
    rhs = np.array((np.dot(p2, p2) - np.dot(p1, p1),
                    np.dot(p3, p3) - np.dot(p1, p1)))
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(p1 - center))


def refine_circle(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Algebraic initialization followed by geometric Gauss-Newton fitting."""
    design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = np.sum(points * points, axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2]
    radius = math.sqrt(max(solution[2] + np.dot(center, center), 0.0))

    for _ in range(15):
        delta = points - center
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        residual = distance - radius
        jacobian = np.column_stack((-delta / distance[:, None], -np.ones(len(points))))
        step, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        center += step[:2]
        radius += step[2]
        if np.linalg.norm(step) < 1e-8:
            break
    residual = np.linalg.norm(points - center, axis=1) - radius
    return center, float(radius), float(np.sqrt(np.mean(residual * residual)))


def fit_circle_ransac(
    points: np.ndarray,
    min_radius: float,
    max_radius: float,
    threshold: float,
    iterations: int,
    min_inliers: int,
    rng: np.random.Generator,
) -> Circle | None:
    if len(points) < max(3, min_inliers):
        return None
    best_indices = np.empty(0, dtype=int)
    best_error = float("inf")
    for _ in range(iterations):
        model = circle_from_three(points[rng.choice(len(points), 3, replace=False)])
        if model is None:
            continue
        center, radius = model
        if not min_radius <= radius <= max_radius:
            continue
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        indices = np.flatnonzero(residual <= threshold)
        error = float(np.mean(residual[indices])) if len(indices) else float("inf")
        if len(indices) > len(best_indices) or (
            len(indices) == len(best_indices) and error < best_error
        ):
            best_indices, best_error = indices, error

    if len(best_indices) < min_inliers:
        return None
    center, radius, _ = refine_circle(points[best_indices])
    if not min_radius <= radius <= max_radius:
        return None
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    inliers = residual <= threshold
    if np.count_nonzero(inliers) >= min_inliers:
        center, radius, rmse = refine_circle(points[inliers])
    else:
        return None
    return Circle(
        x=float(center[0]), y=float(center[1]), radius=radius, rmse=rmse,
        inlier_count=int(np.count_nonzero(inliers)), point_count=len(points),
    )


def extract_sections(
    clusters: Sequence[np.ndarray],
    cone_radius: float,
    cone_height: float,
    min_circle_radius: float,
    circle_threshold: float,
    ransac_iterations: int,
    min_inliers: int,
    seed: int,
    diagnostics: List[str] | None = None,
) -> List[ConeSection]:
    rng = np.random.default_rng(seed)
    sections = []
    for index, cluster in enumerate(clusters, 1):
        circle = fit_circle_ransac(
            cluster, min_circle_radius, cone_radius * 1.05,
            circle_threshold, ransac_iterations, min_inliers, rng,
        )
        if circle is None:
            if diagnostics is not None:
                diagnostics.append(
                    f"cluster {index}: rejected by circle RANSAC ({len(cluster)} points)"
                )
            continue
        # The RANSAC model already limits the radius to 1.05*R. Clamp the
        # small tolerance band to the cone base instead of rejecting it.
        effective_radius = min(circle.radius, cone_radius)
        height = cone_height * (1.0 - effective_radius / cone_radius)
        sections.append(ConeSection(circle, float(height)))
        if diagnostics is not None:
            diagnostics.append(
                f"cluster {index}: circle r={circle.radius:.4f} m, "
                f"rmse={circle.rmse:.5f} m, inliers={circle.inlier_count}/{len(cluster)}"
            )
    return sections


def fit_scan_plane(sections: Sequence[ConeSection]) -> PlaneResult:
    if len(sections) < 3:
        raise RuntimeError("at least three extracted circles are required to fit a plane")
    centers = np.array([(s.circle.x, s.circle.y, s.height) for s in sections])
    design = np.column_stack((centers[:, 0], centers[:, 1], np.ones(len(centers))))
    if np.linalg.matrix_rank(design) < 3:
        raise RuntimeError("circle centers are collinear; place cones in different directions")

    # Better circles receive more weight. Iteratively add Huber plane weights.
    base_weights = np.array([1.0 / max(s.circle.rmse, 1e-4) for s in sections])
    weights = base_weights / np.max(base_weights)
    coefficients = np.zeros(3)
    for _ in range(10):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_z = centers[:, 2] * np.sqrt(weights)
        coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
        residual = centers[:, 2] - design @ coefficients
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-4)
        huber = np.minimum(1.0, (2.0 * scale) / np.maximum(np.abs(residual), 1e-12))
        new_weights = base_weights / np.max(base_weights) * huber
        if np.linalg.norm(new_weights - weights) < 1e-6:
            break
        weights = new_weights

    a, b, c = map(float, coefficients)
    residual = centers[:, 2] - design @ coefficients
    normal = np.array((-a, -b, 1.0))
    normal /= np.linalg.norm(normal)

    # With R = Ry(pitch) Rx(roll), local scan-plane normal in world is
    # [sin(pitch)cos(roll), -sin(roll), cos(pitch)cos(roll)].
    roll = math.asin(float(np.clip(-normal[1], -1.0, 1.0)))
    pitch = math.atan2(float(normal[0]), float(normal[2]))
    return PlaneResult(
        a=a, b=b, c=c, normal=normal, roll=roll, pitch=pitch,
        rmse=float(np.sqrt(np.mean(residual * residual))),
    )


def result_dict(
    plane: PlaneResult, sections: Sequence[ConeSection], cluster_count: int
) -> dict:
    return {
        "success": True,
        "roll_deg": math.degrees(plane.roll),
        "pitch_deg": math.degrees(plane.pitch),
        "z_m": plane.c,
        "plane": {
            "equation": "z = a*x + b*y + c",
            "a": plane.a, "b": plane.b, "c": plane.c,
            "normal": plane.normal.tolist(), "rmse_m": plane.rmse,
        },
        "range_limited_cluster_count": cluster_count,
        "extracted_circle_count": len(sections),
        "cones": [
            {
                "circle_center_x_m": s.circle.x,
                "circle_center_y_m": s.circle.y,
                "circle_radius_m": s.circle.radius,
                "inferred_height_m": s.height,
                "circle_rmse_m": s.circle.rmse,
                "inliers": s.circle.inlier_count,
                "cluster_points": s.circle.point_count,
            }
            for s in sections
        ],
        "warning": "tilted cone sections are not exact circles; validate both RMSE values",
    }


def plot_calibration(
    scans: dict,
    lidar_results: dict,
    calib_config: dict,
    lidar_poses: dict,
    save_path: Path | None,
    realtime: bool = False,
    live_figure=None,
):
    """Render robot-frame XY and fixed-z-range YZ projections."""
    plot_cfg = calib_config.get("plot", {})
    try:
        import matplotlib
        if not bool(plot_cfg.get("show", True)):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("warning: matplotlib is not installed; GUI plot skipped", file=sys.stderr)
        return

    plotted_names = [name for name in lidar_results if name in scans]
    if not plotted_names:
        print("warning: no received LiDAR scan to plot", file=sys.stderr)
        return
    max_points = int(plot_cfg.get("max_points", 5000))
    yz_z_min = float(plot_cfg.get("yz_z_min_m", 0.0))
    yz_z_max = float(plot_cfg.get("yz_z_max_m", 2.0))
    yz_z_to_y_scale = float(plot_cfg.get("yz_z_to_y_scale", 2.0))
    if yz_z_max <= yz_z_min:
        raise ValueError("plot.yz_z_max_m must be greater than plot.yz_z_min_m")
    if yz_z_to_y_scale <= 0.0:
        raise ValueError("plot.yz_z_to_y_scale must be positive")
    min_range = float(calib_config.get("range_filter", {}).get("min_m", 0.1))
    max_range = float(calib_config.get("range_filter", {})["max_m"])
    cone_radius = float(calib_config["cone"]["radius_m"])
    cone_height = float(calib_config["cone"]["height_m"])
    figure = live_figure
    if figure is None:
        figure = plt.figure(figsize=(14, max(5, 4.8 * len(plotted_names))))
    else:
        figure.clear()
        figure.set_size_inches(14, max(5, 4.8 * len(plotted_names)), forward=True)
    axes = figure.subplots(
        len(plotted_names), 2, squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    for row, name in enumerate(plotted_names):
        result = lidar_results[name]
        pose = lidar_poses[name]
        points = scans[name]
        ranges = np.linalg.norm(points, axis=1)
        valid = (
            np.isfinite(points).all(axis=1)
            & (ranges >= min_range) & (ranges <= max_range)
        )
        filtered = points[valid]
        if len(filtered) > max_points:
            indices = np.linspace(0, len(filtered) - 1, max_points).astype(int)
            filtered = filtered[indices]
        filtered_robot = transform_to_robot(
            np.column_stack((filtered, np.zeros(len(filtered)))), pose
        )
        cones = result.get("cones", [])
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(cones), 1)))

        # Preserve the original XY scan/circle diagnostic on the left.
        ax_xy = axes[row, 0]
        # Scale XY from detected cones, not raw scan outliers. Scan points are
        # still drawn, but they must not expand the diagnostic view.
        xy_bounds_points = []
        ax_xy.scatter(filtered_robot[:, 0], filtered_robot[:, 1], s=5, c="0.55",
                      label="range-filtered scan")
        for index, (cone, color) in enumerate(zip(cones, colors), 1):
            cx = cone["circle_center_x_m"]
            cy = cone["circle_center_y_m"]
            radius = cone["circle_radius_m"]
            theta = np.linspace(0.0, 2.0 * math.pi, 100)
            ring_local = np.column_stack((
                cx + radius * np.cos(theta), cy + radius * np.sin(theta),
                np.zeros_like(theta),
            ))
            ring_robot = transform_to_robot(ring_local, pose)
            xy_bounds_points.append(ring_robot[:, :2])
            center_robot = transform_to_robot(np.array(((cx, cy, 0.0),)), pose)[0]
            ax_xy.plot(ring_robot[:, 0], ring_robot[:, 1], color=color, linewidth=2)
            ax_xy.scatter(center_robot[0], center_robot[1], marker="x", s=55, color=color)
            ax_xy.annotate(f"C{index}\nr={radius:.3f}", center_robot[:2], color=color)
        ax_xy.scatter(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                      marker="^", s=80, c="red", label="LiDAR")
        ax_xy.scatter(0.0, 0.0, marker="+", s=90, c="black", label="robot origin")
        ax_xy.set_title(f"{name}: circles in robot TF (XY)")
        ax_xy.set_xlabel("robot x [m]")
        ax_xy.set_ylabel("robot y [m]")
        if xy_bounds_points:
            all_xy = np.vstack(xy_bounds_points)
        else:
            lidar_xy = (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
            all_xy = np.array((lidar_xy, lidar_xy))
        x_min, y_min = np.min(all_xy, axis=0)
        x_max, y_max = np.max(all_xy, axis=0)
        equal_span = max(float(x_max - x_min), float(y_max - y_min), 2.0 * cone_radius)
        equal_span *= 1.08
        x_mid = 0.5 * float(x_min + x_max)
        y_mid = 0.5 * float(y_min + y_max)
        ax_xy.set_xlim(x_mid - 0.5 * equal_span, x_mid + 0.5 * equal_span)
        ax_xy.set_ylim(y_mid - 0.5 * equal_span, y_mid + 0.5 * equal_span)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend(loc="best", fontsize=8)

        if not result.get("success", False):
            reason = result.get("reason", "calibration failed")
            ax_xy.set_title(f"{name}: filtered scan (calibration failed)")
            for axis in axes[row, 1:]:
                axis.text(0.5, 0.55, "Calibration failed", ha="center", va="center",
                          fontsize=16, color="tab:red", transform=axis.transAxes)
                axis.text(0.5, 0.43, str(reason), ha="center", va="center", wrap=True,
                          transform=axis.transAxes)
                axis.set_axis_off()
            continue
        plane_a, plane_b, plane_c = robot_scan_plane(pose)
        scan_z = filtered_robot[:, 2]
        centers_robot = np.array([
            transform_to_robot(
                np.array(((cone["circle_center_x_m"], cone["circle_center_y_m"], 0.0),)),
                pose,
            )[0]
            for cone in cones
        ])
        center_x = centers_robot[:, 0]
        center_y = centers_robot[:, 1]
        raw_y_limits = (float(np.min(center_y) - 1.5 * cone_radius),
                        float(np.max(center_y) + 1.5 * cone_radius))
        yz_axis_limit = 1.05 * max(
            abs(raw_y_limits[0]), abs(raw_y_limits[1]), cone_height,
            abs(float(pose.get("y", 0.0))), abs(float(pose.get("z", 0.0))),
        )
        # Keep robot-y symmetric around zero; robot-z has its own configured
        # fixed range (plot.yz_z_min_m through plot.yz_z_max_m).
        y_limits = (-yz_axis_limit, yz_axis_limit)

        for projection, axis, coordinate, limits, slope, other_centers in (
            ("YZ", axes[row, 1], "y", y_limits, plane_b, center_x),
        ):
            coordinate_index = 0 if coordinate == "x" else 1
            axis.scatter(
                filtered_robot[:, coordinate_index], scan_z, s=5, c="0.45", alpha=0.55,
                label="LaserScan points on fitted plane", zorder=2,
            )
            for index, (cone, color) in enumerate(zip(cones, colors), 1):
                center_coordinate = centers_robot[index - 1, coordinate_index]
                height = centers_robot[index - 1, 2]
                section_radius = cone["circle_radius_m"]
                triangle = np.array([
                    (center_coordinate - cone_radius, 0.0),
                    (center_coordinate, cone_height),
                    (center_coordinate + cone_radius, 0.0),
                ])
                axis.add_patch(Polygon(
                    triangle, closed=True, facecolor=color, edgecolor=color,
                    linewidth=1.4, alpha=0.20,
                ))
                axis.plot(
                    [center_coordinate - section_radius, center_coordinate + section_radius],
                    [height, height], color=color, linewidth=3,
                    label="detected cone section" if index == 1 else None,
                )
                axis.scatter(center_coordinate, height, marker="x", s=55, color=color, zorder=4)
                axis.annotate(f"C{index}\nh={height:.3f}", (center_coordinate, height),
                              xytext=(4, 5), textcoords="offset points", color=color)

            horizontal = np.linspace(limits[0], limits[1], 100)
            other_min, other_max = float(np.min(other_centers)), float(np.max(other_centers))
            other_slope = plane_b if coordinate == "x" else plane_a
            lower = slope * horizontal + plane_c + min(other_slope * other_min,
                                                            other_slope * other_max)
            upper = slope * horizontal + plane_c + max(other_slope * other_min,
                                                            other_slope * other_max)
            axis.fill_between(horizontal, lower, upper, color="deepskyblue", alpha=0.22,
                              label="projected scan plane")
            robot_other_origin = float(pose.get("x" if coordinate == "y" else "y", 0.0))
            axis.plot(horizontal, slope * horizontal + plane_c + other_slope * robot_other_origin, "--",
                      color="dodgerblue", linewidth=2, label="plane center slice")
            axis.scatter(float(pose.get(coordinate, 0.0)), float(pose.get("z", 0.0)),
                         marker="D", s=70, c="red",
                         label="LiDAR", zorder=5)
            axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            axis.set_xlim(*limits)
            axis.set_ylim(yz_z_min, yz_z_max)
            axis.set_xlabel(f"robot {coordinate} [m]")
            axis.set_ylabel("height z [m]")
            # Matplotlib aspect is the displayed z-unit/y-unit length ratio.
            # 2.0 means horizontal y : vertical z = 1 : 2.
            axis.set_aspect(yz_z_to_y_scale, adjustable="box")
            axis.set_title(
                f"{name}: {projection} projection\n"
                f"roll={result['roll_deg']:.3f}°, pitch={result['pitch_deg']:.3f}°, "
                f"z={result['z_m']:.3f} m"
            )
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)

    figure.suptitle("Cone calibration in robot TF: XY circles and YZ projection", fontsize=14)
    if realtime:
        figure.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.92,
                               wspace=0.20, hspace=0.30)
    else:
        figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=int(plot_cfg.get("dpi", 150)), bbox_inches="tight")
        print(f"plot: {save_path}")
    if realtime:
        figure.canvas.draw_idle()
        plt.show(block=False)
        return figure
    if bool(plot_cfg.get("show", True)) and bool(plot_cfg.get("show_2d", True)):
        backend = matplotlib.get_backend().lower()
        non_gui_backends = {"agg", "pdf", "ps", "svg", "template", "cairo"}
        if backend in non_gui_backends or backend.endswith("backend_agg"):
            print(
                f"warning: matplotlib backend '{matplotlib.get_backend()}' has no GUI; PNG was saved only. "
                "Run in a desktop session with DISPLAY/WAYLAND_DISPLAY set.",
                file=sys.stderr,
            )
        else:
            plt.show()
    plt.close(figure)
    return None


def plot_calibration_3d(
    scans: dict,
    lidar_results: dict,
    calib_config: dict,
    lidar_poses: dict,
    save_path: Path,
) -> None:
    """Create an interactive Plotly view of cones and the LiDAR scan plane."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("warning: plotly is not installed; interactive 3D plot skipped", file=sys.stderr)
        return

    names = [
        name for name, result in lidar_results.items()
        if result.get("success", False) and name in scans
    ]
    if not names:
        print("warning: no successful LiDAR result for interactive 3D plot", file=sys.stderr)
        return
    figure = make_subplots(
        rows=len(names), cols=1,
        specs=[[{"type": "scene"}] for _ in names],
        subplot_titles=[f"{name}: cones and fitted LaserScan plane" for name in names],
        vertical_spacing=0.08,
    )
    cone_radius = float(calib_config["cone"]["radius_m"])
    cone_height = float(calib_config["cone"]["height_m"])
    range_cfg = calib_config.get("range_filter", {})
    min_range = float(range_cfg.get("min_m", 0.1))
    max_range = float(range_cfg["max_m"])
    max_points = int(calib_config.get("plot", {}).get("max_points", 5000))
    theta = np.linspace(0.0, 2.0 * math.pi, 50)
    cone_z = np.linspace(0.0, cone_height, 24)
    theta_grid, z_grid = np.meshgrid(theta, cone_z)

    for row, name in enumerate(names, 1):
        result = lidar_results[name]
        pose = lidar_poses[name]
        cones = result["cones"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        centers_robot = np.array([
            transform_to_robot(
                np.array(((cone["circle_center_x_m"], cone["circle_center_y_m"], 0.0),)),
                pose,
            )[0]
            for cone in cones
        ])
        for index, cone in enumerate(cones, 1):
            cx, cy, inferred_z = centers_robot[index - 1]
            radial_grid = cone_radius * (1.0 - z_grid / cone_height)
            x_grid = cx + radial_grid * np.cos(theta_grid)
            y_grid = cy + radial_grid * np.sin(theta_grid)
            color = colors[(index - 1) % len(colors)]
            figure.add_trace(
                go.Surface(
                    x=x_grid, y=y_grid, z=z_grid,
                    surfacecolor=np.full_like(z_grid, index),
                    colorscale=[[0.0, color], [1.0, color]],
                    opacity=0.35, showscale=False, name=f"cone {index}",
                    hoverinfo="skip",
                ), row=row, col=1,
            )
            section_z = float(inferred_z)
            section_r = float(cone["circle_radius_m"])
            figure.add_trace(
                go.Scatter3d(
                    x=cx + section_r * np.cos(theta),
                    y=cy + section_r * np.sin(theta),
                    z=np.full_like(theta, section_z),
                    mode="lines", line={"color": color, "width": 7},
                    name=f"C{index} detected section",
                ), row=row, col=1,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[cx], y=[cy], z=[section_z], mode="markers+text",
                    marker={"size": 5, "color": color}, text=[f"C{index}"],
                    textposition="top center", name=f"C{index} center",
                ), row=row, col=1,
            )

        centers = centers_robot[:, :2]
        x_margin = max(float(np.ptp(centers[:, 0])) * 0.12, cone_radius)
        y_margin = max(float(np.ptp(centers[:, 1])) * 0.12, cone_radius)
        plane_x = np.linspace(np.min(centers[:, 0]) - x_margin, np.max(centers[:, 0]) + x_margin, 22)
        plane_y = np.linspace(np.min(centers[:, 1]) - y_margin, np.max(centers[:, 1]) + y_margin, 22)
        plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
        plane_a, plane_b, plane_c = robot_scan_plane(pose)
        plane_zz = plane_a * plane_xx + plane_b * plane_yy + plane_c
        figure.add_trace(
            go.Surface(
                x=plane_xx, y=plane_yy, z=plane_zz,
                colorscale=[[0.0, "#00a6ff"], [1.0, "#00a6ff"]],
                opacity=0.38, showscale=False, name="fitted scan plane",
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>scan plane</extra>",
            ), row=row, col=1,
        )

        points = scans[name]
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(points).all(axis=1) & (ranges >= min_range) & (ranges <= max_range)
        scan_points = points[valid]
        if len(scan_points) > max_points:
            selected = np.linspace(0, len(scan_points) - 1, max_points).astype(int)
            scan_points = scan_points[selected]
        scan_robot = transform_to_robot(
            np.column_stack((scan_points, np.zeros(len(scan_points)))), pose
        )
        figure.add_trace(
            go.Scatter3d(
                x=scan_robot[:, 0], y=scan_robot[:, 1], z=scan_robot[:, 2],
                mode="markers", marker={"size": 2, "color": "#333333"},
                name=f"{name} scan points",
            ), row=row, col=1,
        )
        figure.add_trace(
            go.Scatter3d(
                x=[float(pose.get("x", 0.0))], y=[float(pose.get("y", 0.0))],
                z=[float(pose.get("z", 0.0))], mode="markers+text",
                marker={"size": 8, "color": "red", "symbol": "diamond"},
                text=["LiDAR"], textposition="top center", name=f"{name} LiDAR",
            ), row=row, col=1,
        )
        scene_name = "scene" if row == 1 else f"scene{row}"
        figure.layout[scene_name].update(
            xaxis_title="robot x [m]", yaxis_title="robot y [m]", zaxis_title="robot z [m]",
            aspectmode="data", camera={"eye": {"x": 1.45, "y": -1.45, "z": 0.9}},
        )

    figure.update_layout(
        title="Cone calibration in robot TF: 3D cones and fitted LaserScan planes",
        height=max(650, 620 * len(names)), showlegend=True,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    show = bool(calib_config.get("plot", {}).get("show", True))
    figure.write_html(str(save_path), include_plotlyjs=True, auto_open=show)
    print(f"interactive 3D plot: {save_path}")


def plot_combined_calibration_3d(
    scans: dict,
    lidar_results: dict,
    calib_config: dict,
    lidar_poses: dict,
    save_path: Path | None,
    auto_open: bool | None = None,
    auto_refresh_sec: float | None = None,
) -> None:
    """Overlay every calibrated LiDAR in one interactive robot-frame scene."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("warning: plotly is not installed; combined 3D plot skipped", file=sys.stderr)
        return
    names = [name for name, result in lidar_results.items()
             if result.get("success", False) and name in scans]
    if not names:
        return
    figure = go.Figure()
    radius = float(calib_config["cone"]["radius_m"])
    height = float(calib_config["cone"]["height_m"])
    palettes = [
        ("#1f77b4", "#17becf", "Blues"),
        ("#ff7f0e", "#d62728", "Oranges"),
        ("#2ca02c", "#9467bd", "Greens"),
    ]
    theta = np.linspace(0.0, 2.0 * math.pi, 28)
    cone_z = np.linspace(0.0, height, 10)
    theta_grid, z_grid = np.meshgrid(theta, cone_z)

    # Associate detections by robot-frame cone-axis XY and render one physical
    # cone per group. Different LiDAR section rings remain visible at their
    # respective heights on that shared cone.
    merge_distance = float(
        calib_config.get("plot", {}).get("combined_cone_merge_distance_m", 0.6)
    )
    detections = []
    for name in names:
        pose = lidar_poses[name]
        for cone_index, cone in enumerate(lidar_results[name]["cones"], 1):
            center = transform_to_robot(
                np.array(((cone["circle_center_x_m"], cone["circle_center_y_m"], 0.0),)), pose
            )[0]
            detections.append((name, cone_index, center))
    groups: List[List[Tuple[str, int, np.ndarray]]] = []
    for detection in detections:
        best_group, best_distance = None, float("inf")
        for group_index, group in enumerate(groups):
            group_xy = np.mean([item[2][:2] for item in group], axis=0)
            distance = float(np.linalg.norm(detection[2][:2] - group_xy))
            if distance < best_distance:
                best_group, best_distance = group_index, distance
        if best_group is not None and best_distance <= merge_distance:
            groups[best_group].append(detection)
        else:
            groups.append([detection])
    associated_xy = {}
    radial = radius * (1.0 - z_grid / height)
    for physical_index, group in enumerate(groups, 1):
        center_xy = np.mean([item[2][:2] for item in group], axis=0)
        for name, cone_index, _ in group:
            associated_xy[(name, cone_index)] = center_xy
        figure.add_trace(go.Surface(
            x=center_xy[0] + radial * np.cos(theta_grid),
            y=center_xy[1] + radial * np.sin(theta_grid), z=z_grid,
            colorscale="Greys", opacity=0.30, showscale=False,
            name=f"physical cone P{physical_index}", hoverinfo="skip",
            legendgroup="physical_cones",
        ))
        figure.add_trace(go.Scatter3d(
            x=[center_xy[0]], y=[center_xy[1]], z=[0.0], mode="markers+text",
            marker={"size": 6, "color": "black"}, text=[f"P{physical_index}"],
            textposition="bottom center", name=f"P{physical_index} base center",
            legendgroup="physical_cones",
        ))

    for lidar_index, name in enumerate(names):
        result, pose = lidar_results[name], lidar_poses[name]
        primary, secondary, _ = palettes[lidar_index % len(palettes)]
        centers = np.array([
            transform_to_robot(np.array(((c["circle_center_x_m"], c["circle_center_y_m"], 0.0),)), pose)[0]
            for c in result["cones"]
        ])
        for cone_index, (cone, center) in enumerate(zip(result["cones"], centers), 1):
            axis_xy = associated_xy[(name, cone_index)]
            section_radius = float(cone["circle_radius_m"])
            figure.add_trace(go.Scatter3d(
                x=axis_xy[0] + section_radius * np.cos(theta),
                y=axis_xy[1] + section_radius * np.sin(theta),
                z=np.full_like(theta, center[2]), mode="lines",
                line={"color": primary, "width": 6},
                name=f"{name} C{cone_index} section", legendgroup=name,
            ))
            figure.add_trace(go.Scatter3d(
                x=[axis_xy[0]], y=[axis_xy[1]], z=[center[2]], mode="markers+text",
                marker={"size": 5, "color": primary}, text=[f"{name}:C{cone_index}"],
                textposition="top center", name=f"{name} C{cone_index} center",
                legendgroup=name,
            ))

        margin_x = max(float(np.ptp(centers[:, 0])) * 0.12, radius)
        margin_y = max(float(np.ptp(centers[:, 1])) * 0.12, radius)
        px = np.linspace(np.min(centers[:, 0]) - margin_x, np.max(centers[:, 0]) + margin_x, 10)
        py = np.linspace(np.min(centers[:, 1]) - margin_y, np.max(centers[:, 1]) + margin_y, 10)
        pxx, pyy = np.meshgrid(px, py)
        a, b, c = robot_scan_plane(pose)
        figure.add_trace(go.Surface(
            x=pxx, y=pyy, z=a * pxx + b * pyy + c,
            colorscale=[[0.0, secondary], [1.0, secondary]], opacity=0.22,
            showscale=False, name=f"{name} scan plane", legendgroup=name,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>" + name + " plane</extra>",
        ))
        figure.add_trace(go.Scatter3d(
            x=[float(pose.get("x", 0.0))], y=[float(pose.get("y", 0.0))],
            z=[float(pose.get("z", 0.0))], mode="markers+text",
            marker={"size": 9, "color": secondary, "symbol": "diamond"},
            text=[name], textposition="top center", name=f"{name} pose", legendgroup=name,
        ))

    figure.update_layout(
        title="Combined cone calibration in robot TF",
        scene={
            "xaxis_title": "robot x [m]", "yaxis_title": "robot y [m]",
            "zaxis_title": "robot z [m]", "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.45, "z": 0.9}},
        },
        height=800, showlegend=True, margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        show = bool(calib_config.get("plot", {}).get("show", True)) if auto_open is None else auto_open
        figure.write_html(str(save_path), include_plotlyjs=True, auto_open=show)
        print(f"combined interactive 3D plot: {save_path}")
    return figure


class _QuietHttpHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass


def start_realtime_plotly_server(
    output_dir: Path,
    interval: float,
    host: str = "0.0.0.0",
    port: int = 8050,
    public_host: str = "localhost",
    auto_open: bool = False,
):
    """Serve one stable Plotly page whose traces update without page reloads."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return None, None, None
    state_name = "realtime_plotly_state.json"
    page_path = output_dir / "realtime_cone_calibration.html"
    poll_ms = max(100, int(interval * 1000.0))
    page_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Realtime cone calibration</title>"
        f"<script>{get_plotlyjs()}</script></head><body style='margin:0'>"
        "<div id='plot' style='width:100vw;height:100vh'></div><script>"
        "let initialized=false; async function update(){try{"
        f"const r=await fetch('/{state_name}?t='+Date.now(),{{cache:'no-store'}});"
        "if(r.ok){const f=await r.json();Plotly.react('plot',f.data,f.layout,f.config||{});"
        "initialized=true;}}catch(e){if(!initialized)console.log(e);}}"
        f"update();setInterval(update,{poll_ms});</script></body></html>",
        encoding="utf-8",
    )
    handler = partial(_QuietHttpHandler, directory=str(output_dir))
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{public_host}:{server.server_port}/{page_path.name}"
    print(f"realtime Plotly URL: {url}", flush=True)
    if auto_open and not webbrowser.open(url):
        print(
            f"warning: could not open a browser; open {url} manually",
            file=sys.stderr,
            flush=True,
        )
    return server, output_dir / state_name, page_path


def update_realtime_plotly_state(figure, state_path: Path) -> None:
    """Atomically replace Plotly state so the browser never reads a partial update."""
    state = json.loads(figure.to_json())
    state["config"] = {"responsive": True, "displaylogo": False}
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    temporary.replace(state_path)


def render_realtime_plotly_state(
    scans: dict, results: dict, config: dict, poses: dict, state_path: Path
) -> None:
    """Build and publish Plotly state away from the ROS/Matplotlib thread."""
    figure = plot_combined_calibration_3d(scans, results, config, poses, None)
    if figure is not None:
        update_realtime_plotly_state(figure, state_path)


class RealtimeLocalPlot:
    """Lightweight persistent Matplotlib artists for low-latency updates."""

    def __init__(self, names: Sequence[str], config: dict):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.plt = plt
        self.names = list(names)
        self.config = config
        self.radius = float(config["cone"]["radius_m"])
        self.height = float(config["cone"]["height_m"])
        self.z_min = float(config.get("plot", {}).get("yz_z_min_m", 0.0))
        self.z_max = float(config.get("plot", {}).get("yz_z_max_m", 2.0))
        self.figure, axes = plt.subplots(
            len(self.names), 2, figsize=(16, max(5.5, 5.4 * len(self.names))), squeeze=False
        )
        self.artists = {}
        self.dynamic_artists = {}
        self.backgrounds = None
        self.success_seen = set()
        self.title_values = {}
        for row, name in enumerate(self.names):
            xy, yz = axes[row]
            xy_scan = xy.scatter([], [], s=4, c="0.55")
            xy_rings = LineCollection([], linewidths=2, colors="tab:blue")
            xy.add_collection(xy_rings)
            xy_centers = xy.scatter([], [], marker="x", s=50, c="tab:blue")
            xy_lidar = xy.scatter([], [], marker="^", s=75, c="red")
            xy.scatter([0.0], [0.0], marker="+", s=80, c="black")
            xy.set_xlabel("robot x [m]"); xy.set_ylabel("robot y [m]")
            xy.grid(True, alpha=0.3); xy.set_aspect("equal", adjustable="box")
            xy.set_xlim(-1.0, 1.0); xy.set_ylim(-1.0, 1.0)
            xy_title = xy.set_title(f"{name}: newest scan / circles", fontsize=10, pad=8)

            yz_scan = yz.scatter([], [], s=4, c="0.45", alpha=0.55)
            yz_cones = LineCollection([], linewidths=1.2, colors="tab:blue", alpha=0.35)
            yz_sections = LineCollection([], linewidths=3, colors="tab:blue")
            yz.add_collection(yz_cones); yz.add_collection(yz_sections)
            yz_centers = yz.scatter([], [], marker="x", s=50, c="tab:blue")
            yz_lidar = yz.scatter([], [], marker="D", s=65, c="red")
            yz_plane, = yz.plot([], [], "--", color="dodgerblue", linewidth=2)
            yz.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            yz.set_ylim(self.z_min, self.z_max)
            yz.set_xlim(-1.0, 1.0)
            yz.set_xlabel("robot y [m]"); yz.set_ylabel("robot z [m]")
            yz.grid(True, alpha=0.3)
            yz.set_aspect(float(config.get("plot", {}).get("yz_z_to_y_scale", 2.0)),
                          adjustable="box")
            yz_title = yz.set_title(f"{name}: waiting for calibration", fontsize=10, pad=8)
            self.artists[name] = (
                xy, yz, xy_scan, xy_rings, xy_centers, xy_lidar,
                yz_scan, yz_cones, yz_sections, yz_centers, yz_lidar, yz_plane,
            )
            dynamic = (xy_scan, xy_rings, xy_centers, xy_lidar, xy_title,
                       yz_scan, yz_cones, yz_sections, yz_centers, yz_lidar,
                       yz_plane, yz_title)
            for artist in dynamic:
                artist.set_animated(True)
            self.dynamic_artists[name] = dynamic
            self.title_values[name] = {
                "xy": f"{name}: newest scan / circles",
                "yz": f"{name}: waiting for calibration",
            }
        self.figure.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92,
                                    wspace=0.34, hspace=0.48)
        self.figure.suptitle("Realtime cone calibration in robot TF")
        self.figure.canvas.mpl_connect("resize_event", self._invalidate_background)
        plt.ion(); plt.show(block=False)
        manager = self.figure.canvas.manager
        if manager is not None:
            manager.show()
            window = getattr(manager, "window", None)
            if window is not None:
                if hasattr(window, "showNormal"): window.showNormal()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _invalidate_background(self, _event=None) -> None:
        self.backgrounds = None

    def _update_title(self, name: str, projection: str, value: str) -> None:
        """Change title text only when its rendered value actually changed."""
        if self.title_values[name][projection] == value:
            return
        self.title_values[name][projection] = value
        title_index = 4 if projection == "xy" else 11
        self.dynamic_artists[name][title_index].set_text(value)

    def update(self, scans: dict, results: dict, poses: dict) -> None:
        newly_successful = {
            name for name in self.names
            if results.get(name, {}).get("success", False) and name not in self.success_seen
        }
        if newly_successful:
            self.success_seen.update(newly_successful)
            self.backgrounds = None
        theta = np.linspace(0.0, 2.0 * math.pi, 64)
        for name in self.names:
            if name not in scans:
                continue
            (xy, yz, xy_scan, xy_rings, xy_centers, xy_lidar,
             yz_scan, yz_cones, yz_sections, yz_centers, yz_lidar, yz_plane) = self.artists[name]
            pose = poses[name]
            points = scans[name]
            valid = np.isfinite(points).all(axis=1)
            scan_robot = transform_to_robot(
                np.column_stack((points[valid], np.zeros(np.count_nonzero(valid)))), pose
            )
            xy_scan.set_offsets(scan_robot[:, :2]); yz_scan.set_offsets(scan_robot[:, 1:3])
            xy_lidar.set_offsets([[float(pose.get("x", 0.0)), float(pose.get("y", 0.0))]])
            yz_lidar.set_offsets([[float(pose.get("y", 0.0)), float(pose.get("z", 0.0))]])
            result = results.get(name, {})
            rings, centers, triangles, sections = [], [], [], []
            if result.get("success", False):
                for cone in result.get("cones", []):
                    cx, cy, radius = (float(cone["circle_center_x_m"]),
                                      float(cone["circle_center_y_m"]),
                                      float(cone["circle_radius_m"]))
                    local = np.column_stack((cx + radius * np.cos(theta),
                                             cy + radius * np.sin(theta),
                                             np.zeros_like(theta)))
                    ring = transform_to_robot(local, pose); rings.append(ring[:, :2])
                    center = transform_to_robot(np.array(((cx, cy, 0.0),)), pose)[0]
                    centers.append(center)
                    triangles.append(np.array(((center[1] - self.radius, 0.0),
                                               (center[1], self.height),
                                               (center[1] + self.radius, 0.0))))
                    sections.append(np.array(((center[1] - radius, center[2]),
                                              (center[1] + radius, center[2]))))
            center_array = np.asarray(centers).reshape((-1, 3))
            xy_rings.set_segments(rings); xy_centers.set_offsets(center_array[:, :2])
            yz_cones.set_segments(triangles); yz_sections.set_segments(sections)
            yz_centers.set_offsets(center_array[:, 1:3])
            # A distant/noisy scan return must not zoom the whole XY view out.
            # Use only fitted cone rings (and therefore their centers/radii).
            bounds = rings
            if self.backgrounds is None and bounds and any(len(item) for item in bounds):
                combined = np.vstack([item for item in bounds if len(item)])
                low, high = np.min(combined, axis=0), np.max(combined, axis=0)
                span = 1.06 * max(float(np.ptp(combined[:, 0])),
                                  float(np.ptp(combined[:, 1])), 2.0 * self.radius)
                middle = 0.5 * (low + high)
                xy.set_xlim(middle[0] - span / 2, middle[0] + span / 2)
                xy.set_ylim(middle[1] - span / 2, middle[1] + span / 2)
            if len(center_array):
                y_margin = 1.5 * self.radius
                y_limits = (float(np.min(center_array[:, 1]) - y_margin),
                            float(np.max(center_array[:, 1]) + y_margin))
                horizontal = np.linspace(*y_limits, 80)
                a, b, c = robot_scan_plane(pose)
                yz_plane.set_data(horizontal, b * horizontal + c + a * float(pose.get("x", 0.0)))
                if self.backgrounds is None:
                    yz.set_xlim(*y_limits)
                self._update_title(name, "xy", f"{name}: newest scan / circles")
                self._update_title(name, "yz",
                    f"{name}: roll={result['roll_deg']:.3f}°, "
                    f"pitch={result['pitch_deg']:.3f}°, z={result['z_m']:.3f} m"
                )
            else:
                yz_plane.set_data([], [])
                self._update_title(name, "xy", f"{name}: calibration failed")
                reason = str(result.get("reason", "waiting"))
                if len(reason) > 64:
                    reason = reason[:61] + "..."
                self._update_title(name, "yz", f"{name}: {reason}")
        if self.backgrounds is None:
            self.figure.canvas.draw()
            renderer = self.figure.canvas.get_renderer()
            self.backgrounds = {
                name: tuple(
                    (
                        axis.get_tightbbox(renderer).expanded(1.01, 1.03),
                        self.figure.canvas.copy_from_bbox(
                            axis.get_tightbbox(renderer).expanded(1.01, 1.03)
                        ),
                    )
                    for axis in self.artists[name][:2]
                )
                for name in self.names
            }
        if self.backgrounds is not None:
            for name in self.names:
                xy, yz = self.artists[name][:2]
                (xy_bbox, xy_background), (yz_bbox, yz_background) = self.backgrounds[name]
                self.figure.canvas.restore_region(xy_background)
                self.figure.canvas.restore_region(yz_background)
                for artist in self.dynamic_artists[name][:5]:
                    xy.draw_artist(artist)
                for artist in self.dynamic_artists[name][5:]:
                    yz.draw_artist(artist)
                self.figure.canvas.blit(xy_bbox); self.figure.canvas.blit(yz_bbox)
            self.figure.canvas.flush_events()


def plot_separate_projections(
    scans: dict,
    lidar_results: dict,
    calib_config: dict,
    lidar_poses: dict,
    xy_path: Path,
    yz_path: Path,
) -> None:
    """Save large XY and equal-scale, tightly laid-out YZ figures separately."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("warning: matplotlib is not installed; projection plots skipped", file=sys.stderr)
        return
    names = [name for name, result in lidar_results.items()
             if result.get("success", False) and name in scans]
    if not names:
        print("warning: no successful LiDAR result to plot", file=sys.stderr)
        return
    cfg = calib_config.get("plot", {})
    range_cfg = calib_config.get("range_filter", {})
    min_range, max_range = float(range_cfg.get("min_m", 0.1)), float(range_cfg["max_m"])
    max_points = int(cfg.get("max_points", 5000))
    cone_radius = float(calib_config["cone"]["radius_m"])
    cone_height = float(calib_config["cone"]["height_m"])

    xy_figure, xy_axes = plt.subplots(
        1, len(names), figsize=(6.5 * len(names), 6.2), squeeze=False,
    )
    yz_figure, yz_axes = plt.subplots(
        len(names), 1, figsize=(14, max(2.2, 1.85 * len(names))), squeeze=False,
    )
    for index_name, name in enumerate(names):
        result, pose = lidar_results[name], lidar_poses[name]
        points = scans[name]
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(points).all(axis=1) & (ranges >= min_range) & (ranges <= max_range)
        filtered = points[valid]
        if len(filtered) > max_points:
            filtered = filtered[np.linspace(0, len(filtered) - 1, max_points).astype(int)]
        scan_robot = transform_to_robot(np.column_stack((filtered, np.zeros(len(filtered)))), pose)
        cones = result["cones"]
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(cones), 1)))
        centers = np.array([
            transform_to_robot(np.array(((c["circle_center_x_m"], c["circle_center_y_m"], 0.0),)), pose)[0]
            for c in cones
        ])

        ax_xy = xy_axes[0, index_name]
        ax_xy.scatter(scan_robot[:, 0], scan_robot[:, 1], s=5, c="0.55", label="range-filtered scan")
        bounds = [scan_robot[:, :2], np.array(((0.0, 0.0),)),
                  np.array(((float(pose.get("x", 0.0)), float(pose.get("y", 0.0))),))]
        theta = np.linspace(0.0, 2.0 * math.pi, 100)
        for cone_index, (cone, color, center) in enumerate(zip(cones, colors, centers), 1):
            r = float(cone["circle_radius_m"])
            local_ring = np.column_stack((
                cone["circle_center_x_m"] + r * np.cos(theta),
                cone["circle_center_y_m"] + r * np.sin(theta), np.zeros_like(theta),
            ))
            ring = transform_to_robot(local_ring, pose)
            bounds.append(ring[:, :2])
            ax_xy.plot(ring[:, 0], ring[:, 1], color=color, linewidth=2)
            ax_xy.scatter(center[0], center[1], marker="x", s=55, color=color)
            ax_xy.annotate(f"C{cone_index}\nr={r:.3f}", center[:2], color=color)
        ax_xy.scatter(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                      marker="^", s=80, c="red", label="LiDAR")
        ax_xy.scatter(0.0, 0.0, marker="+", s=90, c="black", label="robot origin")
        all_xy = np.vstack(bounds)
        low, high = np.min(all_xy, axis=0), np.max(all_xy, axis=0)
        span = 1.08 * max(float(high[0] - low[0]), float(high[1] - low[1]), 2 * cone_radius)
        middle = 0.5 * (low + high)
        ax_xy.set_xlim(middle[0] - span / 2, middle[0] + span / 2)
        ax_xy.set_ylim(middle[1] - span / 2, middle[1] + span / 2)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_title(f"{name}: circles in robot TF (XY)")
        ax_xy.set_xlabel("robot x [m]"); ax_xy.set_ylabel("robot y [m]")
        ax_xy.grid(True, alpha=0.3); ax_xy.legend(loc="best", fontsize=8)

        ax_yz = yz_axes[index_name, 0]
        ax_yz.scatter(scan_robot[:, 1], scan_robot[:, 2], s=5, c="0.45", alpha=0.55,
                      label="LaserScan points", zorder=2)
        for cone_index, (cone, color, center) in enumerate(zip(cones, colors, centers), 1):
            r, h = float(cone["circle_radius_m"]), float(center[2])
            ax_yz.add_patch(Polygon(
                ((center[1] - cone_radius, 0.0), (center[1], cone_height),
                 (center[1] + cone_radius, 0.0)), closed=True,
                facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.20,
            ))
            ax_yz.plot((center[1] - r, center[1] + r), (h, h), color=color, linewidth=3,
                       label="detected cone section" if cone_index == 1 else None)
            ax_yz.scatter(center[1], h, marker="x", s=55, color=color, zorder=4)
            ax_yz.annotate(f"C{cone_index} h={h:.3f}", (center[1], h), color=color)
        y_limits = (float(np.min(centers[:, 1]) - 1.5 * cone_radius),
                    float(np.max(centers[:, 1]) + 1.5 * cone_radius))
        horizontal = np.linspace(*y_limits, 100)
        a, b, c = robot_scan_plane(pose)
        x_min, x_max = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
        lower = b * horizontal + c + min(a * x_min, a * x_max)
        upper = b * horizontal + c + max(a * x_min, a * x_max)
        ax_yz.fill_between(horizontal, lower, upper, color="deepskyblue", alpha=0.22,
                           label="projected scan plane")
        ax_yz.plot(horizontal, b * horizontal + c + a * float(pose.get("x", 0.0)), "--",
                   color="dodgerblue", linewidth=2, label="plane center slice")
        ax_yz.scatter(float(pose.get("y", 0.0)), float(pose.get("z", 0.0)), marker="D",
                      s=70, c="red", label="LiDAR", zorder=5)
        ax_yz.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax_yz.set_xlim(*y_limits); ax_yz.set_ylim(-0.05, cone_height * 1.05)
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.set_title(f"{name}: robot-TF YZ projection")
        ax_yz.set_xlabel("robot y [m]"); ax_yz.set_ylabel("robot z [m]")
        ax_yz.grid(True, alpha=0.3); ax_yz.legend(loc="best", fontsize=8, ncol=3)

    xy_figure.suptitle("Cone calibration: robot-TF XY circle extraction")
    yz_figure.suptitle("Cone calibration: equal-scale robot-TF YZ projections")
    xy_figure.tight_layout(); yz_figure.tight_layout()
    xy_path.parent.mkdir(parents=True, exist_ok=True)
    xy_figure.savefig(xy_path, dpi=int(cfg.get("dpi", 150)), bbox_inches="tight")
    yz_figure.savefig(yz_path, dpi=int(cfg.get("dpi", 150)), bbox_inches="tight")
    print(f"XY plot: {xy_path}"); print(f"YZ plot: {yz_path}")
    if bool(cfg.get("show", True)) and bool(cfg.get("show_2d", True)):
        backend = matplotlib.get_backend().lower()
        if backend in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
            print(f"warning: matplotlib backend '{matplotlib.get_backend()}' has no GUI", file=sys.stderr)
        else:
            plt.show()
    plt.close(xy_figure); plt.close(yz_figure)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--lidar-config", type=Path, default=script_dir / "lidar_config.yaml")
    parser.add_argument("--calib-config", type=Path, default=script_dir / "calib_config.yaml")
    return parser


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def process_lidar(
    points: np.ndarray, config: dict, verbose: bool = True
) -> Tuple[dict, PlaneResult]:
    cone = config["cone"]
    range_cfg = config.get("range_filter", {})
    cluster_cfg = config.get("clustering", {})
    circle_cfg = config.get("circle", {})
    quality_cfg = config.get("quality", {})
    radius = float(cone["radius_m"])
    height = float(cone["height_m"])
    min_range = float(range_cfg.get("min_m", 0.1))
    max_range = float(range_cfg["max_m"])
    if radius <= 0 or height <= 0:
        raise ValueError("cone.radius_m and cone.height_m must be positive")
    if not 0 <= min_range < max_range:
        raise ValueError("range_filter must satisfy 0 <= min_m < max_m")

    clusters = filter_and_cluster(
        points, min_range, max_range,
        float(cluster_cfg.get("gap_m", 0.08)),
        int(cluster_cfg.get("min_points", 5)),
        float(cluster_cfg.get("max_width_m", 0.5)),
    )
    diagnostics: List[str] = []
    sections = extract_sections(
        clusters, radius, height,
        float(circle_cfg.get("min_radius_m", 0.015)),
        float(circle_cfg.get("inlier_threshold_m", 0.01)),
        int(circle_cfg.get("ransac_iterations", 500)),
        int(circle_cfg.get("min_inliers", 5)),
        int(circle_cfg.get("random_seed", 0)),
        diagnostics,
    )
    if verbose:
        print(f"  valid object clusters: {len(clusters)}")
        for message in diagnostics:
            print(f"    {message}")
    max_circle_rmse = float(quality_cfg.get("max_circle_rmse_m", float("inf")))
    before_quality_count = len(sections)
    sections = [section for section in sections if section.circle.rmse <= max_circle_rmse]
    if verbose:
        print(
            f"  fitted circles: {before_quality_count}, "
            f"quality-accepted circles: {len(sections)}"
        )
    min_cones = int(quality_cfg.get("min_cones", 3))
    if len(sections) < min_cones:
        raise RuntimeError(f"accepted {len(sections)} circles, but quality.min_cones is {min_cones}")
    plane = fit_scan_plane(sections)
    max_plane_rmse = float(quality_cfg.get("max_plane_rmse_m", float("inf")))
    if plane.rmse > max_plane_rmse:
        raise RuntimeError(
            f"plane RMSE {plane.rmse:.6f} m exceeds quality.max_plane_rmse_m {max_plane_rmse:.6f} m"
        )
    return result_dict(plane, sections, len(clusters)), plane


def run_realtime(
    lidar_config: dict,
    calib_config: dict,
    lidar_topics: dict,
    calib_config_path: Path,
) -> int:
    """Continuously calibrate from only the newest scan of each LiDAR."""
    try:
        import rclpy
        from sensor_msgs.msg import LaserScan
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("realtime mode requires ROS 2 and matplotlib") from exc

    realtime_cfg = calib_config.get("realtime", {})
    interval = float(realtime_cfg.get("update_interval_sec", 0.5))
    if interval <= 0.0:
        raise ValueError("realtime.update_interval_sec must be positive")
    plotly_interval = float(realtime_cfg.get("plotly_update_interval_sec", 0.5))
    if plotly_interval <= 0.0:
        raise ValueError("realtime.plotly_update_interval_sec must be positive")
    plotly_host = str(realtime_cfg.get("plotly_host", "0.0.0.0"))
    plotly_port = int(realtime_cfg.get("plotly_port", 8050))
    plotly_public_host = str(realtime_cfg.get("plotly_public_host", "localhost"))
    if not 0 <= plotly_port <= 65535:
        raise ValueError("realtime.plotly_port must be between 0 and 65535")
    processing_config = copy.deepcopy(calib_config)
    realtime_ransac_iterations = int(
        realtime_cfg.get(
            "ransac_iterations",
            processing_config.get("circle", {}).get("ransac_iterations", 500),
        )
    )
    if realtime_ransac_iterations <= 0:
        raise ValueError("realtime.ransac_iterations must be positive")
    processing_config.setdefault("circle", {})["ransac_iterations"] = realtime_ransac_iterations
    output_dir = Path(calib_config.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = calib_config_path.resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / str(calib_config.get("output_yaml", "calibrated_output.yaml"))
    calibrated_path = output_dir / str(
        calib_config.get("calibrated_config_yaml", "calibrated_config.yaml")
    )

    latest_scans = {}
    sequence = {name: 0 for name in lidar_topics}
    processed_sequence = {name: -1 for name in lidar_topics}
    calibrated_config = copy.deepcopy(lidar_config)
    last_results = {
        name: {"success": False, "reason": "waiting_for_first_scan"}
        for name in lidar_topics
    }
    last_successful_output = None
    last_successful_config = None
    plotly_server = None
    plotly_state_path = None
    plotly_future = None

    rclpy.init(args=None)
    node = rclpy.create_node("cone_circle_realtime_calibrator")
    subscriptions = []

    def make_callback(name):
        def callback(message: LaserScan) -> None:
            ranges = np.asarray(message.ranges, dtype=float)
            angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
            latest_scans[name] = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
            sequence[name] += 1
        return callback

    for name, topic in lidar_topics.items():
        # Keep only the newest message. A deeper queue makes a realtime viewer
        # process stale scans after a costly calibration/render cycle.
        subscriptions.append(node.create_subscription(LaserScan, topic, make_callback(name), 1))

    local_plot = RealtimeLocalPlot(list(lidar_topics), calib_config)
    viewer = local_plot.figure
    if bool(calib_config.get("plot", {}).get("interactive_3d", True)):
        plotly_server, plotly_state_path, _ = start_realtime_plotly_server(
            output_dir,
            plotly_interval,
            host=plotly_host,
            port=plotly_port,
            public_host=plotly_public_host,
            auto_open=bool(calib_config.get("plot", {}).get("show", True)),
        )
    next_update = time.monotonic()
    next_plotly_update = next_update
    print("realtime mode: newest scan only; close the window or press Ctrl+C to stop")

    calculation_executor = ProcessPoolExecutor(
        max_workers=max(1, len(lidar_topics)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=ignore_worker_sigint,
    )
    plotly_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cone_plotly")
    try:
        while rclpy.ok() and plt.fignum_exists(viewer.number):
            rclpy.spin_once(node, timeout_sec=0.03)
            now = time.monotonic()
            if now < next_update or not latest_scans:
                viewer.canvas.flush_events()
                continue
            if all(sequence[name] == processed_sequence[name] for name in latest_scans):
                next_update = now + interval
                viewer.canvas.flush_events()
                continue
            next_update = now + interval
            scan_snapshot = {
                name: np.array(scan, copy=True) for name, scan in latest_scans.items()
            }
            futures = {
                calculation_executor.submit(process_lidar, scan, processing_config, False): name
                for name, scan in scan_snapshot.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result, plane = future.result()
                    result["roll_rad"] = plane.roll
                    result["pitch_rad"] = plane.pitch
                    last_results[name] = result
                    pose = calibrated_config["lidars"][name]
                    pose["roll"], pose["pitch"], pose["z"] = (
                        float(plane.roll), float(plane.pitch), float(plane.c)
                    )
                except (RuntimeError, ValueError) as error:
                    last_results[name] = {"success": False, "reason": str(error)}
                processed_sequence[name] = sequence[name]

            output = {
                "calibration_mode": "realtime_newest_scan_cone_circle_plane",
                "cone": copy.deepcopy(calib_config.get("cone", {})),
                "success": all(last_results[name].get("success", False) for name in lidar_topics),
                "lidars": copy.deepcopy(last_results),
            }
            render_config = copy.deepcopy(calib_config)
            render_config.setdefault("plot", {})["show"] = True
            render_config["plot"]["show_2d"] = True
            local_plot.update(scan_snapshot, last_results, calibrated_config["lidars"])

            plot_now = time.monotonic()
            if (plotly_state_path is not None and output["success"]
                    and plot_now >= next_plotly_update):
                if plotly_future is None or plotly_future.done():
                    if plotly_future is not None:
                        plotly_future.result()
                    plotly_future = plotly_executor.submit(
                        render_realtime_plotly_state,
                        copy.deepcopy(scan_snapshot), copy.deepcopy(last_results),
                        render_config, copy.deepcopy(calibrated_config["lidars"]),
                        plotly_state_path,
                    )
                    next_plotly_update = plot_now + plotly_interval

            if output["success"]:
                last_successful_output = copy.deepcopy(output)
                last_successful_config = copy.deepcopy(calibrated_config)
            print(
                "realtime update: " + ", ".join(
                    f"{name}={'OK' if last_results[name].get('success') else 'FAIL'}"
                    for name in lidar_topics
                )
            )
    except KeyboardInterrupt:
        print("realtime calibration stopped")
    finally:
        calculation_executor.shutdown(wait=True, cancel_futures=True)
        plotly_executor.shutdown(wait=True, cancel_futures=True)
        if plotly_server is not None:
            plotly_server.shutdown()
            plotly_server.server_close()
        save_final = bool(
            realtime_cfg.get(
                "save_final_result", realtime_cfg.get("save_latest_result", True)
            )
        )
        if save_final:
            if last_successful_output is not None:
                with result_path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(last_successful_output, stream, sort_keys=False, allow_unicode=True)
                with calibrated_path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(last_successful_config, stream, sort_keys=False, allow_unicode=True)
                print(f"final result: {result_path}")
                print(f"final calibrated config: {calibrated_path}")
            else:
                print("warning: no fully successful realtime result to save", file=sys.stderr)
        del subscriptions
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        plt.close(viewer)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    lidar_config = load_yaml(args.lidar_config)
    calib_config = load_yaml(args.calib_config)
    topics = lidar_config.get("topics")
    poses = lidar_config.get("lidars")
    if not isinstance(topics, dict) or not isinstance(poses, dict) or not poses:
        raise ValueError("lidar_config.yaml requires non-empty topics and lidars mappings")
    lidar_topics = {}
    for name in poses:
        if name not in topics:
            raise ValueError(f"topics.{name} is missing from lidar_config.yaml")
        lidar_topics[name] = str(topics[name])
    topic_types = lidar_config.get("topic_types", {})
    for name in lidar_topics:
        if str(topic_types.get(name, "LaserScan")) != "LaserScan":
            raise ValueError(f"topic_types.{name} must be LaserScan")

    if bool(calib_config.get("realtime", {}).get("enabled", False)):
        return run_realtime(lidar_config, calib_config, lidar_topics, args.calib_config)

    duration = float(calib_config.get("collect_duration_sec", 5.0))
    scans = collect_ros_scans(lidar_topics, duration)
    calibrated_config = copy.deepcopy(lidar_config)
    output = {
        "calibration_mode": "stationary_cone_circle_plane",
        "cone": copy.deepcopy(calib_config.get("cone", {})),
        "success": True,
        "lidars": {},
    }
    for name in poses:
        print(f"\n[{name}] {lidar_topics[name]}")
        if name not in scans:
            output["success"] = False
            output["lidars"][name] = {"success": False, "reason": "no_scan_received"}
            continue
        try:
            result, plane = process_lidar(scans[name], calib_config)
        except (RuntimeError, ValueError) as error:
            output["success"] = False
            output["lidars"][name] = {"success": False, "reason": str(error)}
            print(f"  failed: {error}")
            continue
        result["roll_rad"] = plane.roll
        result["pitch_rad"] = plane.pitch
        output["lidars"][name] = result
        calibrated_pose = calibrated_config["lidars"][name]
        calibrated_pose["roll"] = float(plane.roll)
        calibrated_pose["pitch"] = float(plane.pitch)
        calibrated_pose["z"] = float(plane.c)
        print(
            f"  result: roll={math.degrees(plane.roll):.4f} deg, "
            f"pitch={math.degrees(plane.pitch):.4f} deg, z={plane.c:.4f} m, "
            f"plane_rmse={plane.rmse:.6f} m"
        )

    config_dir = args.calib_config.resolve().parent
    output_dir = Path(calib_config.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / str(calib_config.get("output_yaml", "calibrated_output.yaml"))
    calibrated_path = output_dir / str(
        calib_config.get("calibrated_config_yaml", "calibrated_config.yaml")
    )
    # Persist numeric results before opening the blocking GUI. This guarantees
    # YAML output even when the window is closed with Ctrl+C or the process is
    # stopped while plt.show() is waiting.
    with result_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False, allow_unicode=True)
    with calibrated_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(calibrated_config, stream, sort_keys=False, allow_unicode=True)
    print(f"\nresult: {result_path}")
    print(f"calibrated config: {calibrated_path}")
    plot_cfg = calib_config.get("plot", {})
    if bool(plot_cfg.get("enabled", True)):
        # Open/write the non-blocking browser plot first. The Matplotlib plot
        # is intentionally last because plt.show() keeps the process alive
        # until the user closes its window.
        if bool(plot_cfg.get("interactive_3d", True)):
            html_path = output_dir / str(
                plot_cfg.get("output_html", "cone_calibration_3d.html")
            )
            plot_calibration_3d(
                scans, output["lidars"], calib_config, calibrated_config["lidars"], html_path
            )
            combined_html_path = output_dir / str(
                plot_cfg.get("combined_output_html", "cone_calibration_3d_combined.html")
            )
            plot_combined_calibration_3d(
                scans, output["lidars"], calib_config, calibrated_config["lidars"],
                combined_html_path,
            )
        plot_path = output_dir / str(plot_cfg.get("output_png", "cone_calibration.png"))
        plot_calibration(
            scans, output["lidars"], calib_config, calibrated_config["lidars"], plot_path
        )
    return 0 if output["success"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
