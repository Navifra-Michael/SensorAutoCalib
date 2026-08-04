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
import math
import sys
import time
import warnings
from dataclasses import dataclass
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
    save_path: Path,
) -> None:
    """Save XZ/YZ projections of upright cones and the fitted scan plane."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle as CirclePatch
        from matplotlib.patches import Polygon
    except ImportError:
        print("warning: matplotlib is not installed; GUI plot skipped", file=sys.stderr)
        return

    plotted_names = [name for name in lidar_results if name in scans]
    if not plotted_names:
        print("warning: no received LiDAR scan to plot", file=sys.stderr)
        return
    plot_cfg = calib_config.get("plot", {})
    max_points = int(plot_cfg.get("max_points", 5000))
    min_range = float(calib_config.get("range_filter", {}).get("min_m", 0.1))
    max_range = float(calib_config.get("range_filter", {})["max_m"])
    cone_radius = float(calib_config["cone"]["radius_m"])
    cone_height = float(calib_config["cone"]["height_m"])
    figure, axes = plt.subplots(
        len(plotted_names), 3, figsize=(18, max(5, 4.8 * len(plotted_names))), squeeze=False,
    )

    for row, name in enumerate(plotted_names):
        result = lidar_results[name]
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
        cones = result.get("cones", [])
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(cones), 1)))

        # Preserve the original XY scan/circle diagnostic on the left.
        ax_xy = axes[row, 0]
        ax_xy.scatter(filtered[:, 0], filtered[:, 1], s=5, c="0.55",
                      label="range-filtered scan")
        for index, (cone, color) in enumerate(zip(cones, colors), 1):
            cx = cone["circle_center_x_m"]
            cy = cone["circle_center_y_m"]
            radius = cone["circle_radius_m"]
            ax_xy.add_patch(CirclePatch(
                (cx, cy), radius, fill=False, color=color, linewidth=2,
            ))
            ax_xy.scatter(cx, cy, marker="x", s=55, color=color)
            ax_xy.annotate(f"C{index}\nr={radius:.3f}", (cx, cy), color=color)
        ax_xy.scatter(0.0, 0.0, marker="^", s=80, c="red", label="LiDAR")
        ax_xy.set_title(f"{name}: circles in LaserScan (XY)")
        ax_xy.set_xlabel("LiDAR x [m]")
        ax_xy.set_ylabel("LiDAR y [m]")
        ax_xy.axis("equal")
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
        plane = result["plane"]
        scan_z = plane["a"] * filtered[:, 0] + plane["b"] * filtered[:, 1] + plane["c"]
        center_x = np.array([cone["circle_center_x_m"] for cone in cones])
        center_y = np.array([cone["circle_center_y_m"] for cone in cones])
        x_limits = (float(np.min(center_x) - 1.5 * cone_radius),
                    float(np.max(center_x) + 1.5 * cone_radius))
        y_limits = (float(np.min(center_y) - 1.5 * cone_radius),
                    float(np.max(center_y) + 1.5 * cone_radius))

        for projection, axis, coordinate, limits, slope, other_centers in (
            ("XZ", axes[row, 1], "x", x_limits, plane["a"], center_y),
            ("YZ", axes[row, 2], "y", y_limits, plane["b"], center_x),
        ):
            coordinate_index = 0 if coordinate == "x" else 1
            axis.scatter(
                filtered[:, coordinate_index], scan_z, s=5, c="0.45", alpha=0.55,
                label="LaserScan points on fitted plane", zorder=2,
            )
            for index, (cone, color) in enumerate(zip(cones, colors), 1):
                center_coordinate = cone[f"circle_center_{coordinate}_m"]
                height = cone["inferred_height_m"]
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
            other_slope = plane["b"] if coordinate == "x" else plane["a"]
            lower = slope * horizontal + plane["c"] + min(other_slope * other_min,
                                                            other_slope * other_max)
            upper = slope * horizontal + plane["c"] + max(other_slope * other_min,
                                                            other_slope * other_max)
            axis.fill_between(horizontal, lower, upper, color="deepskyblue", alpha=0.22,
                              label="projected scan plane")
            axis.plot(horizontal, slope * horizontal + plane["c"], "--",
                      color="dodgerblue", linewidth=2, label="plane center slice")
            axis.scatter(0.0, result["z_m"], marker="D", s=70, c="red",
                         label="LiDAR", zorder=5)
            axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            axis.set_xlim(*limits)
            axis.set_ylim(min(-0.05, float(np.min(scan_z)) - 0.05),
                          max(cone_height * 1.08, float(np.max(scan_z)) + 0.05))
            axis.set_xlabel(f"LiDAR {coordinate} [m]")
            axis.set_ylabel("height z [m]")
            axis.set_title(
                f"{name}: {projection} projection\n"
                f"roll={result['roll_deg']:.3f}°, pitch={result['pitch_deg']:.3f}°, "
                f"z={result['z_m']:.3f} m"
            )
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)

    figure.suptitle("Cone calibration: XY circles and XZ / YZ projections", fontsize=14)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=int(plot_cfg.get("dpi", 150)), bbox_inches="tight")
    print(f"plot: {save_path}")
    if bool(plot_cfg.get("show_2d", False)):
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


def plot_calibration_3d(
    scans: dict,
    lidar_results: dict,
    calib_config: dict,
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
        cones = result["cones"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for index, cone in enumerate(cones, 1):
            cx, cy = cone["circle_center_x_m"], cone["circle_center_y_m"]
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
            section_z = float(cone["inferred_height_m"])
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

        centers = np.array([
            (cone["circle_center_x_m"], cone["circle_center_y_m"])
            for cone in cones
        ])
        x_margin = max(float(np.ptp(centers[:, 0])) * 0.12, cone_radius)
        y_margin = max(float(np.ptp(centers[:, 1])) * 0.12, cone_radius)
        plane_x = np.linspace(np.min(centers[:, 0]) - x_margin, np.max(centers[:, 0]) + x_margin, 22)
        plane_y = np.linspace(np.min(centers[:, 1]) - y_margin, np.max(centers[:, 1]) + y_margin, 22)
        plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
        plane = result["plane"]
        plane_zz = plane["a"] * plane_xx + plane["b"] * plane_yy + plane["c"]
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
        scan_z = plane["a"] * scan_points[:, 0] + plane["b"] * scan_points[:, 1] + plane["c"]
        figure.add_trace(
            go.Scatter3d(
                x=scan_points[:, 0], y=scan_points[:, 1], z=scan_z,
                mode="markers", marker={"size": 2, "color": "#333333"},
                name=f"{name} scan points",
            ), row=row, col=1,
        )
        figure.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[result["z_m"]], mode="markers+text",
                marker={"size": 8, "color": "red", "symbol": "diamond"},
                text=["LiDAR"], textposition="top center", name=f"{name} LiDAR",
            ), row=row, col=1,
        )
        scene_name = "scene" if row == 1 else f"scene{row}"
        figure.layout[scene_name].update(
            xaxis_title="LiDAR x [m]", yaxis_title="LiDAR y [m]", zaxis_title="height [m]",
            aspectmode="data", camera={"eye": {"x": 1.45, "y": -1.45, "z": 0.9}},
        )

    figure.update_layout(
        title="Cone calibration: 3D cones and fitted LaserScan planes",
        height=max(650, 620 * len(names)), showlegend=True,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    show = bool(calib_config.get("plot", {}).get("show", True))
    figure.write_html(str(save_path), include_plotlyjs=True, auto_open=show)
    print(f"interactive 3D plot: {save_path}")


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


def process_lidar(points: np.ndarray, config: dict) -> Tuple[dict, PlaneResult]:
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
    print(f"  valid object clusters: {len(clusters)}")
    for message in diagnostics:
        print(f"    {message}")
    max_circle_rmse = float(quality_cfg.get("max_circle_rmse_m", float("inf")))
    before_quality_count = len(sections)
    sections = [section for section in sections if section.circle.rmse <= max_circle_rmse]
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
            plot_calibration_3d(scans, output["lidars"], calib_config, html_path)
        plot_path = output_dir / str(plot_cfg.get("output_png", "cone_calibration.png"))
        plot_calibration(scans, output["lidars"], calib_config, plot_path)
    return 0 if output["success"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
