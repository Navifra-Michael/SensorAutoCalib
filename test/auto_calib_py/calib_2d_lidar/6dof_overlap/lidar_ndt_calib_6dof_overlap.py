#!/usr/bin/env python3

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
import time

import numpy as np


_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "6dof",
    "lidar_ndt_calib_6dof.py",
)
_BASE_SPEC = importlib.util.spec_from_file_location("_ndt6_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


NullLogger = _BASE.NullLogger
apply_transform = _BASE.apply_transform
build_ndt_grid = _BASE.build_ndt_grid
downsample_xyz = _BASE.downsample_xyz
load_lidar_entries = _BASE.load_lidar_entries
make_calibrated_config = _BASE.make_calibrated_config
make_initial_result_yaml = _BASE.make_initial_result_yaml
ndt_score = _BASE.ndt_score
normalize_angle = _BASE.normalize_angle
normalize_score_projection = _BASE.normalize_score_projection
optional_stage_values = _BASE.optional_stage_values
pose_to_transform = _BASE.pose_to_transform
rotation_matrix = _BASE.rotation_matrix
stage_values = _BASE.stage_values
transform_accumulated_cloud = _BASE.transform_accumulated_cloud
transform_scan_chunks = _BASE.transform_scan_chunks


def copy_pose(pose):
    return {
        key: float(pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }


def pose_delta(from_pose, to_pose):
    return {
        "x": float(to_pose["x"]) - float(from_pose["x"]),
        "y": float(to_pose["y"]) - float(from_pose["y"]),
        "z": float(to_pose["z"]) - float(from_pose["z"]),
        "roll": float(normalize_angle(
            float(to_pose["roll"]) - float(from_pose["roll"])
        )),
        "pitch": float(normalize_angle(
            float(to_pose["pitch"]) - float(from_pose["pitch"])
        )),
        "yaw": float(normalize_angle(
            float(to_pose["yaw"]) - float(from_pose["yaw"])
        )),
    }


def tilt_delta_magnitude(delta):
    if not delta:
        return None

    return float(np.hypot(
        float(delta.get("roll", 0.0)),
        float(delta.get("pitch", 0.0)),
    ))


def organize_lidar_result(raw_result):
    final_pose = raw_result.get("final_pose")
    if final_pose is None:
        final_pose = {
            key: float(raw_result[key])
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
            if key in raw_result
        }

    initial_pose = raw_result.get("initial_pose")
    stage1_result = raw_result.get("stage1_result", {})
    stage2_result = raw_result.get("stage2_result", {})
    stage1_pose = stage1_result.get(
        "pose",
        raw_result.get("initial_overlap_pose", final_pose),
    )
    stage2_pose = stage2_result.get("pose", final_pose)

    organized = {
        key: float(final_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
        if key in final_pose
    }
    organized["poses"] = {
        "initial": initial_pose,
        "stage1": stage1_pose,
        "stage2": stage2_pose,
        "final": final_pose,
    }
    organized["deltas"] = {
        "stage1_from_initial": stage1_result.get("delta_from_initial"),
        "stage2_from_stage1": stage2_result.get("delta_from_stage1"),
        "final_from_initial": (
            pose_delta(initial_pose, final_pose)
            if initial_pose and final_pose else None
        ),
    }
    organized["tilt_delta_magnitude"] = {
        "stage1_from_initial": tilt_delta_magnitude(
            stage1_result.get("delta_from_initial"),
        ),
        "final_from_initial": tilt_delta_magnitude(
            organized["deltas"]["final_from_initial"],
        ),
    }
    organized["scores"] = {
        "stage1_initial": stage1_result.get("initial_score"),
        "stage1": stage1_result.get("score"),
        "stage1_improvement": stage1_result.get("score_improvement"),
        "stage1_improvement_ratio": stage1_result.get("score_improvement_ratio"),
        "stage1_candidate": stage1_result.get("candidate_score"),
        "stage2_initial": stage2_result.get("initial_score"),
        "stage2_final": stage2_result.get("score"),
        "stage2_improvement": stage2_result.get("score_improvement"),
    }
    organized["timing"] = {
        "stage1_sec": stage1_result.get("time_sec"),
        "stage2_sec": stage2_result.get(
            "time_sec",
            raw_result.get("optimization_time_sec"),
        ),
    }
    organized["status"] = {
        "optimized": bool(raw_result.get("optimized", False)),
        "success": bool(raw_result.get("success", False)),
        "stage1_success": bool(stage1_result.get(
            "success",
            raw_result.get("initial_overlap_success", False),
        )),
        "stage2_success": bool(stage2_result.get(
            "success",
            raw_result.get("success", False),
        )),
    }
    if "reason" in raw_result:
        organized["status"]["reason"] = raw_result["reason"]
    if stage1_result.get("rejected_low_improvement"):
        organized["status"]["stage1_rejected_low_improvement"] = True
        organized["stage1_candidate_pose"] = stage1_result.get("candidate_pose")
    if stage2_result.get("fixed_as_reference"):
        organized["status"]["fixed_as_reference"] = True

    return organized


def config_with_feedback_poses(lidar_config, result_yaml):
    updated_config = copy.deepcopy(lidar_config)
    for name, result_pose in result_yaml.get("lidars", {}).items():
        if name not in updated_config.get("lidars", {}):
            continue

        final_pose = result_pose.get("poses", {}).get("final", result_pose)
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            if key in final_pose:
                updated_config["lidars"][name][key] = float(final_pose[key])

    return updated_config


def feedback_iteration_summary(iteration_idx, result_yaml):
    return {
        "iteration": int(iteration_idx),
        "success": bool(result_yaml.get("success", False)),
        "lidars": {
            name: {
                "final_pose": copy_pose(
                    lidar_result.get("poses", {}).get("final", lidar_result),
                ),
                "scores": copy.deepcopy(lidar_result.get("scores", {})),
                "status": copy.deepcopy(lidar_result.get("status", {})),
            }
            for name, lidar_result in result_yaml.get("lidars", {}).items()
        },
    }


def optional_angle_stage_values(center, stage, range_key, step_key):
    if range_key not in stage and step_key not in stage:
        return np.array([center], dtype=np.float64)
    if range_key not in stage or step_key not in stage:
        raise ValueError(
            f"search stage must define both {range_key} and {step_key}, "
            "or neither to keep this angle fixed."
        )

    return stage_values(
        center,
        np.deg2rad(float(stage[range_key])),
        np.deg2rad(float(stage[step_key])),
    )


def stage_axis_set(stage, default_axes):
    axes = stage.get("axes", stage.get("axis_order", default_axes))
    if isinstance(axes, str):
        axes = [axis.strip() for axis in axes.split(",") if axis.strip()]
    return {str(axis).strip().lower() for axis in axes}


def fixed_stage_values(center):
    return np.array([center], dtype=np.float64)


def selective_stage_values(center, stage, axis_name, range_key, step_key, active_axes):
    if axis_name not in active_axes:
        return fixed_stage_values(center)
    return optional_stage_values(center, stage, range_key, step_key)


def selective_angle_stage_values(
    center,
    stage,
    axis_name,
    range_key,
    step_key,
    active_axes,
):
    if axis_name not in active_axes:
        return fixed_stage_values(center)
    return optional_angle_stage_values(center, stage, range_key, step_key)


def invert_transform(transform):
    inverse = np.eye(4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_single_chunk_to_initial(chunk, pose, initial_odom_inv):
    extrinsic_tf = pose_to_transform(pose)
    lidar_in_base = apply_transform(chunk["points"], extrinsic_tf)
    relative_odom_tf = initial_odom_inv @ chunk["odom_tf"]
    return apply_transform(lidar_in_base, relative_odom_tf)


def source_indices(chunk_count, stride, max_scans):
    stride = max(1, int(stride))
    indices = list(range(1, chunk_count, stride))
    if max_scans is not None and int(max_scans) > 0:
        indices = indices[:int(max_scans)]
    return indices


def initial_scan_overlap_clouds(chunks, pose, indices):
    if not chunks or not indices:
        return {
            "target": np.empty((0, 3)),
            "source": np.empty((0, 3)),
            "target_ranges": np.empty((0,)),
            "source_ranges": np.empty((0,)),
        }

    initial_odom_inv = invert_transform(chunks[0]["odom_tf"])
    target_cloud = transform_single_chunk_to_initial(
        chunks[0],
        pose,
        initial_odom_inv,
    )
    source_clouds = [
        transform_single_chunk_to_initial(chunks[idx], pose, initial_odom_inv)
        for idx in indices
    ]
    source_cloud = np.vstack(source_clouds)
    return {
        "target": target_cloud,
        "source": source_cloud,
        "target_ranges": np.linalg.norm(chunks[0]["points"][:, :2], axis=1),
        "source_ranges": np.concatenate([
            np.linalg.norm(chunks[idx]["points"][:, :2], axis=1)
            for idx in indices
        ]),
    }


def combined_initial_overlap_cloud(chunks, pose, stride, max_scans):
    indices = source_indices(len(chunks), stride, max_scans)
    clouds = initial_scan_overlap_clouds(chunks, pose, indices)
    if len(clouds["target"]) == 0:
        return clouds["source"]
    if len(clouds["source"]) == 0:
        return clouds["target"]
    return np.vstack([clouds["target"], clouds["source"]])


def grid_thickness_score(
    points,
    resolution,
    min_points,
    method="pca_line",
    score_projection="xyz",
    key_points=None,
    point_weights=None,
):
    if len(points) == 0:
        return 1e9

    method = str(method).strip().lower()
    if key_points is None:
        key_points = points
    xy_keys = np.floor(key_points[:, :2] / resolution).astype(np.int64)
    cells = {}
    cell_weights = {}
    if point_weights is None:
        point_weights = np.ones(len(points), dtype=np.float64)
    else:
        point_weights = np.asarray(point_weights, dtype=np.float64)
    for point, weight, key in zip(points, point_weights, map(tuple, xy_keys)):
        cells.setdefault(key, []).append(point)
        cell_weights.setdefault(key, []).append(float(weight))

    weighted_score = 0.0
    total_weight = 0
    for key, cell_points in cells.items():
        cell_points = np.asarray(cell_points, dtype=np.float64)
        count = len(cell_points)
        if count < min_points:
            continue
        cell_weight = float(np.sum(cell_weights[key]))
        if cell_weight <= 0.0:
            continue

        if method == "z_std":
            thickness = float(np.std(cell_points[:, 2]))
        elif method == "xy_std":
            xy_std = np.std(cell_points[:, :2], axis=0)
            thickness = float(np.linalg.norm(xy_std))
        elif method == "pca_plane":
            projected_points = project_points_for_thickness(
                cell_points,
                score_projection,
            )
            centered = projected_points - np.mean(projected_points, axis=0)
            cov = (centered.T @ centered) / max(count - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)
            thickness = float(np.sqrt(max(eigvals[0], 0.0)))
        else:
            projected_points = project_points_for_thickness(
                cell_points,
                score_projection,
            )
            centered = projected_points - np.mean(projected_points, axis=0)
            cov = (centered.T @ centered) / max(count - 1, 1)
            eigvals = np.sort(np.linalg.eigvalsh(cov))
            if len(eigvals) == 2:
                thickness = float(np.sqrt(max(eigvals[0], 0.0)))
            else:
                thickness = float(np.sqrt(max(eigvals[0] + eigvals[1], 0.0)))

        weighted_score += thickness * cell_weight
        total_weight += cell_weight

    if total_weight == 0:
        return 1e9

    return weighted_score / total_weight


def normalize_thickness_projection(score_projection):
    normalized = str(score_projection).strip().lower()
    if normalized in ("xyz", "3d"):
        return "xyz"
    if normalized in ("xy", "floor", "floor_projection"):
        return "xy"
    raise ValueError(
        "thickness score_projection must be one of: xyz, xy"
    )


def project_points_for_thickness(points, score_projection):
    projection = normalize_thickness_projection(score_projection)
    if projection == "xy":
        return points[:, :2]
    return points


def distance_score_weights(ranges, config):
    if not config or not bool(config.get("enabled", False)):
        return None

    ranges = np.asarray(ranges, dtype=np.float64)
    reference_range = max(float(config.get("reference_range", 2.0)), 1e-6)
    power = float(config.get("power", 1.0))
    min_weight = float(config.get("min_weight", 0.25))
    max_weight = float(config.get("max_weight", 4.0))
    weights = np.power(np.maximum(ranges, 0.0) / reference_range, power)
    return np.clip(weights, min_weight, max_weight)


def parse_weighted_score_metric(score_metric, metric_names, default_weight):
    if score_metric in metric_names:
        return float(default_weight)

    for metric_name in metric_names:
        prefix = f"{metric_name}+"
        if score_metric.startswith(prefix):
            return float(score_metric[len(prefix):])

    return None


def pose_tilt_penalty(pose, reference_pose=None, squared=False):
    if reference_pose is None:
        roll = float(pose["roll"])
        pitch = float(pose["pitch"])
    else:
        roll = float(normalize_angle(
            float(pose["roll"]) - float(reference_pose["roll"])
        ))
        pitch = float(normalize_angle(
            float(pose["pitch"]) - float(reference_pose["pitch"])
        ))

    if squared:
        return float((roll * roll) + (pitch * pitch))
    return float(np.hypot(roll, pitch))


def initial_scan_overlap_score(
    chunks,
    pose,
    indices,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
    fixed_grid_pose=None,
    distance_weight_config=None,
):
    clouds = initial_scan_overlap_clouds(chunks, pose, indices)
    target_cloud = clouds["target"]
    source_cloud = clouds["source"]
    if len(target_cloud) == 0 or len(source_cloud) == 0:
        return 1e9
    point_weights = distance_score_weights(
        np.concatenate([clouds["target_ranges"], clouds["source_ranges"]]),
        distance_weight_config,
    )

    score_metric = str(score_metric).strip().lower()
    z_std_weight = parse_weighted_score_metric(
        score_metric,
        ("pca_line_zstd", "pca_xy_zstd", "thickness_zstd"),
        0.005,
    )
    tilt_weight = parse_weighted_score_metric(
        score_metric,
        ("pca_xy_tilt", "pca_line_tilt", "thickness_tilt"),
        0.001,
    )
    tilt2_weight = parse_weighted_score_metric(
        score_metric,
        ("pca_xy_tilt2", "pca_line_tilt2", "thickness_tilt2"),
        0.02,
    )
    if z_std_weight is not None or tilt_weight is not None or tilt2_weight is not None:
        overlap_cloud = np.vstack([target_cloud, source_cloud])
        key_points = None
        if fixed_grid_pose is not None:
            fixed_clouds = initial_scan_overlap_clouds(
                chunks,
                fixed_grid_pose,
                indices,
            )
            key_points = np.vstack([
                fixed_clouds["target"],
                fixed_clouds["source"],
            ])

        score = grid_thickness_score(
            overlap_cloud,
            resolution,
            min_points,
            "pca_line",
            "xy",
            key_points=key_points,
            point_weights=point_weights,
        )
        if z_std_weight is not None:
            score += z_std_weight * grid_thickness_score(
                overlap_cloud,
                resolution,
                min_points,
                "z_std",
                "xyz",
                key_points=key_points,
                point_weights=point_weights,
            )
        if tilt_weight is not None:
            score += tilt_weight * pose_tilt_penalty(
                pose,
                reference_pose=fixed_grid_pose,
            )
        if tilt2_weight is not None:
            score += tilt2_weight * pose_tilt_penalty(
                pose,
                reference_pose=fixed_grid_pose,
                squared=True,
            )
        return float(score)

    if score_metric == "thickness":
        overlap_cloud = np.vstack([target_cloud, source_cloud])
        key_points = None
        if fixed_grid_pose is not None:
            fixed_clouds = initial_scan_overlap_clouds(
                chunks,
                fixed_grid_pose,
                indices,
            )
            key_points = np.vstack([
                fixed_clouds["target"],
                fixed_clouds["source"],
            ])
        return float(grid_thickness_score(
            overlap_cloud,
            resolution,
            min_points,
            thickness_method,
            score_projection,
            key_points=key_points,
            point_weights=point_weights,
        ))

    target_ndt = build_ndt_grid(
        target_cloud,
        resolution,
        min_points,
        score_projection,
    )
    if len(target_ndt) == 0:
        return 1e9

    return float(ndt_score(
        source_cloud,
        target_ndt,
        resolution,
        score_projection,
    ))


def pose_with_rpy(base_pose, roll, pitch, yaw):
    pose = copy_pose(base_pose)
    pose["roll"] = float(normalize_angle(roll))
    pose["pitch"] = float(normalize_angle(pitch))
    pose["yaw"] = float(normalize_angle(yaw))
    return pose


def euler_from_rotation_matrix(rot):
    sy = -float(rot[2, 0])
    sy = min(1.0, max(-1.0, sy))
    pitch = np.arcsin(sy)
    cp = np.cos(pitch)

    if abs(cp) > 1e-9:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rot[0, 1], rot[1, 1])

    return (
        float(normalize_angle(roll)),
        float(normalize_angle(pitch)),
        float(normalize_angle(yaw)),
    )


def axis_angle_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)

    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def tilt_rotation_matrix(direction, tilt):
    tilt_direction = np.array([
        np.cos(direction),
        np.sin(direction),
        0.0,
    ])
    z_axis = np.array([0.0, 0.0, 1.0])
    tilt_axis = np.cross(z_axis, tilt_direction)
    return axis_angle_rotation_matrix(tilt_axis, tilt)


def pose_with_tilt_yaw_delta(
    base_pose,
    tilt_direction,
    tilt,
    yaw_delta,
    tilt_frame="local",
):
    base_rot = rotation_matrix(
        float(base_pose["roll"]),
        float(base_pose["pitch"]),
        float(base_pose["yaw"]),
    )
    delta_rot = (
        rotation_matrix(0.0, 0.0, yaw_delta)
        @ tilt_rotation_matrix(tilt_direction, tilt)
    )
    tilt_frame = str(tilt_frame).strip().lower()
    if tilt_frame in ("local", "lidar", "current"):
        final_rot = base_rot @ delta_rot
    elif tilt_frame in ("global", "base", "robot"):
        final_rot = delta_rot @ base_rot
    else:
        raise ValueError("tilt_frame must be one of: local, global")

    roll, pitch, yaw = euler_from_rotation_matrix(final_rot)
    return pose_with_rpy(base_pose, roll, pitch, yaw)


def pose_plane_normal(pose):
    rot = rotation_matrix(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    return rot @ np.array([0.0, 0.0, 1.0])


def tilt_vector_plot_info(pose, stage_mode=None, tilt_direction=None, tilt=None):
    info = {
        "normal": pose_plane_normal(pose).astype(float).tolist(),
    }
    if stage_mode is not None:
        info["stage_mode"] = stage_mode
    if tilt_direction is not None:
        info["tilt_direction_deg"] = float(np.rad2deg(tilt_direction))
    if tilt is not None:
        info["tilt_deg"] = float(np.rad2deg(tilt))
    return info


def tilt_magnitude_values(stage):
    max_key = "max_tilt_deg"
    if max_key not in stage:
        max_key = "range_tilt_deg"
    if max_key not in stage or "step_tilt_deg" not in stage:
        raise ValueError(
            "tilt_yaw_vector stage must define max_tilt_deg "
            "(or range_tilt_deg) and step_tilt_deg."
        )

    min_tilt = np.deg2rad(float(stage.get("min_tilt_deg", 0.0)))
    max_tilt = np.deg2rad(float(stage[max_key]))
    step = np.deg2rad(float(stage["step_tilt_deg"]))
    if step <= 0.0:
        raise ValueError("step_tilt_deg must be greater than 0.")
    if max_tilt < min_tilt:
        raise ValueError("max_tilt_deg must be greater than or equal to min_tilt_deg.")

    count = int(np.floor((max_tilt - min_tilt) / step + 1e-9)) + 1
    return min_tilt + np.arange(count, dtype=np.float64) * step


def tilt_direction_values(stage):
    for key in (
        "tilt_direction_min_deg",
        "tilt_direction_max_deg",
        "tilt_direction_step_deg",
    ):
        if key not in stage:
            raise ValueError(
                "tilt_yaw_vector stage must define tilt_direction_min_deg, "
                "tilt_direction_max_deg, and tilt_direction_step_deg."
            )

    min_direction = np.deg2rad(float(stage["tilt_direction_min_deg"]))
    max_direction = np.deg2rad(float(stage["tilt_direction_max_deg"]))
    step = np.deg2rad(float(stage["tilt_direction_step_deg"]))
    if step <= 0.0:
        raise ValueError("tilt_direction_step_deg must be greater than 0.")
    if max_direction < min_direction:
        raise ValueError(
            "tilt_direction_max_deg must be greater than or equal to "
            "tilt_direction_min_deg."
        )

    count = int(np.floor((max_direction - min_direction) / step + 1e-9)) + 1
    return min_direction + np.arange(count, dtype=np.float64) * step


def yaw_delta_values(stage):
    if "range_yaw_deg" not in stage and "step_yaw_deg" not in stage:
        return np.array([0.0], dtype=np.float64)
    if "range_yaw_deg" not in stage or "step_yaw_deg" not in stage:
        raise ValueError(
            "tilt_yaw_vector stage must define both range_yaw_deg and "
            "step_yaw_deg, or neither to keep yaw fixed."
        )
    return stage_values(
        0.0,
        np.deg2rad(float(stage["range_yaw_deg"])),
        np.deg2rad(float(stage["step_yaw_deg"])),
    )


def project_cloud_to_xy_plane(points):
    if len(points) == 0:
        return points

    projected = points.copy()
    projected[:, 2] = 0.0
    return projected


INITIAL_OVERLAP_AXIS_CONFIG = {
    "yaw": {
        "pose_key": "yaw",
        "range_key": "range_yaw_deg",
        "step_key": "step_yaw_deg",
    },
    "pitch": {
        "pose_key": "pitch",
        "range_key": "range_pitch_deg",
        "step_key": "step_pitch_deg",
    },
    "roll": {
        "pose_key": "roll",
        "range_key": "range_roll_deg",
        "step_key": "step_roll_deg",
    },
}


def initial_overlap_stage_score_projection(stage, default_projection):
    return stage.get("score_projection", default_projection)


def update_best_initial_overlap(
    best,
    pose,
    score,
    logger,
    lidar_name,
    stage_idx,
    case_idx,
    cases,
    axis_name=None,
):
    if score >= best["score"]:
        return False

    best.update({
        "roll": pose["roll"],
        "pitch": pose["pitch"],
        "yaw": pose["yaw"],
        "score": float(score),
    })
    axis_text = "" if axis_name is None else f", axis={axis_name}"
    logger.info(
        f"{lidar_name} initial-overlap new best: "
        f"stage={stage_idx + 1}{axis_text}, case={case_idx}/{cases}, "
        f"score={score:.4f}, roll={best['roll']:.6f}, "
        f"pitch={best['pitch']:.6f}, yaw={best['yaw']:.6f}"
    )
    return True


def maybe_report_initial_overlap_progress(
    progress_callback,
    progress_state,
    chunks,
    pose,
    indices,
    lidar_name,
    stage_idx,
    case_idx,
    cases,
    best,
    score_projection,
    current_score=None,
    current_vector=None,
    best_vector=None,
    force=False,
):
    if progress_callback is None:
        return
    now = time.perf_counter()
    if (
        not force
        and case_idx != cases
        and now - progress_state["last_time"] < progress_state["interval_sec"]
    ):
        return
    progress_state["last_time"] = now

    current_clouds = initial_scan_overlap_clouds(chunks, pose, indices)
    best_clouds = initial_scan_overlap_clouds(chunks, best, indices)
    progress_callback({
        "lidar_name": f"{lidar_name} initial-overlap",
        "plot_group": "initial_overlap",
        "plot_key": lidar_name,
        "stage_idx": stage_idx,
        "case_idx": case_idx,
        "cases": cases,
        "best": dict(best),
        "current_score": current_score,
        "target_cloud": current_clouds["target"],
        "candidate_cloud": current_clouds["source"],
        "best_target_cloud": best_clouds["target"],
        "best_candidate_cloud": best_clouds["source"],
        "score_projection": score_projection,
        "current_vector": current_vector,
        "best_vector": best_vector,
        "force": force,
    })


def optimize_initial_overlap_full_grid_stage(
    lidar_name,
    chunks,
    indices,
    center,
    best,
    stage,
    stage_idx,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
    distance_weight_config,
    fixed_grid_pose,
    logger,
    progress_callback,
    progress_state,
):
    stage_score_projection = initial_overlap_stage_score_projection(
        stage,
        score_projection,
    )
    rolls = optional_angle_stage_values(
        center["roll"],
        stage,
        "range_roll_deg",
        "step_roll_deg",
    )
    pitches = optional_angle_stage_values(
        center["pitch"],
        stage,
        "range_pitch_deg",
        "step_pitch_deg",
    )
    yaws = optional_angle_stage_values(
        center["yaw"],
        stage,
        "range_yaw_deg",
        "step_yaw_deg",
    )
    cases = len(rolls) * len(pitches) * len(yaws)
    logger.info(
        f"{lidar_name} initial-overlap full-grid stage {stage_idx + 1}: "
        f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
        f"projection={stage_score_projection}, cases={cases}"
    )

    case_idx = 0
    for roll in rolls:
        for pitch in pitches:
            for yaw in yaws:
                case_idx += 1
                pose = pose_with_rpy(center, roll, pitch, yaw)
                score = initial_scan_overlap_score(
                    chunks,
                    pose,
                    indices,
                    resolution,
                    min_points,
                    stage_score_projection,
                    score_metric,
                    thickness_method,
                    fixed_grid_pose=fixed_grid_pose,
                    distance_weight_config=distance_weight_config,
                )
                improved = update_best_initial_overlap(
                    best,
                    pose,
                    score,
                    logger,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    cases,
                )
                maybe_report_initial_overlap_progress(
                    progress_callback,
                    progress_state,
                    chunks,
                    pose,
                    indices,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    cases,
                    best,
                    stage_score_projection,
                    current_score=score,
                    force=improved,
                )


def optimize_initial_overlap_axis_sequential_stage(
    lidar_name,
    chunks,
    indices,
    center,
    best,
    stage,
    stage_idx,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
    distance_weight_config,
    fixed_grid_pose,
    logger,
    progress_callback,
    progress_state,
):
    axis_order = stage.get("axis_order", stage.get("axes", ["yaw", "pitch", "roll"]))
    if isinstance(axis_order, str):
        axis_order = [axis.strip() for axis in axis_order.split(",") if axis.strip()]
    axis_order = [str(axis).strip().lower() for axis in axis_order]
    invalid_axes = [
        axis for axis in axis_order
        if axis not in INITIAL_OVERLAP_AXIS_CONFIG
    ]
    if invalid_axes:
        raise ValueError(f"invalid initial_overlap axis_order: {invalid_axes}")

    passes = max(1, int(stage.get("passes", 1)))
    axis_values_by_pass = []
    total_cases = 0
    current = dict(center)
    for _pass_idx in range(passes):
        pass_values = {}
        for axis_name in axis_order:
            axis_cfg = INITIAL_OVERLAP_AXIS_CONFIG[axis_name]
            values = optional_angle_stage_values(
                current[axis_cfg["pose_key"]],
                stage,
                axis_cfg["range_key"],
                axis_cfg["step_key"],
            )
            pass_values[axis_name] = values
            total_cases += len(values)
        axis_values_by_pass.append(pass_values)

    logger.info(
        f"{lidar_name} initial-overlap axis-sequential stage {stage_idx + 1}: "
        f"axes={axis_order}, passes={passes}, cases={total_cases}"
    )

    case_idx = 0
    current = dict(center)
    for pass_idx in range(passes):
        for axis_name in axis_order:
            axis_cfg = INITIAL_OVERLAP_AXIS_CONFIG[axis_name]
            values = optional_angle_stage_values(
                current[axis_cfg["pose_key"]],
                stage,
                axis_cfg["range_key"],
                axis_cfg["step_key"],
            )
            stage_score_projection = initial_overlap_stage_score_projection(
                stage,
                score_projection,
            )
            logger.info(
                f"{lidar_name} initial-overlap stage {stage_idx + 1} "
                f"pass {pass_idx + 1}/{passes} axis={axis_name}: "
                f"values={len(values)}, projection={stage_score_projection}"
            )
            axis_best_pose = dict(current)
            axis_best_score = best["score"]
            for value in values:
                case_idx += 1
                pose = dict(current)
                pose[axis_cfg["pose_key"]] = float(normalize_angle(value))
                score = initial_scan_overlap_score(
                    chunks,
                    pose,
                    indices,
                    resolution,
                    min_points,
                    stage_score_projection,
                    score_metric,
                    thickness_method,
                    fixed_grid_pose=fixed_grid_pose,
                    distance_weight_config=distance_weight_config,
                )
                if score < axis_best_score:
                    axis_best_pose = dict(pose)
                    axis_best_score = float(score)
                improved = update_best_initial_overlap(
                    best,
                    pose,
                    score,
                    logger,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    total_cases,
                    axis_name=axis_name,
                )
                maybe_report_initial_overlap_progress(
                    progress_callback,
                    progress_state,
                    chunks,
                    pose,
                    indices,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    total_cases,
                    best,
                    stage_score_projection,
                    current_score=score,
                    force=improved,
                )

            current.update(axis_best_pose)


def optimize_initial_overlap_tilt_yaw_vector_stage(
    lidar_name,
    chunks,
    indices,
    center,
    best,
    stage,
    stage_idx,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
    distance_weight_config,
    fixed_grid_pose,
    logger,
    progress_callback,
    progress_state,
):
    stage_score_projection = initial_overlap_stage_score_projection(
        stage,
        score_projection,
    )
    tilt_directions = tilt_direction_values(stage)
    tilts = tilt_magnitude_values(stage)
    yaw_deltas = yaw_delta_values(stage)
    tilt_frame = str(stage.get("tilt_frame", "local")).strip().lower()
    cases = len(tilt_directions) * len(tilts) * len(yaw_deltas)
    logger.info(
        f"{lidar_name} initial-overlap tilt-yaw-vector stage {stage_idx + 1}: "
        f"tilt_directions={len(tilt_directions)}, tilts={len(tilts)}, "
        f"yaw={len(yaw_deltas)}, projection={stage_score_projection}, "
        f"tilt_frame={tilt_frame}, cases={cases}"
    )

    case_idx = 0
    for tilt_direction in tilt_directions:
        for tilt in tilts:
            for yaw_delta in yaw_deltas:
                case_idx += 1
                pose = pose_with_tilt_yaw_delta(
                    center,
                    tilt_direction,
                    tilt,
                    yaw_delta,
                    tilt_frame,
                )
                score = initial_scan_overlap_score(
                    chunks,
                    pose,
                    indices,
                    resolution,
                    min_points,
                    stage_score_projection,
                    score_metric,
                    thickness_method,
                    fixed_grid_pose=fixed_grid_pose,
                    distance_weight_config=distance_weight_config,
                )
                improved = update_best_initial_overlap(
                    best,
                    pose,
                    score,
                    logger,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    cases,
                    axis_name="tilt_yaw_vector",
                )
                maybe_report_initial_overlap_progress(
                    progress_callback,
                    progress_state,
                    chunks,
                    pose,
                    indices,
                    lidar_name,
                    stage_idx,
                    case_idx,
                    cases,
                    best,
                    stage_score_projection,
                    current_score=score,
                    current_vector=tilt_vector_plot_info(
                        pose,
                        stage_mode="tilt_yaw_vector",
                        tilt_direction=tilt_direction,
                        tilt=tilt,
                    ),
                    best_vector=tilt_vector_plot_info(
                        best,
                        stage_mode="tilt_yaw_vector",
                    ),
                    force=improved,
                )


def optimize_lidar_rpy_by_initial_overlap(
    lidar_name,
    chunks,
    init_pose,
    stages,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
    distance_weight_config,
    overlap_stride,
    max_overlap_scans,
    logger,
    progress_interval_sec,
    min_score_improvement_ratio=0.0,
    fixed_grid=False,
    progress_callback=None,
):
    indices = source_indices(len(chunks), overlap_stride, max_overlap_scans)
    fixed_grid_pose = copy_pose(init_pose) if fixed_grid else None
    best = copy_pose(init_pose)
    best["score"] = initial_scan_overlap_score(
        chunks,
        best,
        indices,
        resolution,
        min_points,
        score_projection,
        score_metric,
        thickness_method,
        fixed_grid_pose=fixed_grid_pose,
        distance_weight_config=distance_weight_config,
    )
    initial_score = float(best["score"])

    if best["score"] >= 1e9:
        return {
            **best,
            "success": False,
            "reason": "empty_initial_overlap_score",
        }

    center = dict(best)
    progress_state = {
        "last_time": 0.0,
        "interval_sec": max(float(progress_interval_sec), 0.0),
    }
    for stage_idx, stage in enumerate(stages):
        stage_mode = str(stage.get("mode", "full_grid")).strip().lower()
        if stage_mode in ("full_grid", "grid", "rpy_grid"):
            optimize_initial_overlap_full_grid_stage(
                lidar_name,
                chunks,
                indices,
                center,
                best,
                stage,
                stage_idx,
                resolution,
                min_points,
                score_projection,
                score_metric,
                thickness_method,
                distance_weight_config,
                fixed_grid_pose,
                logger,
                progress_callback,
                progress_state,
            )
        elif stage_mode in ("axis_sequential", "axis", "coordinate_descent"):
            optimize_initial_overlap_axis_sequential_stage(
                lidar_name,
                chunks,
                indices,
                center,
                best,
                stage,
                stage_idx,
                resolution,
                min_points,
                score_projection,
                score_metric,
                thickness_method,
                distance_weight_config,
                fixed_grid_pose,
                logger,
                progress_callback,
                progress_state,
            )
        elif stage_mode in ("tilt_yaw_vector", "tilt_vector", "plane_tilt"):
            optimize_initial_overlap_tilt_yaw_vector_stage(
                lidar_name,
                chunks,
                indices,
                center,
                best,
                stage,
                stage_idx,
                resolution,
                min_points,
                score_projection,
                score_metric,
                thickness_method,
                distance_weight_config,
                fixed_grid_pose,
                logger,
                progress_callback,
                progress_state,
            )
        else:
            raise ValueError(
                "initial_overlap stage mode must be one of: "
                "full_grid, axis_sequential, tilt_yaw_vector"
            )

        center.update(best)
        logger.info(
            f"{lidar_name} initial-overlap stage {stage_idx + 1} best: "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    score_improvement = float(initial_score - best["score"])
    score_improvement_ratio = float(
        score_improvement / max(abs(initial_score), 1e-12)
    )
    if score_improvement_ratio < float(min_score_improvement_ratio):
        rejected = copy_pose(init_pose)
        rejected["score"] = initial_score
        return {
            **rejected,
            "initial_score": initial_score,
            "candidate_pose": {
                key: float(best[key])
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "candidate_score": float(best["score"]),
            "score_improvement": score_improvement,
            "score_improvement_ratio": score_improvement_ratio,
            "rejected_low_improvement": True,
            "success": True,
        }

    return {
        **best,
        "initial_score": initial_score,
        "score_improvement": score_improvement,
        "score_improvement_ratio": score_improvement_ratio,
        "success": True,
    }


def run_initial_overlap_calibration_for_lidar(
    name,
    chunks,
    init_pose,
    overlap_stages,
    overlap_resolution,
    overlap_min_points,
    overlap_score_projection,
    overlap_score_metric,
    overlap_thickness_method,
    overlap_distance_weight_config,
    overlap_stride,
    max_overlap_scans,
    logger,
    progress_interval_sec,
    min_score_improvement_ratio,
    fixed_grid,
    progress_callback,
):
    start_time = time.perf_counter()
    if len(chunks) < 2:
        logger.warn(f"{name}: not enough scans for initial overlap calibration")
        return {
            "name": name,
            "elapsed": time.perf_counter() - start_time,
            "result": {
                **copy_pose(init_pose),
                "score": 1e9,
                "success": False,
                "reason": "not_enough_scans_for_initial_overlap",
            },
        }

    result = optimize_lidar_rpy_by_initial_overlap(
        name,
        chunks,
        init_pose,
        overlap_stages,
        overlap_resolution,
        overlap_min_points,
        overlap_score_projection,
        overlap_score_metric,
        overlap_thickness_method,
        overlap_distance_weight_config,
        overlap_stride,
        max_overlap_scans,
        logger,
        progress_interval_sec,
        min_score_improvement_ratio,
        fixed_grid,
        progress_callback=progress_callback,
    )
    return {
        "name": name,
        "elapsed": time.perf_counter() - start_time,
        "result": result,
    }


def optimize_against_reference_cloud(
    lidar_name,
    source_cloud,
    target_cloud,
    target_ndt,
    init_pose,
    stages,
    resolution,
    score_projection,
    logger,
    progress_interval_sec,
    progress_callback=None,
):
    best = copy_pose(init_pose)
    best["score"] = float(ndt_score(
        source_cloud,
        target_ndt,
        resolution,
        score_projection,
    ))
    center = dict(best)
    last_progress_time = 0.0
    progress_interval_sec = max(float(progress_interval_sec), 0.0)

    for stage_idx, stage in enumerate(stages):
        active_axes = stage_axis_set(
            stage,
            ["x", "y", "z", "roll", "pitch", "yaw"],
        )
        invalid_axes = active_axes - {"x", "y", "z", "roll", "pitch", "yaw"}
        if invalid_axes:
            raise ValueError(f"invalid reference_alignment axes: {sorted(invalid_axes)}")

        xs = selective_stage_values(
            center["x"], stage, "x", "range_x", "step_x", active_axes
        )
        ys = selective_stage_values(
            center["y"], stage, "y", "range_y", "step_y", active_axes
        )
        zs = selective_stage_values(
            center["z"], stage, "z", "range_z", "step_z", active_axes
        )
        rolls = selective_angle_stage_values(
            center["roll"],
            stage,
            "roll",
            "range_roll_deg",
            "step_roll_deg",
            active_axes,
        )
        pitches = selective_angle_stage_values(
            center["pitch"],
            stage,
            "pitch",
            "range_pitch_deg",
            "step_pitch_deg",
            active_axes,
        )
        yaws = selective_angle_stage_values(
            center["yaw"],
            stage,
            "yaw",
            "range_yaw_deg",
            "step_yaw_deg",
            active_axes,
        )
        cases = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitches) * len(yaws)
        logger.info(
            f"{lidar_name} reference-cloud stage {stage_idx + 1}: "
            f"axes={sorted(active_axes)}, "
            f"x={len(xs)}, y={len(ys)}, z={len(zs)}, "
            f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
            f"cases={cases}"
        )

        case_idx = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    for roll in rolls:
                        for pitch in pitches:
                            for yaw in yaws:
                                case_idx += 1
                                candidate = {
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
                                    candidate,
                                )
                                score = ndt_score(
                                    transformed,
                                    target_ndt,
                                    resolution,
                                    score_projection,
                                )
                                improved = score < best["score"]
                                if improved:
                                    best = {
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z),
                                        "roll": float(normalize_angle(roll)),
                                        "pitch": float(normalize_angle(pitch)),
                                        "yaw": float(normalize_angle(yaw)),
                                        "score": float(score),
                                    }
                                    logger.info(
                                        f"{lidar_name} reference-cloud new best: "
                                        f"stage={stage_idx + 1}, "
                                        f"case={case_idx}/{cases}, "
                                        f"score={score:.4f}, "
                                        f"x={best['x']:.4f}, y={best['y']:.4f}, "
                                        f"roll={best['roll']:.6f}, "
                                        f"pitch={best['pitch']:.6f}, "
                                        f"yaw={best['yaw']:.6f}"
                                    )

                                now = time.perf_counter()
                                should_report = (
                                    progress_callback is not None
                                    and (
                                        improved
                                        or case_idx == cases
                                        or now - last_progress_time >= progress_interval_sec
                                    )
                                )
                                if should_report:
                                    last_progress_time = now
                                    best_transformed = transform_accumulated_cloud(
                                        source_cloud,
                                        init_pose,
                                        best,
                                    )
                                    progress_callback({
                                        "lidar_name": lidar_name,
                                        "stage_idx": stage_idx,
                                        "case_idx": case_idx,
                                        "cases": cases,
                                        "best": dict(best),
                                        "current_score": score,
                                        "target_cloud": target_cloud,
                                        "candidate_cloud": transformed,
                                        "best_target_cloud": target_cloud,
                                        "best_candidate_cloud": best_transformed,
                                        "score_projection": score_projection,
                                        "force": improved,
                                    })

        center.update(best)
        logger.info(
            f"{lidar_name} reference-cloud stage {stage_idx + 1} best: "
            f"x={best['x']:.4f}, y={best['y']:.4f}, z={best['z']:.4f}, "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
    }


def calibrate_6dof(
    lidar_config,
    calib_config,
    cloud_buffers,
    collection_elapsed_sec=0.0,
    logger=None,
    progress_callback=None,
):
    logger = logger or NullLogger()
    feedback_calib_count = int(calib_config.get("feedback_calib_count", 0))
    if feedback_calib_count > 0:
        feedback_config = copy.deepcopy(calib_config)
        feedback_config["feedback_calib_count"] = 0
        current_lidar_config = copy.deepcopy(lidar_config)
        feedback_iterations = []
        final_output = None
        total_iterations = feedback_calib_count + 1

        for iteration_idx in range(total_iterations):
            logger.info(
                "Feedback calibration iteration "
                f"{iteration_idx + 1}/{total_iterations}"
            )
            final_output = calibrate_6dof(
                current_lidar_config,
                feedback_config,
                cloud_buffers,
                collection_elapsed_sec=collection_elapsed_sec,
                logger=logger,
                progress_callback=progress_callback,
            )
            feedback_iterations.append(feedback_iteration_summary(
                iteration_idx,
                final_output["result_yaml"],
            ))
            if iteration_idx < total_iterations - 1:
                current_lidar_config = config_with_feedback_poses(
                    current_lidar_config,
                    final_output["result_yaml"],
                )

        final_output["result_yaml"]["feedback_calib_count"] = feedback_calib_count
        final_output["result_yaml"]["feedback_iterations"] = feedback_iterations
        final_output["result_yaml"]["feedback_final_iteration"] = total_iterations - 1
        final_output["calibrated_config"] = make_calibrated_config(
            config_with_feedback_poses(lidar_config, final_output["result_yaml"]),
            final_output["result_yaml"],
        )
        return final_output

    calibration_start_time = time.perf_counter()

    lidars = load_lidar_entries(lidar_config)
    lidar_names = [lidar["name"] for lidar in lidars]
    lidar_poses = {lidar["name"]: copy_pose(lidar["pose"]) for lidar in lidars}
    reference_lidar = calib_config.get("reference_lidar", lidar_names[0])
    if reference_lidar not in lidar_poses:
        raise ValueError(
            f"reference_lidar '{reference_lidar}' is not listed under lidars."
        )

    ndt_cfg = calib_config.get("ndt", {})
    resolution = float(ndt_cfg.get("resolution", 0.5))
    min_points = int(ndt_cfg.get("min_points_per_cell", 5))
    downsample_voxel = float(ndt_cfg.get("downsample_voxel", 0.08))
    score_projection = normalize_score_projection(
        ndt_cfg.get("score_projection", "xy")
    )

    overlap_cfg = calib_config.get("initial_overlap", {})
    overlap_enabled = bool(overlap_cfg.get("enabled", True))
    overlap_stages = overlap_cfg.get("stages", [])
    if overlap_enabled and not overlap_stages:
        raise ValueError("calib_config.yaml must define initial_overlap.stages.")
    overlap_score_projection = normalize_score_projection(
        overlap_cfg.get("score_projection", ndt_cfg.get("self_score_projection", "xyz"))
    )
    overlap_score_metric = str(
        overlap_cfg.get("score_metric", "thickness")
    ).strip().lower()
    overlap_thickness_method = str(
        overlap_cfg.get("thickness_method", "pca_line")
    ).strip().lower()
    overlap_distance_weight_config = overlap_cfg.get("distance_weight")
    overlap_min_score_improvement_ratio = float(
        overlap_cfg.get("min_score_improvement_ratio", 0.0)
    )
    overlap_fixed_grid = bool(overlap_cfg.get("fixed_grid", False))
    overlap_resolution = float(overlap_cfg.get("resolution", resolution))
    overlap_min_points = int(overlap_cfg.get("min_points_per_cell", min_points))
    overlap_stride = int(overlap_cfg.get("scan_stride", 1))
    max_overlap_scans = overlap_cfg.get("max_scans")
    overlap_parallel = bool(overlap_cfg.get("parallel", False))
    overlap_max_workers = int(overlap_cfg.get(
        "max_workers",
        min(len(lidar_names), os.cpu_count() or 1),
    ))
    overlap_max_workers = max(1, min(overlap_max_workers, max(len(lidar_names), 1)))

    reference_cfg = calib_config.get("reference_alignment", {})
    reference_enabled = bool(reference_cfg.get("enabled", True))
    reference_stages = reference_cfg.get("stages", [])
    if reference_enabled and not reference_stages:
        raise ValueError("calib_config.yaml must define reference_alignment.stages.")
    reference_score_projection = normalize_score_projection(
        reference_cfg.get("score_projection", score_projection)
    )
    reference_project_cloud_to_xy = bool(
        reference_cfg.get("project_cloud_to_xy", False)
    )
    plot_cfg = calib_config.get("plot", {})
    progress_interval_sec = float(
        plot_cfg.get("ndt_update_interval_sec", plot_cfg.get("live_update_interval_sec", 0.5))
    )

    scan_chunks_by_lidar = {}
    for name in lidar_names:
        chunks = cloud_buffers.get(name, [])
        downsampled_chunks = []
        for chunk in chunks:
            points = downsample_xyz(chunk["points"], downsample_voxel)
            if len(points) == 0:
                continue
            downsampled_chunks.append({
                "points": points,
                "odom_tf": chunk["odom_tf"],
            })

        scan_chunks_by_lidar[name] = downsampled_chunks
        point_count = sum(len(chunk["points"]) for chunk in downsampled_chunks)
        logger.info(
            f"{name}: {point_count} points after downsample "
            f"from {len(downsampled_chunks)} scans"
        )

    result_yaml = make_initial_result_yaml(reference_lidar, lidars, lidar_poses)
    result_yaml["calibration_mode"] = "initial_overlap_rpy_then_reference_alignment"
    result_yaml["match_mode"] = "initial_overlap_rpy_reference_cloud"
    result_yaml["score_projection"] = score_projection
    result_yaml["initial_overlap_score_metric"] = overlap_score_metric
    result_yaml["initial_overlap_thickness_method"] = overlap_thickness_method
    result_yaml["initial_overlap_distance_weight"] = overlap_distance_weight_config
    result_yaml["initial_overlap_score_projection"] = overlap_score_projection
    result_yaml["initial_overlap_enabled"] = overlap_enabled
    result_yaml["reference_score_projection"] = reference_score_projection
    result_yaml["reference_project_cloud_to_xy"] = reference_project_cloud_to_xy
    result_yaml["reference_alignment_enabled"] = reference_enabled
    result_yaml["reference_alignment_source"] = "initial_overlap_cloud"
    result_yaml["graph_cloud_source"] = "reference_alignment_ndt_input"
    result_yaml["calibration_order"] = lidar_names
    result_yaml["timing_sec"] = {
        "collection": collection_elapsed_sec,
        "initial_overlap_rpy": {},
        "reference_alignment": {},
    }
    result_yaml["success"] = True

    overlap_corrected_poses = copy.deepcopy(lidar_poses)
    for name in lidar_names:
        result_yaml["lidars"][name]["initial_pose"] = copy_pose(lidar_poses[name])

    def apply_initial_overlap_result(stage1_output):
        name = stage1_output["name"]
        elapsed = stage1_output["elapsed"]
        overlap_result = stage1_output["result"]
        result_yaml["timing_sec"]["initial_overlap_rpy"][name] = elapsed
        if overlap_result["success"]:
            overlap_corrected_poses[name]["roll"] = overlap_result["roll"]
            overlap_corrected_poses[name]["pitch"] = overlap_result["pitch"]
            overlap_corrected_poses[name]["yaw"] = overlap_result["yaw"]
        else:
            result_yaml["lidars"][name]["success"] = False
            if "reason" in overlap_result:
                result_yaml["lidars"][name]["reason"] = overlap_result["reason"]

        result_yaml["lidars"][name]["initial_overlap_score"] = float(overlap_result["score"])
        result_yaml["lidars"][name]["initial_overlap_initial_score"] = float(
            overlap_result.get("initial_score", overlap_result["score"])
        )
        result_yaml["lidars"][name]["initial_overlap_score_improvement"] = float(
            overlap_result.get("score_improvement", 0.0)
        )
        result_yaml["lidars"][name]["initial_overlap_score_improvement_ratio"] = float(
            overlap_result.get("score_improvement_ratio", 0.0)
        )
        if overlap_result.get("rejected_low_improvement"):
            result_yaml["lidars"][name]["initial_overlap_rejected_low_improvement"] = True
            result_yaml["lidars"][name]["initial_overlap_candidate_pose"] = (
                overlap_result.get("candidate_pose")
            )
            result_yaml["lidars"][name]["initial_overlap_candidate_score"] = float(
                overlap_result.get("candidate_score", overlap_result["score"])
            )
        result_yaml["lidars"][name]["initial_overlap_success"] = overlap_result["success"]
        result_yaml["lidars"][name]["initial_overlap_time_sec"] = elapsed
        result_yaml["lidars"][name]["initial_overlap_pose"] = {
            key: float(overlap_corrected_poses[name][key])
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        result_yaml["lidars"][name]["stage1_result"] = {
            "pose": copy_pose(overlap_corrected_poses[name]),
            "delta_from_initial": pose_delta(
                lidar_poses[name],
                overlap_corrected_poses[name],
            ),
            "score": float(overlap_result["score"]),
            "initial_score": float(
                overlap_result.get("initial_score", overlap_result["score"])
            ),
            "score_improvement": float(
                overlap_result.get("score_improvement", 0.0)
            ),
            "score_improvement_ratio": float(
                overlap_result.get("score_improvement_ratio", 0.0)
            ),
            "success": bool(overlap_result["success"]),
            "time_sec": elapsed,
        }
        if overlap_result.get("rejected_low_improvement"):
            result_yaml["lidars"][name]["stage1_result"]["rejected_low_improvement"] = True
            result_yaml["lidars"][name]["stage1_result"]["candidate_pose"] = (
                overlap_result.get("candidate_pose")
            )
            result_yaml["lidars"][name]["stage1_result"]["candidate_score"] = float(
                overlap_result.get("candidate_score", overlap_result["score"])
            )

    if overlap_enabled:
        logger.info("Stage 1: per-lidar initial scan overlap roll/pitch/yaw calibration")
        stage1_jobs = [
            (
                name,
                scan_chunks_by_lidar[name],
                lidar_poses[name],
                overlap_stages,
                overlap_resolution,
                overlap_min_points,
                overlap_score_projection,
                overlap_score_metric,
                overlap_thickness_method,
                overlap_distance_weight_config,
                overlap_stride,
                max_overlap_scans,
                logger,
                progress_interval_sec,
                overlap_min_score_improvement_ratio,
                overlap_fixed_grid,
                progress_callback,
            )
            for name in lidar_names
        ]
        if overlap_parallel and overlap_max_workers > 1 and len(stage1_jobs) > 1:
            logger.info(
                "Stage 1 initial-overlap parallel enabled: "
                f"workers={overlap_max_workers}"
            )
            with ThreadPoolExecutor(max_workers=overlap_max_workers) as executor:
                futures = [
                    executor.submit(run_initial_overlap_calibration_for_lidar, *job)
                    for job in stage1_jobs
                ]
                stage1_outputs = [future.result() for future in as_completed(futures)]
            for stage1_output in sorted(stage1_outputs, key=lambda item: lidar_names.index(item["name"])):
                apply_initial_overlap_result(stage1_output)
        else:
            for job in stage1_jobs:
                apply_initial_overlap_result(
                    run_initial_overlap_calibration_for_lidar(*job),
                )
    else:
        logger.info("Stage 1: initial-overlap disabled by config")
        for name in lidar_names:
            result_yaml["lidars"][name]["initial_overlap_score"] = None
            result_yaml["lidars"][name]["initial_overlap_success"] = True
            result_yaml["lidars"][name]["initial_overlap_skipped"] = True
            result_yaml["lidars"][name]["initial_overlap_pose"] = copy_pose(
                overlap_corrected_poses[name]
            )
            result_yaml["lidars"][name]["stage1_result"] = {
                "pose": copy_pose(overlap_corrected_poses[name]),
                "delta_from_initial": pose_delta(
                    lidar_poses[name],
                    overlap_corrected_poses[name],
                ),
                "score": None,
                "success": True,
                "skipped": True,
                "time_sec": 0.0,
            }
            result_yaml["timing_sec"]["initial_overlap_rpy"][name] = 0.0

    logger.info("Stage 2 uses initial-overlap comparison clouds")
    initial_overlap_before_clouds = {
        name: combined_initial_overlap_cloud(
            scan_chunks_by_lidar[name],
            lidar_poses[name],
            overlap_stride,
            max_overlap_scans,
        )
        for name in lidar_names
    }
    overlap_clouds = {
        name: combined_initial_overlap_cloud(
            scan_chunks_by_lidar[name],
            overlap_corrected_poses[name],
            overlap_stride,
            max_overlap_scans,
        )
        for name in lidar_names
    }
    if reference_project_cloud_to_xy:
        logger.info("Stage 2 reference alignment uses XY-projected overlap clouds")
        alignment_clouds = {
            name: project_cloud_to_xy_plane(cloud)
            for name, cloud in overlap_clouds.items()
        }
    else:
        alignment_clouds = overlap_clouds

    if reference_project_cloud_to_xy:
        before_clouds = {
            name: project_cloud_to_xy_plane(cloud)
            for name, cloud in initial_overlap_before_clouds.items()
        }
    else:
        before_clouds = initial_overlap_before_clouds

    reference_cloud = alignment_clouds[reference_lidar]
    after_clouds = {reference_lidar: alignment_clouds[reference_lidar]}
    fused_cloud = reference_cloud

    if reference_enabled:
        logger.info("Stage 2: reference lidar cloud alignment")
        reference_pose = overlap_corrected_poses[reference_lidar]
        result_yaml["lidars"][reference_lidar].update({
            key: reference_pose[key]
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        })
        result_yaml["lidars"][reference_lidar]["optimized"] = True
        result_yaml["lidars"][reference_lidar]["success"] = True
        result_yaml["lidars"][reference_lidar]["score"] = 0.0
        result_yaml["lidars"][reference_lidar]["stage2_result"] = {
            "pose": copy_pose(reference_pose),
            "delta_from_stage1": pose_delta(reference_pose, reference_pose),
            "score": 0.0,
            "success": True,
            "fixed_as_reference": True,
        }
        result_yaml["lidars"][reference_lidar]["final_pose"] = copy_pose(reference_pose)

        for lidar in lidars:
            name = lidar["name"]
            if name == reference_lidar:
                continue

            if len(alignment_clouds[name]) == 0:
                result_yaml["lidars"][name]["success"] = False
                result_yaml["lidars"][name]["reason"] = "no_overlap_cloud"
                continue

            target_ndt = build_ndt_grid(
                fused_cloud,
                resolution,
                min_points,
                reference_score_projection,
            )
            if len(target_ndt) == 0:
                result_yaml["success"] = False
                result_yaml["lidars"][name]["success"] = False
                result_yaml["lidars"][name]["reason"] = "empty_reference_ndt_grid"
                continue

            init_pose = overlap_corrected_poses[name]
            initial_score = float(ndt_score(
                alignment_clouds[name],
                target_ndt,
                resolution,
                reference_score_projection,
            ))
            start_time = time.perf_counter()
            result = optimize_against_reference_cloud(
                name,
                alignment_clouds[name],
                fused_cloud,
                target_ndt,
                init_pose,
                reference_stages,
                resolution,
                reference_score_projection,
                logger,
                progress_interval_sec,
                progress_callback=progress_callback,
            )
            elapsed = time.perf_counter() - start_time
            result_yaml["timing_sec"]["reference_alignment"][name] = elapsed

            result_yaml["lidars"][name].update({
                key: result[key]
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            })
            result_yaml["lidars"][name]["initial_score"] = initial_score
            result_yaml["lidars"][name]["score"] = float(result["score"])
            result_yaml["lidars"][name]["score_improvement"] = float(
                initial_score - result["score"]
            )
            result_yaml["lidars"][name]["optimization_time_sec"] = elapsed
            result_yaml["lidars"][name]["success"] = result["success"]
            result_yaml["lidars"][name]["optimized"] = True
            result_yaml["lidars"][name]["stage2_result"] = {
                "pose": {
                    key: float(result[key])
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                },
                "delta_from_stage1": pose_delta(init_pose, result),
                "initial_score": initial_score,
                "score": float(result["score"]),
                "score_improvement": float(initial_score - result["score"]),
                "success": bool(result["success"]),
                "time_sec": elapsed,
            }
            result_yaml["lidars"][name]["final_pose"] = {
                key: float(result[key])
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            }

            aligned_cloud = transform_accumulated_cloud(
                alignment_clouds[name],
                init_pose,
                result,
            )
            after_clouds[name] = aligned_cloud
            fused_cloud = np.vstack([fused_cloud, aligned_cloud])
            fused_cloud = downsample_xyz(fused_cloud, downsample_voxel)
    else:
        logger.info("Stage 2: reference alignment disabled by config")
        after_clouds = alignment_clouds
        for name in lidar_names:
            pose = overlap_corrected_poses[name]
            result_yaml["lidars"][name].update({
                key: pose[key]
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            })
            result_yaml["lidars"][name]["success"] = True
            result_yaml["lidars"][name]["optimized"] = False
            result_yaml["lidars"][name]["stage2_result"] = {
                "pose": copy_pose(pose),
                "delta_from_stage1": pose_delta(pose, pose),
                "score": None,
                "success": True,
                "skipped": True,
                "time_sec": 0.0,
            }
            result_yaml["lidars"][name]["final_pose"] = copy_pose(pose)
            result_yaml["timing_sec"]["reference_alignment"][name] = 0.0

    finish_time = time.perf_counter()
    for name in lidar_names:
        if "final_pose" not in result_yaml["lidars"][name]:
            result_yaml["lidars"][name]["final_pose"] = {
                key: float(result_yaml["lidars"][name][key])
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
                if key in result_yaml["lidars"][name]
            }

    result_yaml["timing_sec"]["calibration"] = finish_time - calibration_start_time
    result_yaml["timing_sec"]["total"] = (
        result_yaml["timing_sec"]["collection"]
        + result_yaml["timing_sec"]["calibration"]
    )
    for name in lidar_names:
        result_yaml["lidars"][name] = organize_lidar_result(
            result_yaml["lidars"][name],
        )

    return {
        "result_yaml": result_yaml,
        "calibrated_config": make_calibrated_config(lidar_config, result_yaml),
        "before_clouds": before_clouds,
        "after_clouds": after_clouds,
        "extra_graph_clouds": {
            "initial_overlap": {
                "before": initial_overlap_before_clouds,
                "after": overlap_clouds,
            },
        },
        "point_dim": 3,
    }
