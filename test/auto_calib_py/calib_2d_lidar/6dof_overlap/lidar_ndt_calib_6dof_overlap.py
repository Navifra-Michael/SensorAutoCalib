#!/usr/bin/env python3

import copy
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
stage_values = _BASE.stage_values
transform_accumulated_cloud = _BASE.transform_accumulated_cloud
transform_scan_chunks = _BASE.transform_scan_chunks


def copy_pose(pose):
    return {
        key: float(pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
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
        }

    initial_odom_inv = invert_transform(chunks[0]["odom_tf"])
    target_cloud = transform_single_chunk_to_initial(
        chunks[0],
        pose,
        initial_odom_inv,
    )
    source_cloud = np.vstack([
        transform_single_chunk_to_initial(chunks[idx], pose, initial_odom_inv)
        for idx in indices
    ])
    return {
        "target": target_cloud,
        "source": source_cloud,
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
):
    if len(points) == 0:
        return 1e9

    method = str(method).strip().lower()
    xy_keys = np.floor(points[:, :2] / resolution).astype(np.int64)
    cells = {}
    for point, key in zip(points, map(tuple, xy_keys)):
        cells.setdefault(key, []).append(point)

    weighted_score = 0.0
    total_weight = 0
    for cell_points in cells.values():
        cell_points = np.asarray(cell_points, dtype=np.float64)
        count = len(cell_points)
        if count < min_points:
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

        weighted_score += thickness * count
        total_weight += count

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


def initial_scan_overlap_score(
    chunks,
    pose,
    indices,
    resolution,
    min_points,
    score_projection,
    score_metric,
    thickness_method,
):
    clouds = initial_scan_overlap_clouds(chunks, pose, indices)
    target_cloud = clouds["target"]
    source_cloud = clouds["source"]
    if len(target_cloud) == 0 or len(source_cloud) == 0:
        return 1e9

    score_metric = str(score_metric).strip().lower()
    if score_metric == "thickness":
        overlap_cloud = np.vstack([target_cloud, source_cloud])
        return float(grid_thickness_score(
            overlap_cloud,
            resolution,
            min_points,
            thickness_method,
            score_projection,
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
    overlap_stride,
    max_overlap_scans,
    logger,
    progress_interval_sec,
    progress_callback=None,
):
    indices = source_indices(len(chunks), overlap_stride, max_overlap_scans)
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
    )

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
                logger,
                progress_callback,
                progress_state,
            )
        else:
            raise ValueError(
                "initial_overlap stage mode must be one of: "
                "full_grid, axis_sequential"
            )

        center.update(best)
        logger.info(
            f"{lidar_name} initial-overlap stage {stage_idx + 1} best: "
            f"roll={best['roll']:.6f}, pitch={best['pitch']:.6f}, "
            f"yaw={best['yaw']:.6f}, score={best['score']:.4f}"
        )

    return {
        **best,
        "success": True,
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
        xs = optional_stage_values(center["x"], stage, "range_x", "step_x")
        ys = optional_stage_values(center["y"], stage, "range_y", "step_y")
        zs = optional_stage_values(center["z"], stage, "range_z", "step_z")
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
        cases = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitches) * len(yaws)
        logger.info(
            f"{lidar_name} reference-cloud stage {stage_idx + 1}: "
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
    overlap_stages = overlap_cfg.get("stages", [])
    if not overlap_stages:
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
    overlap_resolution = float(overlap_cfg.get("resolution", resolution))
    overlap_min_points = int(overlap_cfg.get("min_points_per_cell", min_points))
    overlap_stride = int(overlap_cfg.get("scan_stride", 1))
    max_overlap_scans = overlap_cfg.get("max_scans")

    reference_cfg = calib_config.get("reference_alignment", {})
    reference_stages = reference_cfg.get("stages", [])
    if not reference_stages:
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

    before_clouds = {
        name: transform_scan_chunks(scan_chunks_by_lidar[name], lidar_poses[name])
        for name in lidar_names
    }

    result_yaml = make_initial_result_yaml(reference_lidar, lidars, lidar_poses)
    result_yaml["calibration_mode"] = "initial_overlap_rpy_then_reference_alignment"
    result_yaml["match_mode"] = "initial_overlap_rpy_reference_cloud"
    result_yaml["score_projection"] = score_projection
    result_yaml["initial_overlap_score_metric"] = overlap_score_metric
    result_yaml["initial_overlap_thickness_method"] = overlap_thickness_method
    result_yaml["initial_overlap_score_projection"] = overlap_score_projection
    result_yaml["reference_score_projection"] = reference_score_projection
    result_yaml["reference_project_cloud_to_xy"] = reference_project_cloud_to_xy
    result_yaml["reference_alignment_source"] = "initial_overlap_cloud"
    result_yaml["calibration_order"] = lidar_names
    result_yaml["timing_sec"] = {
        "collection": collection_elapsed_sec,
        "initial_overlap_rpy": {},
        "reference_alignment": {},
    }
    result_yaml["success"] = True

    logger.info("Stage 1: per-lidar initial scan overlap roll/pitch/yaw calibration")
    overlap_corrected_poses = copy.deepcopy(lidar_poses)
    for name in lidar_names:
        chunks = scan_chunks_by_lidar[name]
        if len(chunks) < 2:
            result_yaml["lidars"][name]["success"] = False
            result_yaml["lidars"][name]["reason"] = "not_enough_scans_for_initial_overlap"
            logger.warn(f"{name}: not enough scans for initial overlap calibration")
            continue

        start_time = time.perf_counter()
        overlap_result = optimize_lidar_rpy_by_initial_overlap(
            name,
            chunks,
            lidar_poses[name],
            overlap_stages,
            overlap_resolution,
            overlap_min_points,
            overlap_score_projection,
            overlap_score_metric,
            overlap_thickness_method,
            overlap_stride,
            max_overlap_scans,
            logger,
            progress_interval_sec,
            progress_callback=progress_callback,
        )
        elapsed = time.perf_counter() - start_time
        result_yaml["timing_sec"]["initial_overlap_rpy"][name] = elapsed
        if overlap_result["success"]:
            overlap_corrected_poses[name]["roll"] = overlap_result["roll"]
            overlap_corrected_poses[name]["pitch"] = overlap_result["pitch"]
            overlap_corrected_poses[name]["yaw"] = overlap_result["yaw"]

        result_yaml["lidars"][name]["initial_overlap_score"] = float(overlap_result["score"])
        result_yaml["lidars"][name]["initial_overlap_success"] = overlap_result["success"]
        result_yaml["lidars"][name]["initial_overlap_time_sec"] = elapsed
        result_yaml["lidars"][name]["initial_overlap_pose"] = {
            key: float(overlap_corrected_poses[name][key])
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }

    logger.info("Stage 2 uses initial-overlap comparison clouds")
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

    logger.info("Stage 2: reference lidar cloud alignment")
    reference_cloud = alignment_clouds[reference_lidar]
    after_clouds = {reference_lidar: overlap_clouds[reference_lidar]}
    fused_cloud = reference_cloud

    reference_pose = overlap_corrected_poses[reference_lidar]
    result_yaml["lidars"][reference_lidar].update({
        key: reference_pose[key]
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    })
    result_yaml["lidars"][reference_lidar]["optimized"] = True
    result_yaml["lidars"][reference_lidar]["success"] = True
    result_yaml["lidars"][reference_lidar]["score"] = 0.0

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

        after_clouds[name] = transform_accumulated_cloud(
            overlap_clouds[name],
            init_pose,
            result,
        )
        aligned_cloud = transform_accumulated_cloud(
            alignment_clouds[name],
            init_pose,
            result,
        )
        fused_cloud = np.vstack([fused_cloud, aligned_cloud])
        fused_cloud = downsample_xyz(fused_cloud, downsample_voxel)

    finish_time = time.perf_counter()
    result_yaml["timing_sec"]["calibration"] = finish_time - calibration_start_time
    result_yaml["timing_sec"]["total"] = (
        result_yaml["timing_sec"]["collection"]
        + result_yaml["timing_sec"]["calibration"]
    )

    return {
        "result_yaml": result_yaml,
        "calibrated_config": make_calibrated_config(lidar_config, result_yaml),
        "before_clouds": before_clouds,
        "after_clouds": after_clouds,
        "point_dim": 3,
    }
