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


def grid_thickness_score(points, resolution, min_points, method="pca_line"):
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
            centered = cell_points - np.mean(cell_points, axis=0)
            cov = (centered.T @ centered) / max(count - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)
            thickness = float(np.sqrt(max(eigvals[0], 0.0)))
        else:
            centered = cell_points - np.mean(cell_points, axis=0)
            cov = (centered.T @ centered) / max(count - 1, 1)
            eigvals = np.sort(np.linalg.eigvalsh(cov))
            thickness = float(np.sqrt(max(eigvals[0] + eigvals[1], 0.0)))

        weighted_score += thickness * count
        total_weight += count

    if total_weight == 0:
        return 1e9

    return weighted_score / total_weight


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
    for stage_idx, stage in enumerate(stages):
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
            f"{lidar_name} initial-overlap rpy stage {stage_idx + 1}: "
            f"roll={len(rolls)}, pitch={len(pitches)}, yaw={len(yaws)}, "
            f"cases={cases}"
        )

        case_idx = 0
        progress_interval = max(cases // 20, 1)
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
                        score_projection,
                        score_metric,
                        thickness_method,
                    )
                    if score < best["score"]:
                        best.update({
                            "roll": pose["roll"],
                            "pitch": pose["pitch"],
                            "yaw": pose["yaw"],
                            "score": float(score),
                        })
                        logger.info(
                            f"{lidar_name} initial-overlap new best: "
                            f"stage={stage_idx + 1}, case={case_idx}/{cases}, "
                            f"score={score:.4f}, roll={best['roll']:.6f}, "
                            f"pitch={best['pitch']:.6f}, yaw={best['yaw']:.6f}"
                        )

                    if (
                        progress_callback is not None
                        and (
                            case_idx % progress_interval == 0
                            or case_idx == cases
                            or score <= best["score"]
                        )
                    ):
                        clouds = initial_scan_overlap_clouds(chunks, pose, indices)
                        progress_callback({
                            "lidar_name": f"{lidar_name} initial-overlap",
                            "stage_idx": stage_idx,
                            "case_idx": case_idx,
                            "cases": cases,
                            "best": dict(best),
                            "target_cloud": clouds["target"],
                            "candidate_cloud": clouds["source"],
                        })

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
        progress_interval = max(cases // 20, 1)
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

                                if (
                                    progress_callback is not None
                                    and (
                                        case_idx % progress_interval == 0
                                        or case_idx == cases
                                        or score <= best["score"]
                                    )
                                ):
                                    progress_callback({
                                        "lidar_name": lidar_name,
                                        "stage_idx": stage_idx,
                                        "case_idx": case_idx,
                                        "cases": cases,
                                        "best": dict(best),
                                        "target_cloud": target_cloud,
                                        "candidate_cloud": transformed,
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

    overlap_clouds = {
        name: transform_scan_chunks(scan_chunks_by_lidar[name], overlap_corrected_poses[name])
        for name in lidar_names
    }

    logger.info("Stage 2: reference lidar cloud alignment")
    reference_cloud = overlap_clouds[reference_lidar]
    after_clouds = {reference_lidar: reference_cloud}
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

        if len(overlap_clouds[name]) == 0:
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
            overlap_clouds[name],
            target_ndt,
            resolution,
            reference_score_projection,
        ))
        start_time = time.perf_counter()
        result = optimize_against_reference_cloud(
            name,
            overlap_clouds[name],
            fused_cloud,
            target_ndt,
            init_pose,
            reference_stages,
            resolution,
            reference_score_projection,
            logger,
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
        fused_cloud = np.vstack([fused_cloud, after_clouds[name]])
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
