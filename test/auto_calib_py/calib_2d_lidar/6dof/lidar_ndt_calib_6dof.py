#!/usr/bin/env python3

import copy
import time

import numpy as np

MAX_LIDAR_COUNT = 10


class NullLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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


def normalize_score_projection(score_projection):
    normalized = str(score_projection).strip().lower()
    if normalized in ("xyz", "3d"):
        return "xyz"
    if normalized in ("xy", "floor", "floor_projection"):
        return "xy"
    raise ValueError("ndt.score_projection must be one of: xyz, xy")


def project_points_for_ndt(points, score_projection):
    if score_projection == "xy":
        return points[:, :2]
    return points


def build_ndt_grid(points, resolution, min_points, score_projection="xyz"):
    points = project_points_for_ndt(points, score_projection)
    if len(points) == 0:
        return {}

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

        dimension = cell_points.shape[1]
        if cov.shape != (dimension, dimension):
            continue

        cov += np.eye(dimension) * 1e-3

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue

        ndt_grid[idx] = {
            "mean": mean,
            "inv_cov": inv_cov,
        }

    return ndt_grid


def ndt_score(points, ndt_grid, resolution, score_projection="xyz"):
    points = project_points_for_ndt(points, score_projection)
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

    lidar_entries = []

    for name, pose in lidar_cfg.items():
        missing_keys = [
            key for key in ("x", "y", "z", "roll", "pitch", "yaw")
            if key not in pose
        ]
        if missing_keys:
            raise ValueError(f"lidars.{name} is missing keys: {missing_keys}")

        lidar_entries.append({
            "name": name,
            "pose": pose,
        })

    return lidar_entries

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


def optional_stage_values(center, stage, range_key, step_key):
    if range_key not in stage and step_key not in stage:
        return np.array([center], dtype=np.float64)
    if range_key not in stage or step_key not in stage:
        raise ValueError(
            f"search stage must define both {range_key} and {step_key}, "
            "or neither to keep this axis fixed."
        )

    return stage_values(center, float(stage[range_key]), float(stage[step_key]))


def optimize_lidar_grid_search(
    scan_chunks,
    target_ndt,
    init_pose,
    resolution,
    score_projection,
    search_stages,
    logger,
    progress_callback=None,
    lidar_name="lidar",
    target_cloud=None,
    initial_score=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    if initial_score is None:
        initial_cloud = transform_scan_chunks(scan_chunks, init_pose)
        initial_score = ndt_score(
            initial_cloud,
            target_ndt,
            resolution,
            score_projection,
        )
    best["score"] = float(initial_score)

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = optional_stage_values(center["z"], stage, "range_z", "step_z")
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
                                if (
                                    progress_callback is not None
                                    and (
                                        case_idx % progress_interval == 0
                                        or case_idx == cases
                                        or score <= best["score"]
                                    )
                                ):
                                    progress_callback(
                                        {
                                            "lidar_name": lidar_name,
                                            "stage_idx": stage_idx,
                                            "case_idx": case_idx,
                                            "cases": cases,
                                            "best": dict(best),
                                            "target_cloud": target_cloud,
                                            "candidate_cloud": transformed,
                                        }
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
    score_projection,
    search_stages,
    logger,
    progress_callback=None,
    lidar_name="lidar",
    target_cloud=None,
    initial_score=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    if initial_score is None:
        initial_score = ndt_score(
            source_cloud,
            target_ndt,
            resolution,
            score_projection,
        )
    best["score"] = float(initial_score)

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = optional_stage_values(center["z"], stage, "range_z", "step_z")
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
                                if (
                                    progress_callback is not None
                                    and (
                                        case_idx % progress_interval == 0
                                        or case_idx == cases
                                        or score <= best["score"]
                                    )
                                ):
                                    progress_callback(
                                        {
                                            "lidar_name": lidar_name,
                                            "stage_idx": stage_idx,
                                            "case_idx": case_idx,
                                            "cases": cases,
                                            "best": dict(best),
                                            "target_cloud": target_cloud,
                                            "candidate_cloud": transformed,
                                        }
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
    score_projection,
    search_stages,
    logger,
    progress_callback=None,
    lidar_name="lidar",
    target_cloud=None,
    initial_score=None,
):
    best = {
        key: float(init_pose[key])
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    if initial_score is None:
        initial_cloud = transform_precomputed_scan_data(scan_data, init_pose)
        initial_score = ndt_score(
            initial_cloud,
            target_ndt,
            resolution,
            score_projection,
        )
    best["score"] = float(initial_score)

    center = dict(best)

    for stage_idx, stage in enumerate(search_stages):
        xs = stage_values(center["x"], float(stage["range_x"]), float(stage["step_x"]))
        ys = stage_values(center["y"], float(stage["range_y"]), float(stage["step_y"]))
        zs = optional_stage_values(center["z"], stage, "range_z", "step_z")
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
                                if (
                                    progress_callback is not None
                                    and (
                                        case_idx % progress_interval == 0
                                        or case_idx == cases
                                        or score <= best["score"]
                                    )
                                ):
                                    progress_callback(
                                        {
                                            "lidar_name": lidar_name,
                                            "stage_idx": stage_idx,
                                            "case_idx": case_idx,
                                            "cases": cases,
                                            "best": dict(best),
                                            "target_cloud": target_cloud,
                                            "candidate_cloud": transformed,
                                        }
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
    lidar_poses = {lidar["name"]: lidar["pose"] for lidar in lidars}
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
        ndt_cfg.get("score_projection", "xyz")
    )
    match_mode = calib_config.get("match_mode", "exact_precomputed")
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

    search_stages = calib_config.get("search", {}).get("stages", [])
    if not search_stages:
        raise ValueError("calib_config.yaml must define search.stages.")

    scan_chunks_by_lidar = {}
    for name in lidar_names:
        chunks = cloud_buffers.get(name, [])
        if len(chunks) == 0:
            logger.warn(f"{name}: no scan received")
            scan_chunks_by_lidar[name] = []
            continue

        downsampled_chunks = []
        for chunk in chunks:
            points = downsample_xyz(chunk["points"], downsample_voxel)
            if len(points) == 0:
                continue
            downsampled_chunks.append({
                "points": points,
                "odom_tf": chunk["odom_tf"],
            })

        if not downsampled_chunks:
            logger.warn(f"{name}: no valid scan points")
            scan_chunks_by_lidar[name] = []
            continue

        scan_chunks_by_lidar[name] = downsampled_chunks
        point_count = sum(len(chunk["points"]) for chunk in downsampled_chunks)
        logger.info(
            f"{name}: {point_count} points after downsample "
            f"from {len(downsampled_chunks)} scans"
        )

    before_clouds = {}
    for name, chunks in scan_chunks_by_lidar.items():
        before_clouds[name] = transform_scan_chunks(chunks, lidar_poses[name])

    reference_cloud = before_clouds[reference_lidar]
    target_ndt = build_ndt_grid(
        reference_cloud,
        resolution,
        min_points,
        score_projection,
    )
    logger.info(f"NDT cells: {len(target_ndt)} ({score_projection} score)")

    result_yaml = make_initial_result_yaml(reference_lidar, lidars, lidar_poses)
    result_yaml["calibration_mode"] = (
        f"sequential_fused_ndt_2d_lidar_odom_6dof_{match_mode}"
    )
    result_yaml["match_mode"] = match_mode
    result_yaml["score_projection"] = score_projection
    result_yaml["calibration_order"] = [lidar["name"] for lidar in lidars]
    result_yaml["timing_sec"] = {
        "collection": collection_elapsed_sec,
        "per_lidar_optimization": {},
    }

    if len(target_ndt) == 0:
        finish_time = time.perf_counter()
        result_yaml["success"] = False
        result_yaml["reason"] = "empty_ndt_grid"
        result_yaml["timing_sec"]["calibration"] = (
            finish_time - calibration_start_time
        )
        result_yaml["timing_sec"]["total"] = (
            result_yaml["timing_sec"]["collection"]
            + result_yaml["timing_sec"]["calibration"]
        )
        return {
            "result_yaml": result_yaml,
            "calibrated_config": make_calibrated_config(lidar_config, result_yaml),
            "before_clouds": before_clouds,
            "after_clouds": before_clouds,
            "point_dim": 3,
        }

    result_yaml["success"] = True
    after_clouds = {reference_lidar: reference_cloud}
    fused_cloud = reference_cloud
    precomputed_scan_data_by_lidar = {
        name: precompute_scan_data(chunks)
        for name, chunks in scan_chunks_by_lidar.items()
    }

    for lidar in lidars:
        name = lidar["name"]
        chunks = scan_chunks_by_lidar[name]

        if name == reference_lidar:
            continue

        if len(chunks) == 0:
            result_yaml["lidars"][name]["success"] = False
            result_yaml["lidars"][name]["reason"] = "no_scan_received"
            continue

        target_ndt = build_ndt_grid(
            fused_cloud,
            resolution,
            min_points,
            score_projection,
        )
        if len(target_ndt) == 0:
            result_yaml["success"] = False
            result_yaml["lidars"][name]["success"] = False
            result_yaml["lidars"][name]["reason"] = "empty_fused_ndt_grid"
            continue

        logger.info(f"Optimizing {name} in 6DOF...")
        initial_score = float(ndt_score(
            before_clouds[name],
            target_ndt,
            resolution,
            score_projection,
        ))
        logger.info(
            f"{name} initial score: {initial_score:.4f} "
            f"(x={float(lidar_poses[name]['x']):.4f}, "
            f"y={float(lidar_poses[name]['y']):.4f}, "
            f"z={float(lidar_poses[name]['z']):.4f}, "
            f"roll={float(lidar_poses[name]['roll']):.6f}, "
            f"pitch={float(lidar_poses[name]['pitch']):.6f}, "
            f"yaw={float(lidar_poses[name]['yaw']):.6f})"
        )
        start_time = time.perf_counter()
        if match_mode == "fast_accumulated_cloud":
            result = optimize_accumulated_cloud_grid_search(
                source_cloud=before_clouds[name],
                target_ndt=target_ndt,
                init_pose=lidar_poses[name],
                resolution=resolution,
                score_projection=score_projection,
                search_stages=search_stages,
                logger=logger,
                progress_callback=progress_callback,
                lidar_name=name,
                target_cloud=fused_cloud,
                initial_score=initial_score,
            )
        elif match_mode == "exact_precomputed":
            result = optimize_precomputed_scan_grid_search(
                scan_data=precomputed_scan_data_by_lidar[name],
                target_ndt=target_ndt,
                init_pose=lidar_poses[name],
                resolution=resolution,
                score_projection=score_projection,
                search_stages=search_stages,
                logger=logger,
                progress_callback=progress_callback,
                lidar_name=name,
                target_cloud=fused_cloud,
                initial_score=initial_score,
            )
        else:
            result = optimize_lidar_grid_search(
                scan_chunks=chunks,
                target_ndt=target_ndt,
                init_pose=lidar_poses[name],
                resolution=resolution,
                score_projection=score_projection,
                search_stages=search_stages,
                logger=logger,
                progress_callback=progress_callback,
                lidar_name=name,
                target_cloud=fused_cloud,
                initial_score=initial_score,
            )
        elapsed = time.perf_counter() - start_time
        result_yaml["timing_sec"]["per_lidar_optimization"][name] = elapsed

        result_yaml["lidars"][name] = {
            key: result[key]
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        result_yaml["lidars"][name]["score"] = float(result["score"])
        result_yaml["lidars"][name]["initial_score"] = initial_score
        result_yaml["lidars"][name]["score_improvement"] = float(
            initial_score - result["score"]
        )
        result_yaml["lidars"][name]["optimization_time_sec"] = elapsed
        result_yaml["lidars"][name]["success"] = result["success"]
        result_yaml["lidars"][name]["optimized"] = True

        if match_mode == "fast_accumulated_cloud":
            after_clouds[name] = transform_accumulated_cloud(
                before_clouds[name],
                lidar_poses[name],
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
