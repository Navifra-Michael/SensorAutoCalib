#!/usr/bin/env python3

import copy
import time

import numpy as np


MAX_LIDAR_COUNT = 10


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class NullLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


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

        lidar_entries.append({
            "name": name,
            "pose": pose,
        })

    return lidar_entries


def pose_roll(pose):
    return float(pose.get("roll", 0.0))


def optimize_lidar_grid_search(
    raw_points,
    target_ndt,
    init_pose,
    resolution,
    search_stages,
    logger,
    initial_score=None,
    progress_callback=None,
    lidar_name="lidar",
    target_cloud=None,
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

        cases = len(xs) * len(ys) * len(yaws)
        case_idx = 0
        progress_interval = max(cases // 20, 1)
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
                    case_idx += 1
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


def calibrate_3dof(
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
    resolution = float(ndt_cfg.get("resolution", 0.2))
    min_points = int(ndt_cfg.get("min_points_per_cell", 3))
    downsample_voxel = float(ndt_cfg.get("downsample_voxel", 0.03))

    search_cfg = calib_config.get("search", {})
    search_stages = search_cfg.get("stages", default_search_stages())

    raw_clouds = {}
    for name in lidar_names:
        chunks = cloud_buffers.get(name, [])
        if len(chunks) == 0:
            logger.warn(f"{name}: no scan received")
            raw_clouds[name] = np.empty((0, 2))
            continue

        points = np.vstack(chunks)
        points = downsample_xy(points, downsample_voxel)
        raw_clouds[name] = points
        logger.info(f"{name}: {len(points)} points after downsample")

    before_clouds = {}
    for name, points in raw_clouds.items():
        pose = lidar_poses[name]
        before_clouds[name] = transform_xy(
            points,
            float(pose["x"]),
            float(pose["y"]),
            float(pose["yaw"]),
            pose_roll(pose),
        )

    reference_cloud = before_clouds[reference_lidar]
    target_ndt = build_ndt_grid(
        reference_cloud,
        resolution=resolution,
        min_points=min_points,
    )
    logger.info(f"NDT cells: {len(target_ndt)}")

    result_yaml = make_initial_result_yaml(reference_lidar, lidars, lidar_poses)
    result_yaml["calibration_mode"] = "sequential_fused_ndt"
    result_yaml["calibration_order"] = [lidar["name"] for lidar in lidars]
    result_yaml["timing_sec"] = {
        "collection": collection_elapsed_sec,
        "per_lidar_optimization": {},
    }

    if len(target_ndt) == 0:
        logger.error(
            "NDT grid is empty. Increase collect_duration_sec, "
            "increase ndt.resolution, or reduce ndt.min_points_per_cell."
        )
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
            "point_dim": 2,
        }

    result_yaml["success"] = True
    after_clouds = {reference_lidar: reference_cloud}
    fused_cloud = reference_cloud

    for lidar in lidars:
        name = lidar["name"]
        points = raw_clouds[name]

        if name == reference_lidar:
            continue

        if len(points) == 0:
            pose = lidar_poses[name]
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
        logger.info(
            f"Optimizing {name} against fused map "
            f"({len(fused_cloud)} points, {len(target_ndt)} NDT cells)..."
        )

        if len(target_ndt) == 0:
            pose = lidar_poses[name]
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

        init_pose = lidar_poses[name]
        lidar_optimization_start = time.perf_counter()
        initial_cloud = transform_xy(
            points,
            float(init_pose["x"]),
            float(init_pose["y"]),
            float(init_pose["yaw"]),
            pose_roll(init_pose),
        )
        initial_score = ndt_score(initial_cloud, target_ndt, resolution)
        result = optimize_lidar_grid_search(
            raw_points=points,
            target_ndt=target_ndt,
            init_pose=init_pose,
            resolution=resolution,
            search_stages=search_stages,
            logger=logger,
            initial_score=initial_score,
            progress_callback=progress_callback,
            lidar_name=name,
            target_cloud=fused_cloud,
        )
        lidar_optimization_sec = time.perf_counter() - lidar_optimization_start
        result_yaml["timing_sec"]["per_lidar_optimization"][name] = (
            lidar_optimization_sec
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
        "point_dim": 2,
    }
