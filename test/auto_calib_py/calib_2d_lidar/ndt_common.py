import os

import yaml
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


CALIB_TARGETS = {
    "3": {
        "label": "3dof",
        "workdir": os.path.join(SCRIPT_DIR, "3dof"),
        "script": "lidar_ndt_calib.py",
    },
    "3dof": {
        "label": "3dof",
        "workdir": os.path.join(SCRIPT_DIR, "3dof"),
        "script": "lidar_ndt_calib.py",
    },
    "6": {
        "label": "6dof",
        "workdir": os.path.join(SCRIPT_DIR, "6dof"),
        "script": "lidar_ndt_calib_6dof.py",
    },
    "6dof": {
        "label": "6dof",
        "workdir": os.path.join(SCRIPT_DIR, "6dof"),
        "script": "lidar_ndt_calib_6dof.py",
    },
    "6overlap": {
        "label": "6dof",
        "command_label": "6dof_overlap",
        "workdir": os.path.join(SCRIPT_DIR, "6dof_overlap"),
        "script": "lidar_ndt_calib_6dof_overlap.py",
    },
    "6dof_overlap": {
        "label": "6dof",
        "command_label": "6dof_overlap",
        "workdir": os.path.join(SCRIPT_DIR, "6dof_overlap"),
        "script": "lidar_ndt_calib_6dof_overlap.py",
    },
}


def load_yaml(path, default=None):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return default if default is not None else {}

    if data is None:
        return default if default is not None else {}

    return data


def deep_merge_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def resolve_output_path(output_dir, path):
    if os.path.isabs(path):
        return path

    return os.path.join(output_dir, path)


def graph_data_path(target, calib_config):
    output_dir = resolve_output_path(
        target["workdir"],
        calib_config.get("output_dir", "output"),
    )
    return resolve_output_path(
        output_dir,
        calib_config.get("graph_data_npz", "calibration_graph_data.npz"),
    )


def save_graph_data(path, lidar_names, before_clouds, after_clouds, point_dim):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    graph_arrays = {
        "lidar_names": np.array(lidar_names),
    }
    empty_cloud = np.empty((0, point_dim))

    for idx, name in enumerate(lidar_names):
        graph_arrays[f"before_{idx}"] = before_clouds.get(name, empty_cloud)
        graph_arrays[f"after_{idx}"] = after_clouds.get(name, empty_cloud)

    np.savez(path, **graph_arrays)


def get_target(dof):
    return CALIB_TARGETS[dof]


def validate_target(target):
    script_path = os.path.join(target["workdir"], target["script"])
    lidar_config_path = os.path.join(target["workdir"], "lidar_config.yaml")
    calib_config_path = os.path.join(target["workdir"], "calib_config.yaml")
    common_calib_config_path = os.path.join(
        SCRIPT_DIR,
        "common_calib_config.yaml",
    )

    missing_paths = [
        path
        for path in (script_path, lidar_config_path, calib_config_path)
        if not os.path.exists(path)
    ]
    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_text}")

    return {
        "script_path": script_path,
        "lidar_config_path": lidar_config_path,
        "calib_config_path": calib_config_path,
        "common_calib_config_path": common_calib_config_path,
    }


def load_merged_calib_config(paths):
    common_calib_config = load_yaml(paths["common_calib_config_path"])
    mode_calib_config = load_yaml(paths["calib_config_path"])
    return deep_merge_dicts(common_calib_config, mode_calib_config)


def env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None

    return value.strip().lower() in ("1", "true", "yes", "on")


def apply_runtime_overrides(config, mode):
    plot_cfg = config.setdefault("plot", {})

    backend = os.environ.get("NDT_PLOT_BACKEND")
    if backend:
        plot_cfg["backend"] = backend

    live_plot = env_bool("NDT_LIVE_PLOT")
    if live_plot is not None:
        plot_cfg["show_live_ndt"] = live_plot
        progress_3d_cfg = plot_cfg.setdefault("progress_3d", {})
        progress_3d_cfg["enabled"] = live_plot

    keep_live_plot = env_bool("NDT_KEEP_LIVE_PLOT")
    if keep_live_plot is not None:
        plot_cfg["keep_live_ndt_open"] = keep_live_plot

    return config
