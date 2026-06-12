#!/usr/bin/env python3

import argparse
import os
import sys

from ndt_common import (
    CALIB_TARGETS,
    get_target,
    load_merged_calib_config,
    validate_target,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 2D LiDAR NDT calibration in 3DOF or 6DOF mode.",
    )
    parser.add_argument(
        "dof",
        nargs="?",
        default="3dof",
        choices=sorted(CALIB_TARGETS.keys()),
        help=(
            "Calibration mode to run: 3, 3dof, 6, 6dof, "
            "6overlap, or 6dof_overlap. Default: 3dof."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Default: current Python.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected command without running it.",
    )
    parser.add_argument(
        "--backend",
        help="Matplotlib backend to use for live plots, for example TkAgg.",
    )

    live_plot_group = parser.add_mutually_exclusive_group()
    live_plot_group.add_argument(
        "--live-plot",
        action="store_true",
        help="Enable live/progress plotting for the selected calibrator.",
    )
    live_plot_group.add_argument(
        "--no-live-plot",
        action="store_true",
        help="Disable live/progress plotting for the selected calibrator.",
    )

    keep_plot_group = parser.add_mutually_exclusive_group()
    keep_plot_group.add_argument(
        "--keep-live-plot",
        action="store_true",
        help="Keep the 3DOF live NDT plot open after calibration.",
    )
    keep_plot_group.add_argument(
        "--close-live-plot",
        action="store_true",
        help="Close the 3DOF live NDT plot automatically after calibration.",
    )
    return parser.parse_args()


def print_run_summary(calibration_run):
    print(f"[ndt_main] mode: {calibration_run.target['label']}")
    print(f"[ndt_main] workdir: {calibration_run.target['workdir']}")
    print(f"[ndt_main] command: {' '.join(calibration_run.command)}")
    print(f"[ndt_main] lidar config: {calibration_run.paths['lidar_config_path']}")
    print(
        "[ndt_main] common calib config: "
        f"{calibration_run.paths['common_calib_config_path']}"
    )
    print(f"[ndt_main] calib config: {calibration_run.paths['calib_config_path']}")
    print(f"[ndt_main] output yaml: {calibration_run.paths['output_yaml']}")


def print_result_summary(result):
    if result.returncode != 0:
        print(f"[ndt_main] calibration process failed: {result.returncode}")
        return

    if not result.result_yaml:
        print("[ndt_main] calibration finished, but result YAML was not found.")
        return

    print(f"[ndt_main] success: {result.result_yaml.get('success')}")
    print(f"[ndt_main] result: {result.paths['output_yaml']}")
    print(f"[ndt_main] calibrated config: {result.paths['calibrated_config_yaml']}")

    lidars = result.result_yaml.get("lidars", {})
    for name, pose in lidars.items():
        optimized = pose.get("optimized", False)
        score = pose.get("score", None)
        score_text = "" if score is None else f", score={score:.4f}"
        print(f"[ndt_main] {name}: optimized={optimized}{score_text}")


def main():
    args = parse_args()
    target = get_target(args.dof)
    preflight_paths = validate_target(target)
    preflight_calib_config = load_merged_calib_config(preflight_paths)
    preflight_plot_config = preflight_calib_config.get("plot", {})

    if args.backend:
        os.environ["NDT_PLOT_BACKEND"] = args.backend
    elif preflight_plot_config.get("backend"):
        os.environ["NDT_PLOT_BACKEND"] = str(preflight_plot_config["backend"])
    if args.live_plot:
        os.environ["NDT_LIVE_PLOT"] = "1"
    elif args.no_live_plot:
        os.environ["NDT_LIVE_PLOT"] = "0"
    if args.keep_live_plot:
        os.environ["NDT_KEEP_LIVE_PLOT"] = "1"
    elif args.close_live_plot:
        os.environ["NDT_KEEP_LIVE_PLOT"] = "0"

    from data_association import prepare_run, run_data_association
    from graph_plot import collect_graph_outputs, print_graph_summary

    calibration_run = prepare_run(target, args)
    print_run_summary(calibration_run)

    if args.dry_run:
        graph_outputs = collect_graph_outputs(
            calibration_run.target,
            calibration_run.calib_config,
        )
        print_graph_summary(graph_outputs)
        return 0

    result = run_data_association(calibration_run)
    print_result_summary(result)

    graph_outputs = collect_graph_outputs(
        calibration_run.target,
        calibration_run.calib_config,
    )
    print_graph_summary(graph_outputs)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
