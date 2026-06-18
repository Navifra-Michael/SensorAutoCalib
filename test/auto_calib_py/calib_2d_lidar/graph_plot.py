import os
import site
import multiprocessing as mp
import queue
import time

import numpy as np

MPL_3D_AVAILABLE = False
MPL_3D_ERROR = None
try:
    import mpl_toolkits

    user_mpl_toolkits_path = os.path.join(
        site.getusersitepackages(),
        "mpl_toolkits",
    )
    if os.path.isdir(user_mpl_toolkits_path):
        mpl_toolkits.__path__ = [
            user_mpl_toolkits_path,
            *[
                path
                for path in mpl_toolkits.__path__
                if path != user_mpl_toolkits_path
            ],
        ]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MPL_3D_AVAILABLE = True
except Exception as exc:
    MPL_3D_ERROR = exc
    MPL_3D_AVAILABLE = False

import matplotlib

plot_backend = os.environ.get("NDT_PLOT_BACKEND")
if plot_backend:
    matplotlib.use(plot_backend, force=True)
elif not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ndt_common import graph_data_path, resolve_output_path


def run_plotter_process(plotter_class, args, kwargs, command_queue):
    plotter = plotter_class(*args, **kwargs)
    while True:
        command = command_queue.get()
        if command is None:
            plotter.close(force=True)
            return

        command_type, command_args, command_kwargs = command
        if command_type == "update":
            plotter.update(*command_args, **command_kwargs)
        elif command_type == "close":
            plotter.close(force=bool(command_kwargs.get("force", False)))
            return


class AsyncPlotter:
    def __init__(self, plotter_class, *args, **kwargs):
        self.enabled = bool(kwargs.get("enabled", False))
        self.command_queue = None
        self.process = None
        if not self.enabled:
            return

        self.command_queue = mp.Queue(maxsize=1)
        self.process = mp.Process(
            target=run_plotter_process,
            args=(plotter_class, args, kwargs, self.command_queue),
            daemon=True,
        )
        self.process.start()

    def update(self, *args, **kwargs):
        if not self.enabled or self.command_queue is None:
            return

        command = ("update", args, kwargs)
        while True:
            try:
                self.command_queue.put_nowait(command)
                return
            except queue.Full:
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    return

    def close(self, force=False):
        if not self.enabled or self.command_queue is None or self.process is None:
            return

        self.enabled = False
        command = ("close", (), {"force": force})
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.command_queue.put_nowait(command)
            except queue.Full:
                pass
        self.process.join(timeout=0.5 if force else 2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)
        try:
            self.command_queue.close()
            self.command_queue.cancel_join_thread()
        except Exception:
            pass
        try:
            self.process.close()
        except Exception:
            pass
        self.command_queue = None
        self.process = None


def sample_points(points, max_points):
    if max_points is None or int(max_points) <= 0:
        return points

    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


def set_axes_equal_3d(ax, cloud_sets):
    all_points = [
        points
        for points in cloud_sets
        if points is not None and len(points) > 0
    ]
    if not all_points:
        return

    points = np.vstack(all_points)
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range, 1.0)
    margin = half_range * 0.05
    half_range += margin

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def axis_limits_2d_from_points(points, min_half_range=1.0):
    if not points:
        return None

    xy_points = np.vstack(points)[:, :2]
    min_values = np.min(xy_points, axis=0)
    max_values = np.max(xy_points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range * 1.05, min_half_range)
    return (
        center[0] - half_range,
        center[0] + half_range,
        center[1] - half_range,
        center[1] + half_range,
    )


def axis_limits_3d_from_points(points, min_half_range=1.0):
    if not points:
        return None

    xyz_points = np.vstack(points)
    min_values = np.min(xyz_points, axis=0)
    max_values = np.max(xyz_points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range * 1.05, min_half_range)
    return (
        center[0] - half_range,
        center[0] + half_range,
        center[1] - half_range,
        center[1] + half_range,
        center[2] - half_range,
        center[2] + half_range,
    )


def merge_axis_limits(old_limits, new_limits):
    if new_limits is None:
        return old_limits
    if old_limits is None:
        return new_limits

    return tuple(
        min(old_limits[idx], new_limits[idx])
        if idx % 2 == 0
        else max(old_limits[idx], new_limits[idx])
        for idx in range(len(old_limits))
    )


def apply_axis_limits_2d(ax, limits):
    if limits is None:
        return

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])


def apply_axis_limits_3d(ax, limits):
    if limits is None:
        return

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_zlim(limits[4], limits[5])


def is_interactive_backend():
    return matplotlib.get_backend().lower() != "agg"


def cloud_buffers_to_clouds(cloud_buffers, mode, max_points):
    clouds = {}
    for name, chunks in cloud_buffers.items():
        if not chunks:
            clouds[name] = np.empty((0, 3 if mode == "6dof" else 2))
            continue

        if mode == "6dof":
            points = [
                chunk["points"]
                for chunk in chunks
                if len(chunk["points"]) > 0
            ]
        else:
            points = [chunk for chunk in chunks if len(chunk) > 0]

        if not points:
            clouds[name] = np.empty((0, 3 if mode == "6dof" else 2))
            continue

        clouds[name] = sample_points(np.vstack(points), max_points)

    return clouds


def rotation_matrix_3d(roll, pitch, yaw):
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


def apply_initial_pose_2d(points, pose):
    if len(points) == 0:
        return points

    roll = float(pose.get("roll", 0.0))
    yaw = float(pose["yaw"])
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotated_points = points.copy()
    rotated_points[:, 1] *= np.cos(roll)
    rot = np.array([[c, -s], [s, c]])
    return rotated_points @ rot.T + np.array([float(pose["x"]), float(pose["y"])])


def apply_initial_pose_3d(points, pose):
    if len(points) == 0:
        return points

    rot = rotation_matrix_3d(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    trans = np.array([float(pose["x"]), float(pose["y"]), float(pose["z"])])
    return points @ rot.T + trans


def transform_collection_clouds(cloud_buffers, mode, lidar_config, max_points):
    poses = lidar_config.get("lidars", {})
    transformed_clouds = {}

    if mode == "6dof":
        for name, chunks in cloud_buffers.items():
            pose = poses.get(name)
            transformed_chunks = []
            for chunk in chunks:
                points = chunk["points"]
                if len(points) == 0:
                    continue
                if pose is not None:
                    points = apply_initial_pose_3d(points, pose)
                odom_tf = chunk.get("odom_tf")
                if odom_tf is not None:
                    points = points @ odom_tf[:3, :3].T + odom_tf[:3, 3]
                transformed_chunks.append(points)

            if transformed_chunks:
                transformed_clouds[name] = sample_points(
                    np.vstack(transformed_chunks),
                    max_points,
                )
            else:
                transformed_clouds[name] = np.empty((0, 3))
        return transformed_clouds

    clouds = cloud_buffers_to_clouds(cloud_buffers, mode, max_points)
    for name, points in clouds.items():
        pose = poses.get(name)
        if pose is None:
            transformed_clouds[name] = points
            continue

        transformed_clouds[name] = apply_initial_pose_2d(points, pose)

    return transformed_clouds


class CollectionLivePlot:
    def __init__(
        self,
        enabled,
        mode,
        title,
        max_points=5000,
        update_interval_sec=0.5,
        keep_open=False,
        lidar_config=None,
    ):
        self.enabled = bool(enabled)
        self.mode = mode
        self.title = title
        self.max_points = int(max_points)
        self.update_interval_sec = float(update_interval_sec)
        self.keep_open = bool(keep_open)
        self.lidar_config = lidar_config or {}
        self.last_update_time = 0.0
        self.fig = None
        self.ax = None
        self.ax_xy = None
        self.axis_limits_3d = None
        self.axis_limits_2d = None

        if not self.enabled:
            return

        if not is_interactive_backend():
            print(
                "[graph_plot] live plot disabled: "
                f"matplotlib backend is non-interactive {matplotlib.get_backend()}"
            )
            self.enabled = False
            return

        if self.mode == "6dof" and not MPL_3D_AVAILABLE:
            print(f"[graph_plot] 3D live plot disabled: {MPL_3D_ERROR}")
            self.enabled = False
            return

        try:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 8))
            if self.mode == "6dof":
                self.ax = self.fig.add_subplot(111, projection="3d")
            else:
                self.ax = self.fig.add_subplot(111)
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as exc:
            print(f"[graph_plot] live plot disabled: {exc}")
            self.close(force=True)
            self.enabled = False

    def update(self, cloud_buffers, elapsed_sec, trajectory=None, force=False):
        if not self.enabled or self.fig is None or self.ax is None:
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return

        self.last_update_time = now
        clouds = transform_collection_clouds(
            cloud_buffers,
            self.mode,
            self.lidar_config,
            self.max_points,
        )
        if trajectory is None:
            trajectory = np.empty((0, 3))
        else:
            trajectory = np.asarray(trajectory, dtype=np.float64)
            trajectory = sample_points(trajectory, self.max_points)
        self.ax.clear()

        if self.mode == "6dof":
            plotted = []
            for name, points in clouds.items():
                if len(points) == 0:
                    continue
                self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, label=name)
                plotted.append(points)
            if len(trajectory) > 0:
                self.ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color="black",
                    linewidth=1.5,
                    alpha=0.75,
                    label="robot path",
                )
                self.ax.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    trajectory[0, 2],
                    s=24,
                    c="tab:green",
                    marker="o",
                    label="path start",
                )
                self.ax.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    trajectory[-1, 2],
                    s=30,
                    c="black",
                    marker="x",
                    label="path current",
                )
                plotted.append(trajectory)
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.set_zlabel("z [m]")
            self.axis_limits_3d = merge_axis_limits(
                self.axis_limits_3d,
                axis_limits_3d_from_points(plotted),
            )
            apply_axis_limits_3d(self.ax, self.axis_limits_3d)
        else:
            plotted = []
            for name, points in clouds.items():
                if len(points) == 0:
                    continue
                self.ax.scatter(points[:, 0], points[:, 1], s=1, label=name)
                plotted.append(points)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.grid(True, color="0.9", linewidth=0.4)
            self.axis_limits_2d = merge_axis_limits(
                self.axis_limits_2d,
                axis_limits_2d_from_points(plotted),
            )
            apply_axis_limits_2d(self.ax, self.axis_limits_2d)

        self.ax.set_title(f"{self.title} | elapsed {elapsed_sec:.1f}s")
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self, force=False):
        if self.keep_open and not force:
            plt.ioff()
            plt.show(block=False)
            return

        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.enabled = False
        self.axis_limits_3d = None
        self.axis_limits_2d = None


class NdtLivePlot2D:
    def __init__(
        self,
        enabled,
        title="NDT optimization",
        max_points=5000,
        update_interval_sec=0.5,
        keep_open=False,
    ):
        self.enabled = bool(enabled)
        self.title = title
        self.max_points = int(max_points)
        self.update_interval_sec = float(update_interval_sec)
        self.keep_open = bool(keep_open)
        self.last_update_time = 0.0
        self.fig = None
        self.ax = None

        if not self.enabled:
            return

        if not is_interactive_backend():
            print(
                "[graph_plot] NDT live plot disabled: "
                f"matplotlib backend is non-interactive {matplotlib.get_backend()}"
            )
            self.enabled = False
            return

        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.show(block=False)
            plt.pause(0.001)
            print("[graph_plot] NDT live plot enabled")
        except Exception as exc:
            print(f"[graph_plot] NDT live plot disabled: {exc}")
            self.close(force=True)
            self.enabled = False

    def update(
        self,
        lidar_name,
        stage_idx,
        case_idx,
        cases,
        best,
        target_cloud,
        candidate_cloud,
        best_target_cloud=None,
        best_candidate_cloud=None,
        current_score=None,
        current_vector=None,
        best_vector=None,
        force=False,
    ):
        if not self.enabled or self.fig is None or self.ax is None:
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return

        self.last_update_time = now
        if target_cloud is None:
            target_cloud = np.empty((0, 2))
        if candidate_cloud is None:
            candidate_cloud = np.empty((0, 2))
        target_cloud = sample_points(target_cloud, self.max_points)
        candidate_cloud = sample_points(candidate_cloud, self.max_points)

        self.ax.clear()
        if len(target_cloud) > 0:
            self.ax.scatter(
                target_cloud[:, 0],
                target_cloud[:, 1],
                s=1,
                c="0.65",
                label="target/fused",
            )
        if len(candidate_cloud) > 0:
            self.ax.scatter(
                candidate_cloud[:, 0],
                candidate_cloud[:, 1],
                s=1,
                c="tab:red",
                label=lidar_name,
            )

        progress = case_idx / max(cases, 1) * 100.0
        self.ax.set_title(
            f"{self.title}: {lidar_name}\n"
            f"stage={stage_idx + 1}, {case_idx}/{cases} ({progress:.1f}%), "
            f"score={best['score']:.4f}"
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, color="0.9", linewidth=0.4)
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self, force=False):
        if self.keep_open and not force:
            plt.ioff()
            plt.show(block=False)
            return

        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.enabled = False


class NdtLivePlot3D:
    def __init__(
        self,
        enabled,
        title="NDT optimization",
        max_points=5000,
        update_interval_sec=0.5,
        keep_open=False,
    ):
        self.enabled = bool(enabled)
        self.title = title
        self.max_points = int(max_points)
        self.update_interval_sec = float(update_interval_sec)
        self.keep_open = bool(keep_open)
        self.last_update_time = 0.0
        self.fig = None
        self.ax = None
        self.ax_best = None
        self.ax_xy = None
        self.ax_best_xy = None
        self.current_limits_3d = None
        self.current_limits_2d = None
        self.best_limits_3d = None
        self.best_limits_2d = None

        if not self.enabled:
            return
        if not is_interactive_backend():
            print(
                "[graph_plot] 3D NDT live plot disabled: "
                f"matplotlib backend is non-interactive {matplotlib.get_backend()}"
            )
            self.enabled = False
            return
        if not MPL_3D_AVAILABLE:
            print(f"[graph_plot] 3D NDT live plot disabled: {MPL_3D_ERROR}")
            self.enabled = False
            return

        try:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 10))
            self.ax = self.fig.add_subplot(221, projection="3d")
            self.ax_xy = self.fig.add_subplot(222)
            self.ax_best = self.fig.add_subplot(223, projection="3d")
            self.ax_best_xy = self.fig.add_subplot(224)
            self.fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.06,
                top=0.88,
                wspace=0.18,
                hspace=0.28,
            )
            plt.show(block=False)
            plt.pause(0.001)
            print("[graph_plot] 3D NDT live plot enabled with current/best views")
        except Exception as exc:
            print(f"[graph_plot] 3D NDT live plot disabled: {exc}")
            self.close(force=True)
            self.enabled = False

    def update(
        self,
        lidar_name,
        stage_idx,
        case_idx,
        cases,
        best,
        target_cloud,
        candidate_cloud,
        best_target_cloud=None,
        best_candidate_cloud=None,
        current_score=None,
        current_vector=None,
        best_vector=None,
        force=False,
    ):
        if (
            not self.enabled
            or self.fig is None
            or self.ax is None
            or self.ax_xy is None
            or self.ax_best is None
            or self.ax_best_xy is None
        ):
            return
        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return
        self.last_update_time = now

        if target_cloud is None:
            target_cloud = np.empty((0, 3))
        if candidate_cloud is None:
            candidate_cloud = np.empty((0, 3))
        if best_target_cloud is None:
            best_target_cloud = target_cloud
        if best_candidate_cloud is None:
            best_candidate_cloud = candidate_cloud
        target_cloud = sample_points(target_cloud, self.max_points)
        candidate_cloud = sample_points(candidate_cloud, self.max_points)
        best_target_cloud = sample_points(best_target_cloud, self.max_points)
        best_candidate_cloud = sample_points(best_candidate_cloud, self.max_points)

        self.ax.clear()
        self.ax_xy.clear()
        self.ax_best.clear()
        self.ax_best_xy.clear()

        progress = case_idx / max(cases, 1) * 100.0
        current_score_text = (
            "n/a"
            if current_score is None
            else f"{float(current_score):.4f}"
        )
        title = (
            f"{self.title}: {lidar_name}\n"
            f"stage={stage_idx + 1}, {case_idx}/{cases} ({progress:.1f}%), "
            f"current={current_score_text}, best={best['score']:.4f}"
        )
        self.draw_3d_and_xy_pair(
            self.ax,
            self.ax_xy,
            "Current 3D view",
            "Current XY top view",
            target_cloud,
            candidate_cloud,
            lidar_name,
            limits_attr_prefix="current",
            vector_info=current_vector,
        )
        self.draw_3d_and_xy_pair(
            self.ax_best,
            self.ax_best_xy,
            "Best 3D view",
            "Best XY top view",
            best_target_cloud,
            best_candidate_cloud,
            lidar_name,
            limits_attr_prefix="best",
            vector_info=best_vector,
        )
        self.fig.suptitle(title)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def draw_3d_and_xy_pair(
        self,
        ax_3d,
        ax_xy,
        title_3d,
        title_xy,
        target_cloud,
        candidate_cloud,
        lidar_name,
        limits_attr_prefix,
        vector_info=None,
    ):
        plotted = []
        if len(target_cloud) > 0:
            ax_3d.scatter(
                target_cloud[:, 0],
                target_cloud[:, 1],
                target_cloud[:, 2],
                s=1,
                c="0.65",
                label="target/fused",
            )
            ax_xy.scatter(
                target_cloud[:, 0],
                target_cloud[:, 1],
                s=1,
                c="0.65",
                label="target/fused",
            )
            plotted.append(target_cloud)
        if len(candidate_cloud) > 0:
            ax_3d.scatter(
                candidate_cloud[:, 0],
                candidate_cloud[:, 1],
                candidate_cloud[:, 2],
                s=1,
                c="tab:red",
                label=lidar_name,
            )
            ax_xy.scatter(
                candidate_cloud[:, 0],
                candidate_cloud[:, 1],
                s=1,
                c="tab:red",
                label=lidar_name,
            )
            plotted.append(candidate_cloud)

        self.draw_vector_3d(ax_3d, plotted, vector_info)

        ax_3d.set_title(title_3d)
        ax_3d.set_xlabel("x [m]")
        ax_3d.set_ylabel("y [m]")
        ax_3d.set_zlabel("z [m]")
        ax_3d.legend(loc="upper right")
        limits_3d_attr = f"{limits_attr_prefix}_limits_3d"
        limits_3d = merge_axis_limits(
            getattr(self, limits_3d_attr),
            axis_limits_3d_from_points(plotted),
        )
        setattr(self, limits_3d_attr, limits_3d)
        apply_axis_limits_3d(ax_3d, limits_3d)

        ax_xy.set_title(title_xy)
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True, color="0.9", linewidth=0.4)
        ax_xy.legend(loc="upper right")
        limits_2d_attr = f"{limits_attr_prefix}_limits_2d"
        limits_2d = merge_axis_limits(
            getattr(self, limits_2d_attr),
            axis_limits_2d_from_points(plotted),
        )
        setattr(self, limits_2d_attr, limits_2d)
        apply_axis_limits_2d(ax_xy, limits_2d)

    def draw_vector_3d(self, ax_3d, plotted, vector_info):
        if not vector_info:
            return

        normal = np.asarray(vector_info.get("normal", []), dtype=np.float64)
        if normal.shape != (3,):
            return
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return
        normal = normal / norm

        if plotted:
            all_points = np.vstack(plotted)
            origin = np.mean(all_points, axis=0)
            span = np.max(np.ptp(all_points, axis=0))
            length = max(float(span) * 0.15, 0.25)
        else:
            origin = np.zeros(3)
            length = 0.5

        ax_3d.quiver(
            origin[0],
            origin[1],
            origin[2],
            normal[0],
            normal[1],
            normal[2],
            length=length,
            normalize=True,
            color="tab:blue",
            linewidth=2.0,
            label="scan normal",
        )

        label_parts = ["normal"]
        if "tilt_direction_deg" in vector_info:
            label_parts.append(f"dir={vector_info['tilt_direction_deg']:.1f}deg")
        if "tilt_deg" in vector_info:
            label_parts.append(f"tilt={vector_info['tilt_deg']:.2f}deg")
        end = origin + normal * length
        ax_3d.text(
            end[0],
            end[1],
            end[2],
            " ".join(label_parts),
            color="tab:blue",
            fontsize=8,
        )

    def close(self, force=False):
        if self.keep_open and not force:
            plt.ioff()
            plt.show(block=False)
            return
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.ax_best = None
        self.ax_xy = None
        self.ax_best_xy = None
        self.current_limits_3d = None
        self.current_limits_2d = None
        self.best_limits_3d = None
        self.best_limits_2d = None
        self.enabled = False


class ProgressPlot3D:
    def __init__(
        self,
        enabled,
        title,
        target_cloud,
        max_points=3000,
        update_interval_sec=1.0,
    ):
        self.enabled = enabled
        self.title = title
        self.target_cloud = sample_points(target_cloud, max_points)
        self.max_points = max_points
        self.update_interval_sec = update_interval_sec
        self.last_update_time = 0.0
        self.last_event_time = 0.0
        self.fig = None
        self.ax = None
        self.target_plot = None
        self.moving_plot = None

        if self.enabled and not MPL_3D_AVAILABLE:
            print(f"3D progress plot disabled: {MPL_3D_ERROR}")
            self.enabled = False
            return

        if self.enabled:
            try:
                plt.ion()
                self.fig = plt.figure(figsize=(9, 7))
                self.ax = self.fig.add_subplot(111, projection="3d")
                self._setup_plot()
            except Exception:
                self.close()
                self.enabled = False

    def _setup_plot(self):
        self.target_plot = self.ax.scatter(
            self.target_cloud[:, 0],
            self.target_cloud[:, 1],
            self.target_cloud[:, 2],
            s=1,
            c="tab:blue",
            alpha=0.35,
            label="target/fused",
        )
        self.moving_plot = self.ax.scatter(
            [],
            [],
            [],
            s=1,
            c="tab:orange",
            alpha=0.55,
            label="current best",
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.legend(loc="upper right")
        set_axes_equal_3d(self.ax, [self.target_cloud])
        self.fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)

    def close(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.target_plot = None
        self.moving_plot = None

    def pump_events(self):
        if not self.enabled or self.fig is None:
            return

        now = time.perf_counter()
        if now - self.last_event_time < 0.2:
            return

        self.last_event_time = now
        try:
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self.enabled = False

    def update(self, stage_idx, case_idx, cases, best, moving_cloud, force=False):
        if (
            not self.enabled
            or self.fig is None
            or self.ax is None
            or self.moving_plot is None
        ):
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        now = time.perf_counter()
        if not force and now - self.last_update_time < self.update_interval_sec:
            return

        self.last_update_time = now
        moving_cloud = sample_points(moving_cloud, self.max_points)

        if len(moving_cloud) > 0:
            self.moving_plot._offsets3d = (
                moving_cloud[:, 0],
                moving_cloud[:, 1],
                moving_cloud[:, 2],
            )
        else:
            self.moving_plot._offsets3d = ([], [], [])

        progress = case_idx / max(cases, 1) * 100.0
        self.ax.set_title(
            f"{self.title}\n"
            f"stage={stage_idx + 1}, {case_idx}/{cases} ({progress:.1f}%), "
            f"score={best['score']:.4f}"
        )
        set_axes_equal_3d(self.ax, [self.target_cloud, moving_cloud])
        self.fig.canvas.draw_idle()
        self.pump_events()


def cloud_axis_limits(cloud_sets, axes):
    all_points = []

    for clouds in cloud_sets:
        for points in clouds.values():
            if len(points) > 0:
                all_points.append(points[:, axes])

    if not all_points:
        return None

    points = np.vstack(all_points)
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    center = (min_values + max_values) / 2.0
    half_range = np.max(max_values - min_values) / 2.0
    half_range = max(half_range, 1.0)
    margin = half_range * 0.05

    return (
        center[0] - half_range - margin,
        center[0] + half_range + margin,
        center[1] - half_range - margin,
        center[1] + half_range + margin,
    )


def plot_clouds_projection(
    title,
    clouds,
    save_path,
    axes,
    axis_labels,
    axis_limits=None,
    max_points=12000,
    show_default_grid=True,
    show_result_plot=False,
    keep_result_plot_open=False,
):
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))

    for name, points in clouds.items():
        if len(points) == 0:
            continue

        points = sample_points(points, max_points)
        plt.scatter(points[:, axes[0]], points[:, axes[1]], s=1, label=name)

    if axis_limits is not None:
        axis_1_min, axis_1_max, axis_2_min, axis_2_max = axis_limits
        plt.xlim(axis_1_min, axis_1_max)
        plt.ylim(axis_2_min, axis_2_max)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel(f"{axis_labels[0]} [m]")
    plt.ylabel(f"{axis_labels[1]} [m]")
    if show_default_grid:
        plt.grid(True, color="0.9", linewidth=0.4)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    if show_result_plot and is_interactive_backend():
        plt.show(block=False)
        plt.pause(0.001)
        if not keep_result_plot_open:
            plt.close(fig)
    else:
        plt.close(fig)


def draw_clouds_projection_on_axis(
    ax,
    title,
    clouds,
    axes,
    axis_labels,
    axis_limits=None,
    max_points=12000,
    show_default_grid=True,
):
    for name, points in clouds.items():
        if len(points) == 0:
            continue

        points = sample_points(points, max_points)
        ax.scatter(points[:, axes[0]], points[:, axes[1]], s=1, label=name)

    if axis_limits is not None:
        axis_1_min, axis_1_max, axis_2_min, axis_2_max = axis_limits
        ax.set_xlim(axis_1_min, axis_1_max)
        ax.set_ylim(axis_2_min, axis_2_max)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{axis_labels[0]} [m]")
    ax.set_ylabel(f"{axis_labels[1]} [m]")
    if show_default_grid:
        ax.grid(True, color="0.9", linewidth=0.4)
    ax.set_title(title)


def plot_6dof_result_summary(
    before_clouds,
    after_clouds,
    save_path,
    max_points=12000,
    show_default_grid=True,
    show_result_plot=False,
    keep_result_plot_open=False,
):
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    limits_xy = cloud_axis_limits([before_clouds, after_clouds], (0, 1))
    limits_xz = cloud_axis_limits([before_clouds, after_clouds], (0, 2))
    limits_yz = cloud_axis_limits([before_clouds, after_clouds], (1, 2))
    plot_specs = [
        ("Before - XY", before_clouds, (0, 1), ("x", "y"), limits_xy),
        ("Before - XZ", before_clouds, (0, 2), ("x", "z"), limits_xz),
        ("Before - YZ", before_clouds, (1, 2), ("y", "z"), limits_yz),
        ("After - XY", after_clouds, (0, 1), ("x", "y"), limits_xy),
        ("After - XZ", after_clouds, (0, 2), ("x", "z"), limits_xz),
        ("After - YZ", after_clouds, (1, 2), ("y", "z"), limits_yz),
    ]

    fig, axes_grid = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes_grid.flatten()
    legend_handles = None
    legend_labels = None
    for ax, (title, clouds, axes, labels, limits) in zip(axes_flat, plot_specs):
        draw_clouds_projection_on_axis(
            ax,
            title,
            clouds,
            axes,
            labels,
            axis_limits=limits,
            max_points=max_points,
            show_default_grid=show_default_grid,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(len(legend_labels), 1),
        )
    fig.suptitle("6DOF Calibration Result", y=1.02)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show_result_plot and is_interactive_backend():
        plt.show(block=False)
        plt.pause(0.001)
        if not keep_result_plot_open:
            plt.close(fig)
    else:
        plt.close(fig)


def plot_6dof_cloud_summary(
    title,
    clouds,
    save_path,
    axis_limits=None,
    max_points=12000,
    show_default_grid=True,
    show_result_plot=False,
    keep_result_plot_open=False,
):
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if axis_limits is None:
        axis_limits = {
            "xy": cloud_axis_limits([clouds], (0, 1)),
            "xz": cloud_axis_limits([clouds], (0, 2)),
            "yz": cloud_axis_limits([clouds], (1, 2)),
        }

    plot_specs = [
        ("XY", clouds, (0, 1), ("x", "y"), axis_limits.get("xy")),
        ("XZ", clouds, (0, 2), ("x", "z"), axis_limits.get("xz")),
        ("YZ", clouds, (1, 2), ("y", "z"), axis_limits.get("yz")),
    ]

    fig, axes_grid = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    legend_handles = None
    legend_labels = None
    for ax, (subplot_title, plot_clouds, axes, labels, limits) in zip(
        axes_grid,
        plot_specs,
    ):
        draw_clouds_projection_on_axis(
            ax,
            subplot_title,
            plot_clouds,
            axes,
            labels,
            axis_limits=limits,
            max_points=max_points,
            show_default_grid=show_default_grid,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(len(legend_labels), 1),
        )
    fig.suptitle(title, y=1.05)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show_result_plot and is_interactive_backend():
        plt.show(block=False)
        plt.pause(0.001)
        if not keep_result_plot_open:
            plt.close(fig)
    else:
        plt.close(fig)


def safe_plot_name(name):
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(name)
    )


def plot_initial_overlap_results(
    target,
    calib_config,
    initial_overlap_clouds,
    max_points=12000,
    show_default_grid=True,
    show_result_plot=False,
    keep_result_plot_open=False,
):
    output_dir = resolve_output_path(
        target["workdir"],
        calib_config.get("output_dir", "output"),
    )
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    before_clouds = initial_overlap_clouds.get("before", {})
    after_clouds = initial_overlap_clouds.get("after", {})
    for name in before_clouds:
        if name not in after_clouds:
            continue

        save_path = resolve_output_path(
            output_dir,
            f"{safe_plot_name(name)}_initial_overlap_result.png",
        )
        plot_6dof_result_summary(
            {name: before_clouds[name]},
            {name: after_clouds[name]},
            save_path,
            max_points=max_points,
            show_default_grid=show_default_grid,
            show_result_plot=show_result_plot,
            keep_result_plot_open=keep_result_plot_open,
        )
        outputs.append({
            "path": save_path,
            "exists": os.path.exists(save_path),
        })

    return outputs


def load_graph_clouds(path):
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    lidar_names = [str(name) for name in data["lidar_names"]]
    before_clouds = {}
    after_clouds = {}
    for idx, name in enumerate(lidar_names):
        before_clouds[name] = data[f"before_{idx}"]
        after_clouds[name] = data[f"after_{idx}"]

    extra_graph_clouds = {}
    if "initial_overlap_names" in data:
        initial_names = [str(name) for name in data["initial_overlap_names"]]
        initial_before_clouds = {}
        initial_after_clouds = {}
        for idx, name in enumerate(initial_names):
            initial_before_clouds[name] = data[f"initial_overlap_before_{idx}"]
            initial_after_clouds[name] = data[f"initial_overlap_after_{idx}"]
        extra_graph_clouds["initial_overlap"] = {
            "before": initial_before_clouds,
            "after": initial_after_clouds,
        }

    return {
        "lidar_names": lidar_names,
        "before_clouds": before_clouds,
        "after_clouds": after_clouds,
        "extra_graph_clouds": extra_graph_clouds,
    }


def pose_map_from_yaml(result_yaml):
    return {
        name: {
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
        }
        for name, pose in result_yaml.get("lidars", {}).items()
        if all(key in pose for key in ("x", "y", "yaw"))
    }


def pose_map_from_lidar_config(lidar_config):
    return {
        name: {
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
        }
        for name, pose in lidar_config.get("lidars", {}).items()
        if all(key in pose for key in ("x", "y", "yaw"))
    }


def draw_ndt_grid_2d(axis_limits, resolution):
    if axis_limits is None or resolution is None or resolution <= 0:
        return

    xmin, xmax, ymin, ymax = axis_limits
    x_start = np.floor(xmin / resolution) * resolution
    x_end = np.ceil(xmax / resolution) * resolution
    y_start = np.floor(ymin / resolution) * resolution
    y_end = np.ceil(ymax / resolution) * resolution

    ax = plt.gca()
    for x in np.arange(x_start, x_end + resolution, resolution):
        ax.axvline(x, color="0.82", linewidth=0.4, zorder=0)
    for y in np.arange(y_start, y_end + resolution, resolution):
        ax.axhline(y, color="0.82", linewidth=0.4, zorder=0)


def draw_pose_arrows_2d(
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
            ax.text(
                x + dx * 1.08,
                y + dy * 1.08,
                name,
                fontsize=label_font_size,
                color=color,
                ha="center",
                va="center",
                zorder=7,
            )


def draw_robot_frame_2d(axis_limits):
    if axis_limits is None:
        axis_len = 0.8
    else:
        xmin, xmax, ymin, ymax = axis_limits
        axis_len = max(xmax - xmin, ymax - ymin) * 0.08

    ax = plt.gca()
    alpha = 0.28
    ax.scatter([0.0], [0.0], s=32, c="0.35", marker="o", alpha=alpha, zorder=2)
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


def draw_scale_bar_2d(axis_limits):
    if axis_limits is None:
        return

    xmin, xmax, ymin, ymax = axis_limits
    width = xmax - xmin
    height = ymax - ymin
    scale_len = 1.0
    x0 = xmin + width * 0.06
    y0 = ymin + height * 0.06
    ax = plt.gca()
    ax.plot([x0, x0 + scale_len], [y0, y0], color="black", linewidth=2.0, zorder=8)
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


def plot_clouds_2d(
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
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    for name, points in clouds.items():
        if len(points) == 0:
            continue
        plt.scatter(points[:, 0], points[:, 1], s=1, label=name)

    if axis_limits is not None:
        xmin, xmax, ymin, ymax = axis_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    if grid_resolution is not None:
        draw_ndt_grid_2d(axis_limits, grid_resolution)

    if poses is not None and bool(options.get("show_lidar_arrows", True)):
        draw_pose_arrows_2d(
            poses,
            axis_limits,
            color=pose_color,
            label_font_size=float(options.get("lidar_label_font_size", 6.0)),
        )
    if secondary_poses is not None and bool(options.get("show_lidar_arrows", True)):
        draw_pose_arrows_2d(
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
        draw_robot_frame_2d(axis_limits)
    draw_scale_bar_2d(axis_limits)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if bool(options.get("show_default_grid", True)):
        plt.grid(True, color="0.9", linewidth=0.4)
    else:
        plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    if bool(options.get("show_result_plot", False)) and is_interactive_backend():
        plt.show(block=False)
        plt.pause(0.001)
        if not bool(options.get("keep_result_plot_open", False)):
            plt.close(fig)
    else:
        plt.close(fig)


def expected_plot_paths(target, calib_config):
    output_dir = resolve_output_path(
        target["workdir"],
        calib_config.get("output_dir", "output"),
    )

    if target["label"] == "6dof":
        paths = [
            resolve_output_path(
                output_dir,
                calib_config.get(
                    "result_plot_png",
                    calib_config.get("after_plot_png", "calibration_result.png"),
                ),
            )
        ]
        if "before_plot_png" in calib_config:
            paths.append(resolve_output_path(
                output_dir,
                calib_config.get("before_plot_png"),
            ))
        if "after_plot_png" in calib_config:
            paths.append(resolve_output_path(
                output_dir,
                calib_config.get("after_plot_png"),
            ))
        initial_pattern_suffix = "_initial_overlap_result.png"
        if os.path.isdir(output_dir):
            for filename in sorted(os.listdir(output_dir)):
                if filename.endswith(initial_pattern_suffix):
                    paths.append(resolve_output_path(output_dir, filename))
        return paths

    plot_keys = [
        ("before_plot_png", "before_calibration.png"),
        ("after_plot_png", "after_calibration.png"),
    ]

    return [
        resolve_output_path(output_dir, calib_config.get(key, default))
        for key, default in plot_keys
    ]


def collect_graph_outputs(target, calib_config):
    plot_paths = expected_plot_paths(target, calib_config)
    return [
        {
            "path": path,
            "exists": os.path.exists(path),
        }
        for path in plot_paths
    ]


def generate_graphs(target, lidar_config, calib_config, result_yaml):
    graph_data = load_graph_clouds(graph_data_path(target, calib_config))
    if graph_data is None:
        print("[graph_plot] graph data missing; skip graph generation")
        return []

    before_clouds = graph_data["before_clouds"]
    after_clouds = graph_data["after_clouds"]
    extra_graph_clouds = graph_data.get("extra_graph_clouds", {})
    plot_cfg = calib_config.get("plot", {})
    outputs = expected_plot_paths(target, calib_config)
    show_result_plot = bool(plot_cfg.get("show_result_plot", False))
    keep_result_plot_open = bool(plot_cfg.get("keep_result_plot_open", False))
    if bool(plot_cfg.get("show_result_plot", False)) and not is_interactive_backend():
        print(
            "[graph_plot] result plot window disabled: "
            f"matplotlib backend is non-interactive {matplotlib.get_backend()}"
        )

    if target["label"] == "3dof":
        ndt_cfg = calib_config.get("ndt", {})
        grid_resolution = None
        if bool(plot_cfg.get("show_ndt_grid", calib_config.get("plot_ndt_grid", True))):
            grid_resolution = float(ndt_cfg.get("resolution", 0.2))

        limits = cloud_axis_limits([before_clouds, after_clouds], (0, 1))
        initial_poses = pose_map_from_lidar_config(lidar_config)
        result_poses = pose_map_from_yaml(result_yaml)
        plot_clouds_2d(
            "Before Calibration",
            before_clouds,
            outputs[0],
            limits,
            grid_resolution,
            initial_poses,
            "red",
            options=plot_cfg,
        )
        plot_clouds_2d(
            "After Calibration",
            after_clouds,
            outputs[1],
            limits,
            grid_resolution,
            result_poses,
            "black",
            initial_poses,
            "red",
            options=plot_cfg,
        )
        if show_result_plot and keep_result_plot_open and is_interactive_backend():
            plt.show(block=True)
        return collect_graph_outputs(target, calib_config)

    max_points = int(plot_cfg.get("max_points", 12000))
    show_default_grid = bool(plot_cfg.get("show_default_grid", True))
    plot_6dof_result_summary(
        before_clouds,
        after_clouds,
        outputs[0],
        max_points=max_points,
        show_default_grid=show_default_grid,
        show_result_plot=show_result_plot,
        keep_result_plot_open=keep_result_plot_open,
    )

    graph_outputs = collect_graph_outputs(target, calib_config)
    xy_axis_limits = cloud_axis_limits([before_clouds, after_clouds], (0, 1))
    if "before_plot_png" in calib_config and len(outputs) > 1:
        plot_clouds_projection(
            "Before Calibration",
            before_clouds,
            outputs[1],
            axes=(0, 1),
            axis_labels=("x", "y"),
            axis_limits=xy_axis_limits,
            max_points=max_points,
            show_default_grid=show_default_grid,
            show_result_plot=False,
            keep_result_plot_open=False,
        )
    if "after_plot_png" in calib_config and len(outputs) > 2:
        plot_clouds_projection(
            "After Calibration",
            after_clouds,
            outputs[2],
            axes=(0, 1),
            axis_labels=("x", "y"),
            axis_limits=xy_axis_limits,
            max_points=max_points,
            show_default_grid=show_default_grid,
            show_result_plot=False,
            keep_result_plot_open=False,
        )
    graph_outputs = collect_graph_outputs(target, calib_config)
    initial_overlap_clouds = extra_graph_clouds.get("initial_overlap")
    if initial_overlap_clouds:
        graph_outputs.extend(plot_initial_overlap_results(
            target,
            calib_config,
            initial_overlap_clouds,
            max_points=max_points,
            show_default_grid=show_default_grid,
            show_result_plot=False,
            keep_result_plot_open=False,
        ))

    if show_result_plot and keep_result_plot_open and is_interactive_backend():
        plt.show(block=True)

    return graph_outputs


def print_graph_summary(graph_outputs):
    print("[graph_plot] graph outputs:")
    for output in graph_outputs:
        status = "ok" if output["exists"] else "missing"
        print(f"[graph_plot] {status}: {output['path']}")
