import os
import site
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


def sample_points(points, max_points):
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
            set_axes_equal_3d(self.ax, plotted)
        else:
            for name, points in clouds.items():
                if len(points) == 0:
                    continue
                self.ax.scatter(points[:, 0], points[:, 1], s=1, label=name)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.grid(True, color="0.9", linewidth=0.4)

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
            self.fig = plt.figure(figsize=(14, 6))
            self.ax = self.fig.add_subplot(121, projection="3d")
            self.ax_xy = self.fig.add_subplot(122)
            plt.show(block=False)
            plt.pause(0.001)
            print("[graph_plot] 3D NDT live plot enabled with XY view")
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
        force=False,
    ):
        if (
            not self.enabled
            or self.fig is None
            or self.ax is None
            or self.ax_xy is None
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
        target_cloud = sample_points(target_cloud, self.max_points)
        candidate_cloud = sample_points(candidate_cloud, self.max_points)

        self.ax.clear()
        self.ax_xy.clear()
        plotted = []
        if len(target_cloud) > 0:
            self.ax.scatter(
                target_cloud[:, 0],
                target_cloud[:, 1],
                target_cloud[:, 2],
                s=1,
                c="0.65",
                label="target/fused",
            )
            self.ax_xy.scatter(
                target_cloud[:, 0],
                target_cloud[:, 1],
                s=1,
                c="0.65",
                label="target/fused",
            )
            plotted.append(target_cloud)
        if len(candidate_cloud) > 0:
            self.ax.scatter(
                candidate_cloud[:, 0],
                candidate_cloud[:, 1],
                candidate_cloud[:, 2],
                s=1,
                c="tab:red",
                label=lidar_name,
            )
            self.ax_xy.scatter(
                candidate_cloud[:, 0],
                candidate_cloud[:, 1],
                s=1,
                c="tab:red",
                label=lidar_name,
            )
            plotted.append(candidate_cloud)

        progress = case_idx / max(cases, 1) * 100.0
        title = (
            f"{self.title}: {lidar_name}\n"
            f"stage={stage_idx + 1}, {case_idx}/{cases} ({progress:.1f}%), "
            f"score={best['score']:.4f}"
        )
        self.ax.set_title(f"{title}\n3D")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.legend(loc="upper right")
        set_axes_equal_3d(self.ax, plotted)

        self.ax_xy.set_title("XY top view")
        self.ax_xy.set_xlabel("x [m]")
        self.ax_xy.set_ylabel("y [m]")
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.grid(True, color="0.9", linewidth=0.4)
        self.ax_xy.legend(loc="upper right")
        if plotted:
            xy_points = np.vstack(plotted)[:, :2]
            min_values = np.min(xy_points, axis=0)
            max_values = np.max(xy_points, axis=0)
            center = (min_values + max_values) / 2.0
            half_range = np.max(max_values - min_values) / 2.0
            half_range = max(half_range * 1.05, 1.0)
            self.ax_xy.set_xlim(center[0] - half_range, center[0] + half_range)
            self.ax_xy.set_ylim(center[1] - half_range, center[1] + half_range)

        self.fig.tight_layout()
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
        self.ax_xy = None
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

    return {
        "lidar_names": lidar_names,
        "before_clouds": before_clouds,
        "after_clouds": after_clouds,
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

    plot_keys = [
        ("before_plot_png", "before_calibration.png"),
        ("after_plot_png", "after_calibration.png"),
    ]
    if target["label"] == "6dof":
        plot_keys.extend([
            ("before_plot_xz_png", "before_calibration_xz.png"),
            ("after_plot_xz_png", "after_calibration_xz.png"),
            ("before_plot_yz_png", "before_calibration_yz.png"),
            ("after_plot_yz_png", "after_calibration_yz.png"),
        ])

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
    limits_xy = cloud_axis_limits([before_clouds, after_clouds], (0, 1))
    limits_xz = cloud_axis_limits([before_clouds, after_clouds], (0, 2))
    limits_yz = cloud_axis_limits([before_clouds, after_clouds], (1, 2))
    plot_specs = [
        ("Before 6DOF Calibration - XY Projection", before_clouds, outputs[0], (0, 1), ("x", "y"), limits_xy),
        ("After 6DOF Calibration - XY Projection", after_clouds, outputs[1], (0, 1), ("x", "y"), limits_xy),
        ("Before 6DOF Calibration - XZ Projection", before_clouds, outputs[2], (0, 2), ("x", "z"), limits_xz),
        ("After 6DOF Calibration - XZ Projection", after_clouds, outputs[3], (0, 2), ("x", "z"), limits_xz),
        ("Before 6DOF Calibration - YZ Projection", before_clouds, outputs[4], (1, 2), ("y", "z"), limits_yz),
        ("After 6DOF Calibration - YZ Projection", after_clouds, outputs[5], (1, 2), ("y", "z"), limits_yz),
    ]
    for title, clouds, path, axes, labels, limits in plot_specs:
        plot_clouds_projection(
            title,
            clouds,
            path,
            axes,
            labels,
            axis_limits=limits,
            max_points=max_points,
            show_default_grid=show_default_grid,
            show_result_plot=show_result_plot,
            keep_result_plot_open=keep_result_plot_open,
        )

    if show_result_plot and keep_result_plot_open and is_interactive_backend():
        plt.show(block=True)

    return collect_graph_outputs(target, calib_config)


def print_graph_summary(graph_outputs):
    print("[graph_plot] graph outputs:")
    for output in graph_outputs:
        status = "ok" if output["exists"] else "missing"
        print(f"[graph_plot] {status}: {output['path']}")
