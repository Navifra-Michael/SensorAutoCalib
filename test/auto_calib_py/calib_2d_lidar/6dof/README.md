# 2D LiDAR + Odom 6DOF Calibration

This folder contains the 6DOF calibration tool for 2D `LaserScan` LiDARs.

The node subscribes to:

- multiple `sensor_msgs/msg/LaserScan` topics
- one `nav_msgs/msg/Odometry` topic

It uses odom motion to accumulate each 2D scan into a session frame, then
optimizes each LiDAR's approximate extrinsic:

```text
x, y, z, roll, pitch, yaw
```

## Files

- `lidar_config.yaml`: odom/LiDAR topics and initial 6DOF poses
- `calib_config.yaml`: collection time, output paths, NDT, search
- `lidar_ndt_calib_6dof.py`: ROS 2 calibration node

## Basic Flow

```text
subscribe odom
-> collect LaserScan data for each LiDAR
-> attach latest odom-relative transform to each scan
-> convert each scan to local 3D points with z=0
-> apply candidate LiDAR 6DOF extrinsic
-> apply odom-relative robot motion into the session frame
-> match against the current fused 3D NDT map
-> add calibrated LiDAR cloud into fused map
-> repeat for the next LiDAR
-> save calibrated YAML and XY/XZ/YZ projection plots
```

## Config

`lidar_config.yaml`:

```yaml
topics:
  odom: /odom
  lidar_1: /navi_robot_1/front_scan
  lidar_2: /navi_robot_1/rear_scan
```

### `topics.odom`

Odometry topic used to accumulate scans into a common session frame.

`topics.<lidar_name>` entries are the 2D LiDAR `LaserScan` topics.

`calib_config.yaml`:

```yaml
reference_lidar: lidar_1
collect_duration_sec: 5.0
require_odom: true
match_mode: exact_precomputed

plot:
  progress_3d:
    enabled: true
    max_points: 3000
    update_interval_sec: 1.0
```

### `require_odom`

If `true`, scans are ignored until odom is received.

If `false`, scans collected before odom use identity motion.

### `match_mode`

Selects how each candidate pose is scored.

- `exact_precomputed`: keeps the correct `odom * extrinsic * scan` transform
  order, but precomputes stacked scan/odom arrays to reduce Python loop overhead.
  This is the recommended mode for roll, pitch, and z calibration.
- `exact_odom_chunks`: recomputes every scan chunk with its odom transform for
  every candidate pose. This is slower, but it keeps the odom/extrinsic math
  exact.
- `fast_accumulated_cloud`: first accumulates the collected scans into one cloud
  per LiDAR, then moves that cloud during matching. This is much faster and works
  like matching one captured "picture" against another, but it is an
  approximation and is not recommended for final roll/pitch calibration.

### `plot.progress_3d`

Shows a live 3D plot during grid search. The blue points are the current
target/fused cloud, and the orange points are the moving LiDAR cloud transformed
by the current best pose.

- `enabled`: turn the live 3D plot on or off
- `max_points`: maximum points drawn for each cloud to keep updates responsive
- `update_interval_sec`: minimum time between plot redraws

## Important Notes

- This is still an observability-limited problem. 2D LiDAR scan points start
  as a flat local plane, so useful 6DOF estimation needs robot motion and 3D
  structure in the scene.
- If odom is mostly planar and the scene has little height variation, `z`,
  `roll`, and `pitch` can be weakly constrained.
- Keep search ranges small. Full 6D grid search grows very quickly.
- LiDAR order under `lidars` is the sequential calibration order.

## Run

From this folder:

```bash
python3 lidar_ndt_calib_6dof.py
```

## Output

Default outputs:

```text
output/calibrated_output.yaml
output/calibrated_config.yaml
output/before_calibration.png
output/after_calibration.png
output/before_calibration_xz.png
output/after_calibration_xz.png
output/before_calibration_yz.png
output/after_calibration_yz.png
```

The default `before_calibration.png` and `after_calibration.png` files are XY
projections. Pitch and roll can be hard to see there, so the tool also saves XZ
and YZ projections. Use XZ to check pitch-related tilt and YZ to check
roll-related tilt.
