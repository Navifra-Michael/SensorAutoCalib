# LiDAR NDT Calibration Guide

This tool calibrates multiple 2D LiDARs from live ROS 2 `LaserScan` topics.
It keeps one reference LiDAR fixed, then optimizes the other LiDARs with a
sequential fused NDT map.

The current matching flow is:

```text
collect LaserScan data
-> convert scans to 2D points
-> apply initial lidar_config poses
-> fix reference_lidar
-> optimize next LiDAR x/y/yaw against fused NDT map
-> add calibrated LiDAR points into fused map
-> repeat for the next LiDAR
-> save YAML results and PNG plots
```

The configuration is split into two files:

- `lidar_config.yaml`: LiDAR topics and initial mounting poses
- `calib_config.yaml`: runtime, output, NDT, search, and plot options

## lidar_config.yaml

```yaml
topics:
  lidar_1: /safety_left_scan_fullframe
  lidar_2: /safety_front_scan_fullframe
  lidar_3: /safety_right_scan_fullframe

lidars:
  lidar_1:
    x: 0.43
    y: 0.58
    roll: 3.141592653589793
    yaw: 1.5708
```

### `topics`

Maps each LiDAR name to a ROS `LaserScan` topic.

The key must match the corresponding key under `lidars`.

### `lidars`

Defines each LiDAR's initial mounting pose.

- `x`: initial x position in robot coordinates, meters
- `y`: initial y position in robot coordinates, meters
- `roll`: fixed roll angle in radians
- `yaw`: initial yaw angle in robot coordinates, radians

Only `x`, `y`, and `yaw` are optimized. `roll` is applied but not optimized.

For an upside-down LiDAR, use either:

```yaml
roll: 3.141592653589793
```

or:

```yaml
roll: -3.141592653589793
```

The transform order is:

```text
local LaserScan point
-> roll correction
-> yaw rotation
-> x/y translation in robot coordinates
```

The order of entries under `lidars` is also the sequential calibration order.
Place LiDARs in overlap order:

```text
reference -> adjacent overlapping LiDAR -> next overlapping LiDAR
```

For example, if the rear LiDAR overlaps more with the side LiDAR than the
front LiDAR, use:

```text
front -> side -> rear
```

## calib_config.yaml

```yaml
reference_lidar: lidar_1
collect_duration_sec: 5.0
output_dir: output
output_yaml: calibrated_output.yaml
calibrated_config_yaml: calibrated_config.yaml
before_plot_png: before_calibration.png
after_plot_png: after_calibration.png
```

### Runtime Parameters

### `reference_lidar`

The fixed LiDAR used to initialize the fused map.

This name must exist in `lidar_config.yaml`.

### `collect_duration_sec`

How long to collect live `LaserScan` data before running calibration.

The robot should stay still while collecting. The code currently accumulates
scans without odometry compensation, so moving during collection can smear the
point cloud.

### Output Parameters

### `output_dir`

Directory where generated files are saved.

### `output_yaml`

Calibration result file. It contains optimized poses, scores, success flags,
calibration order, and timing.

Default output:

```text
output/calibrated_output.yaml
```

### `calibrated_config_yaml`

LiDAR config with calibrated `x`, `y`, `roll`, and `yaw` applied.

This file keeps only:

```yaml
topics:
lidars:
```

Default output:

```text
output/calibrated_config.yaml
```

### `before_plot_png` / `after_plot_png`

Saved plot images.

- `before_calibration.png`: initial pose point cloud
- `after_calibration.png`: calibrated point cloud

The after plot can show both calibrated and previous LiDAR arrows for visual
comparison.

## Plot Parameters

```yaml
plot:
  show_lidar_arrows: true
  show_robot_frame: true
  show_ndt_grid: false
  show_default_grid: true
  show_live_ndt: true
  live_update_interval_sec: 0.2
  live_max_points: 4000
  keep_live_ndt_open: false
  lidar_label_font_size: 10
```

### `show_lidar_arrows`

Shows LiDAR pose arrows on the saved plots.

### `show_robot_frame`

Shows the robot coordinate frame.

- faded gray origin marker: robot origin
- faded x/y arrows: robot coordinate axes

### `show_ndt_grid`

Shows the NDT grid using `ndt.resolution` spacing.

### `show_default_grid`

Shows the normal matplotlib background grid.

### `show_live_ndt`

Shows a live plot while NDT grid search is running.

In the live view:

- gray points: current fused target map
- red points: current candidate LiDAR cloud
- black arrow: current best pose
- title: LiDAR name, stage number, current best score

Live plotting slows calibration. Turn it off for faster runs:

```yaml
show_live_ndt: false
```

### `live_update_interval_sec`

Minimum time between live plot refreshes.

Lower values update more often but slow optimization.

### `live_max_points`

Maximum points shown in the live plot for each cloud.

### `keep_live_ndt_open`

If `false`, the live plot closes automatically after each LiDAR optimization.

### `lidar_label_font_size`

Font size for LiDAR labels on arrows.

Labels are drawn near the arrow tip.

## NDT Parameters

```yaml
ndt:
  resolution: 0.5
  min_points_per_cell: 5
  downsample_voxel: 0.05
```

### `resolution`

NDT grid cell size in meters.

Smaller values use finer local structure, but need enough points per cell.
Larger values are coarser and often more stable when overlap is small.

### `min_points_per_cell`

Minimum number of points required for an NDT cell to be valid.

If this is too high, the NDT grid may become empty. If it is too low, noisy
cells may be used.

### `downsample_voxel`

Voxel size in meters for point downsampling before matching.

This is not the NDT grid size. For example:

```yaml
resolution: 0.5
downsample_voxel: 0.05
```

means:

```text
NDT grid: 0.5 m cells
downsample voxel: 0.05 m point filtering
```

Smaller values keep more points and may improve detail, but increase runtime.

## Search Parameters

```yaml
search:
  stages:
    - range_x: 0.30
      range_y: 0.30
      range_yaw_deg: 30.0
      step_x: 0.05
      step_y: 0.05
      step_yaw_deg: 2.0
```

Search runs in stages. Each stage searches around the best pose from the
previous stage.

Typical use:

```text
Stage 1: wide and coarse
Stage 2: narrower and more precise
Stage 3: very narrow and fine
```

### `range_x` / `range_y`

Search range around the current x/y estimate, in meters.

Example:

```text
current x = 0.40
range_x = 0.30
searches 0.10 to 0.70
```

### `range_yaw_deg`

Search range around the current yaw estimate, in degrees.

### `step_x` / `step_y`

x/y search interval in meters.

Smaller values are more precise but slower.

### `step_yaw_deg`

yaw search interval in degrees.

Smaller values are more precise but slower.

## Score

Each candidate pose gets an NDT score. Lower is better.

For each transformed LiDAR point:

- If it lands in an existing NDT cell, the code measures how well it fits that
  cell's mean and covariance.
- If it lands outside the NDT map, it counts as a miss and receives a penalty.

The final score is roughly:

```text
average NDT distance + miss_ratio * 10.0
```

If no transformed points overlap with valid NDT cells, the score is set very
high.

## Case Count

For each search stage, the number of tested candidates is:

```text
x candidates * y candidates * yaw candidates
```

Example:

```yaml
range_x: 0.30
step_x: 0.05
```

creates about 13 x candidates:

```text
center - 0.30 ... center ... center + 0.30
```

The total cases grow quickly because the search is 3D: `x`, `y`, and `yaw`.

## Output Files

Default outputs:

```text
output/calibrated_output.yaml
output/calibrated_config.yaml
output/before_calibration.png
output/after_calibration.png
```

`calibrated_output.yaml` includes:

- `reference_lidar`
- `lidar_count`
- per-LiDAR optimized pose
- score
- success flag
- optimization time
- total timing
- calibration order

`calibrated_config.yaml` is the next LiDAR config you can reuse.

## Tuning Tips

- If the result is far from correct, increase first-stage `range_x`,
  `range_y`, or `range_yaw_deg`.
- If the result is close but not accurate enough, reduce final-stage `step_x`,
  `step_y`, or `step_yaw_deg`.
- If calibration is too slow, increase step sizes or remove a stage.
- If the NDT grid is empty, increase `collect_duration_sec`, increase
  `resolution`, or reduce `min_points_per_cell`.
- If overlap is small, use a coarser `resolution`, lower
  `min_points_per_cell`, and order LiDARs by overlap.
- Keep the robot still during data collection.
- Use `show_live_ndt: false` for faster unattended runs.
