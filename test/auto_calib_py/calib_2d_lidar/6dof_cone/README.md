# Cone Circle 6DOF Calibration

This folder estimates each stationary 2D LiDAR's `roll`, `pitch`, and `z`
using upright cones with known base radius and height.

## Files

- `lidar_config.yaml`: LaserScan topics and initial LiDAR poses
- `calib_config.yaml`: cone dimensions, range limits, extraction, and output options
- `6dof_cone.py`: ROS 2 collection and calibration program

## Method

```text
collect stationary LaserScan data
-> keep points inside range_filter
-> split consecutive points into object clusters
-> RANSAC circle fit for each cluster
-> circle radius r gives cone height h = H * (1 - r / R)
-> construct (circle center x, circle center y, h)
-> fit z = a*x + b*y + c through all cone centers
-> obtain roll/pitch from the plane normal and LiDAR z from c
```

At least three cones are required. Their centers must not be collinear. Place
the cones in different directions around the robot and keep the robot still
during collection.

## Configuration

Set each LaserScan topic in `lidar_config.yaml`. Keys under `topics` and
`lidars` must match. The calibrated output updates only `z`, `roll`, and
`pitch`; `x`, `y`, and `yaw` remain unchanged.

Enter the physical cone dimensions in `calib_config.yaml`:

```yaml
cone:
  radius_m: 0.14
  height_m: 0.45
```

Limit detection to the area containing the cones:

```yaml
range_filter:
  min_m: 0.5
  max_m: 3.0
```

The most useful detection tuning parameters are:

- `clustering.gap_m`: distance that separates adjacent scan objects
- `clustering.max_width_m`: rejects clusters wider than a cone
- `circle.inlier_threshold_m`: maximum radial error of a circle inlier
- `circle.min_inliers`: minimum scan points required for a circle
- `quality.max_circle_rmse_m`: rejects poorly fitted circles
- `quality.max_plane_rmse_m`: rejects an inconsistent final plane

## Run

From this folder:

```bash
python3 6dof_cone.py
```

Custom config paths can also be supplied:

```bash
python3 6dof_cone.py \
  --lidar-config /path/to/lidar_config.yaml \
  --calib-config /path/to/calib_config.yaml
```

## Output

Default files are written below `output/`:

- `calibrated_output.yaml`: detected circles, inferred heights, plane, RMSE, and results
- `calibrated_config.yaml`: input LiDAR config with calibrated roll/pitch/z
- `cone_calibration.png`: detected circles and fitted center-plane height map
- `cone_calibration_3d.html`: interactive cones, detected sections, scan points, and plane

Angles in `calibrated_config.yaml` are radians.

The plot window is enabled by default:

```yaml
plot:
  enabled: true
  show: true
  show_2d: true
  output_png: cone_calibration.png
  interactive_3d: true
  output_html: cone_calibration_3d.html
  max_points: 5000
  dpi: 150
```

`show: true` opens the interactive 3D HTML in the default browser. Drag to
rotate, use the wheel to zoom, and click legend entries to hide individual
cones or the scan plane. `show_2d: true` keeps the Matplotlib XY/XZ/YZ window
open and the program continues only after that window is closed. Set both to
`false` on a headless computer; PNG and HTML files are still written when
`enabled: true`.

## Geometric limitation

A tilted plane intersects an ideal cone in a conic section rather than an
exact circle. This implementation follows the circular-section approximation,
so always inspect both the circle RMSE and plane RMSE in the result.
