# 2D LiDAR NDT Calibration

Use `ndt_main.py` to run either the 3DOF or 6DOF calibrator from one entry
point.

```bash
cd /home/mic/SensorAutoCalib/test/auto_calib_py/calib_2d_lidar

# 3DOF calibration
python3 ndt_main.py 3dof

# 6DOF calibration
python3 ndt_main.py 6dof
```

Short aliases are also supported:

```bash
python3 ndt_main.py 3
python3 ndt_main.py 6
```

The selected calibrator runs from its own folder, so it uses that folder's
`lidar_config.yaml`, `calib_config.yaml`, and `output/` directory.

## Structure

- `ndt_main.py`: reads the selected mode/config and prints the final summary.
- `data_association.py`: runs the selected internal NDT calibrator and loads the
  result YAML/config back into main.
- `graph_plot.py`: manages expected graph output paths and reports whether the
  saved graph files exist.
- `3dof/lidar_ndt_calib.py`, `6dof/lidar_ndt_calib_6dof.py`: current internal
  NDT calculation engines. These still contain ROS collection and detailed NDT
  logic, and can be reduced further as the engine interfaces are extracted.

To check what will run without starting ROS subscriptions:

```bash
python3 ndt_main.py 6dof --dry-run
```

Common runtime options:

```bash
# Force matplotlib backend for both modes
python3 ndt_main.py 3dof --backend TkAgg

# Disable live/progress plots without editing calib_config.yaml
python3 ndt_main.py 6dof --no-live-plot

# Enable live/progress plots without editing calib_config.yaml
python3 ndt_main.py 6dof --live-plot

# 3DOF only: keep or close live plot after calibration
python3 ndt_main.py 3dof --keep-live-plot
python3 ndt_main.py 3dof --close-live-plot
```
