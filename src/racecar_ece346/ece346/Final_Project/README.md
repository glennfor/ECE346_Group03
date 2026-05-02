# ECE346 Final Project Safety Filter

This package implements a Backup-CBF safety filter for manual driving. It keeps
human commands whenever they satisfy the safety constraint and falls back to a
brake-and-recenter backup policy when a lane departure or obstacle collision is
imminent.

## Architecture Note

The project proposal describes three logical safety-filter roles:
`backup_planner`, `safety_monitor`, and `safety_filter_qp`. This implementation
keeps those roles separated in the Python modules (`backup_policy`, `barrier`,
`margins`, and `qp`) but deploys them inside one ROS node,
`safety_filter_node.py`, to avoid custom inter-node messages and stale timing
between hot-path safety computations.

The same observability is still exposed through ROS topics:
`/safety/backup_traj` corresponds to the fallback planner,
`/safety/value`, `/safety/grad`, and `/safety/binding_constraint` correspond to
the monitor, and `/safety/override`, `/safety/u_human`, `/safety/u_filtered`,
and the filtered command topic correspond to the intervention layer.

## Launch

Simulator:

```bash
ros2 launch racecar_ece346 safety_filter_sim_launch.py
```

Publish simulated human commands to `/human_control` as `racecar_msgs/ServoMsg`.
The filter publishes the safe command to `/control`, which the simulator already
subscribes to.

For PS5/PS4 controller testing on Linux, run the joystick driver in another
terminal:

```bash
ros2 run joy joy_node
```

The launch file starts `joy_to_servo_node.py`, which maps `/joy` to
`/human_control`. The default config requires holding joystick button index `4`
as a deadman before stick input is forwarded. If your controller maps buttons or
axes differently, run `ros2 topic echo /joy` and tune `deadman_button`,
`throttle_axis`, `steer_axis`, `invert_throttle`, and `invert_steer` in
`config/safety_filter_sim.yaml`.

The Docker container must see Linux input devices for `joy_node` to work. After
starting the container, verify:

```bash
ls /dev/input
ros2 topic echo /joy
```

If `/dev/input` is missing in the container, restart with `./start.sh down` and
`./start.sh`; `docker-compose.yml` passes `/dev/input` through for controller
access.

With no controller input, `/control` may show the safety filter publishing max
brake (`throttle: -5.0`) because the human command is stale. That is expected
and should not command forward motion. Use `/slam_pose` to confirm whether the
ego car is moving; `traffic_simulation_node` can also create moving obstacle
cars that are independent of ego control.

Real truck:

```bash
ros2 launch racecar_ece346 safety_filter_real_launch.py
```

The real-truck config listens to `/teleop` and publishes filtered
`AckermannDriveStamped` commands to `/drive`, matching the Lab 5 control-gate
path.

## Debug Topics

- `/safety/value`: implicit barrier value.
- `/safety/grad`: finite-difference barrier gradient.
- `/safety/override`: true when the filter is actively modifying the command.
- `/safety/binding_constraint`: active margin source: lane, obstacle, traffic, or kinematic.
- `/safety/backup_traj`: backup rollout as a `nav_msgs/Path`.
- `/safety/u_human` and `/safety/u_filtered`: controls in `[accel, steering_rate]`.
- `/safety/markers`: RViz marker line and text status.

## Tuning

All high-value knobs are in `config/safety_filter_sim.yaml` and
`config/safety_filter_real.yaml`.

If the truck is too conservative, reduce `r_safe_lane`, `r_safe_obs`, or
`horizon_H`. If it cuts too close, increase those margins or raise
`lambda_cbf`. If override flickers, increase `hysteresis_cycles`.

Measure real stopping time before demo day and set `horizon_H` to cover the
full stop with margin.

