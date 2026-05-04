# CBF Safety Filter (heuristic)

Three-node **heuristic** safety layer for the ECE346 mini-truck: lane and obstacle margins define a scalar `h`, a lane-centering backup publishes `ω`, and the filter **caps throttle** and **blends steering** toward that backup when `h` is low. This is **not** a discrete-time Control Barrier Function QP (no OSQP, no `gᵀu ≥ c` constraint).

## Quick start

### Simulation

```bash
ros2 launch racecar_ece346 cbf_sim_launch.py
```

### Real truck

```bash
ros2 launch racecar_ece346 cbf_real_launch.py enable_qp:=false   # verify topics first
ros2 launch racecar_ece346 cbf_real_launch.py                     # then enable
```

### Toggle intervention at runtime

```bash
ros2 param set /safety_filter_qp_node enable_qp false   # passthrough (human → /control)
ros2 param set /safety_filter_qp_node enable_qp true    # filter active
```

`/safety/value` keeps updating when intervention is off; only steering/throttle shaping changes.

## Architecture

```text
/slam_pose ──► backup_planner_node ──► /safety/backup_u0   [a, ω backup]
         ╲
          ╲──► safety_monitor_node ──► /safety/value       h = min margin (+ human lookahead)
          ╲       ▲ /human_control
           ╲      ▲ /Obstacles/*, map
            ╲
/human_control ──► safety_filter_qp_node ◄── /safety/value, /safety/backup_u0
/slam_pose     ──►                         ◄── /control (steer angle feedback)
                  └──► /control
```

**Important:** `safety_monitor_node` and `backup_planner_node` subscribe to **`/control`** (`filtered_control_topic`) so **steering angle `δ`** matches the command sent to the truck. That keeps human-command lookahead and the backup law consistent with the bicycle model (fixes wrong `ω` and bogus `h` when `δ` was assumed zero).

## Node behavior

### `backup_planner_node`

- Publishes `[a_min, ω]` every tick: strong brake request in `a`, and Stanley-style steering toward the local lane center (`K_e`, `K_p`, `v_eps`).
- Optional first-order smoothing: `backup_omega_lpf_tau_s`.

### `safety_monitor_node`

- `h` = minimum of current lane / static / dynamic margins and the same margins along a short rollout (`horizon_H` steps) using the **human** `ServoMsg` (converted to `[a, ω]` with correct `δ`).
- If the map is not ready, `LaneContext.is_fallback` is true → publishes `h = fallback_h_safe` so the filter does not react to dummy geometry.
- `lane_allow_lane_change` and `route_hysteresis_rad` reduce centerline jumps from Lanelet routing.

### `safety_filter_qp_node` (name kept for compatibility)

- `h ≥ throttle_cap_margin`: passthrough.
- `0 ≤ h < throttle_cap_margin`: **cap zone** — no forward acceleration (`a_out ≤ 0`), steering blends human `ω` toward backup `ω`; optional EMA: `steer_blend_lpf_tau_s`.
- `h < 0`: same throttle cap, backup `ω` only.

## Topics

| Topic | Type | Notes |
|-------|------|--------|
| `/slam_pose` | `nav_msgs/Odometry` | Pose and forward speed |
| `/human_control` | `ServoMsg` | Joystick command |
| `/control` | `ServoMsg` | Filter output; **also** plant steer feedback for monitor/backup |
| `/safety/value` | `std_msgs/Float32` | `h` |
| `/safety/debug_margins` | `Float64MultiArray` | `[lane, obstacle, traffic, lookahead_min]` |
| `/safety/binding_constraint` | `String` | Which term is smallest (`lane`, `obstacle`, `traffic`, `lookahead`, `fallback_map`) |
| `/safety/backup_u0` | `Float64MultiArray` | `[a, ω]` backup |
| `/safety/override` | `Bool` | True if filter modified command |

## Parameters (see `CbfParams` in `cbf_filter/config.py`)

| Parameter | Role |
|-----------|------|
| `throttle_cap_margin` | Below this (meters of margin), throttle capped and steering blended |
| `steer_blend_lpf_tau_s` | Low-pass on blended `ω` during intervention (`0` = off) |
| `K_e`, `K_p`, `v_eps` | Backup lateral law |
| `backup_omega_lpf_tau_s` | Low-pass on published backup `ω` |
| `lane_allow_lane_change` | Passed to Lanelet path/width queries (`false` = stabler centerline) |
| `route_hysteresis_rad` | Stick to previous route if still within this heading of optimal |
| `fallback_h_safe` | `h` when map fallback strip is active |
| `horizon_H`, `dt` | Lookahead length for monitor rollout |

## Tuning

- **Too much sway / chatter:** lower `K_p`, increase `backup_omega_lpf_tau_s` / `steer_blend_lpf_tau_s`, or increase `route_hysteresis_rad`.
- **Cannot accelerate after intervention:** `h` still below `throttle_cap_margin` — check margins, map, and `/safety/binding_constraint`.
- **Tighter lane keeping:** slightly increase `throttle_cap_margin` or `r_safe_lane` (monitor).

## File layout

```text
cbf_filter/
├── cbf_filter/
│   ├── config.py
│   ├── dynamics.py
│   ├── margins.py
│   ├── lane_context.py
│   ├── obstacle_memory.py
│   └── ros_utils.py
├── scripts/
│   ├── cbf_backup_planner_node.py
│   ├── cbf_safety_monitor_node.py
│   └── cbf_safety_filter_qp_node.py
├── config/
│   ├── cbf_filter_sim.yaml
│   └── cbf_filter_real.yaml
├── launch/
│   ├── cbf_sim_launch.py
│   └── cbf_real_launch.py
└── README.md
```

## Relation to formal CBF

A backup-CBF + QP formulation (implicit barrier, gradient, box QP) is documented in the course spec and implemented in `ece346/Final_Project/safety_filter/`. This package intentionally uses a **lighter heuristic** suitable for debugging and demos; tune parameters above before expecting certificate-style guarantees.
