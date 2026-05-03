# CBF Safety Filter

Control Barrier Function (CBF) safety filter for the ECE346 mini-truck. Three explicit nodes enforce that a joystick-driven truck can always reach a safe state by keeping it inside the **backup-safe set**: the region from which the backup policy keeps the truck safe for the full lookahead horizon.

## Quick Start

### Simulation
```bash
ros2 launch racecar_ece346 cbf_sim_launch.py
```

### Real truck
```bash
ros2 launch racecar_ece346 cbf_real_launch.py enable_qp:=false   # verify topics first
ros2 launch racecar_ece346 cbf_real_launch.py                     # then enable
```

### Disable/re-enable intervention at runtime (no restart)
```bash
ros2 param set /safety_filter_qp_node enable_qp false   # passthrough
ros2 param set /safety_filter_qp_node enable_qp true    # CBF active
```
Safety value `/safety/value` keeps updating in both modes — only intervention is off.

---

## Architecture

```
/slam_pose ──► [backup_planner_node] ──► /safety/backup_traj
                                      └► /safety/backup_u0

/slam_pose ──► [safety_monitor_node] ◄── /safety/backup_traj
/Obstacles/Static  ──►     │         ◄── /Obstacles/Dynamic
                           ▼
                  /safety/value   (h_imp scalar)
                  /safety/grad    (∇h_imp, 5-vector)
                  /safety/binding_constraint

/human_control ──► [safety_filter_qp_node] ◄── /safety/value
/slam_pose     ──►          │              ◄── /safety/grad
                            │              ◄── /safety/backup_u0
                            ▼
                       /control
                  /safety/override
```

### Node 1 — `backup_planner_node`
Rolls out the backup policy (max braking + lane-centerline P-controller) from the current pose for `horizon_H` steps. Publishes the full trajectory so the monitor can evaluate it without re-running the rollout.

### Node 2 — `safety_monitor_node`
Receives the backup trajectory and computes the **implicit barrier value** `h_imp = min_{k=0..H} margin(x_k)`, where `margin` is the minimum signed clearance across lane boundaries, static/dynamic obstacles, and kinematic limits. Also computes `∇h_imp` by finite difference (5 extra rollouts with perturbed initial states). Publishes `h_imp` and `∇h_imp`.

### Node 3 — `safety_filter_qp_node`
At each cycle reads the human command `u_h`, then either:
- **Passes through** `u_h` unchanged if `h_imp ≥ 0` and hysteresis has cleared, or
- **Solves a CBF-QP** to find the nearest safe command `u*`:

```
min  ||u - u_h||²_R
s.t. g^T u ≥ c          (CBF constraint)
     u in [u_min, u_max]
```

where `g = B^T ∇h_imp` and `c = -λ·h_imp - ∇h_imp·(f(x,u_h)-x) + g·u_h`.

If the QP is infeasible (very rare; truck already in unsafe set), falls back to `backup_u0`.

---

## Nodes and Topics

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/slam_pose` | `Odometry` | in | Truck pose + velocity |
| `/human_control` | `ServoMsg` | in | Raw joystick command |
| `/Obstacles/Static` | `ObstacleArray` | in | Static obstacles from AprilTags |
| `/Obstacles/Dynamic` | `ObstacleArray` | in | Moving obstacles |
| `/safety/backup_traj` | `Float64MultiArray` | internal | (H+1)×5 state trajectory |
| `/safety/backup_u0` | `Float64MultiArray` | internal | First backup action [a, ω] |
| `/safety/backup_path` | `Path` | out (viz) | Rviz path visualization |
| `/safety/value` | `Float64` | out | h_imp scalar |
| `/safety/grad` | `Float64MultiArray` | out | ∇h_imp, 5-vector |
| `/safety/binding_constraint` | `String` | out | Which constraint is binding |
| `/safety/margins_along_traj` | `Float64MultiArray` | out | Per-step margins (debugging) |
| `/safety/override` | `Bool` | out | True when CBF is intervening |
| `/safety/u_human` | `Float64MultiArray` | out | [a, ω] as received |
| `/safety/u_filtered` | `Float64MultiArray` | out | [a, ω] as sent |
| `/control` | `ServoMsg` | out | Final command to truck |

---

## Launch Each Node Separately

Useful for debugging one piece at a time:

```bash
# Terminal 1 — backup planner only
ros2 run racecar_ece346 cbf_backup_planner_node.py \
  --ros-args --params-file <path>/cbf_filter_sim.yaml

# Terminal 2 — safety monitor (needs backup_traj topic from T1)
ros2 run racecar_ece346 cbf_safety_monitor_node.py \
  --ros-args --params-file <path>/cbf_filter_sim.yaml

# Terminal 3 — QP filter (needs value + grad from T2, human_control)
ros2 run racecar_ece346 cbf_safety_filter_qp_node.py \
  --ros-args --params-file <path>/cbf_filter_sim.yaml
```

---

## Parameters

### Shared across all three nodes

| Parameter | Default (sim) | Default (real) | Notes |
|---|---|---|---|
| `wheelbase_m` | 0.257 | 0.257 | Must match physical truck |
| `truck_radius_m` | 0.13 | 0.13 | Half-width (not 0.274 — that's wrong) |
| `truck_length_m` | 0.40 | 0.40 | |
| `v_max` | 5.0 | 2.0 | Real truck uses lower limit |
| `a_min/a_max` | -5.0 / 5.0 | -3.0 / 1.5 | |
| `horizon_H` | 30 | 30 | Steps at dt=0.05 s → 1.5 s lookahead |
| `dt` | 0.05 | 0.05 | Must match `1/control_rate_hz` |

### Safety margins (`safety_monitor_node`)

| Parameter | Sim | Real | Meaning |
|---|---|---|---|
| `r_safe_lane` | 0.05 | 0.08 | Extra buffer beyond geometric clearance from lane boundary |
| `r_safe_obs` | 0.05 | 0.08 | Extra buffer from static obstacles |
| `r_safe_traf` | 0.10 | 0.12 | Extra buffer from traffic (dynamic) |
| `r_safe_kin` | 0.02 | 0.02 | Buffer inside speed/steering limits |

### CBF gains (`safety_filter_qp_node`)

| Parameter | Default | Meaning |
|---|---|---|
| `lambda_cbf` | 0.4 | CBF decay rate — larger = more aggressive intervention |
| `qp_R_accel` | 1.0 | Cost weight for changing accel vs. human command |
| `qp_R_omega` | 1.0 | Cost weight for changing steering rate |
| `hysteresis_cycles` | 5 (sim) / 8 (real) | Override holds this many cycles after h_imp turns positive |

### Backup policy gains (`backup_planner_node`)

| Parameter | Default | Meaning |
|---|---|---|
| `K_e` | 1.0 | Cross-track error gain |
| `K_p` | 5.0 | Steering-angle P gain |
| `v_eps` | 0.3 | Division guard at near-zero speed |

---

## Tuning Guide

**CBF intervenes too often / annoying:**
- Decrease `lambda_cbf` (try 0.2–0.3)
- Decrease `r_safe_lane`, `r_safe_obs` (smaller safety margins)
- Decrease `horizon_H` (shorter lookahead)

**CBF doesn't intervene soon enough / truck gets too close:**
- Increase `lambda_cbf` (try 0.6–0.8)
- Increase safety margins
- Increase `horizon_H`

**Override chatters (flicker on/off rapidly):**
- Increase `hysteresis_cycles` (try 10–15)

**Backup policy doesn't reach lane center:**
- Increase `K_e` and/or `K_p`
- Decrease `v_max` so the backup stops faster

**Hardware: set `horizon_H` correctly**
1. Drive at max speed and measure braking distance → stopping time `T_stop`
2. Set `horizon_H = ceil(1.5 * T_stop / 0.05)` in `cbf_filter_real.yaml`
3. This ensures the horizon covers the full braking maneuver

---

## Debug Checklist (real truck)

```bash
# Verify localization is live
ros2 topic hz /slam_pose

# Safety value should be positive when truck is safely in lane
ros2 topic echo /safety/value

# Which safety constraint is currently binding
ros2 topic echo /safety/binding_constraint

# Watch whether CBF is overriding
ros2 topic echo /safety/override

# Compare human command vs filtered command
ros2 topic echo /safety/u_human
ros2 topic echo /safety/u_filtered

# Backup trajectory in Rviz
# Add topic /safety/backup_path (nav_msgs/Path)
```

---

## File Layout

```
cbf_filter/
├── cbf_filter/           # Library (importable from nodes)
│   ├── config.py         # CbfParams dataclass + ROS parameter loading
│   ├── dynamics.py       # Bicycle4D step, rollout, control Jacobian
│   ├── backup_policy.py  # brake_and_recenter(x, lane, p)
│   ├── margins.py        # margin_lane, margin_obstacle, margin_total
│   ├── obstacle_memory.py # TTL buffer with age-inflated radii
│   ├── lane_context.py   # LaneContext, LaneletContextBuilder
│   ├── qp.py             # Closed-form box-constrained CBF-QP solver
│   └── ros_utils.py      # Message ↔ numpy conversions
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
└── README.md             # this file
```

---

## Theory in One Paragraph

The backup-CBF defines a safe set `S = {x : h_imp(x) ≥ 0}` where `h_imp(x)` is the minimum safety margin along the rollout of the backup policy starting from `x`. If the truck is in `S`, the backup policy is a certificate that it can always be steered to safety. The CBF-QP keeps `h_imp` from decreasing too fast: it enforces `dh_imp/dt ≥ -λ·h_imp`, which (by Nagumo's theorem) guarantees forward invariance of `S`. The QP only modifies the human command when necessary — the truck drives exactly as commanded whenever it is not approaching the boundary.
