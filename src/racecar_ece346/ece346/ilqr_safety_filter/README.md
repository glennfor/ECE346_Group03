# ILQR Safety Filter

Predictive ADAS safety filter for the ECE 346 1/10-scale RC truck (Princeton, Spring 2026).

Human joystick commands are forwarded to the servo unchanged when they are certified safe. When they would lead to a collision or lane departure, the filter overrides with the first step of a precomputed ILQR fallback trajectory just long enough to recover, then hands control back.

This replaces the earlier Backup-CBF filter (`ece346/Final_Project/safety_filter/`) which did not work reliably.

---

## Architecture

Three logical roles run inside a single ROS 2 node (`ilqr_safety_filter_node.py`) to avoid custom message types and inter-node latency.

```
                      ┌─────────────────────────────────────────┐
/human_control ───────►                                         │
/slam_pose     ───────►   ILQRSafetyFilterNode                  ├──► /control
/Obstacles/*   ───────►                                         │
                      └──┬──────────────────┬────────────────────┘
                         │  background       │  foreground
                         │  thread 10 Hz     │  timer 20 Hz
                         ▼                   ▼
                      Planner            Monitor + Arbiter
                    ILQRSolver          SafetyMonitor.certify()
```

### Planner (background thread, 10 Hz)
Runs `ILQRSolver.solve(x, ctx)` continuously from the latest odometry state. Stores the resulting `(X*, U*)` trajectory under a threading lock. Warm-starts each solve by shifting the previous `U*` forward by one step.

### Monitor (foreground timer, 20 Hz)
Runs `SafetyMonitor.certify(x, u_h, ctx, U*)`:
1. Simulate one step under the human input: `x_next = f(x, u_h)`
2. Shift `U*` by one step to get `U_check`
3. Roll out `X_check` from `x_next` using `U_check` (single JAX rollout, ~2 ms)
4. Evaluate hard safety margins along `X_check`
5. `V̂ = min(margins)` — certified iff `V̂ ≥ 0`

The monitor does **not** run a full ILQR re-solve. A full re-solve would block the 20 Hz timer for 100–500 ms. The shifted-rollout check is conservative (might override when u_h is actually safe) but runs in ~2 ms.

### Arbiter (same 20 Hz timer, after monitor)
- If `V̂ ≥ 0` and hysteresis counter is zero: publish `u_h` (passthrough)
- Otherwise: publish `U*[0]` (ILQR fallback) and arm a 3-cycle hysteresis hold

---

## State and Control Convention

| Symbol | Meaning | Units |
|--------|---------|-------|
| `x[0]` | px — position x | m |
| `x[1]` | py — position y | m |
| `x[2]` | v — longitudinal speed | m/s |
| `x[3]` | ψ — yaw angle | rad |
| `x[4]` | δ — front steering angle | rad |
| `u[0]` | a — longitudinal acceleration | m/s² |
| `u[1]` | ω — steering rate dδ/dt | rad/s |

Dynamics: 5D kinematic bicycle model, Forward Euler, `dt = 0.05 s`.

Trajectory convention throughout: `X` has shape `(H+1, 5)`, `U` has shape `(H, 2)` — **row-major**.

---

## ILQR Cost Function

```
stage_cost(x, u, ctx) =
    w_b  · softplus(−ℓ_soft / τ)²          # safety barrier (dominant)
  + w_c  · e_cross²  + w_ψ · e_heading²    # centerline tracking
  + w_v  · (v − v_ref)²                    # speed preserving
  + R_a  · a²  + R_ω · ω²                  # control regularization

terminal_cost(x, ctx) = stage_cost(x, u=0, ctx)
```

Where:
- `ℓ_soft(x)` = differentiable soft-min (log-sum-exp with β=50) over `[ℓ_lane, ℓ_obs, ℓ_traffic, ℓ_kin]`
- `v_ref = max(v_ref_min, current_speed)` — enforces a minimum forward incentive so the planner never optimally "stays still" (which causes circular drift when any steering is applied)

### Safety Margins

All margins are signed distances: positive = safe, negative = violated.

| Margin | Formula |
|--------|---------|
| `ℓ_lane` | min clearance from 3-point truck footprint to lane boundary |
| `ℓ_obstacle` | min clearance from footprint to any valid static obstacle |
| `ℓ_traffic` | min clearance from footprint to any valid dynamic obstacle |
| `ℓ_kinematic` | min of `(δ−δ_min, δ_max−δ, v, v_max−v) − r_safe_kin` |

The truck footprint is modeled as three inflation circles at offsets `{−0.35, 0, +0.35} × truck_length` along the heading axis.

**Hard min** (for monitor V̂ check) uses `jnp.minimum` chain.
**Soft min** (for ILQR cost gradient) uses `−logsumexp(−β·ℓ) / β`.

---

## File Layout

```
ece346/ilqr_safety_filter/
│
├── README.md                          ← this file
├── __init__.py
│
├── safety_filter/                     # pure Python + JAX, zero ROS imports
│   ├── __init__.py
│   ├── config.py                      # ILQRSafetyParams dataclass + declare_and_load()
│   ├── dynamics.py                    # make_step(), make_rollout(), step_numpy()
│   ├── jax_context.py                 # ILQRContext NamedTuple pytree + build_context()
│   ├── margins.py                     # margin_lane/obstacle/kinematic, hard+soft-min, vmap
│   ├── ilqr_costs.py                  # stage_cost(), terminal_cost() — pure JAX autodiff
│   ├── ilqr_solver.py                 # ILQRSolver: backward/forward pass, LM regularization
│   └── monitor.py                     # SafetyMonitor.certify() — rollout-based V̂ check
│
├── scripts/
│   ├── ilqr_safety_filter_node.py     # main ROS 2 node
│   └── ilqr_safety_viz_node.py        # RViz markers for plan path + V̂ status
│
├── config/
│   ├── ilqr_safety_filter_sim.yaml    # simulator parameters
│   └── ilqr_safety_filter_real.yaml   # real-truck parameters (more conservative margins)
│
└── launch/
    ├── ilqr_safety_filter_sim_launch.py   # sim: simulator + routing + joy + filter + viz
    └── ilqr_safety_filter_real_launch.py  # real: joy + filter + viz
```

### Reused from `ece346/Final_Project/safety_filter/` (imported, never copied)
- `ros_utils`: `odom_to_state`, `servo_msg_to_control`, `control_to_servo_msg`, `marker_array_to_obstacles`, `odometry_array_to_obstacles`
- `obstacle_memory`: `ObstacleMemory`, `Obstacle`
- `lane_context`: `LaneContext`, `LaneletContextBuilder`

---

## Key Design Decisions

### JAX JIT with Fixed-Shape Pytrees
JAX recompiles a JIT function whenever a traced value changes shape. To prevent recompilation when the number of obstacles changes at runtime:
- `ILQRContext` is a `NamedTuple` (auto-registered JAX pytree) with **fixed-size** arrays
- Obstacles are padded to `MAX_OBS = 10` rows; inactive rows carry a `valid = 0.0` mask
- Lane geometry is resampled to exactly `N_LANE = 100` points via `np.linspace` indices
- `cost_params (9,)` and `kin_params (6,)` are fixed-length float arrays, not dicts

### Backward Pass (LM Regularization)
Mirrors Lab 1's `backward_pass()` exactly, transposed to row-major:
```
Q_uu_reg = R_uu[t] + B.T @ (P + λ·I₅) @ B   # Levenberg–Marquardt on state space
```
If `Q_uu_reg` is not PD (negative eigenvalue), the pass returns `ok=False`, the outer loop scales λ up by 10×, and retries.

### Warm-Starting
- **Planner**: shifts previous `U*` forward by 1 step on each 10 Hz tick
- **Monitor**: uses the shifted planner `U*` as the rollout trajectory (no ILQR in monitor)

### Thread Safety
- `self._plan_lock` guards `_current_X` / `_current_U` (planner writes, control timer reads)
- `self._lane_lock` guards `self.lane_context` (odom callback writes, planner thread + control timer read)

---

## ROS Topics

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/slam_pose` | `nav_msgs/Odometry` | Vehicle pose + velocity |
| `/human_control` | `racecar_msgs/ServoMsg` | Joystick command |
| `/Obstacles/Static` | `visualization_msgs/MarkerArray` | Static obstacle markers |
| `/Obstacles/Dynamic` | `racecar_msgs/OdometryArray` | Moving traffic |

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/control` | `racecar_msgs/ServoMsg` | Filtered command to servo |
| `/safety/V_hat` | `std_msgs/Float32` | Safety certificate value (≥0 = certified) |
| `/safety/override` | `std_msgs/Bool` | True when ILQR fallback is active |
| `/safety/binding_constraint` | `std_msgs/String` | Which margin is tightest: `lane/obstacle/traffic/kinematic` |
| `/safety/u_human` | `std_msgs/Float64MultiArray` | Human input `[a, ω]` |
| `/safety/u_filtered` | `std_msgs/Float64MultiArray` | Actual output `[a, ω]` |
| `/safety/plan_states` | `std_msgs/Float64MultiArray` | Flattened `X*` `((H+1)×5,)` |
| `/safety/plan_path` | `nav_msgs/Path` | ILQR plan as RViz path |
| `/safety/markers` | `visualization_msgs/MarkerArray` | Plan trajectory + V̂ text overlay |

---

## Parameters (key ones)

All parameters live under `ilqr_safety_filter_node: ros__parameters:` in the YAML.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon_H` | 30 | ILQR prediction horizon (steps); 30 × 0.05 s = 1.5 s lookahead |
| `dt` | 0.05 | Discretization timestep (s); must match `control_rate_hz` |
| `w_barrier` | 1000.0 | Safety barrier weight; dominates all other cost terms |
| `tau` | 0.1 | Softplus sharpness; smaller = sharper barrier near ℓ = 0 |
| `soft_min_beta` | 50.0 | log-sum-exp β for the differentiable soft-min inside the cost |
| `v_ref_min` | 0.5 | Minimum speed reference (m/s) given to the ILQR planner. Prevents the planner from optimally stopping in place when the car is slow, which would cause circular drift when steering is applied. |
| `hysteresis_cycles` | 3 | Number of 20 Hz cycles to hold override after last unsafe event (~150 ms) |
| `r_safe_lane` | 0.05 | Extra lane-boundary clearance (m) beyond truck radius |
| `r_safe_obs` | 0.05 | Extra clearance around static obstacles |
| `r_safe_traf` | 0.10 | Extra clearance around moving traffic |
| `max_obstacles` | 10 | Fixed JAX array size; **restart required to change** |
| `n_lane_pts` | 100 | Fixed JAX array size; **restart required to change** |
| `planner_rate_hz` | 10.0 | Background ILQR planner rate |
| `control_rate_hz` | 20.0 | Foreground monitor + arbiter rate |

---

## Launch

### Simulation
```bash
cd /ros2_ws
colcon build --packages-select racecar_ece346
source install/setup.bash
ros2 launch racecar_ece346 ilqr_safety_filter_sim_launch.py
```

Starts: simulator, traffic simulation, routing, `joy_to_servo_node`, `ilqr_safety_filter_node`, `ilqr_safety_viz_node`.

Custom param file:
```bash
ros2 launch racecar_ece346 ilqr_safety_filter_sim_launch.py \
  param_file:=/path/to/my_params.yaml
```

### Real Truck
```bash
ros2 launch racecar_ece346 ilqr_safety_filter_real_launch.py
```

Starts: `joy_to_servo_node`, `ilqr_safety_filter_node`, `ilqr_safety_viz_node` (no simulator).

---

## Debugging

The node logs a status line every ~1 s:
```
[ilqr_safety_filter_node] V_hat=0.123  override=False  bind=lane  u_h=[0.5, 0.2]  u_out=[0.5, 0.2]
```

| Value | Meaning |
|-------|---------|
| `V_hat > 0` | Human input certified; passthrough active |
| `V_hat < 0` | Human input not certified; ILQR fallback active |
| `V_hat always < 0, bind=lane` | Lane context is wrong (possibly using `fallback_straight`); check routing node |
| `V_hat always < 0, bind=kinematic` | Kinematic margins violated; check `r_safe_kin` or speed/steering limits |

### RViz visualization
Add a **MarkerArray** display on topic `/safety/markers`:
- Green line = ILQR plan, override inactive
- Red line = ILQR plan, override active
- Text overlay shows `V̂` value and binding constraint

### Forcing passthrough (emergency bypass)
Set `hysteresis_cycles: 0` and `r_safe_lane: 0.0`, `r_safe_obs: 0.0`, `r_safe_traf: 0.0` in the YAML to make certification maximally permissive. The filter will only override when the car is literally outside the lane.

---

## Known Limitations

- **Monitor is conservative**: uses the planner's shifted trajectory rather than re-optimizing from `x_next`. May override safe human inputs if the planner's plan is stale or suboptimal.
- **ILQR is a local optimizer**: can converge to local minima in complex environments. The barrier cost strongly penalizes lane departure but cannot guarantee global safety.
- **Forward Euler integration**: accurate enough for `dt = 0.05 s` and the speed envelope of the 1/10-scale truck but accumulates error over the full horizon.
- **`argmin` has zero gradient through JAX**: the nearest-centerline-point lookup in the cost function has no gradient w.r.t. which point is selected. Gradients flow correctly through the cross-track formula after the lookup, but the ILQR may not replan toward a different segment of centerline if the car drifts far.
