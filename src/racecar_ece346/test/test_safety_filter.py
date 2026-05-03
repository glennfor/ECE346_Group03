import numpy as np

from ece346.Final_Project.safety_filter.barrier import implicit_barrier_value
from ece346.Final_Project.safety_filter.backup_policy import lane_recovery_control
from ece346.Final_Project.safety_filter.config import SafetyFilterParams
from ece346.Final_Project.safety_filter.dynamics import step
from ece346.Final_Project.safety_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.Final_Project.safety_filter.margins import (
    MarginContext,
    margin_kinematic,
    margin_lane,
    margin_obstacle,
)
from ece346.Final_Project.safety_filter.obstacle_memory import Obstacle
from ece346.Final_Project.safety_filter.qp import solve_box_halfspace_qp


def make_params():
    return SafetyFilterParams(
        wheelbase_m=0.257,
        truck_radius_m=0.10,
        truck_length_m=0.40,
        a_min=-2.0,
        a_max=1.0,
        delta_min=-0.35,
        delta_max=0.35,
        v_min=0.0,
        v_max=2.0,
        r_safe_lane=0.05,
        r_safe_obs=0.05,
        r_safe_kin=0.0,
        dt=0.05,
        horizon_H=10,
    )


def make_lane():
    xs = np.linspace(-5.0, 5.0, 100)
    centerline = np.column_stack([xs, np.zeros_like(xs)])
    half_width = np.full_like(xs, 0.5)
    return LaneContext.from_centerline(centerline, half_width, half_width)


def test_dynamics_brake_and_wrap():
    params = make_params()
    x = np.array([0.0, 0.0, 1.0, np.pi - 0.01, 0.0])
    x_next = step(x, np.array([-1.0, 0.0]), params, dt=0.1)

    assert x_next[2] < x[2]
    assert -np.pi <= x_next[3] <= np.pi


def test_lane_and_obstacle_margins():
    params = make_params()
    lane = make_lane()

    x_center = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    x_outside = np.array([0.0, 0.8, 0.0, 0.0, 0.0])
    assert margin_lane(x_center, lane, params) > 0.0
    assert margin_lane(x_outside, lane, params) < 0.0

    contact_distance = params.truck_radius_m + 0.1 + params.r_safe_obs
    obs = Obstacle(1, np.array([contact_distance, 0.0]), 0.1)
    assert abs(margin_obstacle(x_center, [obs], params)) < 1e-9
    assert margin_obstacle(x_center, [], params) == 100.0


def test_lane_query_uses_segment_projection():
    centerline = np.array([[0.0, 0.0], [10.0, 0.0]])
    lane = LaneContext.from_centerline(centerline, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    sample = lane.query(5.0, 0.2)

    assert np.allclose(sample.point, np.array([5.0, 0.0]))
    assert sample.signed_lateral_error > 0.0


def test_lanelet_builder_extracts_yaw_from_full_state():
    state = np.array([1.0, 2.0, 4.0, 1.25, 0.1])
    lane_pose = LaneletContextBuilder._state_to_lane_pose(state)

    assert np.allclose(lane_pose, np.array([1.0, 2.0, 1.25]))


def test_route_selection_uses_yaw_not_speed():
    east_route = np.array([[0.0, 0.0], [1.0, 0.0]])
    north_route = np.array([[0.0, 0.0], [0.0, 1.0]])
    lane_pose = np.array([0.0, 0.0, np.pi / 2.0])

    selected = LaneletContextBuilder._select_route([east_route, north_route], lane_pose)

    assert np.allclose(selected, north_route)


def test_kinematic_margin_allows_stopped_car():
    params = make_params()
    x_stopped = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    assert margin_kinematic(x_stopped, params) > 0.0


def test_lane_recovery_moves_slowly_toward_centerline():
    params = make_params()
    lane = make_lane()
    x_left_of_lane = np.array([0.0, 0.45, 0.0, 0.0, 0.0])

    control = lane_recovery_control(x_left_of_lane, lane, params, np.array([0.5, 0.0]))

    assert control[0] > 0.0
    assert control[1] < 0.0


def test_lane_recovery_does_not_accelerate_without_human_input():
    params = make_params()
    lane = make_lane()
    x_left_of_lane = np.array([0.0, 0.45, 0.0, 0.0, 0.0])

    control = lane_recovery_control(x_left_of_lane, lane, params, np.array([0.0, 0.0]))

    assert control[0] == 0.0
    assert control[1] < 0.0


def test_lane_recovery_brakes_when_rolling_without_human_input():
    params = make_params()
    lane = make_lane()
    x_left_of_lane = np.array([0.0, 0.45, 0.4, 0.0, 0.0])

    control = lane_recovery_control(x_left_of_lane, lane, params, np.array([0.0, 0.0]))

    assert control[0] == params.a_min
    assert control[1] < 0.0


def test_barrier_detects_close_obstacle():
    params = make_params()
    lane = make_lane()
    close_obs = Obstacle(1, np.array([0.15, 0.0]), 0.1)
    ctx = MarginContext(lane, [close_obs], [], params)
    h, label, _, _ = implicit_barrier_value(np.array([0.0, 0.0, 0.4, 0.0, 0.0]), ctx)

    assert h < 0.0
    assert label == "obstacle"


def test_qp_passthrough_projection_and_infeasible():
    params = make_params()
    u_human = np.array([0.0, 0.0])

    feasible = solve_box_halfspace_qp(u_human, np.array([1.0, 0.0]), -0.5, params)
    assert feasible.status == "optimal"
    assert np.allclose(feasible.control, u_human)

    projected = solve_box_halfspace_qp(u_human, np.array([1.0, 0.0]), 0.5, params)
    assert projected.status == "optimal"
    assert np.isclose(projected.control[0], 0.5)

    infeasible = solve_box_halfspace_qp(u_human, np.array([1.0, 0.0]), 2.0, params)
    assert infeasible.status == "infeasible"

