"""Checks of orca_line on encounters whose answer is known."""
import numpy as np
from bluesky.tools.aero import kts, nm

from examples.geometric.orca import det, orca_line, orca_velocity, vertically_close

R, TAU, DT = 5.05 * nm, 300.0, 1.0


def allowed(v, line):
    """Whether velocity v lies in the half-plane (left of the directed line)."""
    point, direction = line
    return det(direction, v - point) >= -1e-9


def closest_in_half_plane(v, line):
    """v if allowed, else its projection onto the line: the smallest change."""
    point, direction = line
    return v if allowed(v, line) else point + ((v - point) @ direction) * direction


def miss_distance(p_rel, v_rel):
    """Closest future distance for p_rel (B - A) and v_rel (A - B), straight flight."""
    t = max(0.0, (p_rel @ v_rel) / (v_rel @ v_rel))
    return np.linalg.norm(p_rel - v_rel * t)


# Two aircraft on a collision course: B 30 NM right and 30 NM ahead, flying left.
P = np.array([30.0, 30.0]) * nm
V_A, V_B = np.array([0.0, 450.0]) * kts, np.array([-450.0, 0.0]) * kts


def test_reciprocal_shares_resolve_exactly_to_the_zone():
    va = closest_in_half_plane(V_A, orca_line(P, V_A, V_B, R, TAU, DT))
    vb = closest_in_half_plane(V_B, orca_line(-P, V_B, V_A, R, TAU, DT))  # B's own view
    assert np.isclose(np.linalg.norm(va - V_A), np.linalg.norm(vb - V_B))  # equal halves
    assert np.isclose(miss_distance(P, va - vb), R, rtol=1e-3)            # together: exactly R


def test_current_velocity_is_allowed_exactly_when_there_is_no_conflict():
    in_conflict = orca_line(P, V_A, V_B, R, TAU, DT)
    diverging = orca_line(np.array([40.0, -40.0]) * nm, V_A, V_B, R, TAU, DT)
    assert not allowed(V_A, in_conflict)
    assert allowed(V_A, diverging)


def test_inside_the_zone_it_steers_apart():
    p = np.array([2.0, 2.0]) * nm                                         # 2.8 NM: a loss of separation
    va = closest_in_half_plane(V_A, orca_line(p, V_A, V_B, R, TAU, DT))
    vb = closest_in_half_plane(V_B, orca_line(-p, V_B, V_A, R, TAU, DT))
    assert (va - vb) @ p < 0.0                                            # A now moves away from B


V_MAX = 500 * kts


def keeps(v, line):
    point, direction = line
    return det(direction, v - point) >= -1e-6


def test_no_conflict_keeps_the_preferred_velocity():
    line = orca_line(np.array([40.0, -40.0]) * nm, V_A, V_B, R, TAU, DT)
    assert np.allclose(orca_velocity([line], V_A, V_MAX), V_A)


def test_two_conflicts_answer_keeps_both_lines():
    lines = [orca_line(np.array([30.0, 30.0]) * nm, V_A, np.array([-450.0, 0.0]) * kts, R, TAU, DT),
             orca_line(np.array([-30.0, 30.0]) * nm, V_A, np.array([450.0, 0.0]) * kts, R, TAU, DT)]
    v = orca_velocity(lines, V_A, V_MAX)
    assert all(keeps(v, line) for line in lines)
    assert np.linalg.norm(v) <= V_MAX + 1e-6
    assert not np.allclose(v, V_A)                              # it had to change


def test_order_of_lines_does_not_change_the_answer():
    rng = np.random.default_rng(0)
    lines = [orca_line(rng.uniform(-20, 20, 2) * nm, V_A, rng.uniform(-400, 400, 2) * kts, R, TAU, DT) for _ in range(4)]
    assert np.allclose(orca_velocity(lines, V_A, V_MAX), orca_velocity(lines[::-1], V_A, V_MAX), atol=1e-6)


def test_boxed_in_still_answers():
    lines = [orca_line(q * nm, V_A, w * kts, R, TAU, DT) for q, w in
             [(np.array([3.0, 3.0]), np.zeros(2)), (np.array([-3.0, 3.0]), np.zeros(2)),
              (np.array([0.0, -3.0]), np.array([0.0, 450.0]))]]
    v = orca_velocity(lines, V_A, V_MAX)
    assert np.all(np.isfinite(v)) and np.linalg.norm(v) <= V_MAX + 1e-6


def test_vertically_close():
    assert vertically_close(500.0, 0.0, 1000.0, 300.0)            # level, 500 ft apart
    assert not vertically_close(3000.0, 0.0, 1000.0, 300.0)       # level, 3000 ft apart
    assert vertically_close(3000.0, -1500.0, 1000.0, 300.0)       # closing at 1500 fpm: inside after 80 s
    assert not vertically_close(3000.0, 1500.0, 1000.0, 300.0)    # diverging
    assert not vertically_close(9000.0, -1500.0, 1000.0, 300.0)   # closing, but only after 320 s
