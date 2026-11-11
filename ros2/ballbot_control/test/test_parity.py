"""Core vs simulator parity for every exported run (g39 goal-centre, recommended; g34 goal at the ball body origin) (the most important check): observations, actor inputs, actions, LQR, authority, torques.

Tolerances: the simulator runs in float32 on the GPU, the core in float64; per-tick differences are at float32 rounding (median about
1e-7 to 1e-6) and the integrators (motion-layer speed, velocity integral, LQR integral) accumulate them to about 1e-4 over 6 s.
A logic error (wrong order, frame, history slot, sign or design constant) shows up as 1e-2 or more.
"""
import numpy as np
import pytest

from ballbot_control.core import load_bundle
from parity_replay import RUNS, bundle_path, full_rollout_path, replay

TOL = dict(cmd=1e-3, goal_rule=1e-5, obs=1e-3, actor_x=1e-3, action=1e-3, lqr=2e-3, rs=1e-3, applied=2e-3, torque=3e-2)


def _check(err, n):
    assert n > 1000
    for k, tol in TOL.items():
        assert err[k].max() < tol, f"{k}: max {err[k].max():.3e} >= {tol}"
        assert np.median(err[k]) < 1e-4, f"{k}: median {np.median(err[k]):.3e}"


def test_parity_fixture(run_case):
    run, bundle, z = run_case
    assert bundle.goal_centre == bool(z["goal_centre"]) if "goal_centre" in z else not bundle.goal_centre
    err, n = replay(z, bundle)
    _check(err, n)


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS)
def test_parity_full_rollout(run):
    p = full_rollout_path(run)
    if not p.exists():
        pytest.skip(f"full rollout of {run} not present (run experiments/warp_ppo/record_parity.py)")
    z = dict(np.load(p))
    b = load_bundle(bundle_path(run))
    err, n = replay(z, b)
    _check(err, n)
    assert z["goal_on"].mean() > 0.2          # goal mode is exercised, not only velocity commands


def test_goal_point_matters(run_case):
    """Feeding the other goal point must break parity in goal mode: proves the test actually checks the goal point."""
    run, bundle, z = run_case
    import copy
    other = copy.copy(bundle); other.data = dict(bundle.data)
    other.data["goal_point"] = dict(bundle.data.get("goal_point") or {}, track_point="ball_body_origin" if bundle.goal_centre else "ball_centre")
    err, n = replay(z, other)
    assert err["obs"].max() > 1e-2


def test_history_layout_in_sim_data(run_case):
    """The simulator's actor vector: history entry 0 = (observation, action) of the previous tick, entries shift by one per tick."""
    z = run_case[2]
    x, o, a = z["x"], z["obs_clean"], z["act"]
    for w in range(x.shape[1]):
        for t in range(9, 40):
            h = x[t, w, 25:25 + 8 * 28].reshape(8, 28)
            for j in range(8):
                np.testing.assert_allclose(h[j, :25], o[t - 1 - j, w], atol=1e-6)
                np.testing.assert_allclose(h[j, 25:], np.clip(a[t - 1 - j, w], -1, 1), atol=1e-6)


def test_design_vectors_match_sim(run_case):
    run, bundle, z = run_case
    for w in range(z["x"].shape[1]):
        d = bundle.design(str(z["designs"][int(z["didx"][w])]))
        np.testing.assert_allclose(z["x"][0, w, 249:255], d["dvec"], atol=1e-5)
        assert d["tau"] == pytest.approx(float(z["tau_w"][w]))
    np.testing.assert_array_equal(bundle.design("nom")["dvec"], np.zeros(6))
