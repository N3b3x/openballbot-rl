"""Estimator: port equals the study's odometry.py; kinematic offsets reproduce the simulator's point velocities."""
import importlib.util

import numpy as np
import pytest

from ballbot_control.core import RobotState, StateEstimator
from ballbot_control.core.estimator import BallbotOdometry
from ballbot_control.core.math3d import mat_to_quat, quat_to_mat
from conftest import REPO


def _study_odometry():
    spec = importlib.util.spec_from_file_location("study_odometry", REPO / "experiments" / "warp_ppo" / "odometry.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_port_matches_study_odometry(bundle):
    ref_mod = _study_odometry()
    C = np.array(bundle["estimator"]["C"])
    rng = np.random.default_rng(0)
    ref = ref_mod.BallbotOdometry(1, C, 0.09)
    mine = BallbotOdometry(C, 0.09)
    for t in range(500):
        still = t < 100      # exercise the zero-velocity bias update first
        wr = rng.normal(0, 0.01 if still else 3.0, 3)
        gy = rng.normal(0, 0.01 if still else 0.5, 3) + 0.004
        acc = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.3, 3)
        p1, v1, e1 = ref.step(wr[None], gy[None], acc[None])
        p2, v2, e2 = mine.step(wr, gy, acc)
        np.testing.assert_allclose(p2, p1[0], atol=1e-10); np.testing.assert_allclose(v2, v1[0], atol=1e-10)
    np.testing.assert_allclose(mine.R, ref.R[0], atol=1e-10)
    np.testing.assert_allclose(mine.bg, ref.bg[0], atol=1e-12)
    assert mine.yaw() == pytest.approx(float(ref.yaw()[0]))


def test_estimator_at_rest(bundle):
    est = StateEstimator(bundle["estimator"], bundle.design("nom")["kinematics"], bundle.dt)
    for _ in range(300):
        s, info = est.step(np.zeros(3), np.array([0.002, -0.001, 0.0005]), np.array([0.0, 0.0, 9.81]))
    assert np.abs(s.quat - [1, 0, 0, 0]).max() < 1e-2
    assert np.linalg.norm(info["vel"]) < 2e-3 and np.linalg.norm(s.v_com) < 2e-3      # residual bias while it converges
    assert np.linalg.norm(info["gyro_bias"] - [0.002, -0.001, 0.0005]) < 2e-3      # zero-velocity bias estimate converging


def test_quaternion_continuity():
    q_prev = np.array([np.cos(3.1 / 2), 0, 0, np.sin(3.1 / 2)])
    R = quat_to_mat(np.array([np.cos(3.2 / 2), 0, 0, np.sin(3.2 / 2)]))      # yaw crosses pi
    q = mat_to_quat(R, q_prev)
    assert np.dot(q, q_prev) > 0 and q[0] < 0      # keeps the integrated (simulator-like) hemisphere


def test_kinematic_offsets_vs_sim(fixture_rollout, bundle):
    """Base-origin velocity and position, and base-COM velocity, from the ball centre and the bundle's rigid offsets, vs the simulator.
    Not exact: the base sits on the ball through compliant contacts, so the offsets are only nominally rigid."""
    z = fixture_rollout
    T, N = z["fell"].shape
    eo, ep, ec = [], [], []
    for w in range(N):
        kin = bundle.design(str(z["designs"][int(z["didx"][w])]))["kinematics"]
        ro, rc, off = (np.array(kin[k]) for k in ("base_origin_from_ball_centre_body", "base_com_from_ball_centre_body", "ball_body_origin_from_centre_ball"))
        for t in range(1, T):
            if z["fell"][:t, w].any():
                break
            qp, qv, cv = (z[k][t, w].astype(np.float64) for k in ("qpos", "qvel", "cvel"))
            R, Rk = quat_to_mat(qp[3:7]), quat_to_mat(qp[13:17])
            wk, wb = Rk @ qv[12:15], R @ qv[3:6]
            c = qp[10:13] - Rk @ off
            vc = qv[9:12] - np.cross(wk, Rk @ off)
            eo.append(np.abs(vc + np.cross(wb, R @ ro) - qv[0:3]).max())
            ep.append(np.abs(c + R @ ro - qp[0:3]).max())
            ec.append(np.abs(vc + np.cross(cv[0:3], R @ rc) - cv[3:6]).max())
    eo, ep, ec = map(np.array, (eo, ep, ec))
    print(f"v_origin err median {np.median(eo):.2e} p99 {np.percentile(eo, 99):.2e}; p_origin {np.median(ep):.2e}/{np.percentile(ep, 99):.2e}; "
          f"v_com {np.median(ec):.2e}/{np.percentile(ec, 99):.2e}")
    assert np.median(eo) < 0.02 and np.median(ep) < 0.01 and np.median(ec) < 0.02


def test_robotstate_from_sim_fields(fixture_rollout):
    z = fixture_rollout
    s = RobotState.from_sim(z["qpos"][50, 0], z["qvel"][50, 0], z["cvel"][50, 0])
    np.testing.assert_allclose(s.v_origin, z["qvel"][50, 0, 0:3]); np.testing.assert_allclose(s.ball_pos, z["qpos"][50, 0, 10:12])


def test_track_point_follows_bundle():
    """goal_centre bundles (g39) put the goal point at the ball centre with no orientation dead-reckoning; older bundles (g34) keep
    the ball body origin. The ControlLoop picks it from the bundle unless overridden."""
    from ballbot_control.core import load_bundle
    from ballbot_control.core.runtime import ControlLoop, LoopConfig
    from parity_replay import bundle_path
    b39, b34 = load_bundle(bundle_path("g39_goalcentre_s0")), load_bundle(bundle_path("g34_mref_rsad_db_s0"))
    assert b39.goal_centre and b39.track_point == "ball_centre"
    assert not b34.goal_centre and b34.track_point == "ball_body_origin"
    assert ControlLoop(b39).est.track == "ball_centre" and ControlLoop(b34).est.track == "ball_body_origin"
    assert ControlLoop(b39, LoopConfig(track_point="ball_body_origin")).est.track == "ball_body_origin"
    est = ControlLoop(b39).est
    wr = np.array([3.0, -1.0, 2.0])
    for _ in range(100):
        s, info = est.step(wr, np.zeros(3), np.array([0.0, 0.0, 9.81]))
    np.testing.assert_allclose(s.ball_pos, info["pos"]); np.testing.assert_allclose(s.ball_vel, info["vel"])
    np.testing.assert_allclose(est.Rk, np.eye(3))       # no ball-orientation integration in centre mode


def test_from_sim_goal_centre(fixture_rollout, bundle):
    z = fixture_rollout
    off = np.array(bundle.design("nom")["kinematics"]["ball_body_origin_from_centre_ball"])
    qp = z["qpos"][100, 0].astype(np.float64)
    s = RobotState.from_sim(qp, z["qvel"][100, 0], z["cvel"][100, 0], goal_centre=True, ball_origin_from_centre=off)
    np.testing.assert_allclose(s.ball_pos, (qp[10:13] - quat_to_mat(qp[13:17]) @ off)[:2])
    assert np.linalg.norm(s.ball_pos - qp[10:12]) <= 0.14 + 1e-6
