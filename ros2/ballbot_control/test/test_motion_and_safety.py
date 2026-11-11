"""Motion layer vs the study's torch MotionRef; safety supervisor and command gate behaviour; controller smoke tests."""
import importlib.util
import math

import numpy as np
import pytest

from ballbot_control.core import ACTIVE, FAULT, HOLD, CommandGate, RobotState, SafetySupervisor, TrackingController
from ballbot_control.core.motion_ref import MotionRef
from conftest import REPO


def test_motion_ref_matches_study(bundle):
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("study_mref", REPO / "experiments" / "warp_ppo" / "motion_ref.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(1)
    for f in (1.0, 0.75):
        ref = m.MotionRef(1, "cpu"); ref.s, ref.th = ref.s.double(), ref.th.double()      # float64 so the comparison is exact
        ref.reset(torch.ones(1, dtype=torch.bool), heading=torch.tensor([0.3], dtype=torch.float64))
        mine = MotionRef(bundle["motion_ref"], f); mine.reset(heading=0.3)
        pos = np.zeros(2); goal = np.array([1.0, -0.6]); fw = torch.tensor([f], dtype=torch.float64)
        for t in range(1500):
            on = (t // 300) % 2 == 1                                   # alternate velocity and goal phases
            v = rng.normal(0, 0.5, 2) if t % 50 == 0 else (v if t else np.zeros(2))
            if t % 400 == 399:
                v = -v                                                # reversals
            c_ref = ref.step(torch.tensor(v[None]), fw, torch.tensor(pos[None]), torch.tensor(goal[None]),
                             goal_on=torch.tensor([on]))[0].numpy()
            c = mine.step(v, pos, goal, goal_on=on)
            np.testing.assert_allclose(c, c_ref, atol=1e-9)
            assert bool(ref.hold[0]) == mine.hold
            pos = pos + c * 0.01                                       # ideal tracking moves the point


def _state(tilt_deg=0.0):
    th = math.radians(tilt_deg)
    return RobotState(quat=np.array([math.cos(th / 2), math.sin(th / 2), 0, 0]), omega_body=np.zeros(3), v_com=np.zeros(3),
                      v_origin=np.zeros(3), ball_omega=np.zeros(3), ball_pos=np.zeros(2), ball_vel=np.zeros(2))


def test_supervisor_startup_hold_and_enable():
    sup = SafetySupervisor(tilt_cutoff_deg=30, engage_tilt_deg=3, state_timeout=0.05)
    assert sup.mode == HOLD
    assert np.all(sup.gate_torque([5.0, -5.0, 1.0]) == 0)            # startup: no torque
    sup.request_enable()
    mode, just = sup.update(tilt_deg=8.0, state_age=0.01)            # too tilted to engage
    assert mode == HOLD and not just
    mode, just = sup.update(tilt_deg=1.0, state_age=0.2)             # stale state
    assert mode == HOLD and not just
    mode, just = sup.update(tilt_deg=1.0, state_age=0.01)
    assert mode == ACTIVE and just
    np.testing.assert_allclose(sup.gate_torque([1.0, 2.0, 3.0]), [1, 2, 3])


def test_supervisor_faults_latch():
    for kw in (dict(tilt_deg=35.0, state_age=0.01), dict(tilt_deg=1.0, state_age=0.5), dict(tilt_deg=float("nan"), state_age=0.01)):
        sup = SafetySupervisor(tilt_cutoff_deg=30)
        sup.request_enable(); sup.update(1.0, 0.0)
        assert sup.mode == ACTIVE
        sup.update(**kw)
        assert sup.mode == FAULT
        sup.update(1.0, 0.0)
        assert sup.mode == FAULT and np.all(sup.gate_torque([1, 1, 1]) == 0)      # latched
        sup.clear_fault(); assert sup.mode == HOLD
        sup.update(1.0, 0.0); assert sup.mode == HOLD                            # needs a new enable request


def test_torque_clamp_and_nonfinite():
    sup = SafetySupervisor(torque_limit=10.0)
    sup.request_enable(); sup.update(0.0, 0.0)
    np.testing.assert_allclose(sup.gate_torque([25.0, -30.0, 3.0]), [10, -10, 3])
    sup.gate_torque([np.inf, 0, 0])
    assert sup.mode == FAULT


def test_command_gate_timeout_and_hold():
    g = CommandGate(cmd_timeout=0.5, hold_on_timeout=True, a_brk=0.5)
    v, wz, goal = g.get(0.0, np.zeros(2), np.zeros(2))
    assert goal is not None and np.allclose(goal, 0)              # nothing received yet: hold where it stands
    g.set_velocity(1.0, [0.4, 0.0], 0.2)
    v, wz, goal = g.get(1.2, np.zeros(2), np.array([0.4, 0.0]))
    assert goal is None and np.allclose(v, [0.4, 0]) and wz == 0.2
    v, wz, goal = g.get(1.6, np.array([1.0, 2.0]), np.array([0.4, 0.0]))
    assert np.allclose(v, 0) and wz == 0.0
    np.testing.assert_allclose(goal, [1.0 + 0.16 / 1.0, 2.0])      # stop point: p + v|v| / (2 a_brk)
    _, _, goal2 = g.get(1.7, np.array([1.1, 2.0]), np.array([0.3, 0.0]))
    np.testing.assert_allclose(goal2, goal)                        # latched
    g.set_velocity(2.0, [0.0, 0.1], 0.0)
    assert g.get(2.1, np.zeros(2), np.zeros(2))[2] is None         # new command cancels the hold


def test_controller_at_rest_is_quiet(bundle):
    """Upright and still with a zero command: torques stay small (no residual kick), first tick uses the reset observation."""
    c = TrackingController(bundle)
    c.reset(state=_state())
    taus = [c.step(_state(), v_des=(0.0, 0.0)).torque for _ in range(200)]
    assert np.abs(np.array(taus)).max() < 0.5 * c.tau
    assert c.obs.hist.shape == (8, 28)


def test_controller_lqr_only_mode(bundle):
    """residual=False: the policy is bypassed (action 0, last_action stays 0) and the LQR alone reacts to a tilt."""
    c = TrackingController(bundle, residual=False)
    s = _state()
    th = math.radians(3.0)
    s.quat = np.array([math.cos(th / 2), 0, math.sin(th / 2), 0])    # top leans toward +x
    for _ in range(3):
        r = c.step(s, v_des=(0.0, 0.0))
    assert np.all(r.action == 0) and np.all(r.obs[12:15] == 0)
    assert np.abs(r.applied).max() > 0.01
