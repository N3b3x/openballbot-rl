"""ControlLoop (what the controller node runs) in plain Python, including a closed-loop CPU MuJoCo run with the estimator in the loop."""
import math

import numpy as np
import pytest

from ballbot_control.core import ACTIVE, FAULT, HOLD
from ballbot_control.core.runtime import ControlLoop, LoopConfig

G_UP = np.array([0.0, 0.0, 9.81])


def test_loop_startup_hold_then_enable(bundle):
    loop = ControlLoop(bundle)
    r = loop.tick(0.0, np.zeros(3), G_UP, np.zeros(3), None)          # no sensor data yet
    assert r.mode == HOLD and np.all(r.torque == 0)
    for k in range(5):
        r = loop.tick(0.01 * k, np.zeros(3), G_UP, np.zeros(3), 0.01 * k)
    assert r.mode == HOLD and np.all(r.torque == 0)                 # startup hold until enabled
    loop.enable()
    r = loop.tick(0.05, np.zeros(3), G_UP, np.zeros(3), 0.05)
    assert r.mode == ACTIVE and r.step is not None


def test_loop_sensor_dropout_faults(bundle):
    loop = ControlLoop(bundle, LoopConfig(state_timeout=0.05))
    loop.enable()
    loop.tick(0.0, np.zeros(3), G_UP, np.zeros(3), 0.0)
    assert loop.sup.mode == ACTIVE
    r = loop.tick(0.1, np.zeros(3), G_UP, np.zeros(3), 0.0)          # last sample 100 ms old
    assert r.mode == FAULT and np.all(r.torque == 0)


def test_cmd_vel_base_link_rotated_to_odom(bundle):
    loop = ControlLoop(bundle, LoopConfig(cmd_vel_frame="base_link", max_speed=0.6))
    yaw = math.pi / 2
    loop.state = type("S", (), {"quat": np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])})()
    loop.on_cmd_vel(0.0, 0.3, 0.0, 0.2)
    np.testing.assert_allclose(loop.gate.v, [0.0, 0.3], atol=1e-12)
    loop.on_cmd_vel(0.0, 2.0, 0.0, 5.0)                             # speed and yaw-rate caps
    assert np.linalg.norm(loop.gate.v) == pytest.approx(0.6) and loop.gate.wz == 1.0


def test_closed_loop_cpu_mujoco():
    pytest.importorskip("mujoco")
    import sim_closed_loop as S
    L = S.run(seconds=10.0, speed=0.3, drive=(1.0, 6.0))
    s = S.summary(L, drive=(1.0, 6.0), speed=0.3)
    print(s)
    assert s["final_mode"] == ACTIVE and s["ticks"] == 1000       # balanced the whole run, no fault
    assert s["max_tilt_deg"] < 15
    assert 0.15 < s["cruise_vx_mean"] < 0.4                        # follows the 0.3 m/s command
    assert s["stopped_speed"] < 0.1                                # command timeout -> stop and hold
