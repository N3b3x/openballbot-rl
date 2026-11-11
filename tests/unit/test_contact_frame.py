"""The Python RK4 stepper must match mj_step, and the frame fix must align wheel contacts."""
import importlib.resources

import mujoco
import numpy as np
import pytest

import ballbot_gym  # noqa: F401
from ballbot_gym.utils.contact_frame import ContactFrameStepper


def _make():
    xml = str(importlib.resources.files("ballbot_gym.models") / "ballbot.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    m.hfield_data = np.zeros(m.hfield_nrow[0] * m.hfield_ncol[0])
    mujoco.mj_resetData(m, d)
    d.joint("base_free_joint").qpos[2] += 0.01
    d.joint("ball_free_joint").qpos[2] += 0.01
    mujoco.mj_forward(m, d)
    return m, d


def test_stepper_matches_mj_step_without_fix():
    m1, d1 = _make()
    m2, d2 = _make()
    stepper = ContactFrameStepper(m2, fix_frames=False)
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.uniform(-10, 10, 3)
        d1.ctrl[:] = u
        d2.ctrl[:] = u
        mujoco.mj_step(m1, d1)
        stepper.step(d2)
    np.testing.assert_allclose(d1.qpos, d2.qpos, atol=1e-9)
    np.testing.assert_allclose(d1.qvel, d2.qvel, atol=1e-9)


def test_wheel_contact_frames_align_with_capsule_axis():
    m, d = _make()
    stepper = ContactFrameStepper(m, fix_frames=True)
    rng = np.random.default_rng(1)
    for _ in range(100):
        d.ctrl[:] = rng.uniform(-3, 3, 3)
        stepper.step(d)
    checked = 0
    for i in range(d.ncon):
        c = d.contact[i]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in c.geom]
        if "the_ball" in names and any("wheel" in n for n in names):
            w = c.geom[[("wheel" in n) for n in names].index(True)]
            axis = d.geom_xmat[w].reshape(3, 3)[:, 2]
            assert abs(axis @ c.frame[3:6]) == pytest.approx(1.0, abs=1e-6)
            checked += 1
    assert checked >= 1


def test_env_velocity_observation_is_linear_velocity():
    """obs['vel'] must be translational velocity (cvel[3:]); the directional reward depends on it."""
    import yaml
    from ballbot_rl.training.utils import make_ballbot_env

    cfg = yaml.safe_load(open("configs/env/flat_directional.yaml"))
    cfg["camera"]["enabled"] = False
    env_config = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                           env_config=env_config, seed=0, eval_env=True)()
    env.reset(seed=0)
    base = env.unwrapped
    body = base.model.body("base").id
    prev = base.data.xpos[body].copy()
    rng = np.random.default_rng(0)
    for _ in range(12):
        obs, *_ = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
    fd = (base.data.xpos[body] - prev) / (12 * 5 * base.model.opt.timestep)
    # linear velocity is small and finite-difference-consistent; angular velocity is not
    assert np.abs(obs["vel"][:2]).max() < 1.0
    assert np.abs(obs["vel"][:2] - fd[:2]).max() < 0.5
    env.close()
