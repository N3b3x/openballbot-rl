"""Ball retainer ("arrestor", after ETH Zurich's Rezero ballbot): passive low-friction rollers on the body, one diametrically opposite each
drive wheel across the ball, so the ball is held in a cage of six contacts and the body cannot lift or slide off it. Simulation only.

Geometry, per design, computed from the compiled model at the reset pose:
  c      ball centre,  R ball radius
  p_k    point of wheel capsule k closest to c (the wheel-ball contact normal is n_k = (p_k - c)/|p_k - c|)
  a_k    arrestor centre = c - (R + r_a + gap) n_k   (sphere of radius r_a, rigid with the base)
With gap = 0 the arrestors just touch the ball at the reset pose (no preload): they add no force while the wheels are seated and only act
when the body starts to lift or slide off. Their forces pass through the ball centre (opposite the wheel contact), so they add no torque
about the ball centre and do not change the planar ballbot model the LQR is built on.
Contact: condim 1 (frictionless, an ideal ball-transfer unit) by default, or condim 3 with sliding friction fric to model roller drag.
Arrestors add no mass (density 0); real units weigh about 10 g each, which is negligible against the body.
Enabled in train_dev.py with RETAINER="gap_mm,fric[,r_a_mm]" (e.g. RETAINER=0,0).
"""
import re
from pathlib import Path
import numpy as np


def settle(m, d, steps=1500):
    """Let the base settle onto the ball under gravity with the body held upright (base and ball horizontal and angular velocities zeroed
    every step, so only the vertical seat compresses). Returns the seated (loaded equilibrium) pose in d."""
    import mujoco
    mujoco.mj_resetData(m, d)
    qa, va = m.jnt_qposadr[m.joint("ball_free_joint").id], m.jnt_dofadr[m.joint("ball_free_joint").id]
    mujoco.mj_forward(m, d)      # put the ball on the floor first (no drop impact): ball centre at height R above z = 0
    gb = m.geom("the_ball").id
    dz = float(m.geom_size[gb][0]) - float(d.geom_xpos[gb][2])
    d.qpos[2] += dz; d.qpos[qa + 2] += dz
    for _ in range(steps):
        mujoco.mj_step(m, d)
        d.qvel[0:2] = 0; d.qvel[3:6] = 0; d.qvel[6:9] = 0; d.qvel[va:va + 2] = 0; d.qvel[va + 3:va + 6] = 0
        d.qvel[2] *= 0.95; d.qvel[va + 2] *= 0.95      # damp the vertical settling
        d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; d.qpos[0:2] = d.qpos[qa:qa + 2]
    mujoco.mj_forward(m, d)


def seat_offset(m):
    """Vertical offset base origin minus ball body origin at the loaded seat (the reset pose starts the wheels 10 mm inside the ball;
    with arrestors that would wedge the ball at t = 0, so retainer runs start from the seated pose)."""
    import copy, mujoco
    m = copy.deepcopy(m)
    if m.nhfield:      # settle on a flat floor: the live model may already carry the tile terrain
        m.hfield_data[:] = 0
    d = mujoco.MjData(m); settle(m, d)
    qa = m.jnt_qposadr[m.joint("ball_free_joint").id]
    return float(d.qpos[2] - d.qpos[qa + 2])


def arrestor_positions(xml_path, gap=0.0, r_a=0.008, phi=0.0):
    """Arrestor centres in the base frame for the LOADED seat: the model is settled under gravity first (the reset pose has the wheels
    10 mm inside the ball, the loaded seat is about 1 cm higher), so gap = 0 means the arrestors just touch the ball when the robot stands
    still: no preload, no force while seated."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path)); d = mujoco.MjData(m)
    settle(m, d)
    gb = m.geom("the_ball").id
    c, R = d.geom_xpos[gb].copy(), float(m.geom_size[gb][0])
    bid = m.body("base").id
    xb, Rb = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
    out, normals = [], []
    for k in range(3):
        g = m.geom(f"wheel_mesh_{k}").id
        gp, gm, hl = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), float(m.geom_size[g][1])
        ax = gm[:, 2]
        s = np.clip(np.dot(c - gp, ax), -hl, hl)
        p = gp + s * ax
        n = (p - c) / np.linalg.norm(p - c)
        if phi > 0:      # mount at polar angle phi (deg from the top of the ball), on the azimuth opposite the wheel
            az = np.arctan2(-n[1], -n[0]); ph = np.deg2rad(phi)
            u = np.array([np.sin(ph) * np.cos(az), np.sin(ph) * np.sin(az), np.cos(ph)])
        else:            # diametrically opposite the wheel contact
            u = -n
        a = c + (R + r_a + gap) * u
        out.append(Rb.T @ (a - xb)); normals.append(n)
    return np.array(out), np.array(normals), R


def patch_xml(src_xml, out_xml, gap=0.0, fric=0.0, r_a=0.008, floor=True, phi=0.0):
    pos, normals, R = arrestor_positions(src_xml, gap, r_a, phi)
    src = Path(src_xml).read_text()
    geoms = "".join(f'\n      <geom name="arrestor_{k}" type="sphere" size="{r_a:.4f}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" density="0" contype="0" conaffinity="0" rgba="0.85 0.85 0.9 1" group="1"/>'
                    for k, p in enumerate(pos))
    src, nsub = re.subn(r'(<freejoint name="base_free_joint"\s*/>)', r'\1' + geoms.replace('\\', '\\\\'), src, count=1)
    assert nsub == 1, "base free joint not found"
    cd = 'condim="1"' if fric <= 0 else f'condim="3" friction="{fric:g} {fric:g}"'
    pairs = "".join(f'\n    <pair name="arrest{k}" geom1="the_ball" geom2="arrestor_{k}" {cd}/>' for k in range(3))
    if floor and 'name="terrain"' in src:      # the arrestors sit about 1.2 cm above the floor: let them hit the terrain (a real unit would scrape on a step)
        pairs += "".join(f'\n    <pair name="arrest_floor{k}" geom1="terrain" geom2="arrestor_{k}" condim="1"/>' for k in range(3))
    src, nsub = re.subn(r'(\s*</contact>)', pairs + r'\1', src, count=1)
    assert nsub == 1, "contact section not found"
    Path(out_xml).write_text(src)
    return dict(pos=pos.tolist(), normals=normals.tolist(), ball_r=R)


if __name__ == "__main__":
    import sys, json
    root = Path(__file__).resolve().parents[2] / "ballbot_gym" / "models" / "variants"
    for n in sys.argv[1:] or ["nom"]:
        info = patch_xml(root / f"dz_{n}.xml", root / f"_ret_check_{n}.xml")
        print(n, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in info.items()}))
