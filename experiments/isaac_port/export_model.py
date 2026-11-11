"""Dump everything the PhysX model needs from the compiled MuJoCo model (world frame at the reset pose, yaw 0).

Also computes the roller (omni-wheel) emulation geometry: at each wheel the MuJoCo contact has a grip direction (high
friction, = wheel circumferential tangent) and a free direction t (friction 0.001).  A passive roller sphere centred on
the wheel side of the contact, spinning about the grip direction, reproduces that anisotropy with isotropic PhysX friction.
"""
import json
import sys

import mujoco
import numpy as np

from common import ROOT, DesignCfg
from backend_mj import MJBackend


def export(design):
    cfg = DesignCfg(design)
    be = MJBackend(cfg)
    m, d = be.m, be.d
    mujoco.mj_forward(m, d)
    # settle a few ms so contacts exist at the reset pose
    for _ in range(120):
        be.stepper.step(d)
    ball_c = d.xpos[be.kb] + d.xmat[be.kb].reshape(3, 3) @ m.geom_pos[be.ball_geom]
    out = dict(design=design, tau_max=cfg.tau_max, damping=cfg.damping, armature=cfg.armature, timestep=m.opt.timestep)
    out["ball"] = dict(radius=float(m.geom_size[be.ball_geom][0]), mass=float(m.body_mass[be.kb]),
                       inertia=m.body_inertia[be.kb].tolist(), ipos=m.body_ipos[be.kb].tolist(),
                       centre0=(m.body_pos[be.kb] + m.geom_pos[be.ball_geom]).tolist())
    R0 = np.eye(3)
    out["base"] = dict(pos_settled=d.xpos[be.bb].tolist(), pos0=m.body_pos[be.bb].tolist(), mass=float(m.body_mass[be.bb]), ipos=m.body_ipos[be.bb].tolist(),
                       iquat=m.body_iquat[be.bb].tolist(), inertia=m.body_inertia[be.bb].tolist())
    # full inertia of base *including* child bodies (cameras, wheels are separate bodies in MuJoCo, so exclude them)
    out["tower"] = dict(r=float(m.geom_size[m.geom("tower_collision").id][0]), hl=float(m.geom_size[m.geom("tower_collision").id][1]),
                        pos=m.geom_pos[m.geom("tower_collision").id].tolist())
    ws = []
    for k in range(3):
        b = m.body(f"wheel_{k}").id
        j = m.joint(f"wheel_joint_{k}").id
        ws.append(dict(body_pos=m.body_pos[b].tolist(), body_quat=m.body_quat[b].tolist(), mass=float(m.body_mass[b]),
                       ipos=m.body_ipos[b].tolist(), iquat=m.body_iquat[b].tolist(), inertia=m.body_inertia[b].tolist(),
                       anchor_world=d.xanchor[j].tolist(), axis_world=d.xaxis[j].tolist(),
                       cap_r=float(m.geom_size[m.geom(f"wheel_mesh_{k}").id][0]), cap_hl=float(m.geom_size[m.geom(f"wheel_mesh_{k}").id][1])))
    # contacts at reset pose (world)
    for c in be.wheel_ball_forces():
        w, p, F, fn, ft = c
        ws[w]["contact_world"] = p.tolist()
        ws[w]["contact_normal_N"] = fn
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {c.geom1, c.geom2}
        if be.ball_geom in pair and pair & set(be.wheel_geoms):
            w = be.wheel_geoms.index((pair & set(be.wheel_geoms)).pop())
            fr = c.frame.reshape(3, 3)
            ws[w]["frame_normal"] = fr[0].tolist()
            ws[w]["frame_t1"] = fr[1].tolist()      # friction[0]=0.001 direction (free)
            ws[w]["frame_t2"] = fr[2].tolist()      # friction[1]=1.0 direction (grip)
            ws[w]["penetration"] = float(-c.dist)
    out["wheels"] = ws
    out["ball_centre_world"] = ball_c.tolist()
    out["pair_friction"] = m.pair_friction[0].tolist()
    return out




def export_all(designs):
    import pathlib
    out = ROOT / "outputs/study/isaac_port/models"
    out.mkdir(parents=True, exist_ok=True)
    for dn in designs:
        json.dump(export(dn), open(out / f"{dn}.json", "w"), indent=1)


if __name__ == "__main__":
    export_all(sys.argv[1:] or ["nom"])
