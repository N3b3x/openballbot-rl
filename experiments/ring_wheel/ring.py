"""CHECK 2: roller-ring wheel variant of dz_nom in plain MuJoCo (asRoBallet-style) and the standard protocol with the LQR.

  python ring.py --n 12 --rho 0.012 --damp 0 --soft 0 --phase 0 --tag r12      (appends one row to outputs/study/ring_wheel/results.csv)
  python ring.py --ideal                                                      (idealised dz_nom, same protocol, same trial counts)

Generates the XML (outputs/study/ring_wheel/xml/*.xml) from dz_nom.xml by script: each wheel capsule is replaced by N passive
capsule rollers on hinge joints; hinge axis = tangent of the rim (90 deg to the wheel axle).  No <pair>, no contact-frame fix,
plain mujoco.mj_step (implicitfast).  Wheel body keeps the original mass (minus the roller mass), inertia, armature, damping.
"""
import argparse, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/isaac_port"))
import scenarios as S                                   # noqa: E402
from common import DesignCfg, REPEAT                     # noqa: E402

OUT = ROOT / "outputs/study/ring_wheel"
SRC = ROOT / "ballbot_gym/models/variants/dz_nom.xml"


def lowprio():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE
    except Exception:
        pass


def settled_geometry():
    """Per-wheel local geometry of the idealised model at its settled reset pose."""
    from ballbot_gym.utils.contact_frame import ContactFrameStepper
    m = mujoco.MjModel.from_xml_path(str(SRC)); m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m); st = ContactFrameStepper(m)
    mujoco.mj_forward(m, d)
    for _ in range(120):
        st.step(d)
    ball = m.geom("the_ball").id
    out = []
    for k in range(3):
        b = m.body(f"wheel_{k}").id; j = m.joint(f"wheel_joint_{k}").id; g = m.geom(f"wheel_mesh_{k}").id
        ci = [i for i in range(d.ncon) if {d.contact[i].geom1, d.contact[i].geom2} == {ball, g}]
        pw = np.mean([d.contact[i].pos for i in ci], 0)
        R = d.xmat[b].reshape(3, 3)
        out.append(dict(contact=R.T @ (pw - d.xpos[b]), anchor=m.jnt_pos[j].copy(), axis=m.jnt_axis[j].copy(),
                        mass=float(m.body_mass[b]), ipos=m.body_ipos[b].copy(), iquat=m.body_iquat[b].copy(), inertia=m.body_inertia[b].copy(),
                        cap_r=float(m.geom_size[g][0])))
    return out


def fmt(v):
    return " ".join(f"{x:.10g}" for x in v)


def make_xml(n=12, rho=0.012, damp=0.0, soft=False, phase=0.0, m_roller=None, rim_r=None, mu=1.0, hl_scale=1.0, tag=None):
    geo = settled_geometry()
    tree = ET.parse(SRC); root = tree.getroot()
    comp = root.find("compiler")
    comp.set("meshdir", str((ROOT / "ballbot_gym/models/meshes").resolve()).replace("\\", "/"))
    comp.set("texturedir", str((ROOT / "ballbot_gym/models/textures").resolve()).replace("\\", "/"))
    if soft:
        o = ET.SubElement(root, "option"); o.set("impratio", "10"); o.set("noslip_iterations", "1")
    ct = root.find("contact")
    root.remove(ct)                                     # drops the anisotropic <pair> entries
    # ball collides with rollers: ball contype 1/conaffinity 3; rollers contype 2/conaffinity 0
    # (so no roller-roller, roller-tower or roller-terrain contacts)
    ball = [g for g in root.iter("geom") if g.get("name") == "the_ball"][0]
    ball.set("contype", "1"); ball.set("conaffinity", "3")
    m_r = m_roller if m_roller is not None else 0.024 / n
    for k in range(3):
        G = geo[k]
        body = [b for b in root.iter("body") if b.get("name") == f"wheel_{k}"][0]
        for g in list(body.findall("geom")):
            body.remove(g)
        # keep the wheel joint first child order irrelevant; add explicit inertial (original mass minus roller mass)
        inertial = ET.Element("inertial"); inertial.set("pos", fmt(G["ipos"])); inertial.set("quat", fmt(G["iquat"]))
        inertial.set("mass", f"{G['mass'] - n * m_r:.10g}"); inertial.set("diaginertia", fmt(G["inertia"]))
        body.insert(0, inertial)
        a = G["axis"] / np.linalg.norm(G["axis"])
        d0 = G["contact"] - G["anchor"]
        s_ax = float(d0 @ a)
        e1 = d0 - s_ax * a; e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
        r_out = rim_r if rim_r is not None else G["cap_r"]
        Rr = r_out - rho
        half = np.pi * Rr / n
        for i in range(n):
            ph = 2 * np.pi * (i + phase) / n
            rad = np.cos(ph) * e1 + np.sin(ph) * e2
            tan = -np.sin(ph) * e1 + np.cos(ph) * e2
            cen = G["anchor"] + s_ax * a + Rr * rad
            rb = ET.SubElement(body, "body"); rb.set("name", f"roll_{k}_{i}"); rb.set("pos", fmt(cen))
            j = ET.SubElement(rb, "joint"); j.set("name", f"roll_joint_{k}_{i}"); j.set("type", "hinge"); j.set("axis", fmt(tan))
            j.set("damping", f"{damp:.6g}"); j.set("armature", "0")
            g = ET.SubElement(rb, "geom"); g.set("name", f"roll_geom_{k}_{i}"); g.set("type", "capsule")
            g.set("fromto", fmt(np.concatenate([-half * tan, half * tan]))); g.set("size", f"{rho:.6g}")
            g.set("mass", f"{m_r:.6g}"); g.set("contype", "2"); g.set("conaffinity", "0"); g.set("friction", f"{mu} 0.005 0.0001")
            g.set("rgba", "0.9 0.3 0.3 1")
            if soft:
                g.set("solimp", "0.85 0.99 0.003")
    name = tag or f"ring_n{n}_rho{rho}_d{damp}_s{int(soft)}_p{phase}"
    p = OUT / "xml" / f"{name}.xml"; p.parent.mkdir(parents=True, exist_ok=True)
    tree.write(p)
    return p, name


class RingBackend:
    """Same interface as isaac_port MJBackend, plain mj_step (implicitfast), no contact-frame fix."""
    name = "MuJoCo-ring"

    def __init__(self, cfg, xml, timestep=None):
        self.cfg = cfg
        m = self.m = mujoco.MjModel.from_xml_path(str(xml))
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * cfg.tau_max
        for k in range(3):
            j = m.joint(f"wheel_joint_{k}").dofadr[0]
            m.dof_armature[j] = cfg.armature; m.dof_damping[j] = cfg.damping
        self.d = mujoco.MjData(m)
        self.bb, self.kb = m.body("base").id, m.body("ball").id
        self.bq = m.joint("base_free_joint").qposadr[0]; self.bd = m.joint("base_free_joint").dofadr[0]
        self.kd = m.joint("ball_free_joint").dofadr[0]; self.kq = m.joint("ball_free_joint").qposadr[0]
        self.wdofs = np.array([m.joint(f"wheel_joint_{k}").dofadr[0] for k in range(3)])
        self.ball_geom = m.geom("the_ball").id
        self.total_mass = float(m.body_mass[1:].sum())
        self._settled = None
        self.reset(0.0)

    def reset(self, yaw=0.0, drop=0.0):
        m, d = self.m, self.d
        if self._settled is None:
            mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
            for _ in range(120):
                mujoco.mj_step(m, d)
            self._settled = (d.qpos[self.bq + 2], d.qpos[self.kq + 2])
        mujoco.mj_resetData(m, d)
        d.qpos[self.bq + 3:self.bq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.bq + 2] = self._settled[0] + drop; d.qpos[self.kq + 2] = self._settled[1] + drop
        mujoco.mj_forward(m, d)

    def state(self):
        d = self.d
        R = d.xmat[self.bb].reshape(3, 3).copy(); Rb = d.xmat[self.kb].reshape(3, 3)
        return dict(R=R, wb_w=R @ d.qvel[self.bd + 3:self.bd + 6], wk_w=Rb @ d.qvel[self.kd + 3:self.kd + 6],
                    vel=d.cvel[self.bb][3:].copy(), pbody=d.xpos[self.bb].copy(),
                    pball=d.xpos[self.kb] + Rb @ self.m.geom_pos[self.ball_geom], wheel_w=d.qvel[self.wdofs].copy(), t=float(d.time))

    def step_ctrl(self, tau, push=None):
        d = self.d
        d.ctrl[:] = tau
        d.xfrc_applied[self.bb, :3] = push if push is not None else 0.0
        for _ in range(REPEAT):
            mujoco.mj_step(self.m, d)
        d.xfrc_applied[self.bb, :3] = 0.0


def speed_detail(be, cfg, dirs=4, steps=400):
    """Like scenarios.speed_table but also returns per-heading outcomes: ratio_<sp> (mean over non-fallen), falls_<sp>, hd_<sp> = e.g. 'F.F.' (F fell, . held)
    and fell step counts."""
    row = {}
    for sp in S.SPEEDS:
        ratios, flags, tf = [], "", []
        for k in range(dirs):
            ang = k * np.pi / 2
            cmd = sp * cfg.f * np.array([np.cos(ang), np.sin(ang), 0.0])
            fell, v = S.run_drive(be, cfg, cmd, steps)
            flags += "F" if fell else "."
            if fell:
                tf.append(len(v))
            else:
                ratios.append(float(v[-150:].mean(0) @ cmd[:2] / (cmd[:2] @ cmd[:2])))
        row[f"ratio_{sp}"] = float(np.mean(ratios)) if ratios else np.nan
        row[f"falls_{sp}"] = flags.count("F")
        row[f"hd_{sp}"] = flags
        row[f"fellstep_{sp}"] = "/".join(map(str, tf))
    return row


def protocol(be, cfg, trials=16, dirs=4):
    t0 = time.time()
    row = {}
    row.update(speed_detail(be, cfg, dirs=dirs))
    row.update(S.push_table(be, cfg, trials=trials))
    row["wall_s"] = round(time.time() - t0, 1)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--rho", type=float, default=0.012)
    ap.add_argument("--damp", type=float, default=0.0); ap.add_argument("--soft", type=int, default=0)
    ap.add_argument("--phase", type=float, default=0.0); ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--rim", type=float, default=None); ap.add_argument("--tag", default=None)
    ap.add_argument("--ideal", action="store_true"); ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="build + 100 control steps only")
    a = ap.parse_args()
    lowprio()
    cfg = DesignCfg("nom")
    if a.ideal:
        from backend_mj import MJBackend
        be = MJBackend(cfg); name = "ideal"
    else:
        xml, name = make_xml(a.n, a.rho, a.damp, bool(a.soft), a.phase, rim_r=a.rim, mu=a.mu, tag=a.tag)
        be = RingBackend(cfg, xml)
        print("mass total", be.total_mass, "nbody", be.m.nbody, "ngeom", be.m.ngeom, flush=True)
    if a.smoke:
        t0 = time.time()
        fell, v = S.run_drive(be, cfg, np.array([0.4 * cfg.f, 0, 0]), 400)
        print("smoke fell", fell, "steps", len(v), "ratio", None if fell else float(v[-150:, 0].mean() / (0.4 * cfg.f)), "time", round(time.time() - t0, 1))
        sys.exit()
    row = dict(name=name, n=a.n, rho=a.rho, damp=a.damp, soft=a.soft, phase=a.phase, mu=a.mu, rim=a.rim)
    row.update(protocol(be, cfg, a.trials))
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    print("ROW " + json.dumps(row, default=float), flush=True)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
