"""Ballbot design space: parametric MuJoCo variants + analytic design metrics from the derived planar model.

A design is a dict with
  s      uniform geometric scale (lengths x s, densities fixed -> masses x s^3, inertias x s^5)
  kball  ball mass multiplier on top of the geometric scaling (ball density)
  kbal   ballast (body mass) multiplier
  zb     ballast height above the base origin at s=1 [m]  (moves the body COM l and the pitch inertia I_a)
  J      wheel rotor inertia at s=1 [kg m^2] (scaled by s^5)
  tau    per-wheel torque limit at s=1 [N m] (scaled by s^4; action scale of the env)
  damp   wheel viscous damping [N m s/rad]
  dz     change of the wheel-ball contact zenith angle [deg] (nominal 51 deg): rotates each wheel assembly about the ball centre

  python scripts/study/design_space.py build nominal --s 1.0           # writes xml + model json + env yaml
  python scripts/study/design_space.py sweep --out outputs/study/results/design_sweep.csv
"""
import argparse
import copy
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
import ballbot_model as bm  # noqa: E402

BASE_XML = ROOT / "ballbot_gym/models/ballbot.xml"
VAR_DIR = ROOT / "ballbot_gym/models/variants"
NOMINAL = dict(s=1.0, kball=1.0, kbal=1.0, zb=0.002, J=0.005, tau=30.0, damp=0.1, dz=0.0)
BALLAST_M0 = 400.0 * 8 * 0.1 ** 3          # nominal ballast mass [kg]
BALL_R0 = 0.09


def _scale_attr(el, key, s):
    if key in el.attrib:
        el.set(key, " ".join(f"{float(v) * s:.8g}" for v in el.get(key).split()))


def _rotate_wheels(root, dz_deg, s):
    """Rotate every wheel assembly (capsule pose, joint axis and position) about the ball centre so that the contact
    zenith angle changes by dz_deg. Done in each wheel body's own frame (the azimuth rotation is about z)."""
    from scipy.spatial.transform import Rotation as Rot
    c = np.array([0.0, 0.0, -0.119 * s])          # ball centre in the wheel-body frame
    for body in root.iter("body"):
        if not body.get("name", "").startswith("wheel_"):
            continue
        g = next(x for x in body.iter("geom"))
        j = next(x for x in body.iter("joint"))
        pos = np.array([float(v) for v in g.get("pos").split()])
        u = pos - c
        n = np.cross([0, 0, 1.0], [u[0], u[1], 0.0])
        n /= np.linalg.norm(n)
        R = Rot.from_rotvec(n * np.radians(dz_deg))
        g.set("pos", " ".join(f"{v:.8g}" for v in c + R.apply(pos - c)))
        eul = [float(v) for v in g.get("euler").split()]
        q = (R * Rot.from_euler("xyz", eul, degrees=True)).as_quat()      # x y z w
        del g.attrib["euler"]
        g.set("quat", f"{q[3]:.8g} {q[0]:.8g} {q[1]:.8g} {q[2]:.8g}")
        ax = np.array([float(v) for v in j.get("axis").split()])
        j.set("axis", " ".join(f"{v:.8g}" for v in R.apply(ax)))
        jp = np.array([float(v) for v in j.get("pos").split()])
        j.set("pos", " ".join(f"{v:.8g}" for v in c + R.apply(jp - c)))


def make_variant_xml(design, path):
    """Write a scaled / re-weighted copy of ballbot.xml next to the base models (assets via ../meshes)."""
    d = {**NOMINAL, **design}
    s = d["s"]
    tree = ET.parse(BASE_XML)
    root = tree.getroot()
    root.find("compiler").set("meshdir", "../meshes")
    root.find("compiler").set("texturedir", "../textures")
    for el in root.iter():
        if el.tag in ("body", "geom", "joint", "site", "camera", "light"):
            for key in ("pos", "fromto") + (("size",) if el.tag in ("geom", "site") else ()):
                _scale_attr(el, key, s)
        if el.tag == "mesh":
            _scale_attr(el, "scale", s)
    # the hfield is the 10 m floor: keep it (do not scale)
    for el in root.iter("hfield"):
        pass
    for el in root.iter("geom"):
        if el.get("name") == "ballast":
            # restore the un-scaled floor-independent hfield; set ballast mass and height explicitly
            el.set("density", f"{BALLAST_M0 * d['kbal'] * s ** 3 / (8 * (0.1 * s) ** 3):.8g}")
            pos = el.get("pos").split()
            el.set("pos", f"{float(pos[0]):.8g} {float(pos[1]):.8g} {d['zb'] * s:.8g}")
        if el.get("name") == "the_ball":
            el.set("density", f"{float(el.get('density')) * d['kball']:.8g}")
        if el.get("name") == "terrain":
            # the floor geom was scaled above; undo for hfield geoms
            el.set("pos", "0 0 0")
            el.set("size", "5 5 0.1")
    if d["dz"]:
        _rotate_wheels(root, d["dz"], s)
    for el in root.iter("joint"):
        if el.get("name", "").startswith("wheel_joint"):
            el.set("armature", f"{d['J'] * s ** 5:.8g}")
            el.set("damping", f"{d['damp'] * s ** 2.5:.8g}")
    # hfield and the world view camera: undo scaling of the hfield size (a size attr on hfield was not touched)
    for el in root.iter("light"):
        el.set("pos", "0 0 10.0")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path)
    return d


def name_of(design):
    d = {**NOMINAL, **design}
    return "d_" + "_".join(f"{k}{d[k]:g}" for k in ("s", "kball", "kbal", "zb", "J", "tau", "damp", "dz") if d[k] != NOMINAL[k]) \
        if any(d[k] != NOMINAL[k] for k in NOMINAL) else "d_nominal"


def model_params(xml_path, design):
    d = {**NOMINAL, **design}
    P = bm.extract_params(wheel_damping=d["damp"] * d["s"] ** 2.5, armature=d["J"] * d["s"] ** 5, xml=xml_path)
    n, J_eff, d_eff, tau_plane_max = bm.planar_effective(P)
    P.update(J_eff=J_eff, d_eff=d_eff)
    return P, n, tau_plane_max


_DER = None


def linear_model(P):
    global _DER
    if _DER is None:
        _DER = bm.derive_planar()
    A, B, *_ = bm.linearize(_DER, P)
    return A, B


def analytic(design, xml_path=None, keep=False):
    """Design metrics from the linearised planar model (no simulation)."""
    d = {**NOMINAL, **design}
    xml_path = xml_path or (VAR_DIR / f"{name_of(d)}.xml")
    make_variant_xml(d, xml_path)
    P, n, tau_plane_max = model_params(xml_path, d)
    A, B = linear_model(P)
    S = bm.structural(A, B, P)
    poles = S["poles"]
    p = float(max(poles.real))
    # closed-form NMP zero of tau -> ball position (matches the numerical zero; the numeric tf is ill-conditioned at some scales)
    z = float(np.sqrt(P["m_a"] * bm.G * P["l"] / (P["I_a"] + P["m_a"] * P["l"] ** 2 + P["m_a"] * P["l"] * P["r"])))
    ms_lb = (z + p) / abs(z - p)     # Skogestad: ||S||_inf >= |z+p|/|z-p| for one RHP pole p and zero z
    tau_max = tau_plane_max(d["tau"] * d["s"] ** 4)   # torque limit scales with m g l ~ s^4 (geometric similarity)
    mgl = P["m_a"] * bm.G * P["l"]
    # LQR peak torque for a 3 deg initial tilt (continuous, nominal weights) -> linear domain of attraction under the limit
    Q = np.diag([1e-6, 50.0, 1.0, 1.0])
    peak = float("nan")
    try:
        K, cl = bm.lqr(A, B, Q, np.array([[1.0]]))
        x0 = np.array([0, np.radians(3), 0, 0.0])
        ts = np.linspace(0, 1.5, 300)
        us = [abs(float((-K @ la.expm((A - B @ K) * t) @ x0)[0])) for t in ts]
        peak = max(us)
        lam_cl = float(np.max(cl.real))
    except Exception:   # noqa: BLE001
        lam_cl = float("nan")
    th_dom = np.degrees(np.radians(3) * tau_max / peak) if peak == peak and peak > 0 else float("nan")
    # steady acceleration capability under the torque limit (lean law: row eqs of the linear model, thetadd = 0)
    # phidd = A[2,1] th + B[2] u = a/r ; thetadd = A[3,1] th + B[3] u = 0  ->  th, u as functions of a
    Mx = np.array([[A[2, 1], B[2, 0]], [A[3, 1], B[3, 0]]])
    th_a, u_a = np.linalg.solve(Mx, np.array([1.0 / P["r"], 0.0]))     # per unit acceleration
    a_torque = tau_max / abs(u_a)
    tilt_cap = np.radians(20.0)
    a_lean = tilt_cap / abs(th_a)
    out = dict(**{k: d[k] for k in NOMINAL}, name=name_of(d), r=P["r"], m_k=P["m_k"], m_a=P["m_a"], l=P["l"], I_a=P["I_a"],
               J_eff=P["J_eff"], mass_ratio=P["m_a"] / P["m_k"], l_over_r=P["l"] / P["r"], p=p, z=z, z_over_p=z / p if z == z else np.nan,
               Ms_lb=ms_lb, kappa_deg=float(np.degrees(abs(th_a))), tau_plane_max=tau_max, mgl=mgl,
               torque_authority=tau_max / mgl, lqr_peak_3deg=peak, tilt_domain_deg=th_dom, a_torque=a_torque, a_lean=a_lean,
               a_max=min(a_torque, a_lean), lam_cl=lam_cl, total_mass=P["m_a"] + P["m_k"])
    if keep:
        return out, P, A, B, n
    return out


def context(m):
    """Normalised morphology context vector seen by a morphology-conditioned policy (O(1) entries)."""
    return [round(float(np.log2(m["s"])), 4), round(float((np.log10(m["mass_ratio"]) - 1.36) / 0.5), 4),
            round(float((m["l_over_r"] - 1.45) / 0.5), 4), round(float((np.log10(m["torque_authority"]) - 1.54) / 0.5), 4)]


def build(design, tag=None):
    """Write the variant xml, a model json (for LQR), and an env yaml based on robs_c; returns paths."""
    d = {**NOMINAL, **design}
    nm = tag or name_of(d)
    xml = VAR_DIR / f"{nm}.xml"
    out, P, A, B, n = analytic(d, xml, keep=True)
    out["ctx"] = context(out)
    # json for LQRPolicy (same structure as ballbot_planar_model.json)
    Q = np.diag([1e-6, 50.0, 1.0, 1.0])
    mj = dict(params={k: v for k, v in P.items() if k != "contacts"}, contacts=P["contacts"], n=n.tolist(), A=A.tolist(),
              B=B.tolist(), poles=[], zeros=[], lean_per_accel=out["kappa_deg"], ctx=out["ctx"], metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in out.items() if k != "ctx"})
    mj_path = VAR_DIR / f"{nm}.json"
    json.dump(mj, open(mj_path, "w"), default=str)
    cfg = yaml.safe_load(open(ROOT / "configs/env/robs_c.yaml"))
    cfg["env"]["max_wheel_velocity"] = float(d["tau"] * d["s"] ** 4)
    cfg["env"]["wheel_damping"] = float(d["damp"] * d["s"] ** 2.5)
    cfg["env"]["wheel_armature"] = float(d["J"] * d["s"] ** 5)
    cfg["env"]["lqr_model_json"] = str(mj_path)
    cfg["env"]["xml_path"] = str(xml)
    # Froude scaling of the command range: v ~ sqrt(s)
    f = d["s"] ** 0.5
    cfg["env"]["cmd_vmax"] = round(1.2 * f, 3)
    cfg["env"]["cmd_wmax"] = 1.0           # spins above 1 rad/s suppressed learning in the nominal robot
    wscale = out["total_mass"] / 4.021     # pushes scale with the robot's weight (nominal 5-20 N)
    cfg["env"]["dr"]["push_force"] = [round(5.0 * wscale, 2), round(20.0 * wscale, 2)]
    yml = ROOT / "configs/env" / f"{nm}.yaml"
    yaml.safe_dump(cfg, open(yml, "w"), sort_keys=False)
    return dict(xml=xml, json=mj_path, yaml=yml, metrics=out)


POOL_SMALL = [dict(), dict(s=0.75), dict(s=1.5), dict(zb=-0.05), dict(zb=0.1), dict(kball=4.0), dict(kball=2.0),
              dict(kbal=0.5), dict(kbal=2.0), dict(tau=6.0), dict(J=0.02), dict(J=0.002)]


def pool_config(tag, designs=POOL_SMALL, hybrid=True, residual=False, base="robs_c"):
    """Env yaml whose reset draws a design from the pool (one policy for a family of ballbots)."""
    pool = []
    for i, dsg in enumerate(designs):
        r = build(dsg, f"{tag}_m{i}")
        d = {**NOMINAL, **dsg}
        pool.append(dict(xml=str(r["xml"]), json=str(r["json"]), tau=float(d["tau"] * d["s"] ** 4),
                         damp=float(d["damp"] * d["s"] ** 2.5), armature=float(d["J"] * d["s"] ** 5), ctx=r["metrics"]["ctx"],
                         name=name_of(d)))
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{base}.yaml"))
    e = cfg["env"]
    e["morph_pool"] = pool
    e["xml_path"] = pool[0]["xml"]
    e["max_wheel_velocity"], e["wheel_damping"], e["wheel_armature"] = pool[0]["tau"], pool[0]["damp"], pool[0]["armature"]
    e["lqr_model_json"] = pool[0]["json"]
    e["cmd_vmax"], e["cmd_wmax"] = 1.2, 1.0
    if hybrid:
        e["hybrid_gate"] = True
    if residual:
        e["residual_lqr"], e["residual_scale"] = True, 0.1
    yml = ROOT / "configs/env" / f"{tag}.yaml"
    yaml.safe_dump(cfg, open(yml, "w"), sort_keys=False)
    print("wrote", yml, "with", len(pool), "designs")
    return yml


def sweep(out):
    rows = []
    ofat = []
    for s in (0.5, 0.75, 1.0, 1.5, 2.0):
        ofat.append(dict(s=s))
    for k in (0.25, 0.5, 2.0, 4.0, 8.0):
        ofat.append(dict(kball=k))
    for k in (0.25, 0.5, 2.0, 4.0):
        ofat.append(dict(kbal=k))
    for zb in (-0.05, 0.05, 0.1, 0.15, 0.2, 0.3):
        ofat.append(dict(zb=zb))
    for J in (0.0005, 0.002, 0.02):
        ofat.append(dict(J=J))
    for tau in (1.5, 3.0, 6.0, 12.0):
        ofat.append(dict(tau=tau))
    ofat.insert(0, dict())
    # two-factor grid: COM height x body/ball mass ratio  (the main passive-dynamics trade)
    grid = [dict(zb=zb, kball=kb) for zb, kb in itertools.product((-0.05, 0.002, 0.1, 0.2, 0.3), (0.5, 1, 2, 4, 8))]
    for dsg in ofat + grid:
        try:
            rows.append(analytic(dsg))
        except Exception as e:   # noqa: BLE001
            print("fail", dsg, e)
    df = pd.DataFrame(rows).drop_duplicates("name")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df[["name", "m_a", "m_k", "l", "p", "z", "z_over_p", "Ms_lb", "kappa_deg", "torque_authority", "tilt_domain_deg", "a_max"]]
          .round(3).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("tag")
    for k, v in NOMINAL.items():
        b.add_argument(f"--{k}", type=float, default=v)
    pl = sub.add_parser("pool")
    pl.add_argument("tag")
    pl.add_argument("--mode", choices=["hybrid", "residual", "ppo"], default="hybrid")
    s = sub.add_parser("sweep")
    s.add_argument("--out", default=str(ROOT / "outputs/study/results/design_sweep.csv"))
    a = ap.parse_args()
    if a.cmd == "build":
        r = build({k: getattr(a, k) for k in NOMINAL}, a.tag)
        print(json.dumps({k: str(v) for k, v in r.items() if k != "metrics"}, indent=1))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["metrics"].items()})
    elif a.cmd == "pool":
        pool_config(a.tag, hybrid=a.mode == "hybrid", residual=a.mode == "residual")
    else:
        sweep(a.out)


if __name__ == "__main__":
    main()
