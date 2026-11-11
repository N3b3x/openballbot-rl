"""Export the MuJoCo-derived ballbot geometry the browser needs (site/data/geometry_table.json).

The Math lab / Design space pages re-implement the planar model in JavaScript. The model parameters
(r, m_k, I_k, m_a, l, I_a, J_eff, wheel contact geometry) come from the MuJoCo XML through
scripts/study/design_space.py. Most of the dependence on the design knobs is closed form:

  s     lengths x s, masses x s^3, inertias x s^5, wheel torque x s^4, damping x s^2.5 (exact geometric similarity)
  kball ball mass and inertia are linear in the ball density multiplier
  kbal  ballast mass linear; zb moves the ballast along z (its moment and inertia about the ball centre follow)
  J     J_eff = J s^5 |n|^2, with |n|^2 the planar wheel-transmission norm (depends on dz only)

so this script only samples the one knob that changes the wheel geometry (contact zenith change dz) and, for each
dz, decomposes the body into a ballast-free part and a ballast part by solving linear fits over kbal and zb.
It verifies the closed forms against direct MuJoCo extraction before writing.

  OPENBLAS_NUM_THREADS=1 python scripts/site/geometry_table.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "study"))
import design_space as ds  # noqa: E402

TMP = ROOT / "ballbot_gym" / "models" / "_site_tmp"
DZ_GRID = list(range(-15, 25, 1))
ZB_NOM = ds.NOMINAL["zb"]


def params(design):
    TMP.mkdir(exist_ok=True)
    d = {**ds.NOMINAL, **design}
    xml = TMP / "v.xml"
    ds.make_variant_xml(d, xml)
    P, n, _ = ds.model_params(xml, d)
    W = np.array(P["W"])
    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    gain = max(np.max(np.abs(W[:, :2] @ np.array([-np.sin(p), np.cos(p)]))) for p in phis)
    nn = float(n @ n)
    return dict(P=P, nn=nn, fac=nn / float(np.max(np.abs(n))), gain=float(gain),
                r_w=float(np.mean([c["r_w"] for c in P["contacts"]])))


def moments(P):
    """Sums about the ball centre: mass, first moment, second moment (pitch plane)."""
    return np.array([P["m_a"], P["m_a"] * P["l"], P["I_a"] + P["m_a"] * P["l"] ** 2])


def decompose(dz):
    base = {"dz": float(dz)}
    runs = {k: params({**base, **kw}) for k, kw in
            dict(b05=dict(kbal=0.5), b1=dict(), b2=dict(kbal=2.0), zlo=dict(zb=-0.05), zhi=dict(zb=0.10)).items()}
    S = {k: moments(v["P"]) for k, v in runs.items()}
    # linear in kbal: S(k) = S_body + k S_ballast
    Sb = (S["b2"] - S["b05"]) / 1.5
    S0 = S["b1"] - Sb
    # check linearity (b05)
    assert np.allclose(S0 + 0.5 * Sb, S["b05"], rtol=1e-9), "not linear in kbal"
    mb = float(Sb[0])
    hb = float(Sb[1] / mb)                      # ballast COM height above the ball centre at zb = nominal
    Ib = float(Sb[2] - mb * hb ** 2)            # ballast inertia about its own COM
    # ballast height follows zb one-to-one
    for key, zb in (("zlo", -0.05), ("zhi", 0.10)):
        pred = S0 + Sb * 0 + np.array([mb, mb * (hb + zb - ZB_NOM), Ib + mb * (hb + zb - ZB_NOM) ** 2])
        assert np.allclose(pred, S[key], rtol=1e-7), f"zb law fails {key} {pred} {S[key]}"
    g = runs["b1"]
    P = g["P"]
    return dict(dz=float(dz), zeta_deg=P["zenith_deg"], r_w=g["r_w"], nn=g["nn"], fac=g["fac"], gain=g["gain"],
                r=P["r"], m_k=P["m_k"], I_k=P["I_k"], S0=S0.tolist(), mb=mb, hb=hb, Ib=Ib,
                m_a=P["m_a"], l=P["l"], I_a=P["I_a"], J_w=P["J_w"], d_w=P["d_w"])


def main():
    rows = [decompose(dz) for dz in DZ_GRID]
    # scaling-law verification against direct extraction at several (s, kball, kbal, zb, J, dz)
    checks = []
    for dsg in (dict(s=0.5), dict(s=1.5), dict(kball=4.0), dict(kbal=0.25, zb=0.2), dict(J=0.02), dict(dz=9.0, s=0.75, kbal=2.0)):
        g = params(dsg)
        d = {**ds.NOMINAL, **dsg}
        row = min(rows, key=lambda r_: abs(r_["dz"] - d["dz"]))
        s = d["s"]
        m_a = row["S0"][0] + d["kbal"] * row["mb"]
        S1 = row["S0"][1] + d["kbal"] * row["mb"] * (row["hb"] + d["zb"] - ZB_NOM)
        S2 = row["S0"][2] + d["kbal"] * (row["Ib"] + row["mb"] * (row["hb"] + d["zb"] - ZB_NOM) ** 2)
        l = S1 / m_a
        I_a = S2 - m_a * l ** 2
        pred = dict(r=row["r"] * s, m_k=row["m_k"] * d["kball"] * s ** 3, m_a=m_a * s ** 3, l=l * s, I_a=I_a * s ** 5,
                    J_eff=d["J"] * s ** 5 * row["nn"])
        got = dict(r=g["P"]["r"], m_k=g["P"]["m_k"], m_a=g["P"]["m_a"], l=g["P"]["l"], I_a=g["P"]["I_a"], J_eff=g["P"]["J_eff"])
        err = {k: abs(pred[k] / got[k] - 1) for k in got}
        checks.append(dict(design=dsg, max_rel_err=max(err.values()), err=err))
        assert max(err.values()) < 2e-3, (dsg, err)
    shutil.rmtree(TMP, ignore_errors=True)
    out = dict(
        generated_by="scripts/site/geometry_table.py",
        source="scripts/study/design_space.py (MuJoCo XML geometry), ballbot_gym/models/ballbot.xml",
        nominal=ds.NOMINAL, ballast_m0=ds.BALLAST_M0, zb_nominal=ZB_NOM, G=9.81,
        scaling_laws=dict(length="x s", mass="x s^3", inertia="x s^5", J_eff="J s^5 |n|^2", d_eff="damp s^2.5 |n|^2",
                          tau_plane_max="tau s^4 |n|^2/max|n|", p="p ~ s^-1/2", z="z ~ s^-1/2", kappa="invariant",
                          m_k="m_k0 kball s^3", I_k="I_k0 kball s^5",
                          body="m_a = (S0_0 + kbal mb) s^3; m_a l = S0_1 + kbal mb (hb + zb - zb_nom); "
                               "m_a l^2 + I_a = S0_2 + kbal (Ib + mb (hb + zb - zb_nom)^2)  (all at s=1, about the ball centre)"),
        dz_grid=DZ_GRID, rows=rows, verification=checks)
    outp = ROOT / "site" / "data" / "geometry_table.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1)
    print("wrote", outp, "rows", len(rows))
    for c in checks:
        print(c["design"], f"max rel err {c['max_rel_err']:.2e}")


SIM_DESIGNS = {   # dz_lqr.csv design tag -> design dict (from ballbot_gym/models/variants/dz_*.json metrics)
    "nom": dict(), "s0.5": dict(s=0.5), "s1.5": dict(s=1.5), "zlow": dict(zb=-0.05), "zhigh": dict(zb=0.15),
    "kb4": dict(kball=4.0), "kb025": dict(kball=0.25), "bal025": dict(kbal=0.25), "bal4": dict(kbal=4.0),
    "tau3": dict(tau=3.0), "tau6": dict(tau=6.0), "J02": dict(J=0.02), "J0005": dict(J=0.0005),
    "zeta41": dict(dz=-10.0), "zeta66": dict(dz=15.0), "nom_motor": dict()}


def tables():
    """Copy the CSV evidence the pages show into one JSON (so the site never reads outside site/)."""
    import pandas as pd
    res = ROOT / "outputs" / "study" / "results"

    def rows(name, cols=None):
        df = pd.read_csv(res / name)
        if cols:
            df = df[cols]
        return json.loads(df.to_json(orient="records"))   # NaN -> null

    sim = rows("dz_lqr.csv") + rows("dz_lqr_motor.csv")
    for r_ in sim:
        r_["design_params"] = SIM_DESIGNS[r_["design"]]
    out = dict(
        generated_by="scripts/site/geometry_table.py",
        sources=["outputs/study/results/design_sweep.csv", "outputs/study/results/design_constraints.csv",
                 "outputs/study/results/dz_lqr.csv", "outputs/study/results/dz_lqr_motor.csv"],
        sweep=rows("design_sweep.csv"), constraints=rows("design_constraints.csv"), sim_lqr=sim,
        protocol="scripts/study/eval_design.py: speeds {0.4,0.8,1.1} sqrt(s) m/s in 4 headings (falls out of 4); "
                 "one 50 ms horizontal push of alpha x weight, alpha in {0.5,1,1.5}, random direction (fraction of trials that fell; "
                 "8 trials, 6 for zeta41); controller = LQR designed on the nominal planar model structure for that design, zero-shot")
    outp = ROOT / "site" / "data" / "designlab_tables.json"
    json.dump(out, open(outp, "w"), indent=1, allow_nan=False)
    print("wrote", outp)


if __name__ == "__main__":
    main()
    tables()
