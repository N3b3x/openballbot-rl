"""Physical feasibility constraints for ballbot designs, as margins (>= 1 means satisfied).

Every constraint is derived from the planar model of design_space.py plus elementary contact mechanics:

  C_struct   z > p            the NMP zero must lie above the unstable pole, else no linear controller stabilises x.
                              Reported as the sensitivity floor Ms >= (z+p)/(z-p) * exp(p T): a SOFT limit that
                              diverges as z -> p. Margin = Ms_ref / Ms_floor with Ms_ref = 20 (about 26 dB).
  C_delay    exp(p T) <= 2    total loop delay T (sample/2 + compute + sensing + actuator lag) must satisfy
                              T <= ln 2 / p; margin = (ln 2 / p) / T_loop.
  C_trac     wheel tangential force <= mu N : per-wheel torque <= mu N r_w, with N = m_a g /(3 cos zeta)
                              (zeta = wheel-ball contact zenith angle from the pole). The transmissible planar torque on
                              the ball is tau_plane = tau_w * |n|^2 / max|n|. Margin = tau_plane_trac / (m_a g l sin(15 deg)):
                              can traction recover a 15 degree tilt?
  C_motor    same ratio with the motor torque limit (what the actuators could do).
  C_lean     lean needed for the target acceleration a* is kappa a*, must stay below 20 deg.
  C_geom     zeta in [35, 75] deg: below, the wheels cannot constrain the ball against the floor and the body is wide; above,
                              the wheels approach the equator, lose lever arm and the ball can be ejected.
  C_speed    v_max = r omega_max / max_phi |W_phi| (wheel speed limit) >= target speed.

  python scripts/study/design_constraints.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "study"))
import design_space as ds  # noqa: E402

G = 9.81
DEFAULTS = dict(mu=1.0, tilt_rec_deg=15.0, a_target=1.5, v_target=1.0, T_loop=0.015, omega_max=150.0, Ms_ref=20.0)


def constraints(design, **kw):
    c = {**DEFAULTS, **kw}
    m, P, A, B, n = ds.analytic(design, keep=True)
    W = np.array(P["W"])
    zeta = np.radians(P["zenith_deg"])
    r_w = np.mean([x["r_w"] for x in P["contacts"]])
    N = P["m_a"] * G / (3 * np.cos(zeta))                       # static normal force per wheel
    tw_trac = c["mu"] * N * r_w                                 # slip-free wheel torque
    fac = float(n @ n) / float(np.max(np.abs(n)))               # planar ball torque per unit wheel torque (min-norm split)
    tau_trac = tw_trac * fac
    mgl_sin = P["m_a"] * G * P["l"] * np.sin(np.radians(c["tilt_rec_deg"]))
    d = {**ds.NOMINAL, **design}
    tau_motor = m["tau_plane_max"]
    # speed limit: wheel rates for a ball rolling at v in direction phi: omega_w = W[:, :2] @ (v/r)(-sin phi, cos phi)
    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    gain = max(np.max(np.abs(W[:, :2] @ np.array([-np.sin(p), np.cos(p)]))) for p in phis)
    v_max = P["r"] * c["omega_max"] / gain
    floor = m["Ms_lb"] * np.exp(m["p"] * c["T_loop"])
    out = dict(name=m["name"], zeta_deg=P["zenith_deg"], r_w=r_w, N_wheel=N, tau_w_trac=tw_trac, tau_plane_trac=tau_trac,
               tau_plane_motor=tau_motor, mgl=m["mgl"], p=m["p"], z=m["z"], Ms_floor=floor,
               C_struct=c["Ms_ref"] / floor, C_delay=np.log(2) / m["p"] / c["T_loop"], T_loop_max=np.log(2) / m["p"],
               C_trac=tau_trac / mgl_sin, C_motor=tau_motor / mgl_sin,
               C_lean=np.radians(20.0) / (np.radians(m["kappa_deg"]) * c["a_target"]),
               C_geom=min((P["zenith_deg"] - 35.0) / 10.0, (75.0 - P["zenith_deg"]) / 10.0) / 1.0 + 1.0 if 35 <= P["zenith_deg"] <= 75 else 0.0,
               C_speed=v_max / c["v_target"], v_max=v_max,
               rec_tilt_trac_deg=float(np.degrees(np.arcsin(min(1.0, tau_trac / (P["m_a"] * G * P["l"]))))),
               a_trac=float(c["mu"] * G))                       # ball-floor horizontal acceleration limit (floor mu ~ 1)
    out["binding"] = min(("C_struct", "C_delay", "C_trac", "C_motor", "C_lean", "C_geom", "C_speed"), key=lambda k: out[k])
    out["feasible"] = all(out[k] >= 1.0 for k in ("C_struct", "C_delay", "C_trac", "C_motor", "C_lean", "C_geom", "C_speed"))
    return out


def run():
    designs = [dict()]
    designs += [dict(dz=v) for v in (-15, -10, -5, 5, 10, 15, 20, 24)]
    designs += [dict(s=v) for v in (0.5, 0.75, 1.5, 2.0)]
    designs += [dict(zb=v) for v in (-0.05, 0.1, 0.2, 0.3)]
    designs += [dict(kbal=v) for v in (0.25, 0.5, 2.0, 4.0)]
    designs += [dict(kball=v) for v in (0.25, 4.0, 8.0)]
    designs += [dict(tau=v) for v in (1.5, 3.0, 6.0, 12.0)]
    rows = [constraints(d) for d in designs]
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/study/results/design_constraints.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = ["name", "zeta_deg", "N_wheel", "tau_plane_trac", "tau_plane_motor", "mgl", "rec_tilt_trac_deg", "C_struct", "C_delay", "C_trac", "C_motor", "C_lean", "C_speed", "binding", "feasible"]
    print(df[cols].round(2).to_string(index=False))
    return df


if __name__ == "__main__":
    run()
