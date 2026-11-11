"""Record scenario trajectories (poses, wheel angles, wheel-ball contact forces, HUD metrics) from any backend to npz.

  python record.py --engine mj --design nom --scenario track04
Scenarios: settle (LQR hold, ball drops 3 cm), pulse (open-loop torque pulse), track04, track08, push (1x weight, 50 ms).
"""
import argparse
from pathlib import Path

import numpy as np

import scenarios as S
from common import DT_CTRL, ROOT, DesignCfg, LQRCore, tilt_deg
from sweep import make_backend

OUT = ROOT / "outputs/study/isaac_port/rec"

PUSH_PHI = float(np.radians(35.0))


def record(be, cfg, scen, push_phi=PUSH_PHI):
    ctrl = LQRCore(cfg)
    be.reset(0.0, 0.02 if scen == 'settle' else 0.0)
    f = cfg.f
    steps = dict(settle=300, pulse=60, track04=500, track08=500, push=450)[scen]
    cmd = np.zeros(3)
    if scen == "track04":
        cmd = np.array([0.4 * f, 0.0, 0.0])
    if scen == "track08":
        cmd = np.array([0.8 * f, 0.0, 0.0])
    if scen == "push":
        cmd = np.array([0.0, 0.4 * f, 0.0])
    F = cfg.m_total * 9.81 * np.array([np.cos(push_phi), np.sin(push_phi), 0.0])
    rows = []
    fell_at = None
    for t in range(steps):
        st = be.state()
        if tilt_deg(st["R"]) > S.FALL_DEG and fell_at is None:
            fell_at = t
        if scen == "pulse":
            on = 10 <= t < 20
            tau = np.array([0.5, 0.0, -0.5]) * (1.0 if on else 0.0)
        else:
            tau = ctrl.act(st, cmd)
        push = F if (scen == "push" and 150 <= t < 155) else None
        pose = be.pose()
        cons = be.wheel_ball_forces()
        rows.append(dict(t=st["t"], tilt=tilt_deg(st["R"]), speed=float(np.linalg.norm(st["vel"][:2])),
                         cmd=float(np.linalg.norm(cmd[:2])), push=float(np.linalg.norm(push)) if push is not None else 0.0,
                         tau=tau, pose=pose, cons=cons, ballxy=st["pball"][:2], torque_on=bool(np.any(tau != 0))))
        be.step_ctrl(tau, push)
        if fell_at is not None and t - fell_at > 60:
            break
    return rows, fell_at


def pack(rows):
    n = len(rows)
    d = dict(t=np.array([r["t"] for r in rows]), tilt=np.array([r["tilt"] for r in rows]),
             speed=np.array([r["speed"] for r in rows]), cmd=np.array([r["cmd"] for r in rows]),
             push=np.array([r["push"] for r in rows]), tau=np.array([r["tau"] for r in rows]),
             base_pos=np.array([r["pose"]["base_pos"] for r in rows]), base_quat=np.array([r["pose"]["base_quat"] for r in rows]),
             ball_pos=np.array([r["pose"]["ball_pos"] for r in rows]), ball_quat=np.array([r["pose"]["ball_quat"] for r in rows]),
             wheel_q=np.array([r["pose"]["wheel_q"] for r in rows]),
             ball_center=np.array([r["pose"]["ball_center"] for r in rows]))
    # contacts padded to 3 per frame: point (3), force on ball side (3), normal N, friction N, present flag
    C = np.zeros((n, 3, 9))
    for i, r in enumerate(rows):
        for (w, p, F, fn, ft) in r["cons"]:
            C[i, w % 3, :] = [*p, *F, fn, ft, 1.0]
    d["contacts"] = C
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mj")
    ap.add_argument("--design", default="nom")
    ap.add_argument("--scenario", nargs="+", default=["settle", "pulse", "track04", "track08", "push"])
    ap.add_argument("--fric", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--push-phi", type=float, default=35.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = DesignCfg(a.design)
    be = make_backend(a.engine, cfg, a.fric)
    for sc in a.scenario:
        rows, fell = record(be, cfg, sc, np.radians(a.push_phi))
        d = pack(rows)
        np.savez_compressed(OUT / f"{a.engine}{a.tag}_{a.design}_{sc}.npz", fell_at=-1 if fell is None else fell, **d)
        print(a.engine, a.design, sc, "frames", len(rows), "fell_at", fell, flush=True)


if __name__ == "__main__":
    main()
