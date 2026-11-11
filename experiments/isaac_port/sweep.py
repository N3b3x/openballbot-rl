"""Per-engine parameter sweep with identical scenarios/metrics.

  python sweep.py --engine mj --designs nom zeta66 zeta41 s0.5 s1.5 kb4 kb025 bal4 bal025 J0005 J02 "nom@fric0.5" "nom@fric2"
Writes outputs/study/isaac_port/sweep_<engine>.csv and traces/<engine>_<design>.npz (step response + torque pulse).
Design token "name@fricX" scales the cross-roller (tangential) ball-wheel friction by X (MuJoCo pair friction / PhysX material).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scenarios as S
from common import ROOT, DesignCfg

OUT = ROOT / "outputs/study/isaac_port"


def make_backend(engine, cfg, fric):
    if engine == "mj":
        from backend_mj import MJBackend
        be = MJBackend(cfg)
        if fric != 1.0:
            be.m.pair_friction[:, 1] *= fric
        return be
    from backend_px import PXBackend
    return PXBackend(cfg, fric=fric, mode="naive" if engine == "pxnaive" else "roller")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mj")
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dirs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    rows = []
    for tok in a.designs:
        print('start', tok, flush=True)
        name, _, fr = tok.partition("@fric")
        fric = float(fr) if fr else 1.0
        cfg = DesignCfg(name)
        t0 = time.time()
        be = make_backend(a.engine, cfg, fric)
        row = dict(engine=a.engine, design=tok)
        row.update(S.speed_table(be, cfg, yaw_dirs=a.dirs))
        row.update(S.push_table(be, cfg, trials=a.trials))
        fell, v = S.cmd_step_response(be, cfg)
        pul = S.torque_pulse(be, cfg)
        row["step_fell"] = int(fell)
        row["step_final_ratio"] = float(v[-100:, 1].mean() / (0.4 * cfg.f)) if not fell else np.nan
        row["wall_s"] = time.time() - t0
        np.savez(OUT / "traces" / f"{a.engine}{a.tag}_{tok}.npz", step=v, pulse=pul, fell=int(fell))
        rows.append(row)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"sweep_{a.engine}{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
