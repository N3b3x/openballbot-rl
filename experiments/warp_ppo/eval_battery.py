"""Standard evaluation battery for one run (Warp, simulation only). Calls eval_tiles2.py with every setting passed explicitly and writes
outputs/study/warp_ppo/<run>/battery.json plus a printed summary. All result files carry their settings in the file name and in env_settings.

  python eval_battery.py --run g9_dr_s0 [--dr-scales 1 1.75 2.5] [--skip-flat]

Protocols (see docs/research/tracking_study_log.md S59, S60, S54):
  flat   : flat 24 m world, nominal design, no DR, no disturbances, 64 worlds per speed (two nominal copies of 32), 800 steps, steady state after 3 s, speeds 0.3 0.6 0.9
  goal   : goal 1.1 m, 11 designs, DR + surfaces, 16 worlds per design and terrain kind, pushes + pulls on and off, for each DR scale
  vel    : velocity 0.3 m/s on tiles, DR + surfaces, pushes + pulls on and off
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "outputs" / "study" / "warp_ppo"
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--dr-scales", type=float, nargs="+", default=[1.0])
ap.add_argument("--skip-flat", action="store_true"); ap.add_argument("--flat-only", action="store_true", help="re-run only the flat benchmark and merge it into an existing battery.json"); ap.add_argument("--skip-vel", action="store_true"); ap.add_argument("--n", type=int, default=16)
a = ap.parse_args()
SUF = f"_mu{os.environ['PAIR_MU1']}" if os.environ.get("PAIR_MU1") else ""      # wheel-ball free-rolling friction override (S98): separate result files


def call(extra_env, args):
    env = dict(os.environ, TRAIN_MOD="train_dev", **extra_env)
    subprocess.run([PY, str(HERE / "eval_tiles2.py"), "--run", a.run, *args], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def load(tag):
    f = RUN / a.run / tag
    return json.load(open(f)) if f.exists() else None


out = dict(run=a.run, time=time.strftime("%Y-%m-%d %H:%M:%S"), flat={}, goal={}, vel={})
if a.flat_only:
    prev = load(f"battery{SUF}.json")
    if prev:
        out["goal"], out["vel"] = prev["goal"], prev["vel"]
    a.skip_vel = True
if not a.skip_flat:
    for sp in (0.3, 0.6, 0.9):
        call(dict(TILES_FLAT="1", EVAL_NOEXIT="1"), ["--task", "vel", "--speed", str(sp), "--n", "32", "--eval-designs", "nom,nom"])
        d = load(f"eval2_vel_{sp}_flat_noexit{SUF}.json")
        if d:
            f = d["by_kind"]["flat"]
            out["flat"][str(sp)] = dict(err=f["late_err"], achieved=f["late_speed_ratio"], falls=f["fall"], falls_at_map_edge=f.get("fall_at_map_edge", 0.0), worlds=64)
            print(f"{a.run} FLAT cmd {sp}: err {f['late_err']:.3f} achieved/command {f['late_speed_ratio']:.2f} falls {f['fall']:.2f}", flush=True)
for sc in ([] if a.flat_only else a.dr_scales):
    for ps, fl, name in ((1, ["--pull"], "pushes+pulls"), (0, [], "none")):
        envd = dict(PUSH_SCALE=str(ps))
        if sc != 1.0:
            envd["DR_SCALE"] = str(sc)
        call(envd, ["--task", "goal", "--n", str(a.n), "--dr", "--surfaces", *fl])
        tag = f"eval2_goal_goal_dr_surf{'_pull' if fl else ''}{'' if ps == 1 else '_ps0'}{'' if sc == 1.0 else f'_drs{sc:g}'}{SUF}.json"
        d = load(tag)
        if d:
            o = d["overall"]
            out["goal"][f"drs{sc:g}_{name}"] = dict(overall=o, by_kind=d["by_kind"], env=d.get("env_settings"))
            print(f"{a.run} GOAL dr_scale {sc} {name}: falls {o['fall']:.2f} success {o['success']:.2f} time {o.get('median_time_s')}", flush=True)
if not a.skip_vel:
    for sc in a.dr_scales:
        for ps, fl, name in ((1, ["--pull"], "pushes+pulls"), (0, [], "none")):
            envd = dict(PUSH_SCALE=str(ps))
            if sc != 1.0:
                envd["DR_SCALE"] = str(sc)
            call(envd, ["--task", "vel", "--speed", "0.3", "--n", str(a.n), "--dr", "--surfaces", *fl])
            tag = f"eval2_vel_0.3_dr_surf{'_pull' if fl else ''}{'' if ps == 1 else '_ps0'}{'' if sc == 1.0 else f'_drs{sc:g}'}{SUF}.json"
            d = load(tag)
            if d:
                o = d["overall"]
                out["vel"][f"drs{sc:g}_{name}"] = dict(overall=o, env=d.get("env_settings"))
                print(f"{a.run} VEL0.3 dr_scale {sc} {name}: falls {o['fall']:.2f} err {o['track_err']:.3f}", flush=True)
json.dump(out, open(RUN / a.run / f"battery{SUF}.json", "w"), indent=1)
print(f"saved battery{SUF}.json", flush=True)
