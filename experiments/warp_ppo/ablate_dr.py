"""Leave-one-out ablation of the domain randomisation terms at a given master scale (Warp, goal task, undisturbed, DR + surfaces, 16 worlds per design and kind).
  python ablate_dr.py --run g9_dr_s0 --scale 2.5
Terms: M mass, F friction, T torque, D damping spread, L latency probability, N sensor noise."""
import argparse, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; RUN = HERE.parents[1] / "outputs" / "study" / "warp_ppo"
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--scale", type=float, default=2.5); ap.add_argument("--n", type=int, default=16)
a = ap.parse_args()
cases = [("all on", {}), ("no mass", {"DR_M": "0"}), ("no friction", {"DR_F": "0"}), ("no torque", {"DR_T": "0"}), ("no damping", {"DR_D": "0"}),
         ("no latency", {"DR_L": "0"}), ("no sensor noise", {"DR_N": "0"}), ("all off", {k: "0" for k in ("DR_M", "DR_F", "DR_T", "DR_D", "DR_L", "DR_N")})]
res = {}
for name, ovr in cases:
    env = dict(os.environ, TRAIN_MOD="train_dev", PUSH_SCALE="0", DR_SCALE=str(a.scale), **ovr)
    subprocess.run([sys.executable, str(HERE / "eval_tiles2.py"), "--run", a.run, "--task", "goal", "--n", str(a.n), "--dr", "--surfaces"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tag = "eval2_goal_goal_dr_surf_ps0_drs%g" % a.scale + ("_" + "".join(f"{k[3:].lower()}{v}" for k, v in ovr.items()) if ovr else "") + ".json"
    d = json.load(open(RUN / a.run / tag)); o = d["overall"]
    res[name] = dict(falls=o["fall"], success=o["success"], time=o.get("median_time_s"))
    print(f"{a.run} scale {a.scale} {name:16s} falls {o['fall']:.2f} success {o['success']:.2f} time {o.get('median_time_s')}", flush=True)
json.dump(res, open(RUN / a.run / f"ablate_dr_{a.scale:g}.json", "w"), indent=1)
