"""Render a navigation trajectory recorded by eval_nav.py --record K (simulation only). Uses the drawing helpers of render_policy.py.
  python render_nav.py --run g18_ret_rs025_s1 --traj nav_oracle_rec --out c6_nav_oracle --title "Perfect map + planner"
  --sensor-view adds what the head ToF sees (sensor_view.py). The sensor yaw is the recorded one ('syaw', eval_nav.py records it now) when the
  recording has it, otherwise the training rule (command direction, body yaw when nearly stopped); recordings carry no proprioception,
  so the learned grid estimate is not shown here.
"""
import argparse, pickle, os, sys, json
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_policy as RP
import train_dev as T
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--traj", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--title", default="navigation"); ap.add_argument("--sub", default=""); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--sensor-view", action="store_true", help="overlay what the head ToF sensor sees (CPU ground-truth casts + the training noise model)")
ap.add_argument("--seed", type=int, default=0, help="sensor noise seed for --sensor-view")
a = ap.parse_args()
d = pickle.load(open(RP.RUN / a.run / f"{a.traj}_traj.pkl", "rb"))
if not os.environ.get("RETAINER"):
    ar = json.load(open(RP.RUN / a.run / "args.json"))
    if ar.get("retainer"): os.environ["RETAINER"] = ar["retainer"]
    if ar.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"): os.environ["PAIR_MU1"] = f"{ar['pair_mu1']:g}"      # contact model of the run (S101 fix)
env = T.GpuBallbot(1, seed=0, dr=False, design=d["design"], tiles=True, tile_kinds=["blocks"])
m = env.mjm; m.vis.global_.offwidth = 1280; m.vis.global_.offheight = 640
r = mujoco.Renderer(m, height=640, width=1280, max_geom=4000); md = mujoco.MjData(m)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = 2.6; cam.elevation = -38
log = d["rec"]
if log and log[-1]["fell"]:
    log = log + [log[-1]] * 30
frames = []
ref = log[0]["qpos"][10:12].copy()
for x in log:
    ref = ref + x["cmd"][:2] * 0.01; x["ref"] = ref.copy()
if a.sensor_view:
    import sensor_view as SV
    sv, _ = RP.sensor_setup(env, m, seed=a.seed, sensor_cfg=d.get("sensor"), estimator=False)
    yaw_src = "recorded sensor yaw" if "syaw" in log[0] else "command direction (recording has no sensor yaw)"
    print("sensor view: yaw from", yaw_src)
for i in range(0, len(log), a.stride):
    e = log[i]
    cam.azimuth = 135 + 20 * np.sin(i * 0.003)
    ov, ins = RP.sensor_frame(sv, None, md, m, e, SV.sensor_yaw(e["qpos"], e["cmd"], e.get("syaw"))) if a.sensor_view else (None, None)
    RP.draw_scene(r, md, m, e, [x["qpos"][10:12] for x in log[max(0, i - 600):i + 1:4]], [x["ref"] for x in log[max(0, i - 600):i + 1:4]], cam, e["goal"], overlay=ov)
    frames.append(RP.compose(r.render().copy(), e, i, None, a.title, a.sub, insets=ins))
codec = RP.write_mp4(frames, RP.VID / f"{a.out}.mp4", 100.0 / a.stride)
frames[len(frames) // 2].save(RP.VID / f"{a.out}.jpg", quality=88)
if a.sensor_view:
    json.dump(dict(name=a.out, run=a.run, traj=a.traj, seconds=len(log) / 100, codec=codec, stride=a.stride, fell=any(x["fell"] for x in log),
                   sensor=d.get("sensor"), sensor_view=dict(yaw=yaw_src, noise_seed=a.seed, estimator=False)), open(RP.VID / f"{a.out}.json", "w"))
print(a.out, len(log) / 100, "s", codec, "fell", any(x["fell"] for x in log))
