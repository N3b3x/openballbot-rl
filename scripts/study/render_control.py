"""Render a velocity-command tracking video: scripted command sequence, green arrow = commanded
velocity, red arrow = achieved velocity, text overlay with numbers.

  python scripts/study/render_control.py --run ctrl_free_c_s0 --config ctrl_free_b --out outputs/study/preview/ctrl_free_c_final.mp4
  python scripts/study/render_control.py --run ctrl_free_c_s0 --config ctrl_free_b --ckpt ppo_2999916_steps --out ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
from policy_io import load_controller  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402

# (label, (vx, vy, wz), seconds); +y is "forward"
SEQ = [
    ("rest", (0, 0, 0), 1.5),
    ("forward 0.5", (0, 0.5, 0), 3),
    ("right 0.5", (0.5, 0, 0), 3),
    ("backward 0.5", (0, -0.5, 0), 3),
    ("left 0.5", (-0.5, 0, 0), 3),
    ("diagonal 0.5", (0.35, 0.35, 0), 3),
    ("fast forward 1.0", (0, 1.0, 0), 3),
    ("spin +2 rad/s", (0, 0, 2.0), 3),
    ("spin -2 rad/s", (0, 0, -2.0), 3),
    ("arc 0.5 m/s + 1 rad/s", (0, 0.5, 1.0), 4),
    ("stop", (0, 0, 0), 2),
]
# square within the trained envelope (|w| <= 1), then a fast leg and a turning leg
SQUARE = [
    ("rest", (0, 0, 0), 1.5),
    ("forward 0.5", (0, 0.5, 0), 3),
    ("right 0.5", (0.5, 0, 0), 3),
    ("backward 0.5", (0, -0.5, 0), 3),
    ("left 0.5", (-0.5, 0, 0), 3),
    ("fast forward 1.0", (0, 1.0, 0), 2.5),
    ("forward 0.5 while turning 1 rad/s", (0, 0.5, 1.0), 4),
    ("diagonal back 0.5", (-0.35, -0.35, 0), 4),
    ("stop", (0, 0, 0), 2),
]
CLIMB = [("rest", (0, 0, 0), 1.0), ("forward 0.7 m/s at the step", (0, 0.7, 0), 3.5), ("stop on top", (0, 0, 0), 2.0)]
TOUR = [
    ("rest", (0, 0, 0), 1.0),
    ("forward 0.5", (0, 0.5, 0), 3),
    ("right 0.5", (0.5, 0, 0), 3),
    ("backward 0.5", (0, -0.5, 0), 3),
    ("left 0.5", (-0.5, 0, 0), 3),
    ("diagonal 0.7", (0.5, 0.5, 0), 3),
    ("diagonal back 0.7", (-0.5, -0.5, 0), 3),
    ("forward 0.5 while turning 0.8 rad/s", (0, 0.5, 0.8), 4),
    ("stop", (0, 0, 0), 1.5),
]
# loop inside a 1.5 m square so the robot never leaves the 10 m floor while being pushed
PUSH = [("rest", (0, 0, 0), 0.5)] + [(n, c, 3) for n, c in (("forward 0.5", (0, 0.5, 0)), ("right 0.5", (0.5, 0, 0)),
         ("backward 0.5", (0, -0.5, 0)), ("left 0.5", (-0.5, 0, 0)))] * 2 + [("stop", (0, 0, 0), 1.5)]
NAV = [("navigate waypoints", (0, 0, 0), 60)]
WAYPOINTS = [(0.0, 1.0), (1.0, 1.6), (1.8, 0.6), (0.8, -0.4), (-0.6, 0.2), (0.0, 0.0)]
SEQS = {"full": SEQ, "square": SQUARE, "climb": CLIMB, "tour": TOUR, "push": PUSH, "nav": NAV}


def arrow(scene, mujoco, p0, vec, rgba, scale=1.0):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(vec) < 1e-3:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    p1 = p0 + scale * np.array([vec[0], vec[1], 0.0])
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, p0, p1)
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="final_model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-35.0)
    ap.add_argument("--distance", type=float, default=2.2)
    ap.add_argument("--vcap", type=float, default=1.0)
    ap.add_argument("--step-height", type=float, default=0.0, help="raised floor from y = 1 m (book-like step) [m]")
    ap.add_argument("--seq", choices=list(SEQS), default="full")
    ap.add_argument("--guide", type=float, default=0.0, help="position-feedback gain [1/s] toward the planned path (0 = pure velocity tracking)")
    ap.add_argument("--perlin-amp", type=float, default=0.0, help="Perlin-noise floor amplitude [m]")
    ap.add_argument("--terrain-seed", type=int, default=3)
    ap.add_argument("--pushes", default="", help="comma list of time:force[N], e.g. 3:30,7:40,11:50 (random direction)")
    ap.add_argument("--vnav", type=float, default=0.5, help="waypoint-navigation speed [m/s]")
    ap.add_argument("--dr", action="store_true", help="keep the config's domain randomisation / pushes")
    a = ap.parse_args()

    import mujoco
    import imageio
    from PIL import Image, ImageDraw

    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{a.config}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    if not a.dr:
        cfg["env"].pop("dr", None)   # benchmark the nominal robot unless asked otherwise
    cfg["env"]["max_ep_steps"] = 10 ** 7   # no time-limit resets mid-video
    if a.step_height > 0:
        cfg["terrain"] = {"type": "book", "config": {"step_y": 1.0, "book_len": 0.0, "rough_amplitude": 0.0,
                                                     "step_height_m": a.step_height, "ramp_len": 0.02, "seed": 0}}
        cfg["env"]["init_yaw_random"] = False
    if a.perlin_amp > 0:
        cfg["terrain"] = {"type": "perlin", "config": {"scale": 25.0, "octaves": 4, "persistence": 0.2, "lacunarity": 2.0,
                                                       "amplitude": a.perlin_amp, "seed": a.terrain_seed}}
        cfg["env"]["init_yaw_random"] = False
    ec = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=ec, seed=0, eval_env=True)()
    run_dir = ROOT / "outputs/study/runs" / a.run
    model, norm = load_controller(a.run, env, a.ckpt)

    base = env.unwrapped
    base.cmd_fixed = np.zeros(3)
    obs, _ = env.reset(seed=0)
    m, d = base.model, base.data
    renderer = mujoco.Renderer(m, height=360, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = a.distance, a.azimuth, a.elevation

    frames, errs, t = [], [], 0
    seq = SEQS[a.seq]
    start = d.xpos[m.body("base").id][:2].copy()
    # full planned path (commands are world-frame velocities) so the viewer sees where the robot should go
    plan, q = [start.copy()], start.copy()
    nav = a.seq == "nav"
    if nav:
        pts = [start] + [start + np.array(w) for w in WAYPOINTS]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(p1 - p0) / (a.vnav * 0.01)))
            plan.extend(list(p0 + (p1 - p0) * (np.arange(1, n + 1) / n)[:, None]))
        wp_abs = [start + np.array(w) for w in WAYPOINTS]
    else:
        for _, c, secs in seq:
            for _ in range(int(secs * 100)):
                q = q + np.asarray(c[:2], float) * 0.01
                plan.append(q.copy())
    plan = np.array(plan)
    push_sched = []
    for tok in [x for x in a.pushes.split(",") if x]:
        tt, ff = tok.split(":")
        push_sched.append([int(float(tt) * 100), float(ff), np.random.default_rng(int(float(tt) * 100)).uniform(0, 2 * np.pi), 0])
    wp_i = 0
    push_vis = [None]
    real_trail, path_err = [], []
    fell = False

    def dot(scene, xy, rgba, size=0.012):
        if scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                                np.array([xy[0], xy[1], 0.01]), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
            scene.ngeom += 1

    def draw(cmd_eff, v, label, seg, banner=None):
        pos = d.xpos[m.body("base").id].copy()
        cam.lookat[:] = pos
        renderer.update_scene(d, camera=cam)
        for xy in plan[::10]:
            dot(renderer.scene, xy, [0.1, 0.8, 0.2, 0.55], 0.009)
        dot(renderer.scene, plan[min(t, len(plan) - 1)], [0.1, 1.0, 0.2, 1.0], 0.03)   # where it should be now
        for xy in real_trail:
            dot(renderer.scene, xy, [0.9, 0.15, 0.1, 0.9])
        if nav:
            for w in wp_abs:
                dot(renderer.scene, w, [0.2, 0.4, 1.0, 1.0], 0.05)
        if push_vis[0] is not None and t - push_vis[0][2] < 40:
            F, ang, _ = push_vis[0]
            arrow(renderer.scene, mujoco, pos + [0, 0, 0.18], np.array([np.cos(ang), np.sin(ang)]) * F / 60.0, [1.0, 0.6, 0.0, 1], 1.0)
        arrow(renderer.scene, mujoco, pos + [0, 0, 0.25], cmd_eff[:2], [0.1, 0.9, 0.2, 1], 0.5)
        arrow(renderer.scene, mujoco, pos + [0, 0, 0.30], v, [0.95, 0.15, 0.1, 1], 0.5)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        tail = seg[-50:] or [(np.zeros(2), 0.0)]
        vm = np.mean([x[0] for x in tail], axis=0)
        wm = np.mean([x[1] for x in tail])
        dr.rectangle([0, 0, 640, 80], fill=(0, 0, 0))
        g = f"  guidance k={a.guide}" if a.guide else ""
        dr.text((8, 4), f"{a.run} [{a.ckpt}]  t={t / 100:5.1f}s   command: {label}{g}", fill=(255, 255, 255))
        dr.text((8, 22), f"cmd  v=({cmd_eff[0]:+.2f},{cmd_eff[1]:+.2f}) m/s  w={cmd_eff[2]:+.2f} rad/s   green: planned path, big dot = target now", fill=(120, 255, 140))
        dr.text((8, 40), f"real v=({vm[0]:+.2f},{vm[1]:+.2f}) m/s  w={wm:+.2f} rad/s   red: actual path", fill=(255, 130, 120))
        pe = path_err[-1] if path_err else 0.0
        dr.text((8, 58), f"distance from planned path point: {pe:.2f} m   off-ball {info_last.get('off_ball_deg', 0):.1f} deg", fill=(230, 230, 120))
        if push_vis[0] is not None and t - push_vis[0][2] < 40:
            dr.text((8, 340), f"PUSH {push_vis[0][0]:.0f} N (orange arrow)", fill=(255, 170, 0))
        if banner:
            dr.rectangle([120, 160, 520, 205], fill=(170, 0, 0))
            dr.text((135, 175), banner, fill=(255, 255, 255))
        frames.append(np.asarray(img))

    info_last = {}
    for label, cmd, secs in seq:
        cmd = np.asarray(cmd, dtype=float)
        seg = []
        for _ in range(int(secs * 100)):
            pos2 = d.xpos[m.body("base").id][:2]
            ref = plan[min(t, len(plan) - 1)]
            cmd_eff = cmd.copy()
            if nav:
                if wp_i < len(wp_abs) and np.linalg.norm(wp_abs[wp_i] - pos2) < 0.15:
                    wp_i += 1
                if wp_i < len(wp_abs):
                    dvec = wp_abs[wp_i] - pos2
                    cmd_eff[:2] = dvec / max(np.linalg.norm(dvec), 1e-6) * min(a.vnav, 1.5 * np.linalg.norm(dvec))
                else:
                    cmd_eff[:2] = 0.0
            if a.guide:   # outer position loop: steer back toward the planned point (path following on top of velocity tracking)
                corr = a.guide * (ref - pos2)
                n = np.linalg.norm(corr)
                cmd_eff[:2] = cmd[:2] + (corr * min(1.0, 0.5 / n) if n > 1e-9 else 0)
                sp = np.linalg.norm(cmd_eff[:2])
                if sp > a.vcap:   # stay inside the speed range the policy tracks reliably
                    cmd_eff[:2] *= a.vcap / sp
            base.cmd = cmd_eff
            obs["cmd"] = cmd_eff.astype(np.float32)
            with torch.no_grad():
                act, _ = model.predict(norm(obs), deterministic=True)
            push_now = None
            for ps in push_sched:
                if ps[0] <= t < ps[0] + 5:
                    push_now = ps
                    d.xfrc_applied[m.body("base").id, :3] = ps[1] * np.array([np.cos(ps[2]), np.sin(ps[2]), 0.0])
            obs, r, term, trunc, info_last = env.step(act)
            v = obs["vel"][:2].copy()
            if push_now is not None:
                push_vis[0] = (push_now[1], push_now[2], t)
            seg.append((v, float(obs["angular_vel"][2])))
            path_err.append(float(np.linalg.norm(d.xpos[m.body("base").id][:2] - plan[min(t + 1, len(plan) - 1)])))
            if t % 10 == 0:
                real_trail.append(d.xpos[m.body("base").id][:2].copy())
            if t % 2 == 0 or term:
                draw(cmd_eff, v, label, seg)
            t += 1
            if term:
                fell = True
                why = "left the ball" if info_last.get("off_ball_deg", 0) > 25 else "tipped over"
                for _ in range(75):   # hold 1.5 s on the failure instead of silently resetting
                    draw(cmd_eff, v, label, seg, banner=f"FELL at t={t / 100:.1f}s ({why}) - run ends here")
                break
        late = seg[len(seg) // 2:]
        if late:
            vm = np.mean([x[0] for x in late], axis=0)
            wm = np.mean([x[1] for x in late])
            errs.append((label, float(np.linalg.norm(vm - cmd[:2])), float(abs(wm - cmd[2]))))
        if fell:
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=50)
    np.savez(str(Path(a.out).with_suffix("")) + "_traj.npz", plan=plan, real=np.array(real_trail),
             path_err=np.array(path_err), fell=fell, guide=a.guide)
    print("saved", a.out, "fell" if fell else "no fall",
          f"| path error mean {np.mean(path_err):.2f} m, max {np.max(path_err):.2f} m, final {path_err[-1]:.2f} m")
    for lab, ev, ew in errs:
        print(f"  {lab:24s} vel err {ev:.2f} m/s  yaw err {ew:.2f} rad/s")


if __name__ == "__main__":
    main()
