"""Cinematic render of a trained policy on the tiled terrain: Warp simulation of one world, CPU MuJoCo renderer for the pictures.
Checkerboard floor, tracking camera, commanded path (cyan) vs actual path (green) drawn in the scene, goal markers, push arrows, HUD. Simulation only.
  python render_policy.py --run NAME --kind step_up --severity 0.03 --task vel --speed 0.6 --design nom --out clip_name [--seconds 8]
  --sensor-view adds what the head ToF sees (sensor_view.py): frustum and zone dots in 3-D, the measured range image and the 7 x 5 height grids.
Scenario: the world is placed on the tile of the requested kind whose severity is closest to --severity, heading chosen so that the robot drives into the feature.
"""
import argparse, json, importlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"
VID = T.ROOT / "outputs" / "study" / "warp_ppo" / "videos"
FD = Path("C:/Windows/Fonts")
C = dict(ink=(236, 241, 248), dim=(150, 163, 182), bad=(255, 107, 107), good=(94, 224, 138), warn=(255, 180, 84), bg=(10, 14, 22))
_f = {}


def font(sz, kind="reg"):
    k = (sz, kind)
    if k not in _f:
        n = {"reg": "segoeui.ttf", "bold": "segoeuib.ttf", "mono": "consola.ttf"}[kind]
        try:
            _f[k] = ImageFont.truetype(str(FD / n), sz)
        except OSError:
            _f[k] = ImageFont.load_default()
    return _f[k]


def load_policy(run, dev="cuda"):
    ck = torch.load(RUN / run / "model.pt", map_location=dev)
    sdim = ck.get("sdim", 0)
    net = T.Net(sdim, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to(dev)
    net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
    mean, var = ck["mean"].to(dev), ck["var"].to(dev)
    return ck, (lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1))


def simulate(a):
    ck, pol = load_policy(a.run)
    _ap0 = RUN / a.run / "args.json"
    _a0 = json.load(open(_ap0)) if _ap0.exists() else {}
    if os.environ.get("RETAINER") == "off":      # same rule as the evaluation scripts: training setting unless overridden
        os.environ.pop("RETAINER")
    elif not os.environ.get("RETAINER") and _a0.get("retainer"):
        os.environ["RETAINER"] = _a0["retainer"]
    if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):      # contact model of the run (S101 fix)
        os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
    sdim = ck.get("sdim", 0); sensor, mount = {0: ("none", "head"), 35: ("scan", "head"), 252: ("depth", "head"), 504: ("depth", "cones")}[sdim]
    designs = ck.get("designs") or ["nom"]
    dname = a.design if a.design in designs else designs[0]
    env = T.GpuBallbot(1, seed=a.seed, dr=a.dr, design=dname, designs=None if len(designs) == 1 else designs, tiles=True, sensor=sensor, mount=mount,
                       hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=a.dr, pull=getattr(a, "pull", False), goal_p=1.0 if a.task == "goal" else ck.get("goal_p", 0.0),
                       design_cond=ck.get("ddim", 0) > 0)
    if env.K > 1:
        env.set_design(torch.zeros(1, dtype=torch.long, device="cuda"), torch.tensor([designs.index(dname)], device="cuda"))
    _ap = T.ROOT / "outputs" / "study" / "warp_ppo" / a.run / "args.json"
    _args = json.load(open(_ap)) if _ap.exists() else {}
    if _args.get("lqr_base", 0) > 0:      # residual policy: rebuild the LQR base controller as in training
        from lqr_gpu import GpuLQR
        env.lqr, env.rs = GpuLQR(env, env.design_names, a_max=_args.get("lqr_amax", 0.5), r=_args.get("lqr_r", 1.0)), _args["lqr_base"]
        if _args.get("rs_adapt"): env.rs_adapt = [float(x) for x in _args["rs_adapt"].split(",")]      # S111 adaptive authority, as in the evals
    env.goal_centre = bool(_args.get("goal_centre")); env.db_centre = bool(_args.get("db_centre"))      # S118: goal logic on the true ball centre if the run was trained that way
    if _args.get("motion_ref") and not os.environ.get("NO_MREF"):      # S105 motion layer the run was trained with
        env.enable_motion_ref(curve_p=0.0)
    # choose the tile
    ids = [i for i, t in enumerate(env.tile_info) if t["kind"] == a.kind]
    tid = min(ids, key=lambda i: abs(env.tile_info[i]["severity"] - a.severity)) if a.severity is not None else ids[0]
    tinfo = env.tile_info[tid]
    cx, cy = tinfo["cx"], tinfo["cy"]
    env.resample, env.max_steps = 10 ** 9, 10 ** 9
    env.cur_tile[:] = tid
    ang = np.radians(a.heading)
    x0, y0 = cx - 0.45 * np.cos(ang), cy - 0.45 * np.sin(ang)
    q = env.q0_tab[env.didx].clone()
    q[:, 0] += x0; q[:, 1] += y0; q[:, 10] += x0; q[:, 11] += y0
    hz = env.hgt(torch.tensor([x0], device="cuda"), torch.tensor([y0], device="cuda")); q[:, 2] += hz; q[:, 12] += hz
    yaw = ang
    q[:, 3] = float(np.cos(yaw / 2)); q[:, 6] = float(np.sin(yaw / 2))
    env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
    f = float(env.f_w[0])
    if a.task == "vel":
        env.goal_mode[:] = False
        env.cmd = torch.tensor([[a.speed * f * np.cos(ang), a.speed * f * np.sin(ang), 0.0]], device="cuda", dtype=torch.float32)
    else:
        env.goal_mode[:] = True; env.path_left[:] = 0
        env.goal[:] = torch.tensor([[cx + 1.0 * np.cos(ang), cy + 1.0 * np.sin(ang)]], device="cuda", dtype=torch.float32)
        env.update_goal_cmd()
    if getattr(env, "mref", None) is not None:      # shaped from rest, robot facing the desired direction (same as eval_tiles2)
        env.cmd_des = env.cmd[:, :2].clone(); env.cmd_w[:] = 0
        env.mref.reset(torch.ones(1, dtype=torch.bool, device="cuda"), heading=torch.tensor([ang], device="cuda", dtype=torch.float32))
    _frc = {"f": np.zeros(3)}
    _bstep = env.be.step
    def _step_rec(*k, **kw):      # capture the applied push/pull force before the physics step consumes it
        _frc["f"] = env.xfrc[0, env.base, :3].cpu().numpy().copy()
        return _bstep(*k, **kw)
    env.be.step = _step_rec
    o = env.noisy(env.reset_obs())
    log, ref = [], np.array([x0, y0])
    n = int(a.seconds * 100)
    for t in range(n):
        with torch.no_grad():
            act = pol(o)
        o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
        qp = env.qpos[0].cpu().numpy().copy()
        cmd = env.cmd[0].cpu().numpy().copy()
        ref = ref + cmd[:2] * 0.01
        log.append(dict(qpos=qp, cmd=cmd, ref=ref.copy(), goal=env.goal[0].cpu().numpy().copy() if a.task == "goal" else None,
                        tilt=float(info["tilt"][0]), speed=float(np.linalg.norm(env._vel[0, :2].cpu().numpy())), fell=bool(fell[0]),
                        prop=o[0, :T.OBS_DIM].cpu().numpy().copy() if getattr(a, "sensor_view", False) else None,
                        push=_frc["f"].copy()))      # 25-D proprio for the grid estimator
        if bool(fell[0]):
            log.extend([log[-1]] * 30); break
    return env, log, tinfo


def draw_scene(r, md, m, e, trail_act, trail_ref, cam, goal, overlay=None, look_off=None):
    """overlay: optional callable(add, line) drawing extra scene geoms (sensor_view.draw_3d); look_off: optional (dx, dy) camera target shift."""
    md.qpos[:] = e["qpos"]; mujoco.mj_forward(m, md)
    ball = md.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")]
    cam.lookat[:] = [ball[0], ball[1], ball[2] + 0.1]
    if look_off is not None:
        cam.lookat[:2] += np.asarray(look_off, float)
    r.update_scene(md, camera=cam)
    sc = r.scene

    def add(kind, size, pos, rgba):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        sc.ngeom += 1
        return g

    def line(p0, p1, rgba, w=0.006):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(p0, float), np.asarray(p1, float))
        sc.ngeom += 1
        return g

    zz = float(ball[2]) - 0.085
    for a_, b_ in zip(trail_ref[:-1], trail_ref[1:]):
        line([a_[0], a_[1], zz + 0.012], [b_[0], b_[1], zz + 0.012], [0.1, 0.85, 1.0, 1.0])
    for a_, b_ in zip(trail_act[:-1], trail_act[1:]):
        line([a_[0], a_[1], zz + 0.006], [b_[0], b_[1], zz + 0.006], [0.2, 1.0, 0.4, 1.0], 0.007)
    if goal is not None:
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [goal[0], goal[1], zz + 0.06], [1.0, 0.3, 0.2, 0.9])
    pf = e.get("push") if isinstance(e, dict) else None
    if pf is not None and np.linalg.norm(pf[:2]) > 0.5:      # push / pull arrow: red, pointing in the force direction, 3 cm per newton
        tip = np.array([ball[0], ball[1], ball[2] + 0.35]); d = np.asarray(pf[:2], float)
        tail = tip - np.r_[d * 0.03, 0.0]
        line(tail, tip, [1.0, 0.15, 0.15, 1.0], 0.012)
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.022, 0, 0], tip, [1.0, 0.15, 0.15, 1.0])
    if overlay is not None:
        overlay(add, line)


def compose(frame, e, i, a, title, sub, insets=None):
    """insets: optional list of RGBA PIL images, laid out left to right in a strip below the 3-D view (so they never cover the scene)."""
    W, H = frame.shape[1], frame.shape[0]
    S = (max(x.size[1] for x in insets) + 16) if insets else 0
    im = Image.new("RGB", (W, H + 80 + S), C["bg"])
    im.paste(Image.fromarray(frame), (0, 80))
    d = ImageDraw.Draw(im, "RGBA")
    d.text((16, 10), title, font=font(26, "bold"), fill=C["ink"])
    d.text((16, 46), sub, font=font(15), fill=C["dim"])
    d.text((W - 16, 14), "SIMULATION ONLY", font=font(15, "bold"), fill=C["warn"], anchor="ra")
    d.rectangle([12, 90, 360, 168], fill=(10, 14, 22, 190))
    d.text((22, 96), f"t {i*0.01:5.2f} s   speed {e['speed']:.2f} m/s", font=font(15, "mono"), fill=C["ink"])
    d.text((22, 118), f"cmd {np.linalg.norm(e['cmd'][:2]):.2f} m/s   tilt {e['tilt']:5.1f} deg", font=font(15, "mono"), fill=C["bad"] if e["tilt"] > 25 else C["ink"])
    d.line([22, 150, 52, 150], fill=(26, 217, 255), width=5); d.text((58, 142), "commanded path", font=font(13), fill=C["dim"])
    d.line([190, 150, 220, 150], fill=(51, 255, 102), width=5); d.text((226, 142), "actual path", font=font(13), fill=C["dim"])
    if e["fell"] or e["tilt"] > 70:
        d.text((W // 2, H // 2 + 80), "FELL", font=font(40, "bold"), fill=C["bad"], anchor="mm")
    if insets:
        gap = max(8, (W - sum(x.size[0] for x in insets)) // (len(insets) + 1))
        x = gap
        for img in insets:
            im.paste(img, (x, H + 80 + 8), img); x += img.size[0] + gap
    return im


def sensor_setup(env, m, seed=0, sensor_cfg=None, estimator=True):
    """--sensor-view: CPU sensor caster on the render model plus (optional) the learned grid estimator."""
    import sensor_view as SV
    kw = {}
    if sensor_cfg:      # recorded sensor config of a navigation run
        kw = dict(res=tuple(int(x) for x in sensor_cfg["res"].split("x")), fov=tuple(np.radians(float(x)) for x in sensor_cfg["fov"].split(",")),
                  pitch=np.radians(sensor_cfg["pitch"]), up=sensor_cfg["up"], noise=sensor_cfg["noise"])
    sv = SV.SensorView(m, hmap=env.hmap.cpu().numpy(), seed=seed, **kw)
    ep = RUN / "percep" / "tof18_gov.pt"
    est = SV.Estimator(ep) if (estimator and ep.exists()) else None
    return sv, est


def sensor_frame(sv, est, md, m, e, yaw):
    """One frame of the sensor view: reading, 3-D overlay callable, and the insets for compose()."""
    import sensor_view as SV
    md.qpos[:] = e["qpos"]; mujoco.mj_forward(m, md)
    rd = sv.read(md, e["qpos"], yaw)
    seen, _ = sv.seen_grid(md, rd, e["qpos"])
    grids = [("robot: from points", seen), ("true: height map", sv.true_grid(e["qpos"], yaw))]
    if est is not None and e.get("prop") is not None:
        grids.insert(1, ("learned estimate", est(rd["meas"], e["prop"])))
    insets = [SV.range_inset(rd, font), SV.grids_inset(grids, font), SV.legend_inset(font, sv, rd)]
    return (lambda add, line: SV.draw_3d(add, line, rd)), insets


def write_mp4(frames, path, fps):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = frames[0].size
    for codec, extra in (("h264_nvenc", ["-preset", "p5", "-cq", "24"]), ("libx264", ["-preset", "slow", "-crf", "23"])):
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-c:v", codec, *extra, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for f in frames:
                p.stdin.write(np.asarray(f).tobytes())
            p.stdin.close(); rc = p.wait()
        except (BrokenPipeError, OSError):
            rc = 1
        if rc == 0:
            return codec
    raise RuntimeError("ffmpeg failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--kind", default="step_up"); ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--task", choices=["vel", "goal"], default="vel"); ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--design", default="nom"); ap.add_argument("--heading", type=float, default=0.0); ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dr", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1280); ap.add_argument("--h", type=int, default=640); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pull", action="store_true", help="also apply the sustained 3 to 12 N pulls (needs --dr)"); ap.add_argument("--title", default=None); ap.add_argument("--sub", default="")
    ap.add_argument("--sensor-view", action="store_true", help="overlay what the head ToF sensor sees (CPU ground-truth casts + the training noise model)")
    a = ap.parse_args()
    VID.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env, log, tinfo = simulate(a)
    m = env.mjm
    m.vis.global_.offwidth = max(a.w, 640); m.vis.global_.offheight = max(a.h, 480)
    r = mujoco.Renderer(m, height=a.h, width=a.w, max_geom=4000)
    md = mujoco.MjData(m)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = 1.8; cam.elevation = -22
    frames = []
    if a.sensor_view:
        import sensor_view as SV
        sv, est = sensor_setup(env, m, seed=a.seed)
        print("sensor view:", "learned estimate shown" if est is not None else "no estimator file, grid estimate omitted")
    for i in range(0, len(log), a.stride):
        e = log[i]
        cam.azimuth = 150 + 25 * np.sin(i * 0.004)
        trail_act = [x["qpos"][10:12] for x in log[max(0, i - 400):i + 1:4]]
        trail_ref = [x["ref"] for x in log[max(0, i - 400):i + 1:4]]
        ov, ins, lo = None, None, None
        if a.sensor_view:      # rear three-quarter camera so the sensed ground ahead is in view and above the bottom insets
            syaw = SV.sensor_yaw(e["qpos"], e["cmd"])
            ov, ins = sensor_frame(sv, est, md, m, e, syaw)
            cam.azimuth = np.degrees(syaw) + 32 + 10 * np.sin(i * 0.004); cam.distance = 2.1; cam.elevation = -27
            lo = 0.55 * np.array([np.cos(syaw), np.sin(syaw)])
        draw_scene(r, md, m, e, trail_act, trail_ref, cam, e["goal"], overlay=ov, look_off=lo)
        title = a.title or f"{a.kind.replace('_', ' ')} {tinfo['severity']*100:.0f} cm" if a.kind not in ("ramp_up", "ramp_down") else f"{a.kind.replace('_', ' ')} {tinfo['severity']:.0f} deg"
        frames.append(compose(r.render().copy(), e, i, a, title, a.sub or f"policy {a.run}, design {a.design}, task {a.task}" + (", ball arrestors ON" if os.environ.get("RETAINER") else ", no arrestors"), insets=ins))
    codec = write_mp4(frames, VID / f"{a.out}.mp4", 100.0 / a.stride)
    frames[len(frames) // 2].save(VID / f"{a.out}.jpg", quality=88)
    fell = any(x["fell"] for x in log)
    meta = dict(name=a.out, run=a.run, kind=a.kind, severity=tinfo["severity"], task=a.task, design=a.design, fell=fell, seconds=len(log) / 100, codec=codec,
                dr=bool(a.dr), arrestors=bool(os.environ.get("RETAINER")), stride=a.stride, speed=(a.speed if a.task == "vel" else None))
    if a.sensor_view:
        meta["sensor_view"] = dict(sensor="head ToF 18x14, FoV 55x42 deg, pitch -25 deg, 0.15 m above base, 0.05 to 4 m", noise="sd 0.005*d, 2% dropout",
                                   yaw="command direction (training rule)", estimator=est is not None)
    json.dump(meta, open(VID / f"{a.out}.json", "w"))
    print(meta, "%.0fs" % (time.time() - t0))
