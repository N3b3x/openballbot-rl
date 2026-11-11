"""Side-by-side videos: CPU MuJoCo (left) vs MuJoCo-Warp (right), same scenario and controller gains, contact points and
contact forces drawn. Simulation only. The Warp trajectory (float32, GPU) is exported to a CPU MjData for rendering.
python render_videos.py [scenario ...]    -> outputs/study/mjx_port/videos/*.mp4 and videos/clips.json (consumed by make_site_data.py)"""
import sys, json, subprocess, time
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from warp_sim import *      # noqa
import cpu_ref

VID = OUT / "videos"
PW, PH, TOP, BOT = 640, 480, 74, 40
C = dict(ink=(236, 241, 248), dim=(150, 163, 182), bad=(255, 107, 107), good=(94, 224, 138), warn=(255, 180, 84),
         model=(74, 168, 255), warp=(118, 235, 140), bg=(10, 14, 22))
FD = Path("C:/Windows/Fonts")
_f = {}


def font(sz, kind="reg"):
    k = (sz, kind)
    if k not in _f:
        n = {"reg": "segoeui.ttf", "bold": "segoeuib.ttf", "mono": "consola.ttf", "monob": "consolab.ttf"}[kind]
        try:
            _f[k] = ImageFont.truetype(str(FD / n), sz)
        except OSError:
            _f[k] = ImageFont.load_default()
    return _f[k]


def contacts_of(be, kind):
    """List of (pos, frame9, force in contact frame (n,t1,t2), geom0, geom1) after the last step."""
    if kind == "cpu":
        d = be.ds[0]
        m = be.m
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            out.append((c.pos.copy(), c.frame.copy(), f6[:3].copy(), int(c.geom[0]), int(c.geom[1])))
        return out
    n = int(be.d.nacon.numpy()[0])
    ct = be.d.contact
    pos, fr, ad, ge = ct.pos.numpy()[:n], ct.frame.numpy()[:n], ct.efc_address.numpy()[:n], ct.geom.numpy()[:n]
    ef = be.d.efc.force.numpy()[0]
    return [(pos[i].astype(float), fr[i].reshape(9).astype(float),
             np.array([ef[a] if a >= 0 else 0.0 for a in ad[i][:3]]), int(ge[i][0]), int(ge[i][1])) for i in range(n)]


def simulate(kind, m, design, sc):
    be = cpu_ref.CpuBackend(m, 1) if kind == "cpu" else WarpBallbot(m, 1)
    q, v = reset_state(m)
    be.reset(q, v)
    ctl = LQRBatch(1, model_json=design.json, tau_max=design.tau)
    cmd = np.array([sc.get("cmd", [0, 0, 0])], float)
    log = []
    qp, qv, cv, bx, gx = be.state()
    for t in range(sc["steps"]):
        tt = t * 0.01
        if sc["mode"] == "lqr":
            tau = ctl(qp, qv, cv[:, 3:6], cmd)
        elif sc["mode"] == "step":
            tau = (np.array([2.0, 0, 0]) * design.tau / TAU_MAX)[None] if 0.5 <= tt < 0.8 else np.zeros((1, 3))
        else:
            tau = np.zeros((1, 3))
        be.set_ctrl(tau)
        push = np.zeros((1, 3))
        if "push" in sc and sc["push"]["t0"] <= t < sc["push"]["t0"] + 5:
            push = np.array([sc["push"]["F"]])
        be.set_push(push)
        be.step(ACTION_REPEAT)
        qp, qv, cv, bx, gx = be.state()
        log.append(dict(qpos=qp[0].copy(), t=(t + 1) * 0.01, speed=float(np.hypot(cv[0, 3], cv[0, 4])),
                        tilt=float(tilt_deg(qp[0])), contacts=contacts_of(be, kind), push=push[0].copy()))
    return log


def draw_scene(r, md, m, entry):
    d = md
    d.qpos[:] = entry["qpos"]
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ball = entry["qpos"][KQ:KQ + 3]
    cam.lookat[:] = [ball[0], ball[1], 0.13]
    cam.distance, cam.azimuth, cam.elevation = 0.85, 55.0, -20.0
    r.update_scene(d, camera=cam)
    sc = r.scene

    def add(typ, size, pos):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.array([1, 1, 1, 1], np.float32))
        sc.ngeom += 1

    def arrow(p0, p1, w, rgba):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, w, np.asarray(p0, float), np.asarray(p1, float))
        sc.ngeom += 1

    SC = 0.015     # m per N
    for pos, fr, f, g0, g1 in entry["contacts"]:
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], pos)
        n, t1, t2 = fr[0:3], fr[3:6], fr[6:9]
        arrow(pos, pos + SC * f[0] * n, 0.010, [1.0, 0.55, 0.1, 1])
        ft = f[1] * t1 + f[2] * t2
        if np.linalg.norm(ft) > 0.05:
            arrow(pos, pos + SC * ft, 0.010, [0.1, 0.85, 1.0, 1])
    if np.linalg.norm(entry["push"]) > 0:
        b = entry["qpos"][:3]
        arrow(b - 0.004 * entry["push"], b, 0.012, [1.0, 0.2, 0.2, 1])


def compose(fl, fr, el, er, sc, title):
    out = []
    for i in range(min(len(fl), len(fr))):
        im = Image.new("RGB", (2 * PW, TOP + PH + BOT), C["bg"])
        im.paste(Image.fromarray(fl[i]), (0, TOP))
        im.paste(Image.fromarray(fr[i]), (PW, TOP))
        d = ImageDraw.Draw(im, "RGBA")
        d.text((14, 8), title, font=font(24, "bold"), fill=C["ink"])
        d.text((14, 42), sc["sub"], font=font(14), fill=C["dim"])
        d.text((2 * PW - 14, 14), "SIMULATION ONLY", font=font(15, "bold"), fill=C["warn"], anchor="ra")
        for k, (e, col, name, sub) in enumerate(((el[i], C["model"], "CPU MuJoCo 3.17", "float64 + ContactFrameStepper"),
                                                 (er[i], C["warp"], "MuJoCo-Warp 3.17 (GPU)", "float32 + contact-frame kernel"))):
            ox = k * PW
            d.rectangle([ox + 8, TOP + 8, ox + 330, TOP + 96], fill=(10, 14, 22, 190))
            d.text((ox + 16, TOP + 12), name, font=font(17, "bold"), fill=col)
            d.text((ox + 16, TOP + 34), sub, font=font(12), fill=C["dim"])
            d.text((ox + 16, TOP + 52), f"t {e['t']:5.2f} s   speed {e['speed']:.2f} m/s", font=font(14, "mono"), fill=C["ink"])
            d.text((ox + 16, TOP + 72), f"tilt {e['tilt']:5.1f} deg   ctrl {sc['ctrl']}", font=font(14, "mono"),
                   fill=C["bad"] if e["tilt"] > 25 else C["ink"])
            if e["tilt"] > 70:
                d.text((ox + PW // 2, TOP + 130), "FELL (tilt > 70 deg)", font=font(26, "bold"), fill=C["bad"], anchor="mm")
        d.line([PW, TOP, PW, TOP + PH], fill=(255, 255, 255, 60), width=2)
        yy = TOP + PH + 10
        d.ellipse([14, yy + 4, 24, yy + 14], fill=(255, 255, 255))
        d.text((30, yy), "contact point", font=font(14), fill=C["dim"])
        d.line([150, yy + 9, 180, yy + 9], fill=(255, 140, 26), width=5)
        d.text((188, yy), "normal force", font=font(14), fill=C["dim"])
        d.line([300, yy + 9, 330, yy + 9], fill=(26, 217, 255), width=5)
        d.text((338, yy), "friction force", font=font(14), fill=C["dim"])
        d.text((460, yy), "arrow = 1.5 cm per N   |   same LQR gains, same initial state", font=font(14), fill=C["dim"])
        out.append(im)
    return out


def write_mp4(frames, path, fps):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = frames[0].size
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f).tobytes())
    p.stdin.close()
    p.wait()


SCEN = {
    "settle": dict(title="Free fall onto the ball, open loop",
                   sub="Robot dropped 1 cm onto the ball, zero wheel torque. The upright pose is unstable: both engines topple, timing depends on rounding.",
                   mode="settle", steps=130, ctrl="none (open loop)", stride=3, slow=1.0, window=(0, 130)),
    "torque_step": dict(title="Open-loop wheel-torque step",
                        sub="2 N m on wheel 0 for 0.3 s (t = 0.5 to 0.8 s), no feedback. Forced response matches; later tilt diverges (unstable plant). Half speed.",
                        mode="step", steps=100, ctrl="none (open loop)", stride=2, slow=0.5, window=(40, 100)),
    "lqr_0.4": dict(title="LQR tracking 0.4 m/s", sub="Model-based LQR (dz_nom gains), command 0.4 m/s along +x.", mode="lqr", steps=400,
                    cmd=[0.4, 0, 0], ctrl="LQR", stride=4, slow=1.0, window=(0, 300)),
    "lqr_0.8": dict(title="LQR tracking 0.8 m/s", sub="Model-based LQR (dz_nom gains), command 0.8 m/s along +x.", mode="lqr", steps=400,
                    cmd=[0.8, 0, 0], ctrl="LQR", stride=4, slow=1.0, window=(0, 300)),
}


def push_scen(idx, alpha, mt, title, sub):
    phi = idx * 2 * np.pi / 16
    F = alpha * mt * G * np.array([np.cos(phi), np.sin(phi), 0.0])
    return dict(title=title, sub=sub, mode="lqr", steps=300, cmd=[0, 0.4, 0], push=dict(t0=200, F=F.tolist()), ctrl="LQR", stride=2,
                slow=0.5, window=(190, 300), idx=idx)


def main(names):
    lowprio()
    VID.mkdir(parents=True, exist_ok=True)
    design = Design("nom")
    m = load_cpu_model(design=design)
    rc, rw = np.load(OUT / "sweep_nom_cpu.npz"), np.load(OUT / "sweep_nom_warp.npz")
    fc, fw = rc["push_1.0__fell"], rw["push_1.0__fell"]
    scen = dict(SCEN)
    mt = design.m_total
    sb = [i for i in range(16) if not fc[i] and not fw[i]]
    fb = [i for i in range(16) if fc[i] and fw[i]]
    dis = [i for i in range(16) if fc[i] != fw[i]]
    print("push dirs: survive both", sb, "fall both", fb, "disagree", dis, [bool(fc[i]) for i in dis])
    if sb:
        scen["push_survive"] = push_scen(sb[0], 1.0, mt, "Push: both engines recover",
                                         f"1x body weight (39 N), 50 ms, direction {sb[0] * 22.5:.0f} deg, while driving 0.4 m/s along +y. Half speed.")
    if fb:
        scen["push_fall"] = push_scen(fb[0], 1.0, mt, "Push: both engines lose the ball",
                                      f"1x body weight (39 N), 50 ms, direction {fb[0] * 22.5:.0f} deg, while driving 0.4 m/s along +y. Half speed.")
    if dis:
        scen["push_disagree"] = push_scen(dis[0], 1.0, mt, "Push at the edge: engines disagree",
                                          f"1x body weight, 50 ms, direction {dis[0] * 22.5:.0f} deg: CPU {'falls' if fc[dis[0]] else 'recovers'}, "
                                          f"Warp {'falls' if fw[dis[0]] else 'recovers'}. Borderline case. Half speed.")
    meta = []
    r = mujoco.Renderer(m, height=PH, width=PW, max_geom=2000)
    md = mujoco.MjData(m)
    for name in (names or list(scen)):
        sc = scen[name]
        t0 = time.time()
        logs = {k: simulate(k, m, design, sc) for k in ("cpu", "warp")}
        a, b = sc["window"]
        idxs = list(range(a, min(b, sc["steps"]), sc["stride"]))
        fr = {}
        for k in logs:
            fr[k] = []
            for i in idxs:
                draw_scene(r, md, m, logs[k][i])
                fr[k].append(r.render().copy())
        ent = {k: [logs[k][i] for i in idxs] for k in logs}
        fell = {k: bool(max(e["tilt"] for e in logs[k]) > 70) for k in logs}
        if name == "push_disagree":
            word = lambda f: "falls" if f else "recovers"
            sc["sub"] = (f"1x body weight, 50 ms, direction {sc['idx'] * 22.5:.0f} deg: in the sweep run CPU falls, Warp recovers; in this render "
                         f"CPU {word(fell['cpu'])}, Warp {word(fell['warp'])}. Borderline case (GPU run is not bit-reproducible). Half speed.")
        frames = compose(fr["cpu"], fr["warp"], ent["cpu"], ent["warp"], sc, sc["title"])
        fps = 100.0 / sc["stride"] * sc["slow"]
        path = VID / f"mjx_{name}.mp4"
        write_mp4(frames, path, fps)
        frames[len(frames) // 2].save(VID / f"mjx_{name}.jpg", quality=85)
        meta.append(dict(name=name, title=sc["title"], description=sc["sub"], file=f"outputs/study/mjx_port/videos/mjx_{name}.mp4",
                         poster=f"outputs/study/mjx_port/videos/mjx_{name}.jpg", fps=fps, speed_factor=sc["slow"], frames=len(frames),
                         fell=fell, final_tilt={k: logs[k][-1]["tilt"] for k in logs}, mb=round(path.stat().st_size / 1e6, 2)))
        print(name, meta[-1], "%.0fs" % (time.time() - t0), flush=True)
    old = json.load(open(VID / "clips.json")) if (VID / "clips.json").exists() else []
    keep = [o for o in old if o["name"] not in {x["name"] for x in meta}]
    json.dump(keep + meta, open(VID / "clips.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
