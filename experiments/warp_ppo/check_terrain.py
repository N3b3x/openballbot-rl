"""Terrain smoke test on MuJoCo-Warp: does a shared height-field (book step) behave like the CPU engine?
LQR drives the nominal ballbot at 0.4 m/s towards +y over a step at y=0.6 m. CPU (ContactFrameStepper) vs Warp (N worlds,
start x offsets spread to emulate per-world variety). Reports ball height/y trajectories, falls, NaNs, speed.
python check_terrain.py [step_cm] [nworld]"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mjx_port"))
from common import *  # noqa
import mujoco
import cpu_ref


def generate_book_terrain(n, step_height_m=0.03, step_y=0.6, hz=2.0):
    # same as ballbot_gym/terrain/book.py curb case (inlined: the package import needs gymnasium)
    c = np.linspace(-5.0, 5.0, n)
    Y, _ = np.meshgrid(c, c, indexing="ij")
    return ((Y >= step_y).astype(float) * (step_height_m / hz)).flatten()

step_cm = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 256
T, SPEED = 5.0, 0.4
design = Design("nom")

m = load_cpu_model(design=design)
n = m.hfield_nrow[0]
m.hfield_data[:] = generate_book_terrain(n, step_height_m=step_cm / 100.0, step_y=0.6)
q0, v0 = reset_state(m)
nsteps = int(T / 0.01)


def tilt(qp):
    w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def run_cpu():
    be = cpu_ref.CpuBackend(m, 1)
    d = be.ds[0]
    be.reset(q0, v0)
    ctl = LQRBatch(1, model_json=design.json, tau_max=design.tau)
    cmd = np.array([[0.0, SPEED, 0.0]])
    out = []
    for t in range(nsteps):
        qp = d.qpos[None].copy(); qv = d.qvel[None].copy()
        cv = np.zeros((1, 6)); cv[0] = d.cvel[be.base]
        be.set_ctrl(ctl(qp, qv, cv[:, 3:6], cmd))
        for _ in range(ACTION_REPEAT): be.st.step(d)
        out.append((d.geom_xpos[be.bg].copy(), tilt(d.qpos[None])[0]))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def run_warp(N):
    from warp_sim import WarpBallbot
    import warp as wp
    be = WarpBallbot(m, N)
    # spread start x so worlds hit the step edge at slightly different contact geometry
    xs = np.linspace(-0.4, 0.4, N) if N > 1 else np.zeros(1)
    q = np.broadcast_to(q0, (N, m.nq)).copy()
    q[:, BQ] += xs; q[:, KQ] += xs
    be.reset(q, v0)
    ctl = LQRBatch(N, model_json=design.json, tau_max=design.tau)
    cmd = np.tile([0.0, SPEED, 0.0], (N, 1))
    out = np.zeros((nsteps, N, 3)); tl = np.zeros((nsteps, N))
    t0 = time.perf_counter()
    for t in range(nsteps):
        qp, qv, cv, bx, gx = be.state()
        be.set_ctrl(ctl(qp, qv, cv[:, 3:6], cmd))
        be.step(ACTION_REPEAT)
        qp, qv, cv, bx, gx = be.state()
        out[t] = gx; tl[t] = tilt(qp)
    wp.synchronize()
    return out, tl, time.perf_counter() - t0, xs, int(be.d.nacon.numpy()[0]), int(be.d.naconmax)


gc, tc = run_cpu()
gw, tw, secs, xs, nacon, naconmax = run_warp(NW)

res = dict(step_cm=step_cm, nworld=NW)
res["cpu"] = dict(final_y=float(gc[-1, 1]), max_ball_z=float(gc[:, 2].max()), min_ball_z=float(gc[:, 2].min()), max_tilt=float(tc.max()),
                  fell=bool(tc.max() > 70), t_cross=float(np.argmax(gc[:, 1] > 0.6) * 0.01))
fin = np.isfinite(gw).all((0, 2))
fell = (tw.max(0) > 70) | ~fin
ctr = np.argmin(np.abs(xs))
res["warp"] = dict(finite_frac=float(fin.mean()), fall_frac=float(fell.mean()), steps_per_s=float(NW * nsteps / secs),
                   centre_world=dict(final_y=float(gw[-1, ctr, 1]), max_ball_z=float(gw[:, ctr, 2].max()), max_tilt=float(tw[:, ctr].max()),
                                     t_cross=float(np.argmax(gw[:, ctr, 1] > 0.6) * 0.01)),
                   median_final_y=float(np.median(gw[-1, fin & ~fell, 1])) if (fin & ~fell).any() else None,
                   nacon_last=nacon, naconmax=naconmax)
# per-time ball height difference centre world vs CPU
res["centre_vs_cpu"] = dict(rms_dz_m=float(np.sqrt(np.mean((gw[:, ctr, 2] - gc[:, 2]) ** 2))),
                            rms_dy_m=float(np.sqrt(np.mean((gw[:, ctr, 1] - gc[:, 1]) ** 2))))
k = int(1.9 / 0.01)
res["centre_vs_cpu"]["pre_cross_rms_dz_m"] = float(np.sqrt(np.mean((gw[:k, ctr, 2] - gc[:k, 2]) ** 2)))
res["centre_vs_cpu"]["pre_cross_rms_dy_m"] = float(np.sqrt(np.mean((gw[:k, ctr, 1] - gc[:k, 1]) ** 2)))
res["centre_vs_cpu"]["pre_cross_tilt_rms_deg"] = float(np.sqrt(np.mean((tw[:k, ctr] - tc[:k]) ** 2)))
print(json.dumps(res, indent=1))
OUTD = ROOT / "outputs" / "study" / "warp_ppo"
json.dump(res, open(OUTD / f"terrain_check_{step_cm:g}cm.json", "w"), indent=1)
