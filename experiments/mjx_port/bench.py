"""Throughput: MuJoCo-Warp (GPU, CUDA graph) vs CPU ContactFrameStepper. Control step = 5 physics steps of 2 ms.
python bench.py warp 1 256 4096   |  python bench.py cpu"""
import sys, time, json
import numpy as np
from common import *  # noqa


def bench_warp(ns, reps=200):
    import warp as wp
    import mujoco_warp as mjw
    from warp_sim import WarpBallbot
    m = load_cpu_model()
    res = {}
    for n in ns:
        be = WarpBallbot(m, n, fix_frames=True)
        q, v = reset_state(m)
        be.reset(q, v)
        rng = np.random.default_rng(0)
        be.set_ctrl(rng.uniform(-1, 1, (n, 3)))     # small random torques keep contacts active
        for _ in range(3): be.step(ACTION_REPEAT)    # compile / warm
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for _ in range(ACTION_REPEAT):
                mjw.step(be.m, be.d)
        for _ in range(5): wp.capture_launch(cap.graph)
        wp.synchronize()
        qp0 = wp.array(np.broadcast_to(q, (n, m.nq)).astype(np.float32).copy(), dtype=float)
        t0 = time.perf_counter()
        for k in range(reps):
            if k % 50 == 0:          # periodic reset (no episode is allowed to run past ~0.5 s of free toppling)
                wp.copy(be.d.qpos, qp0); be.d.qvel.zero_(); be.d.qacc_warmstart.zero_()
            wp.capture_launch(cap.graph)
        wp.synchronize()
        dt = time.perf_counter() - t0
        fin = np.isfinite(be.d.qpos.numpy()).all(1).mean()
        qp = be.d.qpos.numpy()
        res[n] = dict(control_steps_per_s=n * reps / dt, physics_steps_per_s=n * reps * ACTION_REPEAT / dt,
                      ms_per_control_step_batch=1e3 * dt / reps, finite_frac=float(fin), nacon_mean=float(be.d.nacon.numpy()[0]) / n)
        print(n, res[n], flush=True)
        del be, cap
    return res


def bench_cpu(secs=10):
    import mujoco
    import cpu_ref
    lowprio()
    m = load_cpu_model()
    st = cpu_ref.cf.ContactFrameStepper(m)
    d = mujoco.MjData(m)
    q, v = reset_state(m)
    d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    n = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        d.ctrl[:] = rng.uniform(-1, 1, 3)
        for _ in range(ACTION_REPEAT): st.step(d)
        n += 1
        if n % 300 == 0:
            d.qpos[:] = q; d.qvel[:] = v   # keep it from toppling forever (reset)
    dt = time.perf_counter() - t0
    r = dict(control_steps_per_s=n / dt, physics_steps_per_s=n * ACTION_REPEAT / dt)
    # also stock mj_step (no Python frame hook) for reference
    d2 = mujoco.MjData(m); d2.qpos[:] = q; d2.qvel[:] = v
    n2 = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3:
        for _ in range(ACTION_REPEAT): mujoco.mj_step(m, d2)
        n2 += 1
        if n2 % 300 == 0: d2.qpos[:] = q; d2.qvel[:] = v
    r["stock_mj_step_control_steps_per_s"] = n2 / (time.perf_counter() - t0)
    print(r)
    return r


def bench_warp_lqr(ns=(256, 4096), steps=300):
    """Closed loop: numpy LQR on the host every control step (host<->device copy included), no resets. Reports throughput and falls."""
    import warp as wp
    from warp_sim import WarpBallbot
    m = load_cpu_model(design=Design("nom"))
    d = Design("nom")
    res = {}
    for n in ns:
        be = WarpBallbot(m, n)
        q, v = reset_state(m)
        be.reset(q, v)
        ctl = LQRBatch(n, model_json=d.json, tau_max=d.tau)
        rng = np.random.default_rng(0)
        ang = rng.uniform(0, 2 * np.pi, n)
        cmd = np.stack([0.4 * np.cos(ang), 0.4 * np.sin(ang), np.zeros(n)], 1)
        qp, qv, cv, bx, gx = be.state()
        be.step(ACTION_REPEAT); be.reset(q, v); qp, qv, cv, bx, gx = be.state()
        wp.synchronize(); t0 = time.perf_counter()
        for t in range(steps):
            be.set_ctrl(ctl(qp, qv, cv[:, 3:6], cmd))
            be.step(ACTION_REPEAT)
            qp, qv, cv, bx, gx = be.state()
        wp.synchronize(); dt = time.perf_counter() - t0
        fell = (tilt_deg(qp) > 70) | ~np.isfinite(qp).all(1)
        res[n] = dict(control_steps_per_s=n * steps / dt, fall_frac=float(fell.mean()), nonfinite_frac=float((~np.isfinite(qp).all(1)).mean()),
                      mean_speed_ratio=float(np.mean(np.sum(cv[~fell, 3:5] * cmd[~fell, :2], 1) / 0.16)))
        print("lqr-closed-loop", n, res[n], flush=True)
        del be
    return res


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "warp_lqr":
        r = bench_warp_lqr()
    elif which == "warp":
        r = bench_warp([int(x) for x in sys.argv[2:]])
    else:
        r = bench_cpu()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(OUT / f"bench_{which}.json", "w"), indent=1)
