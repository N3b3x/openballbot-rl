"""PhysX throughput: N cloned ballbots, GPU or CPU PhysX, warp tensor frontend, batched LQR in numpy on the host.

  isaac_venv/python bench_px.py --n 1 1024 4096 --device gpu --ctrl-steps 100
Reports physics steps/s per env summed over envs (env-steps/s, one env-step = one 2 ms physics step) for
 (a) sim only: constant wheel torque, no state read
 (b) closed loop: read state, batched LQR (numpy), write torques each 10 ms control step.
Short GPU runs only (the GPU is shared with training).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[1])
ap.add_argument("--device", default="gpu")
ap.add_argument("--ctrl-steps", type=int, default=60)
ap.add_argument("--design", default="nom")
ap.add_argument("--rollers", type=int, default=16)
ap.add_argument("--out", default=None)
a = ap.parse_args()
app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

import build_usd  # noqa: E402
from common import ROOT, DesignCfg, LQRCore  # noqa: E402

gpu = a.device == "gpu"
cfg = DesignCfg(a.design)
M = json.load(open(ROOT / "outputs/study/isaac_port/models" / f"{a.design}.json"))
results = []


def run(N):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
    ps.CreateTimeStepsPerSecondAttr(500)
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateEnableGPUDynamicsAttr(gpu)
    ps.CreateBroadphaseTypeAttr("GPU" if gpu else "MBP")
    if gpu:
        ps.CreateGpuMaxRigidContactCountAttr(2 ** 24)
        ps.CreateGpuMaxRigidPatchCountAttr(2 ** 22)
        ps.CreateGpuFoundLostPairsCapacityAttr(2 ** 22)
        ps.CreateGpuFoundLostAggregatePairsCapacityAttr(2 ** 24)
        ps.CreateGpuTotalAggregatePairsCapacityAttr(2 ** 24)
        ps.CreateGpuHeapCapacityAttr(2 ** 28)
        ps.CreateGpuTempBufferCapacityAttr(2 ** 28)
    UsdGeom.Xform.Define(stage, "/World/envs")
    paths = build_usd.build(stage, "/World/envs/env_0", M, mode="roller", n_rollers=a.rollers, roller_cap_r=0.02, with_ground=False)
    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    g.AddScaleOp().Set(Gf.Vec3f(4000, 4000, 1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    t0 = time.time()
    if N > 1:
        from isaacsim.core.cloner import GridCloner
        cl = GridCloner(spacing=1.5)
        cl.define_base_env("/World/envs")
        tgt = cl.generate_paths("/World/envs/env", N)
        cl.clone(source_prim_path="/World/envs/env_0", prim_paths=tgt, replicate_physics=True, copy_from_source=False)
    build_s = time.time() - t0
    import carb.settings
    carb.settings.get_settings().set_bool("/physics/suppressReadback", bool(gpu))
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    t1 = time.time()
    app.update()
    tl.pause()
    start_s = time.time() - t1
    enable_extension("omni.physx.tensors")
    import importlib
    T = importlib.import_module("omni.physics.tensors")
    sid = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    frontend = "warp" if gpu else "numpy"
    sv = T.create_simulation_view(frontend, sid)
    art = sv.create_articulation_view("/World/envs/env_*/robot/base")
    ball = sv.create_rigid_body_view("/World/envs/env_*/ball")
    n = art.count
    names = [p.split("/")[-1] for p in art.dof_paths[0]]
    wdof = [names.index(f"wheel_joint_{k}") for k in range(3)]
    ndof = len(names)
    sim = omni.physx.get_physx_simulation_interface()
    dev = "cuda:0" if gpu else "cpu"
    idx_np = np.arange(n, dtype=np.uint32)
    if gpu:
        idx = wp.array(idx_np, dtype=wp.uint32, device=dev)
    else:
        idx = idx_np.astype(np.int32)
    ctrl = LQRCore(cfg)

    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def put_forces(tau):  # tau (n,3) numpy
        f = np.zeros((n, ndof), np.float32)
        f[:, wdof] = tau
        art.set_dof_actuation_forces(wp.array(f, dtype=wp.float32, device=dev) if gpu else f, idx)

    def step_phys(k):
        dt = 0.002
        for _ in range(k):
            sim.simulate(dt, 0.0)
            sim.fetch_results()

    res = dict(N=int(n), device=a.device, build_s=build_s, start_s=start_s, links_per_env=int(art.max_links))
    # warm-up
    put_forces(np.zeros((n, 3), np.float32))
    step_phys(20)
    # (a) sim only
    put_forces(np.full((n, 3), 0.01, np.float32))
    t = time.perf_counter()
    K = a.ctrl_steps * 5
    step_phys(K)
    if gpu:
        wp.synchronize()
    dtm = time.perf_counter() - t
    res["sim_only_envsteps_per_s"] = n * K / dtm
    res["sim_only_phys_step_ms"] = 1e3 * dtm / K
    # (b) closed loop: read state, LQR on host, write torques; one control step = 5 physics steps
    cmd = np.zeros(3)
    Wm = ctrl.W
    t = time.perf_counter()
    for c in range(a.ctrl_steps):
        rt = to_np(art.get_root_transforms()).reshape(n, 7)
        rv = to_np(art.get_root_velocities()).reshape(n, 6)
        bv = to_np(ball.get_velocities()).reshape(n, 6)
        # vectorised LQR (same law as LQRCore.act, no reference governor/integral for speed: cmd = 0)
        q = rt[:, [6, 3, 4, 5]]
        w, x, y, z = q.T
        R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
        up = R[:, :, 2]
        ty, tx = np.arctan2(up[:, 0], up[:, 2]), np.arctan2(up[:, 1], up[:, 2])
        wb, wk = rv[:, 3:6], bv[:, 3:6]
        K_ = ctrl.K
        sy = np.stack([0 * ty, ty, wk[:, 1], wb[:, 1]], 1)
        sx = np.stack([0 * tx, tx, -wk[:, 0], -wb[:, 0]], 1)
        ty_ = -(sy @ K_)
        tx_ = -(sx @ K_)
        T_ = np.stack([-tx_, ty_, -5.0 * (0 - wb[:, 2])], 1)
        Tb = np.einsum("nji,nj->ni", R, T_)
        tau = np.clip(np.linalg.solve(Wm.T, Tb.T).T, -cfg.tau_max, cfg.tau_max).astype(np.float32)
        put_forces(tau)
        step_phys(5)
    if gpu:
        wp.synchronize()
    dtc = time.perf_counter() - t
    res["closed_loop_envsteps_per_s"] = n * a.ctrl_steps * 5 / dtc
    res["closed_loop_ctrl_ms"] = 1e3 * dtc / a.ctrl_steps
    print("RESULT", json.dumps(res), flush=True)
    results.append(res)
    del art, ball, sv


for N in a.n:
    try:
        run(N)
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        print("RESULT", json.dumps(dict(N=N, device=a.device, error=repr(e)[:300])), flush=True)
if a.out:
    json.dump(results, open(a.out, "w"), indent=1)
app.close()
