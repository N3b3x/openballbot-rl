"""PhysX (Isaac Sim 6.2) backend, single environment on CPU PhysX through the omni.physics.tensors numpy frontend.

Must be imported inside the Isaac python (C:/Users/Owner/isaac_venv).  The SimulationApp is a process singleton.
"""
import os
import sys

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

_APP = None


def app():
    global _APP
    if _APP is None:
        _APP = SimulationApp({"headless": True})
    return _APP


from common import REPEAT, ROOT, DesignCfg  # noqa: E402


def _qwxyz(xyzw):
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _R(q):  # (w,x,y,z)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PXBackend:
    name = "PhysX"

    def __init__(self, cfg: DesignCfg, fric=1.0, mode="roller", solver_pos=32, solver_vel=4, hz=500, solver="TGS",
                 gpu=False, mu_grip=1.0, tag="/World/env", n_rollers=16, rho=0.02, element="roller"):
        app()
        import omni.usd
        import omni.physx
        import omni.timeline
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils
        import build_usd
        self.cfg = cfg
        import json
        self.M = json.load(open(ROOT / 'outputs/study/isaac_port/models' / f'{cfg.design}.json'))
        self.mode = mode
        ctx = omni.usd.get_context()
        ctx.new_stage()
        self.stage = stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr(9.81)
        ps = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
        ps.CreateTimeStepsPerSecondAttr(hz)
        ps.CreateSolverTypeAttr(solver)
        ps.CreateEnableGPUDynamicsAttr(gpu)
        ps.CreateBroadphaseTypeAttr("GPU" if gpu else "MBP")
        ps.CreateEnableCCDAttr(False)
        ps.CreateEnableStabilizationAttr(False)
        self.paths = build_usd.build(stage, tag + "_0", self.M, mode=mode, fric=fric, mu_grip=mu_grip,
                                     solver_pos=solver_pos, solver_vel=solver_vel, n_rollers=n_rollers, roller_cap_r=rho, element=element)
        self.hz = hz
        self.n_rollers = n_rollers
        self.sub = int(round(hz * 0.002))   # physics substeps per MuJoCo physics step (1 at 500 Hz)
        self.px = omni.physx.get_physx_interface()
        self.sim = omni.physx.get_physx_simulation_interface()
        self.tl = omni.timeline.get_timeline_interface()
        self.tl.play()
        app().update()
        self.tl.pause()
        self.t = 0.0
        self._views(stage, tag, UsdUtils)
        self._init_pose = None
        self.reset(0.0)

    def _views(self, stage, tag, UsdUtils):
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.physx.tensors")
        import importlib
        T = importlib.import_module("omni.physics.tensors")
        sid = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self.sv = T.create_simulation_view("numpy", sid)
        self.art = self.sv.create_articulation_view(self.paths["base"])
        self.ball = self.sv.create_rigid_body_view(self.paths["ball"])
        self.dofn = list(self.art.dof_paths[0]) if hasattr(self.art, "dof_paths") else None
        names = [p.split("/")[-1] for p in self.art.dof_paths[0]]
        self.wdof = [names.index(f"wheel_joint_{k}") for k in range(3)]
        self.rdof = [i for i, n in enumerate(names) if n.startswith("roller_joint")]
        pass
        self.ndof = len(names)
        self.rbv = []
        if self.mode == "roller":
            self.cview = self.sv.create_rigid_contact_view(self.paths["base"].rsplit("/", 1)[0] + "/rl_*_*",
                                                          filter_patterns=[self.paths["ball"]])
            self.sensor_paths = list(self.cview.sensor_paths) if hasattr(self.cview, "sensor_paths") else None
            self.rbv = self.sv.create_rigid_body_view(self.paths["base"].rsplit("/", 1)[0] + "/rl_*_*")
            self.roller_paths = list(self.rbv.prim_paths)
        else:
            self.cview = None
        self.sim_idx = np.array([0], dtype=np.int32)

    def reset(self, yaw=0.0, drop=0.0):
        """Start from the MuJoCo-settled configuration, optionally lifted by `drop` metres (settling test)."""
        assert abs(yaw) < 1e-9, "PX backend builds the rig at yaw 0"
        M = self.M
        bp = np.array(M["base"]["pos_settled"]) + np.array([0, 0, drop])
        q = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)   # xyzw
        rt = np.zeros((1, 7), np.float32)
        rt[0, :3] = bp
        rt[0, 3:] = q
        self.art.set_root_transforms(rt, self.sim_idx)
        self.art.set_root_velocities(np.zeros((1, 6), np.float32), self.sim_idx)
        z = np.zeros((1, self.ndof), np.float32)
        self.art.set_dof_positions(z, self.sim_idx)
        self.art.set_dof_velocities(z, self.sim_idx)
        bt = np.zeros((1, 7), np.float32)
        bt[0, :3] = np.array(M["ball_centre_world"]) + np.array([0, 0, drop])
        bt[0, 6] = 1.0
        self.ball.set_transforms(bt, self.sim_idx)
        self.ball.set_velocities(np.zeros((1, 6), np.float32), self.sim_idx)
        if self.mode == "roller":
            # rollers are articulation links: their placement follows the articulation joint positions (zeros) set above
            pass
        self.t = 0.0
        self.last_forces = [None] * 3

    # --- state in the MuJoCo-backend convention
    def state(self):
        rt = self.art.get_root_transforms()[0]
        rv = self.art.get_root_velocities()[0]
        q = _qwxyz(rt[3:])
        R = _R(q)
        bt = self.ball.get_transforms()[0]
        bv = self.ball.get_velocities()[0]
        dq = self.art.get_dof_velocities()[0]
        return dict(R=R, wb_w=rv[3:6].astype(float), wk_w=bv[3:6].astype(float), vel=rv[:3].astype(float),
                    pbody=rt[:3].astype(float), pball=bt[:3].astype(float), wheel_w=dq[self.wdof].astype(float), t=self.t)

    def step_ctrl(self, tau, push=None):
        f = np.zeros((1, self.ndof), np.float32)
        f[0, self.wdof] = tau
        self.art.set_dof_actuation_forces(f, self.sim_idx)
        dt = 1.0 / self.hz
        for _ in range(REPEAT * self.sub):
            if push is not None:
                F = np.zeros((1, self.art.max_links, 3), np.float32)
                F[0, 0] = push
                self.art.apply_forces_and_torques_at_position(F, None, None, self.sim_idx, True)
            self.sim.simulate(dt, self.t)
            self.sim.fetch_results()
            self.t += dt
        self.sv.clear_forces() if False else None

    def pose(self):
        rt = self.art.get_root_transforms()[0]
        bt = self.ball.get_transforms()[0]
        dp = self.art.get_dof_positions()[0]
        bq = _qwxyz(rt[3:])
        kq = _qwxyz(bt[3:])
        origin = bt[:3] + _R(kq) @ np.array([0, 0, 0.14])      # MuJoCo ball body origin is 0.14 above the sphere centre
        return dict(base_pos=rt[:3].astype(float), base_quat=bq, ball_pos=origin, ball_quat=kq, wheel_q=dp[self.wdof].astype(float),
                    ball_center=bt[:3].astype(float))

    def wheel_ball_forces(self):
        """Per wheel: force-weighted contact point and total force on the ball from the wheel's rollers (contact reporting)."""
        if self.cview is None:
            return []
        fm = np.asarray(self.cview.get_contact_force_matrix(1.0 / self.hz)).reshape(-1, 3).astype(float)   # force on roller from ball
        pos = self.rbv.get_transforms()[:, :3].astype(float)
        bc = self.ball.get_transforms()[0][:3].astype(float)
        wid = np.array([int(pth.split("/rl_")[-1].split("_")[0]) for pth in self.roller_paths])
        out = []
        for k in range(3):
            sel = wid == k
            F = -fm[sel].sum(0)
            mag = np.linalg.norm(fm[sel], axis=1)
            if mag.sum() < 1e-6:
                continue
            c = (pos[sel] * mag[:, None]).sum(0) / mag.sum()
            u = (c - bc) / np.linalg.norm(c - bc)
            p = bc + self.M["ball"]["radius"] * u
            fn = float(-(F @ u)) * -1
            ft = float(np.linalg.norm(F - (F @ u) * u))
            out.append((k, p, F, abs(float(F @ u)), ft))
        return out
