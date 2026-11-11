"""GPU-resident PPO for the nominal ballbot on MuJoCo-Warp (simulation only).

Whole loop on the GPU: Warp physics (contact-frame fix, CUDA graph) -> torch views of the Warp arrays -> observation, reward,
termination, command resampling, resets and PPO updates, with no host copies. Mirrors the CPU env of configs/env/dz_nom.yaml:
command-tracking task, reward "err" with lean term, obs = orientation rotvec, angular vel, vel, motor state, last action, cmd,
heading, velocity-error integral (22-D). Domain randomisation kept: obs noise, 1-step latency (p=0.2), torque scale +-15 %, random pushes.
NOT randomised (needs per-world model fields, untested in Warp): mass, ball friction, wheel damping.

  python train.py --name wp_nom_s0 --nworld 2048 --steps 50e6
"""
import os, argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mjx_port"))
from common import *  # noqa
from warp_sim import WarpBallbot  # noqa

@wp.kernel
def _motor_kernel(qvel: wp.array2d[float], cmd: wp.array2d[float], tau_act: wp.array2d[float], ctrl: wp.array2d[float],
                  dt: float, lag: float, tau_stall: float, omega_nl: float, fc: float, vmax: float):
    i = wp.tid()
    for k in range(3):
        w = qvel[i, 6 + k]
        ta = tau_act[i, k] + (cmd[i, k] - tau_act[i, k]) * (dt / (lag + dt))
        tau_act[i, k] = ta
        tlim = tau_stall * wp.clamp(1.0 - wp.abs(w) / omega_nl, 0.0, 1.0)
        t = ta
        if ta * w > 0.0:
            t = wp.clamp(ta, -tlim, tlim)
        t = t - fc * wp.tanh(w / 0.1)
        ctrl[i, k] = wp.clamp(t, -vmax, vmax)


DEV = "cuda"
STEP_Y = (0.5, 1.1, 1.7)          # edge positions (m) and cumulative heights (m); mirrored for negative y
STEP_H = (0.01, 0.03, 0.06)
START_Y = (0.0, 0.8, 1.4, 2.2)    # centres of the flat segments, used (mirrored) as start rows


# VL53L9CX-like depth sensor (datasheet: 54x42 zones, ~55x42 deg FoV, 5 cm to ~9 m); we bin it to cut ray count.
SENSOR_FOV = (np.radians(55.0), np.radians(42.0))
SENSOR_PITCH = np.radians(-25.0)       # head-mounted, pitched down
SENSOR_UP = 0.15                       # metres above the base origin
SENSOR_RANGE = (0.05, 4.0)
SCAN_FWD = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)   # teacher height-scan grid in the command-aligned frame
SCAN_LAT = (-0.3, -0.15, 0.0, 0.15, 0.3)


# floor material classes: name -> (sliding friction, contact time constant s). Plausible values, NOT measured; softness is a contact-model proxy.
SURFACES = {"concrete": (0.9, 0.02), "hardwood": (0.6, 0.02), "carpet": (1.1, 0.045), "grass": (0.8, 0.05),
            "gravel": (0.7, 0.03), "wet_tile": (0.3, 0.02), "rubber": (1.3, 0.02)}


def terrain_height_t(y):
    ay, h = y.abs(), torch.zeros_like(y)
    for e, hh in zip(STEP_Y, STEP_H):
        h = torch.where(ay >= e, torch.full_like(h, hh), h)
    return h


def terrain_height(y):
    ay, h = np.abs(y), np.zeros_like(y, dtype=float)
    for e, hh in zip(STEP_Y, STEP_H):
        h = np.where(ay >= e, hh, h)
    return h
OBS_DIM, ACT_DIM = 25, 3      # 22 proprio/command dims + goal vector (2) + goal flag (1)


class GpuBallbot:
    def __init__(self, nworld, seed=0, dr=True, design="nom", cfg_name=None, terrain=False, sensor="none", depth_res=(18, 14), depth_noise=0.005, mount="head", edge_p=0.0, tiles=False, tile_seed=0, tile_kinds=None, surfaces=False, hist_k=0, asym=False, goal_p=0.0, pull=False, designs=None, design_cond=False):
        self.N = N = nworld
        self.g = torch.Generator(device=DEV); self.g.manual_seed(seed)
        self.dr = dr
        import yaml
        design_o = Design(design)
        cfg = yaml.safe_load(open(ROOT / "configs" / "env" / f"dz_{cfg_name or design.split('+')[0]}.yaml"))["env"]
        self.motor = cfg.get("motor")
        self.tau, self.cmd_vmax_cfg, self.cmd_wmax_cfg = float(cfg["max_wheel_velocity"]), float(cfg["cmd_vmax"]), float(cfg["cmd_wmax"])
        self.drc = cfg["dr"]
        self.tiles = tiles
        def _load(name):
            d_o = Design(name)
            if tiles:      # big tiled height-field map (see tiles.py): patch the xml hfield size / resolution
                import re, tiles as TL
                src = open(d_o.xml).read()
                src = re.sub(r'<hfield name="terrain" nrow="\d+" ncol="\d+" size="[^"]*"', f'<hfield name="terrain" nrow="{TL.NCELL}" ncol="{TL.NCELL}" size="{TL.HALF} {TL.HALF} {TL.HZ} 0.1"', src)
                src = re.sub(r'(<geom name="terrain" type="hfield" hfield="terrain" size=")[^"]*"', rf'\g<1>{TL.HALF} {TL.HALF} 0.1"', src)
                if os.environ.get("PLANE_FLOOR"):      # diagnostics (nan_hunt.py): flat plane instead of the height field; terrain features vanish
                    src = re.sub(r'<geom name="terrain" type="hfield" hfield="terrain" size="[^"]*"', f'<geom name="terrain" type="plane" size="{TL.HALF} {TL.HALF} 0.1"', src)
                patched = d_o.xml.parent / f"_tiles_{d_o.xml.name}"
                open(patched, "w").write(src); d_o.xml = patched
            if os.environ.get("RETAINER"):      # Rezero-style ball arrestors (see retainer.py): RETAINER="gap_mm,fric"
                from retainer import patch_xml
                _rs = os.environ["RETAINER"].split(",")      # gap_mm, roller friction, floor collision (1/0), mount polar angle deg (0 = opposite the wheel)
                _rv = [float(x) for x in _rs] + [0.0, 0.0, 1.0, 0.0][len(_rs):]
                patched = d_o.xml.parent / f"_ret_{d_o.xml.name}"
                patch_xml(d_o.xml, patched, gap=_rv[0] / 1000.0, fric=_rv[1], floor=_rv[2] > 0, phi=_rv[3]); d_o.xml = patched
            return load_cpu_model(design=d_o), d_o
        self.design_names = list(designs) if designs else [design]
        loaded = [_load(n) for n in self.design_names]
        m, design_o = loaded[0]
        # S118: offset of the ball centre (the_ball geom) from the ball body origin, per design. The origin sits 0.14 m (nominal)
        # from the centre and rotates with the ball, so goal logic measured at the origin chases a point that swings by up to
        # +-0.14 m as the ball rolls; with goal_centre=True the goal logic uses the true centre (ball_xy).
        self.ball_off_tab = torch.tensor(np.stack([mk.geom_pos[mk.geom('the_ball').id] for mk, _ in loaded]), dtype=torch.float32, device=DEV)
        self.goal_centre = False
        if tiles:
            terrain = True
        self.mjm = m
        self.terrain = terrain
        self.edge_p = edge_p if terrain else 0.0
        if tiles:
            import tiles as TL
            H, self.tile_info = TL.build(tile_seed, tile_kinds)
            if os.environ.get('TILES_FLAT'):      # evaluation only: a perfectly flat 24 m world (clean steady-state tracking benchmark)
                H = np.full_like(H, TL.BASE)
            m.hfield_data[:] = TL.data_for_mujoco(H)
            self.hmap = torch.tensor(H, dtype=torch.float32, device=DEV)
            self.tile_xy = torch.tensor([[t["cx"], t["cy"]] for t in self.tile_info], dtype=torch.float32, device=DEV)
            self.tile_kind = torch.tensor([(TL.TYPES + ["blocks"]).index(t["kind"]) for t in self.tile_info], device=DEV)      # "blocks" (navigation test) is index len(TYPES)
            self.tile_sev = torch.tensor([t["severity"] for t in self.tile_info], dtype=torch.float32, device=DEV)
            self.cur_tile = torch.zeros(nworld, dtype=torch.long, device=DEV)
        elif terrain:      # one shared height-field (Warp has no per-world hfield): stepped ridge in y, mirrored, single steps of 1, 2, 3 cm
            n = int(m.hfield_nrow[0]); c = np.linspace(-5.0, 5.0, n)
            m.hfield_data[:] = (terrain_height(np.broadcast_to(c[:, None], (n, n))) / 2.0).flatten()
        self.be = WarpBallbot(m, N, fix_frames=os.environ.get("FIX_FRAMES", "1") == "1", nconmax=int(os.environ.get("NCONMAX", "16")), njmax=int(os.environ.get("NJMAX", "64")))      # diagnostics knobs (nan_hunt.py); defaults unchanged
        d = self.be.d
        # ---- per-world robot design (fields that differ between the compiled designs are batched; see explore_designs.py)
        self.K = K = len(self.design_names)
        dts = [dd for _, dd in loaded]
        self.didx = torch.zeros(N, dtype=torch.long, device=DEV)
        self.tau_tab = torch.tensor([dd.tau for dd in dts], dtype=torch.float32, device=DEV)
        self.f_tab = torch.tensor([dd.f for dd in dts], dtype=torch.float32, device=DEV)
        self.q0_tab = torch.tensor(np.stack([reset_state(mk)[0] for mk, _ in loaded]), dtype=torch.float32, device=DEV)
        if os.environ.get("RETAINER"):      # start from the loaded seat so the arrestors do not wedge the ball at t = 0 (retainer.seat_offset)
            from retainer import seat_offset
            for i, (mk, _) in enumerate(loaded):
                self.q0_tab[i, 2] = self.q0_tab[i, 12] + seat_offset(mk)
        self.mass_tab = torch.tensor(np.stack([mk.body_mass for mk, _ in loaded]), dtype=torch.float32, device=DEV)
        self.damp_tab = torch.tensor(np.stack([mk.dof_damping for mk, _ in loaded]), dtype=torch.float32, device=DEV)
        bi, ki = m.body("base").id, m.body("ball").id
        ref = [np.log(self.f_tab[0].item()) * 0, 0, 0, 0, 0, 0]
        dv = []
        # design vector reference = the nominal design (as in every training run, where 'nom' is loaded first); evaluating held-out
        # designs without nom in the list used to make the first evaluated design the reference, which mislabelled every design
        if self.design_names[0] == "nom":
            ref_d, ref_m = dts[0], m
        else:
            ref_d = Design("nom"); ref_m = load_cpu_model(design=ref_d)
        for (mk, dd) in loaded:
            dv.append([np.log(dd.f), np.log(dd.m_total / ref_d.m_total), np.log(dd.tau / ref_d.tau),
                       np.log(mk.body_mass[ki] / ref_m.body_mass[ki]), np.log(mk.dof_armature[6] / ref_m.dof_armature[6]), np.log(mk.body_mass[bi] / ref_m.body_mass[bi])])
        self.dvec_tab = torch.tensor(dv, dtype=torch.float32, device=DEV)
        self.dstack, self.dw = {}, {}
        if K > 1:
            assert not self.motor, "design randomisation with the motor model is not implemented"
            import dataclasses, mujoco_warp as mjw
            mws = [self.be.m] + [mjw.put_model(mk) for mk, _ in loaded[1:]]
            skipped = []
            for f in dataclasses.fields(self.be.m):
                arrs = [getattr(w, f.name) for w in mws]
                if not all(isinstance(a, wp.array) for a in arrs) or any(a.shape != arrs[0].shape for a in arrs):
                    continue
                nps = [a.numpy() for a in arrs]
                if all(np.array_equal(nps[0], x) for x in nps[1:]):
                    continue
                if arrs[0].shape[0] != 1 or nps[0].dtype != np.float32:
                    skipped.append(f.name); continue
                st = torch.tensor(np.stack([x[0] for x in nps]), device=DEV)
                self.dstack[f.name] = st
                self.dw[f.name] = st[0:1].repeat(N, *([1] * (st.dim() - 1))).contiguous()
                setattr(self.be.m, f.name, wp.from_torch(self.dw[f.name], dtype=arrs[0].dtype))
            self.design_fields_skipped = skipped
        self.sensor, self.depth_noise, self.mount = sensor, depth_noise, mount
        self.sdim = {"none": 0, "scan": len(SCAN_FWD) * len(SCAN_LAT), "depth": depth_res[0] * depth_res[1] * (2 if mount == "cones" else 1)}[sensor]
        self.goal_p, self.pull = goal_p, pull
        self.push_scale = 1.0
        self.cross_p = 0.0
        self.lqr, self.rs = None, 1.0
        self.fall_pen = 2.0
        # S101 test: deliberate stall termination (commanded > 0.15 m/s but achieved along the command < 1/5 of it for stall_steps) with penalty stall_pen; 0 = off
        self.stall_pen, self.stall_steps = 0.0, 100
        self.stall_cnt = torch.zeros(N, dtype=torch.long, device=DEV)
        self.dr_mbody = [1.0, 1.0]
        self.dr_inertia = bool(os.environ.get('DR_INERTIA'))      # scale body inertia with the randomised mass (physically consistent); default off keeps older runs reproducible
        self.dr_d = self.dr_l = self.dr_n = 1.0      # scales of the damping spread / latency probability / observation noise
        self.rw_track, self.rw_pen = 1.0, 1.0      # multipliers: tracking reward / (lean, action-rate, action-size penalties)
        import yaml as _y      # per-design push strength: each design config carries its own push range (scaled with robot mass), use it relative to the nominal 5 to 20 N
        _pt = []
        for _n in ([design] if not designs else list(designs)):
            try:
                _pt.append(float(_y.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "env" / f"dz_{_n}.yaml"))["env"]["dr"]["push_force"][1]) / 20.0)
            except Exception:
                _pt.append(1.0)
        self.push_tab = torch.tensor(_pt, device=DEV)
        self.dr_m = self.dr_f = self.dr_t = 1.0     # scale of the mass / friction / torque randomisation ranges
        self.goal_mode = torch.zeros(N, dtype=torch.bool, device=DEV)
        self.goal = torch.zeros(N, 2, device=DEV)
        self.path_left = torch.zeros(N, dtype=torch.long, device=DEV)
        self.pull_left, self.pull_t, self.pull_f = torch.zeros(N, dtype=torch.long, device=DEV), torch.zeros(N, device=DEV), torch.zeros(N, 3, device=DEV)
        self.hist_k, self.asym = hist_k, asym
        self.hdim = hist_k * (OBS_DIM + ACT_DIM)
        self.ddim = 6 if design_cond else 0
        self.pdim = (40 + (6 if self.K > 1 else 0)) if asym else 0
        self.obs_dim = OBS_DIM + self.sdim + self.hdim + self.ddim + self.pdim
        self.hist = torch.zeros(N, max(hist_k, 1), OBS_DIM + ACT_DIM, device=DEV)
        self._last_noisy = torch.zeros(N, OBS_DIM, device=DEV)
        if sensor == "depth":
            import mujoco_warp as mjw
            w, h = depth_res
            az, el = np.meshgrid((np.arange(w) + 0.5) / w * SENSOR_FOV[0] - SENSOR_FOV[0] / 2,
                                 SENSOR_PITCH + SENSOR_FOV[1] / 2 - (np.arange(h) + 0.5) / h * SENSOR_FOV[1])
            dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
            off = np.zeros_like(dirs)
            if mount == "cones":     # the two yellow sensor cones of ballbot.xml: base-frame (+-0.17, -0.01, -0.06), rigid with the body, aimed outward and 25 deg down
                dirs = np.concatenate([dirs, dirs * np.array([-1.0, -1.0, 1.0])])
                off = np.concatenate([np.tile([0.17, -0.01, -0.06], (len(dirs) // 2, 1)), np.tile([-0.17, -0.01, -0.06], (len(dirs) // 2, 1))])
            self.ray_dirs = torch.tensor(dirs, dtype=torch.float32, device=DEV)
            self.ray_off = torch.tensor(off, dtype=torch.float32, device=DEV)
            R = dirs.shape[0]
            self.rc = mjw.create_render_context(m, nworld=N, cam_res=(8, 8), render_rgb=False, render_depth=False, enabled_geom_groups=[0, 1, 2])
            self.r_dist = wp.empty((N, R), dtype=float); self.r_gid = wp.empty((N, R), dtype=int); self.r_nrm = wp.empty((N, R), dtype=wp.vec3)
            self.r_ex = wp.array(np.full(R, m.body("base").id, dtype=np.int32), dtype=int)
            from mujoco_warp._src.types import vec6
            self.r_gg = vec6(-1, -1, -1, -1, -1, -1)
        if self.motor:      # realistic actuator (first-order lag, torque-speed curve, Coulomb friction) evaluated every physics substep inside the CUDA graph
            import mujoco_warp as mjw
            mm = self.motor
            self.motor_cmd = torch.zeros(N, 3, device=DEV)
            self.tau_act = torch.zeros(N, 3, device=DEV)
            wc, wt, wq, wctrl = wp.from_torch(self.motor_cmd), wp.from_torch(self.tau_act), d.qvel, d.ctrl
            args = [wq, wc, wt, wctrl, float(m.opt.timestep), float(mm.get("lag", 0.002)), float(mm.get("tau_stall", self.tau)),
                    float(mm.get("omega_nl", 1e9)), float(mm.get("fc", 0.0)), float(self.tau)]
            wp.launch(_motor_kernel, dim=N, inputs=args)           # compile before capture
            self.tau_act.zero_()
            with wp.ScopedCapture() as cap:
                for _ in range(ACTION_REPEAT):
                    wp.launch(_motor_kernel, dim=N, inputs=args)
                    mjw.step(self.be.m, d)
            self.be._graph = cap.graph
        if dr:      # per-world physical randomisation: mass (base, ball), ball-wheel transverse friction, wheel damping
            import mujoco_warp as mjw
            from mujoco_warp._src.types import vec5
            self.nom_mass = torch.tensor(m.body_mass, dtype=torch.float32, device=DEV)
            self.nom_fric = torch.tensor(m.pair_friction, dtype=torch.float32, device=DEV)
            self.nom_damp = torch.tensor(m.dof_damping, dtype=torch.float32, device=DEV)
            self.t_mass = self.mass_tab[0:1].repeat(N, 1).contiguous()
            self.t_fric = self.nom_fric.repeat(N, 1, 1).contiguous()
            self.t_damp = self.damp_tab[0:1].repeat(N, 1).contiguous()
            self.be.m.body_mass = wp.from_torch(self.t_mass, dtype=wp.float32)
            self.par = [int(x) for x in m.body_parentid]
            self.t_sub = None
            if 'body_subtreemass' not in self.dw:      # single design: the field is not batched yet
                self.t_sub = self._subtree(self.t_mass).contiguous()
                self.be.m.body_subtreemass = wp.from_torch(self.t_sub, dtype=wp.float32)
            else:
                self.dw['body_subtreemass'].copy_(self._subtree(self.t_mass))
            self.be.m.pair_friction = wp.from_torch(self.t_fric, dtype=vec5)
            self.be.m.dof_damping = wp.from_torch(self.t_damp, dtype=wp.float32)
            self.body_ids = [m.body("base").id, m.body("ball").id]
        self.surfaces = surfaces and dr
        if self.surfaces:      # floor material classes: ball-floor sliding friction and contact softness per world
            from mujoco_warp._src.types import vec5
            ng = m.ngeom
            self.surf_geoms = [m.geom("the_ball").id, m.geom("terrain").id]
            self.t_gfric = torch.tensor(m.geom_friction, dtype=torch.float32, device=DEV).repeat(N, 1, 1).contiguous()
            self.t_gsolref = torch.tensor(m.geom_solref, dtype=torch.float32, device=DEV).repeat(N, 1, 1).contiguous()
            self.be.m.geom_friction = wp.from_torch(self.t_gfric, dtype=wp.vec3)
            self.be.m.geom_solref = wp.from_torch(self.t_gsolref, dtype=wp.vec2)
            self.surf_tab = torch.tensor([[v[0], v[1]] for v in SURFACES.values()], dtype=torch.float32, device=DEV)
            self.surf_id = torch.zeros(N, dtype=torch.long, device=DEV)
        self.qpos, self.qvel = wp.to_torch(d.qpos), wp.to_torch(d.qvel)
        self.cvel, self.xpos, self.gxpos = wp.to_torch(d.cvel), wp.to_torch(d.xpos), wp.to_torch(d.geom_xpos)
        self.xfrc, self.ctrl, self.warm = wp.to_torch(d.xfrc_applied), wp.to_torch(d.ctrl), wp.to_torch(d.qacc_warmstart)
        self.base, self.ball_g = self.be.base, self.be.ball_gid
        q0, _ = reset_state(m)
        self.q0 = torch.tensor(q0, dtype=torch.float32, device=DEV)
        self.cmd_vmax, self.cmd_wmax, self.zero_p = self.cmd_vmax_cfg, self.cmd_wmax_cfg, 0.1
        self.tau_w = self.tau_tab[self.didx]; self.f_w = self.f_tab[self.didx]
        self.max_steps, self.resample = 800, 300
        z = lambda *s: torch.zeros(*s, device=DEV)
        self.cmd, self.vint, self.last_a, self.prev_a = z(N, 3), z(N, 2), z(N, 3), z(N, 3)
        self.has_prev = torch.zeros(N, dtype=torch.bool, device=DEV)
        self.tq = torch.ones(N, device=DEV)
        self.push_t, self.push_left, self.push_f = z(N), torch.zeros(N, device=DEV, dtype=torch.long), z(N, 3)
        self.steps = torch.zeros(N, dtype=torch.long, device=DEV)
        self.reset(torch.ones(N, dtype=torch.bool, device=DEV))
        self.steps = torch.randint(0, self.max_steps, (N,), device=DEV, generator=self.g)   # desynchronise episode phase

    def U(self, *s, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*s, device=DEV, generator=self.g)

    def enable_motion_ref(self, curve_p=0.5, w_curve=1.0):
        """S105: train on commands shaped by the motion-reference layer (motion_ref.py) that the robot will see at deployment:
        slew- and turn-rate-limited velocity, braking into goals and a position hold at arrival. A fraction curve_p of the resampled
        velocity commands also turns steadily at up to w_curve rad/s, so smoothly curving commands are in the training distribution."""
        from motion_ref import MotionRef
        self.mref = MotionRef(self.N, DEV)
        self.curve_p, self.w_curve = curve_p, w_curve
        self.cmd_des = self.cmd[:, :2].clone()
        self.cmd_w = torch.zeros(self.N, device=DEV)

    def _shape_cmd(self, newly):
        """Pass the desired command through the motion-reference layer (no-op unless enable_motion_ref was called)."""
        if getattr(self, "mref", None) is None:
            return
        N = self.N
        w_new = torch.where(self.U(N) < self.curve_p, self.U(N, lo=-self.w_curve, hi=self.w_curve), torch.zeros(N, device=DEV))
        self.cmd_w = torch.where(newly, w_new, self.cmd_w)
        self.cmd_des = torch.where(newly[:, None], self.cmd[:, :2], self.cmd_des)
        c, s = torch.cos(self.cmd_w * 0.01), torch.sin(self.cmd_w * 0.01)
        self.cmd_des = torch.stack([c * self.cmd_des[:, 0] - s * self.cmd_des[:, 1], s * self.cmd_des[:, 0] + c * self.cmd_des[:, 1]], 1)
        gm = self.goal_mode if self.goal_p > 0 else torch.zeros(N, dtype=torch.bool, device=DEV)
        vd = torch.where(gm[:, None], self.cmd[:, :2], self.cmd_des)
        out = self.mref.step(vd, self.f_w, self.ball_xy(), self.goal, goal_on=gm)
        self.cmd = torch.cat([out, self.cmd[:, 2:]], 1)

    def sample_cmd(self, mask):
        """Resample the command (and goal mode) of the worlds in mask (bool over N). Computed for all worlds and selected with where: no host sync."""
        N = self.N
        sp, ang, wz = self.U(N, hi=self.cmd_vmax) * self.f_w, self.U(N, hi=2 * np.pi), self.U(N, lo=-self.cmd_wmax, hi=self.cmd_wmax)
        c = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang), wz], 1)
        c = torch.where((self.U(N) < self.zero_p)[:, None], torch.zeros_like(c), c)
        if self.edge_p > 0:      # edge-focused: with prob edge_p drive across the ridge (outward while |y| < 1.8 m, back inward beyond)
            y = self.qpos[:, 11]
            out = torch.where(y.abs() < 0.02, torch.where(self.U(N) < 0.5, -1.0, 1.0), torch.sign(y))
            tow = torch.where(y.abs() < 1.8, out, -out)
            ang_e = tow * (np.pi / 2) + self.U(N, lo=-0.6, hi=0.6)
            spd = self.U(N, lo=0.15, hi=self.cmd_vmax) * self.f_w
            m_e = (self.U(N) < self.edge_p)[:, None]
            ce = torch.stack([spd * torch.cos(ang_e), spd * torch.sin(ang_e), c[:, 2]], 1)
            c = torch.where(m_e, ce, c)
        self.cmd = torch.where(mask[:, None], c, self.cmd)
        if self.goal_p > 0:
            gm = self.U(N) < self.goal_p
            self.goal_mode = torch.where(mask, gm, self.goal_mode)
            self.path_left = torch.where(mask, torch.randint(0, 4, (N,), device=DEV, generator=self.g), self.path_left)
            self.new_goal(mask & gm)

    def new_goal(self, mask):
        """Draw a goal 0.7 to 1.8 m away (clamped to the tile square) for the worlds in mask; computed for all worlds, no sync."""
        N = self.N
        ang = self.U(N, hi=2 * np.pi); r = self.U(N, lo=0.7, hi=1.8) if self.tiles else self.U(N, lo=1.0, hi=3.0)
        _bp = self.ball_xy(); gx = _bp[:, 0] + r * torch.cos(ang); gy = _bp[:, 1] + r * torch.sin(ang)
        if self.tiles:
            c = self.tile_xy[self.cur_tile]
            gx = torch.minimum(torch.maximum(gx, c[:, 0] - 1.3), c[:, 0] + 1.3); gy = torch.minimum(torch.maximum(gy, c[:, 1] - 1.3), c[:, 1] + 1.3)
        g = torch.stack([gx, gy], 1)
        if self.tiles and self.cross_p > 0:      # cross focus: with prob cross_p the goal is 1.1 to 1.3 m from the tile centre along one of the four tile axes, so reaching it means crossing the terrain feature
            c = self.tile_xy[self.cur_tile]
            ax = torch.randint(0, 4, (N,), device=DEV, generator=self.g)
            dirs = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=DEV)[ax]
            gc = c + dirs * self.U(N, lo=1.1, hi=1.3)[:, None]
            g = torch.where((self.U(N) < self.cross_p)[:, None], gc, g)
        self.goal = torch.where(mask[:, None], g, self.goal)

    def update_goal_cmd(self):
        """Goal-directed worlds: the velocity command is a saturating proportional rule on the goal error (speed free up to cmd_vmax), chained waypoints on arrival."""
        if self.goal_p <= 0:
            return
        e = self.goal - self.ball_xy()
        d = e.norm(dim=1)
        arrived = self.goal_mode & (d < 0.2) & (self.path_left > 0)
        self.path_left = self.path_left - arrived.long()
        self.new_goal(arrived)
        e = self.goal - self.ball_xy(); d = e.norm(dim=1)
        v = e / d.clamp_min(1e-6)[:, None] * torch.minimum(1.5 * d, self.cmd_vmax * self.f_w)[:, None]
        v = torch.where((d < 0.05)[:, None], torch.zeros_like(v), v)
        cx = torch.where(self.goal_mode[:, None], v, self.cmd[:, :2])
        cz = torch.where(self.goal_mode, torch.zeros_like(self.cmd[:, 2]), self.cmd[:, 2])
        self.cmd = torch.cat([cx, cz[:, None]], 1)

    def ball_xy(self):
        """Planar position used by the goal logic: the ball body origin (default, all runs before S118) or, with goal_centre,
        the true ball centre = origin + R(ball quaternion) @ offset."""
        if not self.goal_centre:
            return self.qpos[:, 10:12]
        q = self.qpos[:, 13:17]; w, u = q[:, :1], q[:, 1:4]
        o = self.ball_off_tab[self.didx]
        t = 2.0 * torch.cross(u, o, dim=1)
        c = self.qpos[:, 10:13] + o + w * t + torch.cross(u, t, dim=1)
        return c[:, :2]

    def goal_feat(self):
        e = ((self.goal - self.ball_xy()) / 2).clamp(-2, 2) * self.goal_mode[:, None]
        return torch.cat([e, self.goal_mode[:, None].float()], 1)

    def _subtree(self, mass):
        """Subtree masses (N, nbody) from per-world body masses. The engine divides mass-weighted positions by the stored body_subtreemass to find each subtree's centre of
        mass (the reference point of the dynamics), so it must follow every mass randomisation; a stale value corrupts the dynamics in proportion to the distance from the world origin."""
        sub = mass.clone()
        for i in range(len(self.par) - 1, 0, -1):
            sub[:, self.par[i]] += sub[:, i]
        return sub

    def set_dr_scale(self, x):
        """Master randomisation scale: widens every domain randomisation range at once (mass, friction, torque, damping spread, latency probability, sensor noise)."""
        self.dr_m = self.dr_f = self.dr_t = self.dr_d = self.dr_l = self.dr_n = float(x)

    def set_design(self, ii, dn):
        """Assign design indices dn to the worlds ii (long tensor)."""
        old = self.didx[ii].clone()
        self.didx[ii] = dn
        if self.dr:     # keep the per-world mass / damping randomisation factor, swap the design baseline (mass_tab and damp_tab are per design)
            self.t_mass[ii] = self.t_mass[ii] / self.mass_tab[old].clamp_min(1e-9) * self.mass_tab[dn]
            self.t_damp[ii] = self.t_damp[ii] / self.damp_tab[old].clamp_min(1e-12) * self.damp_tab[dn]
        for nm, st in self.dstack.items():
            self.dw[nm][ii] = st[dn]
        if self.dr:     # the design copy above overwrote body_subtreemass with the design's nominal value: recompute it from the randomised masses (see _subtree)
            _sub = self._subtree(self.t_mass)
            if self.t_sub is not None:
                self.t_sub.copy_(_sub)
            elif 'body_subtreemass' in self.dw:
                self.dw['body_subtreemass'].copy_(_sub)
        self.tau_w[ii] = self.tau_tab[dn]; self.f_w[ii] = self.f_tab[dn]

    def reset(self, idx):
        """Reset the worlds in idx (bool mask over N). Everything is drawn for all N worlds and merged with where / masked_fill_, so there is no host sync.
        Tensors that share memory with Warp arrays are updated in place with copy_."""
        N, mk = self.N, idx
        m1, m2, m3 = mk[:, None], mk[:, None, None], mk[:, None, None, None]
        if self.K > 1:      # new design for these worlds
            if getattr(self, "design_p", None) is not None:      # S114: weighted design sampling (--design-w), more practice on hard designs
                dn = torch.multinomial(self.design_p, N, replacement=True, generator=self.g)
            else:
                dn = torch.randint(0, self.K, (N,), device=DEV, generator=self.g)
            self.didx = torch.where(mk, dn, self.didx)
            for nm, st in self.dstack.items():
                cur = self.dw[nm]
                msk = mk.view(-1, *([1] * (cur.dim() - 1)))
                cur.copy_(torch.where(msk, st[dn], cur))
            self.tau_w = self.tau_tab[self.didx]; self.f_w = self.f_tab[self.didx]
        yaw = self.U(N, lo=-np.pi, hi=np.pi)
        q = self.q0_tab[self.didx].clone()
        q[:, 3] = torch.cos(yaw / 2); q[:, 4] = 0; q[:, 5] = 0; q[:, 6] = torch.sin(yaw / 2)
        if self.tiles:      # random tile, random start inside the flat start zone, z lifted by the local map height
            tid = torch.randint(0, self.tile_xy.shape[0], (N,), device=DEV, generator=self.g)
            if getattr(self, "nav_frac", 0) > 0:      # planner-in-the-loop worlds start on a 'blocks' tile, all others on the normal tiles
                blk_ids = self.nav_blk_ids; oth_ids = self.nav_oth_ids
                newnav = self.U(N) < self.nav_frac
                tb = blk_ids[torch.randint(0, len(blk_ids), (N,), device=DEV, generator=self.g)]
                to = oth_ids[torch.randint(0, len(oth_ids), (N,), device=DEV, generator=self.g)]
                tid = torch.where(newnav, tb, to)
                self.nav_mask = torch.where(mk, newnav, self.nav_mask)
            self.cur_tile = torch.where(mk, tid, self.cur_tile)
            xs = self.tile_xy[tid, 0] + self.U(N, lo=-0.3, hi=0.3); ys = self.tile_xy[tid, 1] + self.U(N, lo=-0.3, hi=0.3)
            if getattr(self, "nav_frac", 0) > 0:
                nm = mk & self.nav_mask
                xs = torch.where(nm, self.tile_xy[tid, 0] + self.U(N, lo=-0.1, hi=0.1), xs); ys = torch.where(nm, self.tile_xy[tid, 1] + self.U(N, lo=-0.1, hi=0.1), ys)
                ax = torch.randint(0, 4, (N,), device=DEV, generator=self.g)
                self.nav.set_key(nm, self.nav_tile2blk[tid].clamp_min(0) * 4 + ax)
                self.goal_mode = torch.where(nm, torch.zeros_like(self.goal_mode), self.goal_mode)
            hz = self.hgt(xs, ys)
            for j in (0, 10):
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += hz
        elif self.terrain:      # random segment of the ridge, random x; z lifted by the local terrain height
            seg = torch.randint(0, len(START_Y), (N,), device=DEV, generator=self.g)
            sgn = torch.where(self.U(N) < 0.5, -1.0, 1.0)
            ys = torch.tensor(START_Y, device=DEV)[seg] * sgn + self.U(N, lo=-0.12, hi=0.12)
            if self.edge_p > 0:      # start shortly before one of the edges
                ed = torch.tensor(STEP_Y, device=DEV)[torch.randint(0, len(STEP_Y), (N,), device=DEV, generator=self.g)]
                ys = torch.where(self.U(N) < self.edge_p, sgn * (ed - self.U(N, lo=0.2, hi=0.45)), ys)
            xs = self.U(N, lo=-1.0, hi=1.0)
            hz = torch.zeros(N, device=DEV)
            for e, hh in zip(STEP_Y, STEP_H):
                hz = torch.where(ys.abs() >= e, torch.full_like(hz, hh), hz)
            for j in (0, 10):     # base and ball free joints (qpos 0:7, 10:17)
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += hz
        self.qpos.copy_(torch.where(m1, q, self.qpos))
        self.qvel.masked_fill_(m1, 0.0)
        self.warm.masked_fill_(m1, 0.0)
        self.ctrl.masked_fill_(m1, 0.0)
        if self.motor:
            self.tau_act.masked_fill_(m1, 0.0)
        self.xfrc.masked_fill_(m2, 0.0)
        self.sample_cmd(mk)
        self.update_goal_cmd()
        if getattr(self, "mref", None) is not None:
            self.mref.reset(mk, heading=torch.atan2(self.cmd[:, 1], self.cmd[:, 0]))
        self._shape_cmd(mk)
        self.vint = torch.where(m1, torch.zeros_like(self.vint), self.vint)
        self.last_a = torch.where(m1, torch.zeros_like(self.last_a), self.last_a)
        self.has_prev = self.has_prev & ~mk
        self.steps = torch.where(mk, torch.zeros_like(self.steps), self.steps)
        self.stall_cnt = torch.where(mk, torch.zeros_like(self.stall_cnt), self.stall_cnt)
        if self.lqr is not None:
            self.lqr.reset(mk)
        self.tq = torch.where(mk, 1 + (self.dr_t * self.U(N, lo=-0.15, hi=0.15) if self.dr else torch.zeros(N, device=DEV)), self.tq)
        if self.dr:
            new = self.mass_tab[self.didx].clone()
            for k_, b in enumerate(self.body_ids):      # body_ids = [base, ball]; dr_mbody switches the two masses on and off separately (analysis only)
                new[:, b] = new[:, b] * (1 + self.dr_m * self.dr_mbody[k_] * self.U(N, lo=-0.15, hi=0.15))
            self.t_mass.copy_(torch.where(m1, new, self.t_mass))
            _sub = self._subtree(self.t_mass)      # keep body_subtreemass consistent with the randomised masses (see _subtree)
            if self.t_sub is not None:
                self.t_sub.copy_(_sub)
            else:
                self.dw['body_subtreemass'].copy_(_sub)
            if self.dr_inertia and 'body_inertia' in self.dw:
                ratio = new / self.mass_tab[self.didx].clamp_min(1e-9)
                scaled = self.dstack['body_inertia'][self.didx] * ratio[:, :, None]
                self.dw['body_inertia'].copy_(torch.where(m2, scaled, self.dw['body_inertia']))
            nf = self.nom_fric.unsqueeze(0).repeat(N, 1, 1)
            nf[:, :, 1] = self.nom_fric[:, 1][None] * (1 + self.dr_f * self.U(N, lo=-0.2, hi=0.2))[:, None]
            self.t_fric.copy_(torch.where(m2, nf, self.t_fric))
            nd = self.damp_tab[self.didx].clone()
            lo, hi = self.drc["damping"]
            lo, hi = max(0.1 - (0.1 - lo) * self.dr_d, 0.01), 0.1 + (hi - 0.1) * self.dr_d
            for k in range(3):
                nd[:, 6 + k] = self.U(N, lo=lo, hi=hi) * self.damp_tab[self.didx, 6 + k] / 0.1
            self.t_damp.copy_(torch.where(m1, nd, self.t_damp))
        if self.surfaces:
            sid = torch.randint(0, len(SURFACES), (N,), device=DEV, generator=self.g)
            self.surf_id = torch.where(mk, sid, self.surf_id)
            mu = self.surf_tab[sid, 0] * (1 + self.U(N, lo=-0.15, hi=0.15)); tc = self.surf_tab[sid, 1]
            for g in self.surf_geoms:
                self.t_gfric[:, g, 0] = torch.where(mk, mu, self.t_gfric[:, g, 0])
                self.t_gsolref[:, g, 0] = torch.where(mk, tc, self.t_gsolref[:, g, 0])
        self.hist = torch.where(m2, torch.zeros_like(self.hist), self.hist)
        self.push_left = torch.where(mk, torch.zeros_like(self.push_left), self.push_left)
        self.push_t = torch.where(mk, self.U(N, lo=200, hi=400), self.push_t)

    @staticmethod
    def rotvec(q):
        w, v = q[:, 0], q[:, 1:4]
        n = v.norm(dim=1).clamp_min(1e-9)
        ang = 2 * torch.atan2(n, w)
        return v / n[:, None] * ang[:, None]

    def enable_nav(self, frac):
        """Planner-in-the-loop training (call right after construction): fraction `frac` of worlds start on 'blocks' tiles with a goal
        1.2 m away behind a block and receive the smooth wavefront planner's command (nav.WavePlanner on the privileged map)."""
        import tiles as TL
        from nav import WavePlanner
        bi = len(TL.TYPES)
        self.nav_blk_ids = (self.tile_kind == bi).nonzero()[:, 0]; self.nav_oth_ids = (self.tile_kind != bi).nonzero()[:, 0]
        assert len(self.nav_blk_ids) > 0, "no blocks tiles in the map"
        self.nav_mask = torch.zeros(self.N, dtype=torch.bool, device=DEV); self.nav_reached = torch.zeros(self.N, dtype=torch.bool, device=DEV)
        z = torch.zeros(self.N, 2, device=DEV)
        self.nav = WavePlanner(self, torch.arange(self.N, device=DEV), z, z, half_w=0.22, carrot=0.3, w_max=2.0)
        d4 = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=DEV)
        cen = self.tile_xy[self.nav_blk_ids].repeat_interleave(4, 0); gol = cen + d4.repeat(len(self.nav_blk_ids), 1) * 1.2
        self.nav.build_static(cen, gol)          # one distance field per (blocks tile, goal axis): the map is static
        self.nav_tile2blk = torch.full((self.tile_xy.shape[0],), -1, dtype=torch.long, device=DEV)
        self.nav_tile2blk[self.nav_blk_ids] = torch.arange(len(self.nav_blk_ids), device=DEV)
        self.nav_frac = frac
        self.reset(torch.ones(self.N, dtype=torch.bool, device=DEV))

    def sensor_yaw(self):
        q = self.qpos[:, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        body = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if getattr(self, "yaw_override", None) is not None:      # evaluation hook: a navigation layer aims the gimbal (e.g. at its target heading)
            return self.yaw_override
        c = self.cmd[:, :2]
        return torch.where(c.norm(dim=1) > 0.05, torch.atan2(c[:, 1], c[:, 0]), body)   # assumption: sensor looks along the commanded direction

    def hgt(self, x, y):
        import tiles as TL
        ix = ((x + TL.HALF) / (2 * TL.HALF) * (TL.NCELL - 1)).round().long().clamp(0, TL.NCELL - 1)
        iy = ((y + TL.HALF) / (2 * TL.HALF) * (TL.NCELL - 1)).round().long().clamp(0, TL.NCELL - 1)
        return self.hmap[iy, ix]

    def scan_feat(self):
        yaw = self.sensor_yaw(); cy, sy = torch.cos(yaw), torch.sin(yaw)
        if not self.terrain:
            return torch.zeros(self.N, len(SCAN_FWD) * len(SCAN_LAT), device=DEV)
        f = torch.tensor([a for a in SCAN_FWD for _ in SCAN_LAT], device=DEV); l = torch.tensor(list(SCAN_LAT) * len(SCAN_FWD), device=DEV)
        bx, by = self.qpos[:, 10:11], self.qpos[:, 11:12]
        py = by + f[None] * sy[:, None] + l[None] * cy[:, None]
        if self.tiles:
            px = bx + f[None] * cy[:, None] - l[None] * sy[:, None]
            return (self.hgt(px, py) - self.hgt(bx, by)) / 0.06
        return (terrain_height_t(py) - terrain_height_t(by)) / 0.06

    def priv_obs(self):
        """Privileged critic inputs: floor material, mass and torque scales, wheel damping, local height scan."""
        N = self.N
        mu = self.t_gfric[:, self.surf_geoms[0], 0] if self.surfaces else torch.ones(N, device=DEV)
        tc = (self.t_gsolref[:, self.surf_geoms[0], 0] * 20) if self.surfaces else torch.ones(N, device=DEV)
        if self.dr:
            mb = (self.t_mass[:, self.body_ids[0]] / self.mass_tab[self.didx, self.body_ids[0]] - 1) * 5
            mk = (self.t_mass[:, self.body_ids[1]] / self.mass_tab[self.didx, self.body_ids[1]] - 1) * 5
            dm = (self.t_damp[:, 6] / self.damp_tab[self.didx, 6] - 1)
        else:
            mb = mk = dm = torch.zeros(N, device=DEV)
        p = torch.cat([torch.stack([mu, tc, mb, mk, (self.tq - 1) * 5], 1), dm[:, None], self.scan_feat()[:, :34]], 1)[:, :40]
        return torch.cat([p, self.dvec_tab[self.didx]], 1) if self.K > 1 else p

    def sensor_obs(self):
        if self.sensor == "none":
            return None
        yaw = self.sensor_yaw(); cy, sy = torch.cos(yaw), torch.sin(yaw)
        if self.sensor == "scan":      # privileged teacher: terrain height ahead relative to the height under the ball
            if not self.terrain:
                return torch.zeros(self.N, self.sdim, device=DEV)
            f = torch.tensor([a for a in SCAN_FWD for _ in SCAN_LAT], device=DEV); l = torch.tensor(list(SCAN_LAT) * len(SCAN_FWD), device=DEV)
            bx, by = self.qpos[:, 10:11], self.qpos[:, 11:12]
            py = by + f[None] * sy[:, None] + l[None] * cy[:, None]
            if self.tiles:
                px = bx + f[None] * cy[:, None] - l[None] * sy[:, None]
                return (self.hgt(px, py) - self.hgt(bx, by)) / 0.06
            return (terrain_height_t(py) - terrain_height_t(by)) / 0.06
        import mujoco_warp as mjw
        R = self.ray_dirs
        if self.mount == "cones":      # rigid body mount: full base rotation (tilt included)
            qw, qx, qy, qz = self.qpos[:, 3], self.qpos[:, 4], self.qpos[:, 5], self.qpos[:, 6]
            Rm = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                              2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                              2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1).reshape(-1, 3, 3)
            v = torch.einsum("nij,rj->nri", Rm, R).contiguous()
            p = (self.qpos[:, None, 0:3] + torch.einsum("nij,rj->nri", Rm, self.ray_off)).contiguous()
        else:
            v = torch.stack([R[:, 0][None] * cy[:, None] - R[:, 1][None] * sy[:, None],
                             R[:, 0][None] * sy[:, None] + R[:, 1][None] * cy[:, None],
                             R[:, 2][None].expand(self.N, -1)], -1).contiguous()
            p = self.qpos[:, 0:3].clone(); p[:, 2] += SENSOR_UP
            p = p[:, None, :].expand(-1, R.shape[0], -1).contiguous()
        mjw.rays(self.be.m, self.be.d, wp.from_torch(p, dtype=wp.vec3), wp.from_torch(v, dtype=wp.vec3), self.r_gg, True, self.r_ex,
                 self.r_dist, self.r_gid, self.r_nrm, self.rc)
        dist = wp.to_torch(self.r_dist).clone()
        dist = torch.where(dist < 0, torch.full_like(dist, SENSOR_RANGE[1]), dist).clamp(*SENSOR_RANGE)
        return dist

    def obs(self):
        q = self.qpos[:, 3:7]
        cv = self.cvel[:, self.base]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._ang, self._vel = cv[:, :3].clamp(-8, 8), cv[:, 3:].clamp(-8, 8)
        # CPU env quirk kept on purpose: motor_state reads qvel[joint.id] (ids 1..3 = base free-joint velocity), not the wheel dofs 6:9
        motor = (self.qvel[:, 1:4] / self.tau_w[:, None]).clamp(-2, 2)
        o = torch.cat([self.rotvec(q), self._ang, self._vel, motor, self.last_a, self.cmd,
                       torch.stack([torch.sin(yaw), torch.cos(yaw)], 1), self.vint, self.goal_feat()], 1)
        so = self.sensor_obs()
        return o if so is None else torch.cat([o, so], 1)

    def reset_obs(self):
        q = self.qpos[:, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        z3 = torch.zeros(self.N, 3, device=DEV)
        o = torch.cat([self.rotvec(q), z3, z3, z3, self.last_a, self.cmd, torch.stack([torch.sin(yaw), torch.cos(yaw)], 1), self.vint, self.goal_feat()], 1)
        so = self.sensor_obs()
        return o if so is None else torch.cat([o, so], 1)

    def step(self, a):
        a = a.clamp(-1, 1)
        rs = self.rs
        if getattr(self, "rs_adapt", None) is not None and self.lqr is not None:
            # S111: adaptive residual authority. Low (rs_lo) while cruising steadily, where the LQR tracks best (S110); high (rs_hi) when
            # the command changes fast, the command is near zero (arrive, stop, hold) or the body tilts, where the residual adds robustness.
            lo, hi = self.rs_adapt[0], self.rs_adapt[1]
            v_db = self.rs_adapt[2] if len(self.rs_adapt) > 2 else 0.0      # S112: hold dead band: the near-zero-command term only while the robot still moves faster than v_db
            c2 = self.cmd[:, :2]
            dc = (c2 - getattr(self, "_rs_prev_cmd", c2)).norm(dim=1) / 0.01
            self._rs_prev_cmd = c2.clone()
            qv = self.qpos[:, 3:7]
            tilt = torch.rad2deg(torch.acos((1 - 2 * (qv[:, 1] ** 2 + qv[:, 2] ** 2)).clamp(-1, 1)))
            near0 = (1 - c2.norm(dim=1) / (0.1 * self.f_w)).clamp(0, 1)
            if v_db > 0:
                near0 = near0 * (self.qvel[:, 9:11].norm(dim=1) > v_db * self.f_w).float()
            need = torch.maximum(torch.maximum((dc / 0.3).clamp(0, 1), near0), ((tilt - 4.0) / 6.0).clamp(0, 1))
            w = getattr(self, "_rs_w", torch.ones_like(need))
            w = torch.maximum(need, w * 0.98)      # rise at once, decay with a 0.5 s time constant
            self._rs_w = w
            rs = (lo + (hi - lo) * w)[:, None]
        applied = a.clone() if self.lqr is None else (self.lqr.act() + rs * a).clamp(-1, 1)      # LQR base controller + bounded learned residual (rs = residual authority)
        sent = applied.clone()
        if self.dr:
            late = self.has_prev & (self.U(self.N) < min(self.drc.get('latency_p', 0.2) * self.dr_l, 0.6))
            applied = torch.where(late[:, None], self.prev_a, applied)
        self.prev_a = sent; self.has_prev[:] = True
        tw = self.tau_w[:, None]
        c = torch.maximum(torch.minimum(applied * tw * self.tq[:, None], tw), -tw)
        (self.motor_cmd if self.motor else self.ctrl)[:] = -c
        if self.dr:   # random base pushes, 5 control steps long (sync free: sampled for all worlds, applied where a push starts)
            N = self.N
            self.push_t = self.push_t - 1
            start = (self.push_t <= 0) & (self.push_left <= 0)
            ang = self.U(N, hi=2 * np.pi); f = self.push_scale * self.push_tab[self.didx] * self.U(N, lo=self.drc['push_force'][0], hi=self.drc['push_force'][1])
            self.push_f = torch.where(start[:, None], torch.stack([f * torch.cos(ang), f * torch.sin(ang), torch.zeros_like(f)], 1), self.push_f)
            self.push_left = torch.where(start, torch.full_like(self.push_left, 5), self.push_left)
            self.push_t = torch.where(start, self.U(N, lo=200, hi=400), self.push_t)
            on = self.push_left > 0
            self.xfrc[:, self.base, :3] = torch.where(on[:, None], self.push_f, torch.zeros_like(self.push_f))
            self.push_left = self.push_left - on.long()
            if self.pull:      # sustained pulls: 3 to 12 N for 0.5 to 2 s, every 4 to 10 s
                self.pull_t = self.pull_t - 1
                st = (self.pull_t <= 0) & (self.pull_left <= 0)
                ang = self.U(N, hi=2 * np.pi); f = self.push_scale * self.U(N, lo=3.0, hi=12.0)
                self.pull_f = torch.where(st[:, None], torch.stack([f * torch.cos(ang), f * torch.sin(ang), torch.zeros_like(f)], 1), self.pull_f)
                self.pull_left = torch.where(st, torch.randint(50, 200, (N,), device=DEV, generator=self.g), self.pull_left)
                self.pull_t = torch.where(st, self.U(N, lo=400, hi=1000), self.pull_t)
                on2 = self.pull_left > 0
                self.xfrc[:, self.base, :3] += torch.where(on2[:, None], self.pull_f, torch.zeros_like(self.pull_f))
                self.pull_left = self.pull_left - on2.long()
        if getattr(self, "ext_push", None) is not None:      # evaluation hook: extra push force (N,3) on while ext_on is set
            self.xfrc[:, self.base, :3] += self.ext_push * float(self.ext_on)
        self.be.step(ACTION_REPEAT)
        self.xfrc[:, self.base, :3] = 0
        self.last_a = a
        # observation (clean values for reward / termination, noisy copy for the policy)
        if self.hist_k:      # shift in the (policy-visible obs, action) pair that produced this step
            self.hist = torch.cat([torch.cat([self._last_noisy, a], 1)[:, None], self.hist[:, :-1]], 1)
        self.steps += 1
        rs = (self.steps % self.resample == 0)
        self.sample_cmd(rs)
        self.update_goal_cmd()
        self._shape_cmd(rs)
        if getattr(self, "nav_frac", 0) > 0:      # planner-in-the-loop worlds: the smooth wavefront planner (nav.py) sets their command
            self.goal_mode = self.goal_mode & ~self.nav_mask
            v = self.nav.command(self.ball_xy(), self.f_w, self.nav.occupancy_oracle)
            self.cmd = torch.where(self.nav_mask[:, None], torch.cat([v, torch.zeros_like(v[:, :1])], 1), self.cmd)
            self.nav_reached = self.nav_mask & ((self.nav.goal - self.ball_xy()).norm(dim=1) < 0.2)
        o = self.obs()
        vel, wz = self._vel[:, :2], self._ang[:, 2]
        self.vint = (self.vint + (self.cmd[:, :2] - vel) * 0.01).clamp(-1, 1)
        o[:, 20:22] = self.vint
        c2, cw = self.cmd[:, :2], self.cmd[:, 2]
        cn = c2.norm(dim=1)
        e = (vel - c2).norm(dim=1)
        has = cn > 0.05
        u = torch.where(has[:, None], c2 / cn.clamp_min(1e-9)[:, None], torch.zeros_like(c2))
        p = torch.where(has, (vel * u).sum(1), torch.zeros_like(cn))
        ew = (wz - cw).abs()
        raw = (cn - e) + 0.5 * torch.exp(-e ** 2 / 0.25 ** 2) + 0.5 * (cw.abs() - ew) + 0.25 * torch.exp(-ew ** 2 / 0.5 ** 2)
        rel = self.xpos[:, self.base] - self.gxpos[:, self.ball_g]
        lean = (rel[:, :2] * u).sum(1) / rel[:, 2].clamp_min(1e-3)
        raw = raw + 0.5 * torch.where(has, (1 - (p / cn.clamp_min(1e-9)).clamp(0, 1)) * (lean / 0.3).clamp(-1, 1), torch.zeros_like(cn))
        r = raw * 0.5 * 5 * self.rw_track
        r = r - self.rw_pen * 1e-4 * 5 * (a ** 2).sum(1)
        rate = ((a - self._prev_applied_cmd) ** 2).sum(1) if hasattr(self, "_prev_applied_cmd") else torch.zeros_like(r)
        self._prev_applied_cmd = a.clone()
        r = r - self.rw_pen * 0.02 * 5 * rate
        off = torch.rad2deg(torch.atan2(rel[:, :2].norm(dim=1), rel[:, 2].clamp_min(1e-6)))
        r = r - self.rw_pen * 0.05 * 5 * (off / 30.0) ** 2
        q = self.qpos[:, 3:7]
        up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        tilt = torch.nan_to_num(torch.rad2deg(torch.acos(up_z.clamp(-1, 1))), nan=90.0)
        bad = ~(torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1) & torch.isfinite(o).all(1))
        o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)      # a non-finite world is a fall and gets reset; keep its values out of the running statistics and history
        fell = (tilt > 70) | (off > 30) | bad
        r = torch.where(fell, r - self.fall_pen, r + 0.02 * 5)
        r = torch.nan_to_num(r, nan=-2.0)
        trunc = (self.steps >= self.max_steps) & ~fell
        if self.stall_pen > 0:      # S101: stalled against a feature = episode ends with a penalty (deliberate version of the accidental blow-up curriculum)
            stalled = has & (cn > 0.15) & (p < 0.2 * cn)
            self.stall_cnt = torch.where(stalled & ~rs, self.stall_cnt + 1, torch.zeros_like(self.stall_cnt))
            stall_term = (self.stall_cnt >= self.stall_steps) & ~fell
            r = torch.where(stall_term, r - self.stall_pen, r)
            trunc = trunc | stall_term
            self.stall_cnt = torch.where(stall_term, torch.zeros_like(self.stall_cnt), self.stall_cnt)
        if self.tiles:      # leaving the 2.9 m square around the tile centre ends the episode without a fall penalty
            import tiles as TL
            c = self.tile_xy[self.cur_tile]
            out = ((self.qpos[:, 10] - c[:, 0]).abs() > TL.EXIT_R) | ((self.qpos[:, 11] - c[:, 1]).abs() > TL.EXIT_R)
            self.exited = out & ~fell
            trunc = trunc | self.exited
            if getattr(self, "nav_frac", 0) > 0:      # a planner world that reached its goal ends without penalty
                trunc = trunc | (self.nav_reached & ~fell)
        return o, r, fell, trunc, dict(tilt=tilt, speed_err=torch.nan_to_num(e, nan=0.0, posinf=0.0))

    def noisy(self, o):
        o = self._noisy(o)
        self._last_noisy = o[:, :OBS_DIM].clone()
        parts = [o]
        if self.hist_k:
            parts.append(self.hist.flatten(1))
        if self.ddim:
            parts.append(self.dvec_tab[self.didx])
        if self.asym:
            parts.append(self.priv_obs())
        return torch.nan_to_num(torch.cat(parts, 1) if len(parts) > 1 else o, nan=0.0, posinf=0.0, neginf=0.0)

    def _noisy(self, o):
        if not self.dr:
            return o
        n = torch.randn(self.N, 9, device=DEV, generator=self.g)
        o = o.clone()
        sd = float(self.drc.get('obs_noise', 0.01)) * self.dr_n
        o[:, 0:3] += n[:, 0:3] * sd * 0.25
        o[:, 3:6] += n[:, 3:6] * sd * 2.0
        o[:, 6:9] += n[:, 6:9] * sd
        if self.sensor == "depth":      # range-proportional noise (sd = depth_noise * d) and 2 % zone dropout (reads as max range)
            dd = o[:, OBS_DIM:]
            dd = dd + torch.randn_like(dd) * self.depth_noise * dd
            drop = torch.rand(dd.shape, device=DEV, generator=self.g) < 0.02
            o[:, OBS_DIM:] = torch.where(drop, torch.full_like(dd, SENSOR_RANGE[1]), dd.clamp(*SENSOR_RANGE))
        return o


class RunNorm:
    def __init__(self, d):
        self.mean = torch.zeros(d, device=DEV); self.var = torch.ones(d, device=DEV); self.n = 1e-4

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        dlt, tot = bm - self.mean, self.n + bn
        self.mean = self.mean + dlt * bn / tot
        self.var = (self.var * self.n + bv * bn + dlt ** 2 * self.n * bn / tot) / tot
        self.n = tot
        if getattr(self, "fix", None) is not None:      # columns with externally fixed statistics (frozen pretrained encoder input)
            lo, hi, fm, fv = self.fix
            self.mean[lo:hi], self.var[lo:hi] = fm, fv

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class Net(nn.Module):
    """MLP on the 22 proprioceptive inputs plus (optionally) a learned encoder of the sensor vector."""
    def __init__(self, sdim, out, enc=32, hdim=0, pdim=0, use_priv=False, ddim=0, width=256, nlayers=2):
        super().__init__()
        self.sdim, self.hdim, self.pdim, self.ddim, self.use_priv = sdim, hdim, pdim, ddim, use_priv and pdim > 0
        self.enc = nn.Sequential(nn.Linear(sdim, 128), nn.Tanh(), nn.Linear(128, enc), nn.Tanh()) if sdim else None
        layers, d_in = [], OBS_DIM + (enc if sdim else 0) + hdim + ddim + (pdim if self.use_priv else 0)
        for _ in range(nlayers):
            layers += [nn.Linear(d_in, width), nn.Tanh()]; d_in = width
        self.body = nn.Sequential(*layers, nn.Linear(d_in, out))

    def forward(self, x):
        a, b = OBS_DIM, OBS_DIM + self.sdim
        c = b + self.hdim
        e = c + self.ddim
        parts = [x[:, :a]]
        if self.sdim:
            parts.append(self.enc(x[:, a:b]))
        if self.hdim:
            parts.append(x[:, b:c])
        if self.ddim:
            parts.append(x[:, c:e])
        if self.use_priv:
            parts.append(x[:, e:e + self.pdim])
        return self.body(torch.cat(parts, 1))


class AC(nn.Module):
    def __init__(self, log_std=-1.0, sdim=0, hdim=0, pdim=0, ddim=0, width=256, nlayers=2):
        super().__init__()
        self.pi, self.v = Net(sdim, ACT_DIM, hdim=hdim, pdim=pdim, use_priv=False, ddim=ddim, width=width, nlayers=nlayers), Net(sdim, 1, hdim=hdim, pdim=pdim, use_priv=True, ddim=ddim, width=width, nlayers=nlayers)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), log_std))
        for mm in (self.pi.body[-1],):
            nn.init.orthogonal_(mm.weight, 0.01)

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--nworld", type=int, default=2048)
    ap.add_argument("--steps", type=float, default=50e6)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--design", default="nom")
    ap.add_argument("--sensor", choices=["none", "scan", "depth"], default="none")
    ap.add_argument("--goal-p", type=float, default=0.0, help="fraction of goal-directed (position goals, chained waypoints) episodes")
    ap.add_argument("--pull", action="store_true", help="sustained pulls on the base")
    ap.add_argument("--lqr-r", type=float, default=1.0, help="LQR control cost (higher = gentler base controller)")
    ap.add_argument("--lqr-amax", type=float, default=0.5, help="LQR velocity reference acceleration limit m/s^2")
    ap.add_argument("--dr-scale", type=float, default=1.0, help="final master scale of all domain randomisation ranges (1 = the standard ranges)")
    ap.add_argument("--dr-ramp", type=float, default=0.0, help="fraction of training over which the master scale ramps from 1 to --dr-scale (0 = from the start)")
    ap.add_argument("--fall-pen", type=float, default=2.0, help="reward penalty on a fall (default 2.0)")
    ap.add_argument("--lqr-base", type=float, default=0.0, help="residual authority rs in (0,1]: the applied action is clip(LQR action + rs * policy action); 0 = pure learned policy")
    ap.add_argument("--rw-track", type=float, default=1.0, help="multiplier on the velocity tracking reward")
    ap.add_argument("--rw-pen", type=float, default=1.0, help="multiplier on the lean, action-rate and action-size penalties")
    ap.add_argument("--cross-focus", type=float, default=0.0, help="tiles + goal task: probability that a goal is placed across the tile feature (1.1 to 1.3 m from the tile centre along a tile axis)")
    ap.add_argument("--enc-init", default=None, help="name of a pretrained depth encoder (percep/NAME_net.pt from train_percep.py); loaded into the policy and critic encoders and frozen")
    ap.add_argument("--push-ramp", type=float, default=0.0, help="push/pull strength ramps linearly from 0.2 to 1 over this fraction of training (0 = off)")
    ap.add_argument("--designs", default=None, help="comma list of dz_* design names sampled per world")
    ap.add_argument("--design-cond", action="store_true", help="policy sees a 6-D design vector")
    ap.add_argument("--width", type=int, default=256); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hist", type=int, default=0, help="stack of the last K (obs, action) pairs as policy input")
    ap.add_argument("--asym", action="store_true", help="asymmetric critic with privileged inputs")
    ap.add_argument("--surfaces", action="store_true", help="randomise floor material class per world (friction, softness)")
    ap.add_argument("--tiles", action="store_true", help="tiled multi-terrain map (steps, drops, ramps, stairs, bumps, rough)")
    ap.add_argument("--edge-focus", type=float, default=0.0)
    ap.add_argument("--mount", choices=["head", "cones"], default="head")
    ap.add_argument("--depth-noise", type=float, default=0.005)
    ap.add_argument("--terrain", action="store_true", help="shared stepped-ridge height-field with random start rows")
    ap.add_argument("--cfg", default=None, help="env config suffix, e.g. nom_real (motor model, stronger noise/latency)")
    ap.add_argument("--zero-p", type=float, default=0.1, help="probability that a resampled velocity command is zero (standing still); the default 0.1 gives little long-horizon standing practice")
    ap.add_argument("--ep-steps", type=int, default=800, help="episode length in control steps (100 Hz); 2000 = 20 s, trains long-horizon balance")
    ap.add_argument("--nav-frac", type=float, default=0.0, help="fraction of worlds on obstacle (blocks) tiles driven by the smooth wavefront planner (nav.py, privileged map); the policy learns to follow its commands near obstacles")
    ap.add_argument("--retainer", default=None, help='Rezero-style ball arrestors (retainer.py): "gap_mm,fric", e.g. "0,0"; recorded in args.json, evaluation scripts re-apply it')
    ap.add_argument("--rs-adapt", default=None, help="S111: adaptive residual authority 'lo,hi' (e.g. 0.15,0.3): lo while cruising, hi when the command changes or nears zero or the body tilts; recorded in args.json and re-applied by the evaluation scripts")
    ap.add_argument("--motion-ref", action="store_true", help="S105: train on commands shaped by the motion-reference layer (motion_ref.py), including arrival, position hold and curving commands")
    ap.add_argument("--curve-p", type=float, default=0.5, help="with --motion-ref: fraction of resampled velocity commands that turn steadily (up to 1 rad/s)")
    ap.add_argument("--stall-pen", type=float, default=0.0, help="S101 test: end an episode with this penalty when the robot stalls against a feature for 1 s (commanded > 0.15 m/s, achieved along the command < 1/5 of it); 0 = off")
    ap.add_argument("--pair-mu1", type=float, default=0.01, help="wheel-ball free-rolling friction coefficient; default 0.01 since S98 (the xml's 0.001 blows up the float32 solver when standing). Recorded in args.json, evaluation scripts re-apply it; runs before S98 have no entry and keep 0.001")
    ap.add_argument("--push-train", type=float, default=1.0, help="S116: push/pull strength multiplier during training (1 = the evaluation strength); with --push-ramp the ramp ends at this value")
    ap.add_argument("--goal-centre", action="store_true", help="S118: goal logic (goal rule, goal observation, motion-layer hold, navigation) on the true ball centre instead of the ball body origin, which swings by up to 0.14 m as the ball rolls; recorded in args.json, evaluation scripts re-apply it")
    ap.add_argument("--design-w", default=None, help="S114: per-design sampling weights 'name:w,...' (others 1), e.g. 'bal4:2,tau6:2,kb025:2'; training only, the evaluations stay uniform per design")
    a = ap.parse_args()
    if a.retainer:
        os.environ["RETAINER"] = a.retainer
    if a.pair_mu1 is not None:
        os.environ["PAIR_MU1"] = f"{a.pair_mu1:g}"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision('high')
    out = ROOT / "outputs" / "study" / "warp_ppo" / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"))
    env = GpuBallbot(a.nworld, a.seed, dr=not a.no_dr, design=a.design, cfg_name=a.cfg, terrain=a.terrain, sensor=a.sensor, depth_noise=a.depth_noise, mount=a.mount, edge_p=a.edge_focus, tiles=a.tiles, surfaces=a.surfaces, hist_k=a.hist, asym=a.asym, goal_p=a.goal_p, pull=a.pull, designs=(a.designs.split(',') if a.designs else None), design_cond=a.design_cond, tile_kinds=(None if not a.nav_frac else __import__('tiles').TYPES + ['blocks'] * 5))
    env.max_steps = a.ep_steps; env.zero_p = a.zero_p; env.stall_pen = a.stall_pen
    if a.nav_frac > 0:
        env.enable_nav(a.nav_frac)
    if a.motion_ref:
        env.enable_motion_ref(curve_p=a.curve_p)
    if a.rs_adapt:
        env.rs_adapt = [float(x) for x in a.rs_adapt.split(",")]
    env.goal_centre = bool(a.goal_centre)
    if a.design_w and env.K > 1:
        wd = {k: float(v) for k, v in (p.split(":") for p in a.design_w.split(","))}
        assert set(wd) <= set(env.design_names), f"--design-w names must be among {env.design_names}"
        env.design_p = torch.tensor([wd.get(n, 1.0) for n in env.design_names], device=DEV, dtype=torch.float32)
        print("design weights", dict(zip(env.design_names, env.design_p.tolist())), flush=True)
    N, T = a.nworld, a.n_steps
    env.cross_p = a.cross_focus; env.rw_track, env.rw_pen = a.rw_track, a.rw_pen; env.fall_pen = a.fall_pen
    if a.lqr_base > 0:
        from lqr_gpu import GpuLQR
        env.lqr, env.rs = GpuLQR(env, env.design_names, a_max=a.lqr_amax, r=a.lqr_r), a.lqr_base
    ac = AC(sdim=env.sdim, hdim=env.hdim, pdim=env.pdim, ddim=env.ddim, width=a.width, nlayers=a.layers).to(DEV)
    on, rn = RunNorm(env.obs_dim), RunNorm(1)
    if a.enc_init:      # frozen pretrained depth encoder: load into policy and critic, fix the input statistics of the depth columns to the dataset ones
        pe = torch.load(ROOT / "outputs" / "study" / "warp_ppo" / "percep" / f"{a.enc_init}_net.pt", map_location=DEV)
        for net_ in (ac.pi, ac.v):
            net_.enc.load_state_dict(pe["net"])
            for q_ in net_.enc.parameters():
                q_.requires_grad_(False)
        lo_, hi_ = OBS_DIM, OBS_DIM + env.sdim
        on.fix = (lo_, hi_, pe["dmean"].to(DEV), pe["dstd"].to(DEV) ** 2)
        on.mean[lo_:hi_], on.var[lo_:hi_] = on.fix[2], on.fix[3]
    opt = torch.optim.Adam([q_ for q_ in ac.parameters() if q_.requires_grad], lr=a.lr, eps=1e-5, fused=True)
    ret = torch.zeros(N, device=DEV)
    o_raw = env.noisy(env.obs())
    on.update(o_raw)
    total, it, t0 = 0, 0, time.time()
    n_iter = int(a.steps // (N * T))
    log = open(out / "log.csv", "w"); log.write("iter,steps,time_s,ep_rew_step,fall_rate,speed_err,tilt_mean,loss_pi,loss_v,kl,log_std\n")
    for it in range(n_iter):
        lr = a.lr + (a.lr_end - a.lr) * it / max(1, n_iter - 1)
        if a.dr_scale != 1.0:
            env.set_dr_scale(1.0 + (a.dr_scale - 1.0) * (min(1.0, it / (a.dr_ramp * n_iter)) if a.dr_ramp > 0 else 1.0))
        if a.push_ramp > 0: env.push_scale = a.push_train * min(1.0, 0.2 + 0.8 * it / (a.push_ramp * n_iter))
        else: env.push_scale = a.push_train
        for gp in opt.param_groups: gp["lr"] = lr
        B = dict(o=[], a=[], lp=[], v=[], r=[], d=[])
        z0 = torch.zeros((), device=DEV)
        falls, rsum, errs, tilts = z0.clone(), z0.clone(), z0.clone(), z0.clone()
        fk, ek = torch.zeros(11, device=DEV), torch.zeros(11, device=DEV);      # 10 terrain kinds + 'blocks' (planner worlds)
        fd, ed = torch.zeros(max(env.K, 1), device=DEV), torch.zeros(max(env.K, 1), device=DEV)
        for t in range(T):
            x = on(o_raw)
            with torch.no_grad():
                dist = ac.dist(x); act = dist.sample(); lp = dist.log_prob(act).sum(1); v = ac.v(x)[:, 0]
            o_next, r, fell, trunc, info = env.step(act)
            o_next_n = env.noisy(o_next)
            r_true = r.clone()
            with torch.no_grad():      # bootstrap time-limit truncations (computed every step: no host sync)
                vb = ac.v(on(o_next_n))[:, 0]
            r = torch.where(trunc, r + a.gamma * vb, r)
            done = fell | trunc
            ret = ret * a.gamma + r_true
            rn.update(ret[:, None])
            ret = torch.where(done, torch.zeros_like(ret), ret)
            rs = r / torch.sqrt(rn.var[0] + 1e-8)
            B["o"].append(x); B["a"].append(act); B["lp"].append(lp); B["v"].append(v); B["r"].append(rs.clamp(-10, 10)); B["d"].append(done)
            if getattr(env, 'tiles', False):
                kk = env.tile_kind[env.cur_tile]; fk.scatter_add_(0, kk, fell.float()); ek.scatter_add_(0, kk, torch.ones_like(fell, dtype=torch.float))
                dd = env.didx; fd.scatter_add_(0, dd, fell.float()); ed.scatter_add_(0, dd, torch.ones_like(fell, dtype=torch.float))
            falls += fell.sum(); rsum += r_true.mean(); errs += info["speed_err"].mean(); tilts += info["tilt"].mean()
            env.reset(done)
            o_next = torch.where(done[:, None], env.reset_obs(), o_next)   # reset worlds: obs from the known reset state
            o_raw = env.noisy(o_next)
            on.update(o_raw)
        with torch.no_grad():
            lastv = ac.v(on(o_raw))[:, 0]
        r_, v_, d_ = torch.stack(B["r"]), torch.stack(B["v"]), torch.stack(B["d"]).float()
        adv = torch.zeros_like(r_); g = torch.zeros(N, device=DEV)
        for t in reversed(range(T)):
            nv = lastv if t == T - 1 else v_[t + 1]
            delta = r_[t] + a.gamma * nv * (1 - d_[t]) - v_[t]
            g = delta + a.gamma * 0.95 * (1 - d_[t]) * g
            adv[t] = g
        tgt = adv + v_
        fo, fa, flp = torch.cat(B["o"]), torch.cat(B["a"]), torch.cat(B["lp"])
        fadv, ftgt = adv.reshape(-1), tgt.reshape(-1)
        M = fo.shape[0]; mb = M // a.minibatches
        stop = False
        for ep in range(a.epochs):
            perm = torch.randperm(M, device=DEV)
            for k in range(a.minibatches):
                ix = perm[k * mb:(k + 1) * mb]
                dist = ac.dist(fo[ix]); lpn = dist.log_prob(fa[ix]).sum(1)
                ratio = (lpn - flp[ix]).exp()
                ad = fadv[ix]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                lpi = -torch.min(ratio * ad, ratio.clamp(0.8, 1.2) * ad).mean()
                lv = 0.5 * ((ac.v(fo[ix])[:, 0] - ftgt[ix]) ** 2).mean()
                loss = lpi + 0.5 * lv
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
                kl = float(((ratio - 1) - (lpn - flp[ix])).mean())
                if kl > 0.03:
                    stop = True; break
            if stop: break
        total += N * T
        if it % 5 == 0 or it == n_iter - 1:
            row = f"{it},{total},{time.time()-t0:.0f},{float(rsum)/T:.4f},{float(falls)/(N*T):.5f},{float(errs)/T:.4f},{float(tilts)/T:.3f},{float(lpi):.4f},{float(lv):.4f},{kl:.4f},{float(ac.log_std.mean()):.3f}"
            log.write(row + "\n"); log.flush()
            if env.tiles:
                with open(out / "falls_by.csv", "a") as ff:
                    ff.write(f"{it},{total},kind," + ",".join(f"{float(x):.5f}" for x in (fk / ek.clamp_min(1))) + "\n")
                    ff.write(f"{it},{total},design," + ",".join(f"{float(x):.5f}" for x in (fd / ed.clamp_min(1))) + "\n")
            print(row, f" | {total/(time.time()-t0)/1e3:.0f}k env-steps/s", flush=True)
        if it % 50 == 0 or it == n_iter - 1:
            torch.save(dict(ac=ac.state_dict(), sdim=env.sdim, hdim=env.hdim, pdim=env.pdim, ddim=env.ddim, width=a.width, nlayers=a.layers, hist=env.hist_k, asym=env.asym, designs=env.design_names, goal_p=env.goal_p, mean=on.mean.cpu(), var=on.var.cpu(), steps=total), out / "model.pt")
    print("done", total, "steps in", time.time() - t0, "s")


if __name__ == "__main__":
    main()
