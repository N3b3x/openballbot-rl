"""GPU-resident PPO for the nominal ballbot on MuJoCo-Warp (simulation only).

Whole loop on the GPU: Warp physics (contact-frame fix, CUDA graph) -> torch views of the Warp arrays -> observation, reward,
termination, command resampling, resets and PPO updates, with no host copies. Mirrors the CPU env of configs/env/dz_nom.yaml:
command-tracking task, reward "err" with lean term, obs = orientation rotvec, angular vel, vel, motor state, last action, cmd,
heading, velocity-error integral (22-D). Domain randomisation kept: obs noise, 1-step latency (p=0.2), torque scale +-15 %, random pushes.
NOT randomised (needs per-world model fields, untested in Warp): mass, ball friction, wheel damping.

  python train.py --name wp_nom_s0 --nworld 2048 --steps 50e6
"""
import argparse, json, sys, time
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
OBS_DIM, ACT_DIM = 22, 3


class GpuBallbot:
    def __init__(self, nworld, seed=0, dr=True, design="nom", cfg_name=None, terrain=False, sensor="none", depth_res=(18, 14), depth_noise=0.005, mount="head", edge_p=0.0, tiles=False, tile_seed=0, tile_kinds=None, surfaces=False, hist_k=0, asym=False):
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
        if tiles:      # big tiled height-field map (see tiles.py): patch the xml hfield size / resolution
            import re, tiles as TL
            src = open(design_o.xml).read()
            src = re.sub(r'<hfield name="terrain" nrow="\d+" ncol="\d+" size="[^"]*"', f'<hfield name="terrain" nrow="{TL.NCELL}" ncol="{TL.NCELL}" size="{TL.HALF} {TL.HALF} {TL.HZ} 0.1"', src)
            src = re.sub(r'(<geom name="terrain" type="hfield" hfield="terrain" size=")[^"]*"', rf'\g<1>{TL.HALF} {TL.HALF} 0.1"', src)
            patched = design_o.xml.parent / f"_tiles_{design_o.xml.name}"
            open(patched, "w").write(src); design_o.xml = patched
            terrain = True
        m = load_cpu_model(design=design_o)
        self.mjm = m
        self.terrain = terrain
        self.edge_p = edge_p if terrain else 0.0
        if tiles:
            import tiles as TL
            H, self.tile_info = TL.build(tile_seed, tile_kinds)
            m.hfield_data[:] = TL.data_for_mujoco(H)
            self.hmap = torch.tensor(H, dtype=torch.float32, device=DEV)
            self.tile_xy = torch.tensor([[t["cx"], t["cy"]] for t in self.tile_info], dtype=torch.float32, device=DEV)
            self.tile_kind = torch.tensor([TL.TYPES.index(t["kind"]) for t in self.tile_info], device=DEV)
            self.tile_sev = torch.tensor([t["severity"] for t in self.tile_info], dtype=torch.float32, device=DEV)
            self.cur_tile = torch.zeros(nworld, dtype=torch.long, device=DEV)
        elif terrain:      # one shared height-field (Warp has no per-world hfield): stepped ridge in y, mirrored, single steps of 1, 2, 3 cm
            n = int(m.hfield_nrow[0]); c = np.linspace(-5.0, 5.0, n)
            m.hfield_data[:] = (terrain_height(np.broadcast_to(c[:, None], (n, n))) / 2.0).flatten()
        self.be = WarpBallbot(m, N, fix_frames=True)
        d = self.be.d
        self.sensor, self.depth_noise, self.mount = sensor, depth_noise, mount
        self.sdim = {"none": 0, "scan": len(SCAN_FWD) * len(SCAN_LAT), "depth": depth_res[0] * depth_res[1] * (2 if mount == "cones" else 1)}[sensor]
        self.hist_k, self.asym = hist_k, asym
        self.hdim = hist_k * (OBS_DIM + ACT_DIM)
        self.pdim = 40 if asym else 0
        self.obs_dim = OBS_DIM + self.sdim + self.hdim + self.pdim
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
            self.t_mass = self.nom_mass.repeat(N, 1).contiguous()
            self.t_fric = self.nom_fric.repeat(N, 1, 1).contiguous()
            self.t_damp = self.nom_damp.repeat(N, 1).contiguous()
            self.be.m.body_mass = wp.from_torch(self.t_mass, dtype=wp.float32)
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

    def sample_cmd(self, idx):
        n = int(idx.sum())
        if n == 0:
            return
        sp, ang, wz = self.U(n, hi=self.cmd_vmax), self.U(n, hi=2 * np.pi), self.U(n, lo=-self.cmd_wmax, hi=self.cmd_wmax)
        c = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang), wz], 1)
        c[self.U(n) < self.zero_p] = 0.0
        if self.edge_p > 0:      # edge-focused: with prob edge_p drive across the ridge (outward while |y| < 1.8 m, back inward beyond)
            y = self.qpos[idx, 11]
            out = torch.where(y.abs() < 0.02, torch.where(self.U(n) < 0.5, -1.0, 1.0), torch.sign(y))
            tow = torch.where(y.abs() < 1.8, out, -out)
            ang_e = tow * (np.pi / 2) + self.U(n, lo=-0.6, hi=0.6)
            spd = self.U(n, lo=0.15, hi=self.cmd_vmax)
            m_e = self.U(n) < self.edge_p
            c[m_e, 0] = (spd * torch.cos(ang_e))[m_e]; c[m_e, 1] = (spd * torch.sin(ang_e))[m_e]
        self.cmd[idx] = c

    def reset(self, idx):
        n = int(idx.sum())
        if n == 0:
            return
        yaw = self.U(n, lo=-np.pi, hi=np.pi)
        q = self.q0.repeat(n, 1)
        q[:, 3] = torch.cos(yaw / 2); q[:, 4] = 0; q[:, 5] = 0; q[:, 6] = torch.sin(yaw / 2)
        if self.tiles:      # random tile, random start inside the flat start zone, z lifted by the local map height
            import tiles as TL
            tid = torch.randint(0, self.tile_xy.shape[0], (n,), device=DEV, generator=self.g)
            self.cur_tile[idx] = tid
            xs = self.tile_xy[tid, 0] + self.U(n, lo=-0.3, hi=0.3); ys = self.tile_xy[tid, 1] + self.U(n, lo=-0.3, hi=0.3)
            hz = self.hgt(xs, ys) - TL.BASE
            for j in (0, 10):
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += self.hgt(xs, ys)
        elif self.terrain:      # random segment of the ridge, random x; z lifted by the local terrain height
            seg = torch.randint(0, len(START_Y), (n,), device=DEV, generator=self.g)
            sgn = torch.where(self.U(n) < 0.5, -1.0, 1.0)
            ys = torch.tensor(START_Y, device=DEV)[seg] * sgn + self.U(n, lo=-0.12, hi=0.12)
            if self.edge_p > 0:      # start shortly before one of the edges
                ed = torch.tensor(STEP_Y, device=DEV)[torch.randint(0, len(STEP_Y), (n,), device=DEV, generator=self.g)]
                ys = torch.where(self.U(n) < self.edge_p, sgn * (ed - self.U(n, lo=0.2, hi=0.45)), ys)
            xs = self.U(n, lo=-1.0, hi=1.0)
            hz = torch.zeros(n, device=DEV)
            for e, hh in zip(STEP_Y, STEP_H):
                hz = torch.where(ys.abs() >= e, torch.full_like(hz, hh), hz)
            for j in (0, 7 + 3):     # base and ball free joints (qpos 0:7, 10:17)
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += hz
        self.qpos[idx] = q
        self.qvel[idx] = 0
        self.warm[idx] = 0
        self.ctrl[idx] = 0
        if self.motor:
            self.tau_act[idx] = 0
        self.xfrc[idx] = 0
        self.sample_cmd(idx)
        self.vint[idx] = 0; self.last_a[idx] = 0; self.has_prev[idx] = False; self.steps[idx] = 0
        self.tq[idx] = 1 + (self.U(n, lo=-0.15, hi=0.15) if self.dr else 0.0)
        if self.dr:
            ii = idx.nonzero()[:, 0]
            self.t_mass[ii] = self.nom_mass
            for b in self.body_ids:
                self.t_mass[ii, b] = self.nom_mass[b] * (1 + self.U(n, lo=-0.15, hi=0.15))
            self.t_fric[ii] = self.nom_fric
            self.t_fric[ii, :, 1] = self.nom_fric[:, 1] * (1 + self.U(n, lo=-0.2, hi=0.2))[:, None]
            self.t_damp[ii] = self.nom_damp
            lo, hi = self.drc["damping"]
            for k in range(3):
                self.t_damp[ii, 6 + k] = self.U(n, lo=lo, hi=hi) * self.nom_damp[6 + k] / 0.1
        if self.surfaces:
            ii = idx.nonzero()[:, 0]
            sid = torch.randint(0, len(SURFACES), (n,), device=DEV, generator=self.g)
            self.surf_id[ii] = sid
            mu = self.surf_tab[sid, 0] * (1 + self.U(n, lo=-0.15, hi=0.15)); tc = self.surf_tab[sid, 1]
            for g in self.surf_geoms:
                self.t_gfric[ii, g, 0] = mu
                self.t_gsolref[ii, g, 0] = tc
        self.hist[idx] = 0
        self.push_left[idx] = 0
        self.push_t[idx] = self.U(n, lo=200, hi=400)

    @staticmethod
    def rotvec(q):
        w, v = q[:, 0], q[:, 1:4]
        n = v.norm(dim=1).clamp_min(1e-9)
        ang = 2 * torch.atan2(n, w)
        return v / n[:, None] * ang[:, None]

    def sensor_yaw(self):
        q = self.qpos[:, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        body = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
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
            mb = (self.t_mass[:, self.body_ids[0]] / self.nom_mass[self.body_ids[0]] - 1) * 5
            mk = (self.t_mass[:, self.body_ids[1]] / self.nom_mass[self.body_ids[1]] - 1) * 5
            dm = (self.t_damp[:, 6] / self.nom_damp[6] - 1)
        else:
            mb = mk = dm = torch.zeros(N, device=DEV)
        return torch.cat([torch.stack([mu, tc, mb, mk, (self.tq - 1) * 5], 1), dm[:, None] * 0 + dm[:, None], self.scan_feat()[:, :34]], 1)[:, :40]

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
        motor = (self.qvel[:, 1:4] / self.tau).clamp(-2, 2)
        o = torch.cat([self.rotvec(q), self._ang, self._vel, motor, self.last_a, self.cmd,
                       torch.stack([torch.sin(yaw), torch.cos(yaw)], 1), self.vint], 1)
        so = self.sensor_obs()
        return o if so is None else torch.cat([o, so], 1)

    def reset_obs(self):
        q = self.qpos[:, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        z3 = torch.zeros(self.N, 3, device=DEV)
        o = torch.cat([self.rotvec(q), z3, z3, z3, self.last_a, self.cmd, torch.stack([torch.sin(yaw), torch.cos(yaw)], 1), self.vint], 1)
        so = self.sensor_obs()
        return o if so is None else torch.cat([o, so], 1)

    def step(self, a):
        a = a.clamp(-1, 1)
        applied = a.clone()
        if self.dr:
            late = self.has_prev & (self.U(self.N) < self.drc.get('latency_p', 0.2))
            applied[late] = self.prev_a[late]
        self.prev_a = a.clone(); self.has_prev[:] = True
        c = (applied * self.tau * self.tq[:, None]).clamp(-self.tau, self.tau)
        (self.motor_cmd if self.motor else self.ctrl)[:] = -c
        if self.dr:   # random base pushes, 5 control steps long
            self.push_t -= 1
            start = (self.push_t <= 0) & (self.push_left <= 0)
            n = int(start.sum())
            if n:
                ang = self.U(n, hi=2 * np.pi)
                f = self.U(n, lo=self.drc['push_force'][0], hi=self.drc['push_force'][1])
                self.push_f[start] = torch.stack([f * torch.cos(ang), f * torch.sin(ang), torch.zeros_like(f)], 1)
                self.push_left[start] = 5
                self.push_t[start] = self.U(n, lo=200, hi=400)
            on = self.push_left > 0
            self.xfrc[:, self.base, :3] = torch.where(on[:, None], self.push_f, torch.zeros_like(self.push_f))
            self.push_left -= on.long()
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
        r = raw * 0.5 * 5
        r = r - 1e-4 * 5 * (a ** 2).sum(1)
        rate = ((a - self._prev_applied_cmd) ** 2).sum(1) if hasattr(self, "_prev_applied_cmd") else torch.zeros_like(r)
        self._prev_applied_cmd = a.clone()
        r = r - 0.02 * 5 * rate
        off = torch.rad2deg(torch.atan2(rel[:, :2].norm(dim=1), rel[:, 2].clamp_min(1e-6)))
        r = r - 0.05 * 5 * (off / 30.0) ** 2
        q = self.qpos[:, 3:7]
        up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        tilt = torch.rad2deg(torch.acos(up_z.clamp(-1, 1)))
        bad = ~torch.isfinite(self.qpos).all(1)
        fell = (tilt > 70) | (off > 30) | bad
        r = torch.where(fell, r - 2.0, r + 0.02 * 5)
        r = torch.nan_to_num(r, nan=-2.0)
        trunc = (self.steps >= self.max_steps) & ~fell
        if self.tiles:      # leaving the 2.9 m square around the tile centre ends the episode without a fall penalty
            import tiles as TL
            c = self.tile_xy[self.cur_tile]
            out = ((self.qpos[:, 10] - c[:, 0]).abs() > TL.EXIT_R) | ((self.qpos[:, 11] - c[:, 1]).abs() > TL.EXIT_R)
            self.exited = out & ~fell
            trunc = trunc | self.exited
        return o, r, fell, trunc, dict(tilt=tilt, speed_err=torch.nan_to_num(e, nan=0.0, posinf=0.0))

    def noisy(self, o):
        o = self._noisy(o)
        self._last_noisy = o[:, :OBS_DIM].clone()
        parts = [o]
        if self.hist_k:
            parts.append(self.hist.flatten(1))
        if self.asym:
            parts.append(self.priv_obs())
        return torch.cat(parts, 1) if len(parts) > 1 else o

    def _noisy(self, o):
        if not self.dr:
            return o
        n = torch.randn(self.N, 9, device=DEV, generator=self.g)
        o = o.clone()
        sd = float(self.drc.get('obs_noise', 0.01))
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

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class Net(nn.Module):
    """MLP on the 22 proprioceptive inputs plus (optionally) a learned encoder of the sensor vector."""
    def __init__(self, sdim, out, enc=32, hdim=0, pdim=0, use_priv=False):
        super().__init__()
        self.sdim, self.hdim, self.pdim, self.use_priv = sdim, hdim, pdim, use_priv and pdim > 0
        self.enc = nn.Sequential(nn.Linear(sdim, 128), nn.Tanh(), nn.Linear(128, enc), nn.Tanh()) if sdim else None
        self.body = nn.Sequential(nn.Linear(OBS_DIM + (enc if sdim else 0) + hdim + (pdim if self.use_priv else 0), 256), nn.Tanh(),
                                  nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, out))

    def forward(self, x):
        a, b = OBS_DIM, OBS_DIM + self.sdim
        c = b + self.hdim
        parts = [x[:, :a]]
        if self.sdim:
            parts.append(self.enc(x[:, a:b]))
        if self.hdim:
            parts.append(x[:, b:c])
        if self.use_priv:
            parts.append(x[:, c:c + self.pdim])
        return self.body(torch.cat(parts, 1))


class AC(nn.Module):
    def __init__(self, log_std=-1.0, sdim=0, hdim=0, pdim=0):
        super().__init__()
        self.pi, self.v = Net(sdim, ACT_DIM, hdim=hdim, pdim=pdim, use_priv=False), Net(sdim, 1, hdim=hdim, pdim=pdim, use_priv=True)
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
    ap.add_argument("--hist", type=int, default=0, help="stack of the last K (obs, action) pairs as policy input")
    ap.add_argument("--asym", action="store_true", help="asymmetric critic with privileged inputs")
    ap.add_argument("--surfaces", action="store_true", help="randomise floor material class per world (friction, softness)")
    ap.add_argument("--tiles", action="store_true", help="tiled multi-terrain map (steps, drops, ramps, stairs, bumps, rough)")
    ap.add_argument("--edge-focus", type=float, default=0.0)
    ap.add_argument("--mount", choices=["head", "cones"], default="head")
    ap.add_argument("--depth-noise", type=float, default=0.005)
    ap.add_argument("--terrain", action="store_true", help="shared stepped-ridge height-field with random start rows")
    ap.add_argument("--cfg", default=None, help="env config suffix, e.g. nom_real (motor model, stronger noise/latency)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = ROOT / "outputs" / "study" / "warp_ppo" / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"))
    env = GpuBallbot(a.nworld, a.seed, dr=not a.no_dr, design=a.design, cfg_name=a.cfg, terrain=a.terrain, sensor=a.sensor, depth_noise=a.depth_noise, mount=a.mount, edge_p=a.edge_focus, tiles=a.tiles, surfaces=a.surfaces, hist_k=a.hist, asym=a.asym)
    N, T = a.nworld, a.n_steps
    ac = AC(sdim=env.sdim, hdim=env.hdim, pdim=env.pdim).to(DEV)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr, eps=1e-5)
    on, rn = RunNorm(env.obs_dim), RunNorm(1)
    ret = torch.zeros(N, device=DEV)
    o_raw = env.noisy(env.obs())
    on.update(o_raw)
    total, it, t0 = 0, 0, time.time()
    n_iter = int(a.steps // (N * T))
    log = open(out / "log.csv", "w"); log.write("iter,steps,time_s,ep_rew_step,fall_rate,speed_err,tilt_mean,loss_pi,loss_v,kl,log_std\n")
    for it in range(n_iter):
        lr = a.lr + (a.lr_end - a.lr) * it / max(1, n_iter - 1)
        for gp in opt.param_groups: gp["lr"] = lr
        B = dict(o=[], a=[], lp=[], v=[], r=[], d=[])
        falls, rsum, errs, tilts = 0, 0.0, 0.0, 0.0
        for t in range(T):
            x = on(o_raw)
            with torch.no_grad():
                dist = ac.dist(x); act = dist.sample(); lp = dist.log_prob(act).sum(1); v = ac.v(x)[:, 0]
            o_next, r, fell, trunc, info = env.step(act)
            o_next_n = env.noisy(o_next)
            r_true = r.clone()
            if trunc.any():       # bootstrap time-limit truncations
                with torch.no_grad():
                    vb = ac.v(on(o_next_n))[:, 0]
                r = torch.where(trunc, r + a.gamma * vb, r)
            done = fell | trunc
            ret = ret * a.gamma + r_true
            rn.update(ret[:, None])
            ret = torch.where(done, torch.zeros_like(ret), ret)
            rs = r / torch.sqrt(rn.var[0] + 1e-8)
            B["o"].append(x); B["a"].append(act); B["lp"].append(lp); B["v"].append(v); B["r"].append(rs.clamp(-10, 10)); B["d"].append(done)
            falls += int(fell.sum()); rsum += float(r_true.mean()); errs += float(info["speed_err"].mean()); tilts += float(info["tilt"].mean())
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
            row = f"{it},{total},{time.time()-t0:.0f},{rsum/T:.4f},{falls/(N*T):.5f},{errs/T:.4f},{tilts/T:.3f},{float(lpi):.4f},{float(lv):.4f},{kl:.4f},{float(ac.log_std.mean()):.3f}"
            log.write(row + "\n"); log.flush()
            print(row, f" | {total/(time.time()-t0)/1e3:.0f}k env-steps/s", flush=True)
        if it % 50 == 0 or it == n_iter - 1:
            torch.save(dict(ac=ac.state_dict(), sdim=env.sdim, hdim=env.hdim, pdim=env.pdim, hist=env.hist_k, asym=env.asym, mean=on.mean.cpu(), var=on.var.cpu(), steps=total), out / "model.pt")
    print("done", total, "steps in", time.time() - t0, "s")


if __name__ == "__main__":
    main()
