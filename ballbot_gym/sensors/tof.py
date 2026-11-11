"""Simulated multizone time-of-flight (ToF) sensors, e.g. ST VL53L9-class (up to 54x42 zones,
about 54x42 deg FoV, 5 cm to about 9 m). Implemented with MuJoCo ray casts against the terrain
only, so it costs microseconds instead of a render.

Each sensor: mount position / yaw / pitch in the base frame, zone grid, FoV, range, noise model
(Gaussian: sigma = noise_abs + noise_rel * range), per-zone dropout and out-of-range handling.
Output per sensor: distances normalised to [0, 1] by max_range (1 = nothing seen).
"""
import mujoco
import numpy as np


def _rot_yaw_pitch(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rz @ Rx


class ToFArray:

    def __init__(self, model, sensors, terrain_geom="terrain", base_body="base"):
        self.m = model
        self.base = model.body(base_body).id
        self.terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, terrain_geom)
        self.cfg = []
        for s in sensors:
            nx, ny = s.get("zones", [8, 8])
            fx, fy = np.radians(s.get("fov", [54.0, 42.0]))
            # forward axis is +y in the sensor frame; x right, z up
            ax = np.tan(np.linspace(-fx / 2, fx / 2, nx) * (nx - 1) / nx)
            az = np.tan(np.linspace(-fy / 2, fy / 2, ny) * (ny - 1) / ny)
            gx, gz = np.meshgrid(ax, az)
            dirs = np.stack([gx.ravel(), np.ones(gx.size), gz.ravel()], 1)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            R = _rot_yaw_pitch(s.get("yaw_deg", 0.0), s.get("pitch_deg", 0.0))
            self.cfg.append(dict(pos=np.array(s.get("pos", [0, 0, 0.0])), dirs_local=dirs @ R.T,
                                 shape=(ny, nx), max_range=float(s.get("max_range", 3.0)),
                                 min_range=float(s.get("min_range", 0.05)),
                                 noise_abs=float(s.get("noise_abs", 0.005)),
                                 noise_rel=float(s.get("noise_rel", 0.01)),
                                 dropout=float(s.get("dropout", 0.0))))
        self.size = sum(c["shape"][0] * c["shape"][1] for c in self.cfg)

    def __call__(self, data, rng, extra_noise=0.0, extra_dropout=0.0):
        R = data.xmat[self.base].reshape(3, 3)
        origin_b = data.xpos[self.base]
        outs = []
        for c in self.cfg:
            n = len(c["dirs_local"])
            pnt = origin_b + R @ c["pos"]
            vec = c["dirs_local"] @ R.T
            dist = np.zeros(n)
            gid = np.zeros(n, dtype=np.int32)
            mujoco.mj_multiRay(self.m, data, pnt, vec.ravel(), None, 1, self.base, gid, dist, None, n, c["max_range"])
            d = np.where((gid == self.terrain) & (dist >= 0), dist, c["max_range"])
            sig = c["noise_abs"] + c["noise_rel"] * d + extra_noise
            d = d + rng.normal(0.0, 1.0, n) * sig
            drop = rng.random(n) < (c["dropout"] + extra_dropout)
            d = np.where(drop | (d < c["min_range"]), c["max_range"], d)
            outs.append(np.clip(d / c["max_range"], 0.0, 1.0).astype(np.float32))
        return np.concatenate(outs)
