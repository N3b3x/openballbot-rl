"""Tiled terrain library for per-world terrain variety on MuJoCo-Warp.

Warp has one height-field array shared by all worlds. We make that array a big map (24 m, 3.4 cm cells) holding an 8x8 grid of 3 m tiles,
each with its own terrain type and severity; each world picks a tile (and a start pose inside it) at reset, so 4096 worlds see 64+ different
terrains at once. Features are placed symmetrically around the tile centre (|x|,|y| >= 0.5 m from it) so any travel direction meets them:
    h(x, y) = B + g(x) + g(y)        (additive, g depends on the signed distance t from the centre along each axis)
Types (g, with severity s in the stated range):
  flat; step_up (1-6 cm); step_down (1-6 cm: robot stands on a plateau); small_drop (3-8 cm); large_drop (10-20 cm);
  ramp_up / ramp_down (3-12 deg slope, over 1.0 m); stairs (2-3 treads, riser 1-4 cm, tread 0.4 m); bump (slab 0.3 m wide, 1-5 cm);
  rough (smooth noise, 0.5-3 cm amplitude, plus a flat 0.5 m start zone).
B is a base elevation so that negative features stay inside the [0, z_top] range of a MuJoCo height-field.
"""
import numpy as np

HALF = 12.0            # map half-extent (m), must match the patched xml
NCELL = 707            # cells per side -> 3.40 cm
TILE = 3.0
NSIDE = 8
BASE = 0.30            # base elevation (m)
HZ = 2.0               # hfield z_top (m)
START_R = 0.5          # flat start zone radius
EXIT_R = 1.45          # a world is truncated when it leaves this radius (square) around its tile centre

TYPES = ["flat", "step_up", "step_down", "small_drop", "large_drop", "ramp_up", "ramp_down", "stairs", "bump", "rough"]
RANGE = {"flat": (0, 0), "step_up": (0.01, 0.06), "step_down": (0.01, 0.06), "small_drop": (0.03, 0.08), "large_drop": (0.10, 0.20),
         "ramp_up": (3.0, 12.0), "ramp_down": (3.0, 12.0), "stairs": (0.01, 0.04), "bump": (0.01, 0.05), "rough": (0.005, 0.03),
         "blocks": (0.3, 0.5)}
BLOCK_H = 0.08         # obstacle block height (m): well above the ~2.6 cm traction limit of the 9 cm ball (tracking_study_log S88)


def blocks_patch(tx, ty, w, rng):
    """'blocks' tile (navigation test, not part of TYPES): flat ground with four BLOCK_H blocks, one across each tile axis, centred
    0.75 m from the tile centre, w metres wide (perpendicular to the axis) and 0.2 m deep, plus two smaller random blocks between
    the axes. A goal placed on an axis beyond its block can only be reached by going around a block end."""
    X, Y = np.meshgrid(tx, ty)
    m = np.zeros_like(X, dtype=bool)
    for (ux, uy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        along = X * ux + Y * uy; across = X * (-uy) + Y * ux
        m |= (np.abs(along - 0.75) <= 0.1) & (np.abs(across) <= w / 2)
    for _ in range(2):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.7, 1.1); bx, by = r * np.cos(a), r * np.sin(a)
        if min(abs(np.cos(a)), abs(np.sin(a))) < 0.35:      # keep the axes (goal corridors' approaches) for the axis blocks
            continue
        m |= (np.abs(X - bx) <= 0.12) & (np.abs(Y - by) <= 0.12)
    return BLOCK_H * m


def g_profile(kind, s, t, rng_noise=None):
    """Height contribution along one axis at signed distance t (array, metres)."""
    a = np.abs(t)
    if kind == "flat":
        return np.zeros_like(t)
    if kind == "step_up":
        return s * (a >= 0.6)
    if kind in ("step_down", "small_drop", "large_drop"):
        return -s * (a >= 0.6)
    if kind == "ramp_up":
        return np.tan(np.radians(s)) * np.clip(a - 0.5, 0.0, 1.0)
    if kind == "ramp_down":
        return -np.tan(np.radians(s)) * np.clip(a - 0.5, 0.0, 1.0)
    if kind == "stairs":
        return s * np.clip(np.floor((a - 0.5) / 0.4) + 1.0, 0.0, 3.0) * (a >= 0.5)
    if kind == "bump":
        return s * ((a >= 0.6) & (a < 0.9))
    if kind == "rough":
        return np.zeros_like(t)
    raise ValueError(kind)


def smooth_noise(n, rng, cells=14):
    """Bilinearly interpolated random grid, two octaves, zero mean, unit-ish scale."""
    out = np.zeros((n, n))
    for k, w in ((cells, 1.0), (cells * 2, 0.5)):
        g = rng.standard_normal((k + 1, k + 1))
        u = np.linspace(0, k, n)
        i = np.minimum(u.astype(int), k - 1); f = u - i
        a = g[i][:, i] * (1 - f)[None] * (1 - f)[:, None] + g[i][:, i + 1] * f[None] * (1 - f)[:, None]
        b = g[i + 1][:, i] * (1 - f)[None] * f[:, None] + g[i + 1][:, i + 1] * f[None] * f[:, None]
        out += w * (a + b)
    return out / 1.5


def build(seed=0, kinds=None):
    """Returns (height_m (NCELL,NCELL) rows=y cols=x, info list of dict(kind, severity, cx, cy))."""
    rng = np.random.default_rng(seed)
    kinds = kinds or TYPES
    ax = np.linspace(-HALF, HALF, NCELL)
    H = np.full((NCELL, NCELL), BASE)
    info = []
    for iy in range(NSIDE):
        for ix in range(NSIDE):
            cx, cy = -HALF + TILE * (ix + 0.5), -HALF + TILE * (iy + 0.5)
            kind = kinds[(iy * NSIDE + ix) % len(kinds)] if len(kinds) <= NSIDE * NSIDE else kinds[rng.integers(len(kinds))]
            lo, hi = RANGE[kind]
            s = float(rng.uniform(lo, hi))
            xs = (ax >= cx - TILE / 2) & (ax < cx + TILE / 2)
            ys = (ax >= cy - TILE / 2) & (ax < cy + TILE / 2)
            tx, ty = ax[xs] - cx, ax[ys] - cy
            if kind == "blocks":
                patch = blocks_patch(tx, ty, s, rng)
            else:
                patch = g_profile(kind, s, tx)[None, :] + g_profile(kind, s, ty)[:, None]
            if kind == "rough":
                nz = smooth_noise(max(len(tx), len(ty)), rng)[:len(ty), :len(tx)]
                zone = (np.maximum(np.abs(tx)[None, :], np.abs(ty)[:, None]) > START_R)
                patch = patch + s * nz * zone
            H[np.ix_(ys, xs)] = BASE + patch
            info.append(dict(kind=kind, severity=s, cx=cx, cy=cy))
    return H, info


def data_for_mujoco(H):
    """Normalised hfield data (0..1 of z_top); flattened row-major."""
    return np.clip(H / HZ, 0.0, 1.0).flatten()
