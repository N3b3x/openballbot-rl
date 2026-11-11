"""Pure-NumPy gradient-noise fallback for the C-extension ``noise`` package.

``noise`` has no Windows wheels and needs a C++ toolchain to build. This module
provides a vectorised 2D fractal gradient noise with the same call convention
(octaves / persistence / lacunarity / base seed) so Perlin terrains can be
generated anywhere. Output is roughly in [-1, 1], like ``snoise2``.
"""
import numpy as np


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _gradient_noise(x, y, perm, grads):
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    mask = len(perm) - 1

    def corner(ix, iy, dx, dy):
        g = grads[perm[(perm[ix & mask] + iy) & mask] % len(grads)]
        return g[..., 0] * dx + g[..., 1] * dy

    u = _fade(xf)
    v = _fade(yf)
    n00 = corner(xi, yi, xf, yf)
    n10 = corner(xi + 1, yi, xf - 1, yf)
    n01 = corner(xi, yi + 1, xf, yf - 1)
    n11 = corner(xi + 1, yi + 1, xf - 1, yf - 1)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * np.sqrt(2.0)  # scale to ~[-1, 1]


def fractal_noise_grid(n: int,
                       scale: float,
                       octaves: int,
                       persistence: float,
                       lacunarity: float,
                       seed: int) -> np.ndarray:
    """n x n fractal gradient noise on integer lattice coordinates / scale."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(1024).astype(np.int64)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x0 = ii / scale
    y0 = jj / scale

    total = np.zeros((n, n))
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        # per-octave offset decorrelates octaves
        off = 17.31 * (o + 1)
        total += amp * _gradient_noise(x0 * freq + off, y0 * freq + off, perm, grads)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm
