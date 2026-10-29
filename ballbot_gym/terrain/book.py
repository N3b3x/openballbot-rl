"""Book / curb terrain: a raised slab across the path (plus optional Perlin roughness).

Heights are returned normalised by the height-field z size (2.0 m in ballbot.xml),
so ``step_height_m=0.03`` is a 3 cm book. The robot starts at the origin and drives
towards +y; array index [i, j] <-> (x, y).
"""
import numpy as np

from ballbot_gym.terrain.perlin import generate_perlin_terrain

HFIELD_Z = 2.0  # metres, must match <hfield size="5 5 2.0 ..."> in ballbot.xml


def generate_book_terrain(
        n: int,
        step_height_m: float = 0.03,
        step_y: float = 0.6,
        book_len: float = 0.0,
        book_width: float = 0.0,
        rough_amplitude: float = 0.0,
        ramp_len: float = 0.0,
        seed: int = 0,
        step_height_range=None,
        **_ignored) -> np.ndarray:
    """book_len = 0 gives a curb (raised plateau); > 0 a slab of that length in y.
    book_width = 0 spans the full width; > 0 a slab of that width centred on x=0.
    ramp_len > 0 makes the front edge a ramp of that length (a soft lip)."""
    half = 5.0
    coords = np.linspace(-half, half, n)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    y1 = step_y + book_len if book_len > 0 else np.inf
    if ramp_len > 0:
        up = np.clip((Y - step_y) / ramp_len, 0.0, 1.0)
    else:
        up = (Y >= step_y).astype(float)
    mask = up * (Y < y1)
    if book_width > 0:
        mask = mask * (np.abs(X) <= book_width / 2)
    terrain = mask * (step_height_m / HFIELD_Z)
    if rough_amplitude > 0:
        terrain = terrain + generate_perlin_terrain(n, amplitude=rough_amplitude, seed=seed).reshape(n, n)
    return np.clip(terrain, 0.0, 1.0).flatten()
