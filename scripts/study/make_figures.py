"""Reproducible MuJoCo renders and diagrams for the paper.

    python scripts/study/make_figures.py            # writes paper/figures/*.png

Figures
    fig_robot_views.png   robot on flat / easy / hard terrain (MuJoCo third-person render)
    fig_episode_strip.png filmstrip of one episode of the trained policy with the two depth views
    fig_trajectories.png  top-down trajectories on the terrain height map (proprio vs depth)
    fig_tilt_time.png     tilt angle versus time on the same terrains
    fig_terrain_levels.png height maps and slope histograms per amplitude
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "study")]
import ballbot_gym  # noqa: F401,E402
from ballbot_gym.terrain.perlin import generate_perlin_terrain  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
from evaluate import _load_policy, _normalize  # noqa: E402

OUT = ROOT / "paper" / "figures"
SAMPLE = ROOT / "outputs" / "study" / "sample"
OUT.mkdir(parents=True, exist_ok=True)


_RENDERERS = {}


def render_scene(base, dist=1.5, azimuth=135, elevation=-22, size=(360, 360)):
    """Third-person MuJoCo render that follows the robot (own free camera)."""
    import mujoco
    key = id(base.model)
    if key not in _RENDERERS:
        _RENDERERS[key] = mujoco.Renderer(base.model, height=size[0], width=size[1])
    r = _RENDERERS[key]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    body = base.model.body("base").id
    cam.lookat[:] = base.data.xpos[body] + np.array([0, 0, -0.05])
    cam.distance, cam.azimuth, cam.elevation = dist, azimuth, elevation
    r.update_scene(base.data, camera=cam)
    return r.render().copy()


def build(cameras, flat=False, amp=0.08, seed=0):
    cfg = yaml.safe_load(open(ROOT / "configs/env/perlin_directional_train.yaml"))
    cfg["camera"]["enabled"] = cameras
    if flat:
        cfg["terrain"] = {"type": "flat", "config": {}}
    else:
        cfg["terrain"]["config"].update(seed=int(seed), amplitude=float(amp))
        cfg["terrain"]["config"].pop("amplitude_range", None)
    env_config = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                           env_config=env_config, seed=seed, eval_env=True)()
    return env


def rollout(run, variant, seed, amp, frames_at=()):
    """Run one deterministic episode; return trajectory, tilt, and rendered frames."""
    model, vn = _load_policy(SAMPLE / run, variant, "final_model")
    env = build(variant != "proprio", amp=amp, seed=seed)
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    xy, tilt, frames, depths = [info["pos2d"].copy()], [], {}, {}
    step = 0
    while True:
        if step in frames_at:
            frames[step] = render_scene(base)
            if variant != "proprio":
                depths[step] = (obs["rgbd_0"][0].copy(), obs["rgbd_1"][0].copy())
        with torch.no_grad():
            a, _ = model.predict(_normalize(obs, vn), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        step += 1
        xy.append(info["pos2d"].copy())
        tilt.append(float(np.degrees(np.linalg.norm(obs["orientation"]))))
        if term or trunc:
            break
    fell = bool(info.get("failure", False))
    env.close()
    return np.array(xy), np.array(tilt), frames, depths, fell


def fig_robot_views():
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, (title, kw) in zip(ax, (("flat", dict(flat=True)), ("amplitude 0.08", dict(amp=0.08, seed=100000)),
                                   ("amplitude 0.18", dict(amp=0.18, seed=100000)))):
        env = build(False, **kw)
        env.reset(seed=kw.get("seed", 0))
        for _ in range(60):
            env.step(np.zeros(3, dtype=np.float32))
        a.imshow(render_scene(env.unwrapped))
        a.set_title(title, fontsize=9)
        a.axis("off")
        env.close()
    fig.suptitle("Simulated ballbot (MuJoCo render) on terrains of increasing difficulty", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robot_views.png", dpi=170)
    plt.close(fig)


def fig_episode_strip():
    steps = (0, 100, 250, 400, 550, 700)
    xy, tilt, frames, depths, fell = rollout("depth_frozen_s0", "depth_frozen", 100000, 0.12, frames_at=steps)
    keys = [s for s in steps if s in frames]
    fig, ax = plt.subplots(3, len(keys), figsize=(2.1 * len(keys), 6.2))
    ax = np.atleast_2d(ax).reshape(3, -1)
    for j, s in enumerate(keys):
        ax[0, j].imshow(frames[s])
        ax[0, j].set_title(f"t = {s * 0.01:.1f} s", fontsize=8)
        ax[1, j].imshow(depths[s][0], vmin=0, vmax=1, cmap="viridis")
        ax[2, j].imshow(depths[s][1], vmin=0, vmax=1, cmap="viridis")
        for i in range(3):
            ax[i, j].axis("off")
    ax[0, 0].text(-0.05, 0.5, "scene", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    ax[1, 0].text(-0.05, 0.5, "depth cam 0", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    ax[2, 0].text(-0.05, 0.5, "depth cam 1", transform=ax[2, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("One held-out episode (amplitude 0.12): scene and the two depth inputs", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_episode_strip.png", dpi=170)
    plt.close(fig)


def fig_trajectories():
    seeds = [100000 + 7 * i for i in range(4)]
    amp = 0.12
    fig, ax = plt.subplots(1, len(seeds), figsize=(2.9 * len(seeds), 3.2))
    tilts = {}
    for j, sd in enumerate(seeds):
        t = generate_perlin_terrain(293, 25.0, 4, 0.2, 2.0, amp, sd).reshape(293, 293) * 2.0
        # env transposes nothing: heightfield row index = x; show with x horizontal, y vertical
        ax[j].imshow(t.T, extent=[-5, 5, -5, 5], origin="lower", cmap="terrain", alpha=0.85)
        for run, var, col in (("proprio_s0", "proprio", "#1f4e9c"), ("depth_frozen_s0", "depth_frozen", "#d9541e")):
            xy, tilt, _, _, fell = rollout(run, var, sd, amp)
            ax[j].plot(xy[:, 0], xy[:, 1], color=col, lw=2, label=("proprio" if var == "proprio" else "proprio+depth"))
            ax[j].plot(*xy[-1], marker="x" if fell else "o", color=col, ms=7)
            tilts[(j, var)] = tilt
        ax[j].plot(0, 0, "k*", ms=9)
        ax[j].set_title(f"seed {sd}", fontsize=8)
        ax[j].set_xlim(-2.5, 2.5)
        ax[j].set_ylim(-1, 4)
        ax[j].set_aspect("equal")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Top-down trajectories (start = star, o = episode end, x = fall); target direction is +y", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_trajectories.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(seeds), figsize=(2.9 * len(seeds), 2.6), sharey=True)
    for j in range(len(seeds)):
        for var, col in (("proprio", "#1f4e9c"), ("depth_frozen", "#d9541e")):
            t = tilts[(j, var)]
            ax[j].plot(np.arange(len(t)) * 0.01, t, color=col, lw=1.2)
        ax[j].axhline(20, color="r", ls="--", lw=1)
        ax[j].set_xlabel("time [s]", fontsize=8)
        ax[j].set_title(f"seed {seeds[j]}", fontsize=8)
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("tilt [deg]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tilt_time.png", dpi=170)
    plt.close(fig)


def fig_terrain_levels():
    amps = (0.04, 0.08, 0.12, 0.18)
    cell = 10 / 293
    fig, ax = plt.subplots(2, len(amps), figsize=(2.8 * len(amps), 5.2))
    for j, A in enumerate(amps):
        t = generate_perlin_terrain(293, 25.0, 4, 0.2, 2.0, A, 100000).reshape(293, 293) * 2.0
        im = ax[0, j].imshow(t, extent=[-5, 5, -5, 5], cmap="terrain")
        ax[0, j].set_title(f"A = {A}", fontsize=9)
        gy, gx = np.gradient(t, cell)
        sl = np.degrees(np.arctan(np.hypot(gx, gy))).ravel()
        ax[1, j].hist(sl, bins=40, color="#555")
        ax[1, j].axvline(20, color="r", ls="--")
        ax[1, j].set_xlabel("slope [deg]  (red: 20 deg tilt limit)", fontsize=7)
        plt.colorbar(im, ax=ax[0, j], fraction=0.046, label="m")
    fig.tight_layout()
    fig.savefig(OUT / "fig_terrain_levels.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    for f in (fig_terrain_levels, fig_robot_views, fig_episode_strip, fig_trajectories):
        print("making", f.__name__, flush=True)
        f()
    print("done ->", OUT)
