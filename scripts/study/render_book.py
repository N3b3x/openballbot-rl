"""Roll out a book-climbing policy; print where it stalls, save an mp4 (side view) and optionally open the live viewer.

  python scripts/study/render_book.py --run book_proprio_s0 --config book_proprio --height 0.02 --video outputs/study/preview/book_proprio_2cm.mp4
  python scripts/study/render_book.py --run book_proprio_s0 --config book_proprio --height 0.02 --viewer
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
from evaluate import _load_policy, _normalize  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402


def build(config, height, seed):
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{config}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["terrain"]["config"].update(seed=int(seed), step_height_m=float(height))
    cfg["terrain"]["config"].pop("ranges", None)
    ec = {k: cfg[k] for k in ("camera", "env", "logging", "tof", "height_scan") if k in cfg}
    return make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=ec,
                            seed=seed, eval_env=True)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--height", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--ckpt", default="final_model")
    ap.add_argument("--video")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--azimuth", type=float, default=0.0, help="0 or 180 = side view (drive direction is +y)")
    a = ap.parse_args()

    model, vn = _load_policy(ROOT / "outputs/study/runs" / a.run, "proprio", a.ckpt)
    env = build(a.config, a.height, a.seed)
    base = env.unwrapped
    obs, info = env.reset(seed=a.seed)
    m, d = base.model, base.data
    frames, ys, zs, tilts = [], [], [], []
    renderer = cam = None
    if a.video:
        import mujoco
        renderer = mujoco.Renderer(m, height=360, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 1.1, a.azimuth, -6

    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 60, -20

    step = 0
    while True:
        with __import__("torch").no_grad():
            act, _ = model.predict(_normalize(obs, vn), deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        step += 1
        ys.append(float(d.qpos[1]))
        zs.append(float(d.qpos[2]))
        if renderer is not None and step % 2 == 0:
            cam.lookat[:] = d.xpos[m.body("base").id] + np.array([0, 0, -0.05])
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.cam.lookat[:] = d.xpos[m.body("base").id]
            viewer.sync()
            time.sleep(0.01)
        if term or trunc:
            break
    print(f"{a.run} h={a.height * 100:.0f}cm steps={step} fell={bool(info.get('failure', False))} "
          f"final_y={ys[-1]:.2f} max_y={max(ys):.2f} (book edge at y=0.60) progress={info['progress']:.2f}")
    # where did y stop moving? last 100 steps displacement
    print(f"  displacement over last 100 steps: {ys[-1] - ys[max(0, len(ys) - 100)]:.3f} m")
    if a.video and frames:
        import imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, frames, fps=50)
        print("saved", a.video)
    if viewer is not None:
        while viewer.is_running():
            time.sleep(0.2)


if __name__ == "__main__":
    main()
