# Perception-guided ballbot study — scripts

Everything needed to reproduce the experiments in `paper/main.tex`.

| Script | Purpose |
|---|---|
| `train_ppo.py` | PPO training. `--variant proprio \| depth_frozen \| depth_e2e`, any `configs/env/*.yaml` |
| `evaluate.py` | Held-out evaluation (seeds >= 100000) incl. depth-noise / dropout / occlusion conditions |
| `analyze.py` | Tables, paired statistics (bootstrap CI + Wilcoxon) and figures |
| `make_notebook.py` | Regenerates `notebooks/ballbot_perception_rl_draft.ipynb` |

```bash
# train (4 runs: 2 variants x 2 seeds)
python scripts/study/train_ppo.py --name proprio_s0 --variant proprio \
    --env-config configs/env/perlin_directional_train.yaml --total-timesteps 1.5e6 \
    --seed 0 --n-envs 5 --n-steps 1024
python scripts/study/train_ppo.py --name depth_frozen_s0 --variant depth_frozen \
    --env-config configs/env/perlin_directional_train.yaml --total-timesteps 1.5e6 \
    --seed 0 --n-envs 5 --n-steps 1024

# evaluate on held-out terrains (clean) and under perception corruption
python scripts/study/evaluate.py --runs outputs/study/runs/proprio_s0:proprio \
    outputs/study/runs/depth_frozen_s0:depth_frozen --episodes 30 --out outputs/study/results/h1.csv
python scripts/study/evaluate.py --runs outputs/study/runs/depth_frozen_s0:depth_frozen \
    --conditions clean noise_0.05 noise_0.15 noise_0.30 dropout_0.2 occlusion_0.25 occlusion_0.50 \
    --episodes 30 --out outputs/study/results/h2.csv

python scripts/study/analyze.py --h1 outputs/study/results/h1.csv --h2 outputs/study/results/h2.csv
```

## Fixes to the upstream environment that these experiments depend on

1. **Omniwheel contact frames** (`ballbot_gym/utils/contact_frame.py`): re-implements `tools/mujoco_fix.patch`
   in Python so stock `pip install mujoco` works (bit-identical to `mj_step` when the fix is off).
2. **Velocity bug** (`ballbot_env.py`): MuJoCo's `cvel` is `(angular, linear)`; the code had the halves swapped, so the
   directional reward paid for angular velocity.
3. **Terrain amplitude** (`terrain/perlin.py`, `amplitude`/`amplitude_range`): default Perlin terrain has ~46 deg mean slope.
4. **Missing assets/configs**: `ballbot_gym/models/meshes/cone.stl` (ignored by `.gitignore`, restored from git history) and
   `configs/env/*.yaml` (hidden by the `env/` ignore rule).
5. `noise` fallback (`terrain/_noise_fallback.py`) for platforms without a C toolchain.
6. Time-limit truncation + goal success (`env.goal_progress`), action repeat, depth corruption, optional waypoint mode
   (`env.goal_mode`).
