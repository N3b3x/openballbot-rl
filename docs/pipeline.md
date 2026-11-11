# Pipeline walkthrough

A technical tour of the current ballbot pipeline: what each stage does, which files implement it, and the commands,
flags and environment variables that matter. Everything is simulation; there is no physical robot. Numbers and their
reasons are in the study log ([research/tracking_study_log.md](research/tracking_study_log.md)); entry numbers (S-numbers)
below point there.

Conventions used in the commands:

```bash
cd experiments/warp_ppo
PY=../mjx_port/.venv/Scripts/python.exe        # the study venv (Windows layout); see the top-level Readme for setup
```

Run outputs go to `outputs/study/warp_ppo/<run>/`.

## 1. Robot spec to simulated design

| File | Role |
|---|---|
| [specs/nominal.yaml](../specs/nominal.yaml), [specs/sweep_one_at_a_time.yaml](../specs/sweep_one_at_a_time.yaml) | Specs in physical units: ball radius and mass, body mass and COM height above the ball centre, per-wheel torque limit, rotor inertia, wheel damping, contact zenith angle |
| [scripts/study/robot_spec.py](../scripts/study/robot_spec.py) | Maps a spec onto the parametric generator, reports derived physics (unstable pole, non-minimum-phase zero, lean per acceleration, torque authority) and feasibility margins, and with `--eval` simulates the design with a trained controller |
| [scripts/study/design_space.py](../scripts/study/design_space.py) | Parametric design `(s, kball, kbal, zb, J, tau, damp, dz)`: writes the MuJoCo variant `ballbot_gym/models/variants/dz_<name>.xml`, the planar model and LQR json, and an env yaml (S16) |
| [scripts/study/design_constraints.py](../scripts/study/design_constraints.py) | Analytic margins: structure, loop delay, traction, motor, lean, contact geometry, wheel speed ([research/design_rules.md](research/design_rules.md)) |
| [ballbot_gym/models/ballbot.xml](../ballbot_gym/models/ballbot.xml) | Base MuJoCo model of the three-omniwheel ballbot |

```bash
python scripts/study/robot_spec.py build specs/nominal.yaml                     # files, physics, margins
python scripts/study/robot_spec.py build specs/nominal.yaml --eval              # plus simulation (GPU)
python scripts/study/robot_spec.py sweep specs/sweep_one_at_a_time.yaml --eval   # one property at a time (S119)
```

Results go to `outputs/study/specs/`. `--eval` uses `g34_mref_rsad_db_s0` (set as `EVAL_RUN` in the script) and calls
the study venv by its Windows path. The controller was trained on 11 designs, so a poor result on a design far from them
can mean extrapolation rather than physics (S119). Guidelines derived from the sweep:
[research/design_guidelines.md](research/design_guidelines.md).

The 11 training designs: `nom`, `zhigh` and `zlow` (COM height), `kb4` and `kb025` (ball mass x4 and x0.25), `bal4` and
`bal025` (body ballast x4 and x0.25), `J02` (rotor inertia), `tau6` (6 N m torque limit), `zeta41` and `zeta66` (contact
angle 41 and 66 deg). Their files are `ballbot_gym/models/variants/dz_<name>.{xml,json}`.

## 2. Simulation

| File | Role |
|---|---|
| [experiments/mjx_port/warp_sim.py](../experiments/mjx_port/warp_sim.py) | `WarpBallbot`: batched MuJoCo-Warp model, CUDA graph, the omniwheel contact-frame fix as a Warp kernel ([experiments/mjx_port/REPORT.md](../experiments/mjx_port/REPORT.md)) |
| [experiments/mjx_port/common.py](../experiments/mjx_port/common.py) | Shared model paths, design loading, numpy LQR law |
| [experiments/warp_ppo/train_dev.py](../experiments/warp_ppo/train_dev.py) | `GpuBallbot`: observations, reward, terminations, command and goal sampling, domain randomisation, pushes and pulls, sensor noise and latency, floor surfaces, sensors; and the PPO loop |
| [experiments/warp_ppo/tiles.py](../experiments/warp_ppo/tiles.py) | One 24 m height field holding 8 x 8 tiles of 3 m: flat, step up and down, small and large drop, ramp up and down, stairs, bump, rough; plus `blocks` tiles for navigation |
| [experiments/warp_ppo/retainer.py](../experiments/warp_ppo/retainer.py) | Passive ball arrestors on the body, after the Rezero ballbot (S79) |

The control rate is 100 Hz. Warp is float32 only; the wheel-ball free-rolling friction default became 0.01 in S98
because the model's 0.001 made the float32 solver blow up when standing.

## 3. Training the controller

The controller is an LQR base per design ([lqr_gpu.py](../experiments/warp_ppo/lqr_gpu.py), discrete LQR on the derived
planar model) plus a PPO residual: applied action = clip(LQR action + rs x policy action). The policy sees a stack of
recent (observation, action) pairs and a 6-D design vector, so one network serves all 11 designs.

Recommended recipe (g39, S121), as launched for `g39_goalcentre_s0`:

```bash
$PY train_dev.py --name g39_goalcentre_s0 --seed 0 --nworld 4096 --steps 200e6 --goal-p 0.5 --pull \
  --dr-scale 1.75 --dr-ramp 0.5 --lqr-base 0.25 --rw-track 2.0 --cross-focus 0.6 \
  --designs nom,zhigh,zlow,kb4,bal4,J02,tau6,bal025,kb025,zeta41,zeta66 --design-cond \
  --width 512 --layers 3 --hist 8 --asym --surfaces --tiles --retainer 0,0,0,100 \
  --motion-ref --rs-adapt 0.15,0.3,0.05 --goal-centre
```

The g41 runs in progress (S122) add `--db-centre` to the same command.

### Training flags that define the recipe

| Flag | Meaning | Log |
|---|---|---|
| `--lqr-base RS` | Residual authority on the LQR base; 0 = pure learned policy | S58, S83 |
| `--rs-adapt lo,hi[,v]` | Adaptive authority: `lo` while cruising, `hi` when the command changes fast, nears zero or the body tilts; optional third value is the dead band (the near-zero term only raises authority while the ball moves faster than `v` m/s) | S112, S113 |
| `--motion-ref` | Train on commands shaped by the motion-reference layer (arrival, hold, curving commands); `--curve-p` sets the fraction of curving velocity commands | S110 |
| `--goal-centre` | All goal logic (goal rule, goal observation, motion-layer hold, navigation) on the true ball centre instead of the ball body origin, which is 0.14 m from the centre and swings as the ball rolls | S118 |
| `--db-centre` | The adaptive-authority dead band reads the ball-centre velocity instead of the body-origin velocity | S122 |
| `--hist K` | Stack of the last K (observation, action) pairs; 8 = 80 ms, 24 = 240 ms | S117, S120 |
| `--designs a,b,...`, `--design-cond` | Design pool sampled per world; policy sees the design vector | S33 onward |
| `--design-w name:w,...` | Per-design sampling weights in training only (for example `bal4:2,tau6:2,kb025:2`) | S114 |
| `--push-train X` | Push and pull strength multiplier during training (1 = evaluation strength) | S116 |
| `--pull` | Sustained pulls on the base in addition to the short pushes | S42, S116 |
| `--dr-scale`, `--dr-ramp` | Final scale of all randomisation ranges, and the fraction of training over which it ramps up | S67 |
| `--goal-p`, `--cross-focus` | Fraction of goal-directed episodes; probability that a goal lies across a tile feature | S51, S57 |
| `--rw-track` | Weight on the velocity-tracking reward | S57 |
| `--tiles`, `--surfaces` | Tiled terrain map; randomised floor material per world | S33 |
| `--retainer gap_mm,fric,floor,angle` | Ball arrestors (the recipe uses `0,0,0,100`) | S79 |
| `--pair-mu1` | Wheel-ball free-rolling friction (default 0.01) | S98 |
| `--asym`, `--width`, `--layers` | Asymmetric critic with privileged inputs; network size | |
| `--ep-steps`, `--zero-p` | Episode length (800 = 8 s) and probability of a zero (stand-still) command | S94 |
| `--nav-frac` | Fraction of worlds driven by the wavefront planner during training (a null result in S91) | S91 |
| `--sensor scan or depth` | Privileged height-scan teacher or depth-ray observations | S40, S84 |

### What a run writes, and how evaluation re-applies it

`train_dev.py` writes `args.json` (every command-line argument), `log.csv` (per-iteration training statistics),
`falls_by.csv` and `model.pt` (actor-critic weights plus the network shape, history length, design list and observation
normalisation) into `outputs/study/warp_ppo/<run>/`. `model.pt` and `log.csv` are git-ignored.

Every evaluation and render script rebuilds the network from `model.pt` and re-applies the behaviour settings from
`args.json`, so a run is always evaluated the way it was trained:

| Setting in args.json | Re-applied by |
|---|---|
| `lqr_base`, `lqr_amax`, `lqr_r`, `rs_adapt` | eval_tiles2.py, eval_motion.py, eval_nav.py, render_policy.py |
| `motion_ref` | eval_tiles2.py, eval_nav.py, render_policy.py (off with `NO_MREF=1`); eval_motion.py takes it as `--mode ref` |
| `goal_centre`, `db_centre` | eval_tiles2.py, eval_motion.py, eval_nav.py, render_policy.py |
| `retainer`, `pair_mu1` | every evaluation script, unless `RETAINER` or `PAIR_MU1` is set explicitly |

eval_battery.py calls eval_tiles2.py with every setting passed explicitly. The ROS 2 export reads the same `args.json`
into the bundle.

## 4. Evaluation

| Script | What it measures |
|---|---|
| [eval_battery.py](../experiments/warp_ppo/eval_battery.py) | Standard battery: flat tracking at 0.3 / 0.6 / 0.9 m/s; goal task (1.1 m) and velocity task (0.3 m/s) on the terrain tiles, 11 designs, DR and surfaces on, pushes and pulls off and on, for each `--dr-scales` value; writes `battery.json` |
| [eval_tiles2.py](../experiments/warp_ppo/eval_tiles2.py) | Per terrain kind and per design: falls, crossings or goals reached, tracking error (`--task vel or goal`, `--dr`, `--surfaces`, `--pull`, `--eval-designs`) |
| [eval_motion.py](../experiments/warp_ppo/eval_motion.py) | Goal arrival and hold, curve and zigzag tracking, raw or through the motion layer (`--scenario goal, curve or zigzag`, `--mode raw or ref`); goal metrics always at the true ball centre |
| [eval_nav.py](../experiments/warp_ppo/eval_nav.py) | Navigation (section 5) |
| [render_policy.py](../experiments/warp_ppo/render_policy.py), [render_nav.py](../experiments/warp_ppo/render_nav.py), [sensor_view.py](../experiments/warp_ppo/sensor_view.py) | Videos of one world on the CPU renderer, with push and pull arrows (`--pull`) and the ToF view (`--sensor-view`) |

```bash
$PY eval_motion.py --run g39_goalcentre_s0 --scenario goal --mode ref --out motion_goal_ref
TILES_FLAT=1 EVAL_NOEXIT=1 PUSH_SCALE=0 $PY eval_tiles2.py --run g39_goalcentre_s0 --task vel --speed 0.3 \
  --n 32 --dr --surfaces --steps 2000                    # 20 s driving on a fully flat map
(cd ../.. && experiments/mjx_port/.venv/Scripts/python.exe experiments/warp_ppo/eval_battery.py \
  --run g39_goalcentre_s0 --dr-scales 1.0 1.75 2.5)
$PY render_policy.py --run g39_goalcentre_s0 --kind step_up --severity 0.03 --task vel --speed 0.6 --design nom --out clip_step
```

### Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `PUSH_SCALE` | eval_tiles2.py, eval_nav.py (default 1), eval_motion.py (default 0) | Scales pushes and pulls; 0 turns them off |
| `TILES_FLAT=1` | train_dev.py (used by evaluations) | A fully flat map instead of the terrain tiles |
| `EVAL_NOEXIT=1` | eval_tiles2.py | Do not end an episode when the ball leaves its tile |
| `DR_SCALE`, `DR_OFF` | eval_tiles2.py | Master randomisation scale; switch single terms off (`mass,friction,damping,torque,latency_p,obs_noise`) |
| `NJMAX`, `NCONMAX` | train_dev.py | Per-world constraint and contact buffers (defaults 64 and 16); 256 / 32 changed no battery number (S113) |
| `NO_MREF=1` | eval_tiles2.py, eval_nav.py, render_policy.py | Bypass the motion layer for a run trained with it |
| `MREF_KW` | eval_nav.py | Motion-layer limits, for example `MREF_KW=a_acc=0.6,a_brk=0.8,a_lat=0.5` (defaults a_acc 0.4, a_brk 0.5, a_lat 0.35, w_max 1.5) (S114) |
| `NAV_ORIGIN=1` | eval_nav.py | Planner, hold and success measured at the ball body origin even for goal-centre runs (S120) |
| `RS_ADAPT` | eval_motion.py | Override the trained adaptive-authority setting (S115) |
| `LQR_ONLY=1` | eval_tiles2.py, eval_motion.py | Zero the residual: the LQR base alone |
| `RETAINER`, `PAIR_MU1` | all evaluation scripts | Override the arrestors or the wheel-ball friction of the run (eval_tiles2.py and render_policy.py accept `RETAINER=off`) |
| `TRAIN_MOD` | evaluation scripts | Module providing the environment (default `train_dev`) |

## 5. Navigation and state estimation

| File | Role |
|---|---|
| [nav.py](../experiments/warp_ppo/nav.py) | `WavePlanner`: obstacle map, wavefront distance field, smooth follower (carrot point, turn-rate and acceleration limits) (S90) |
| [motion_ref.py](../experiments/warp_ppo/motion_ref.py) | `MotionRef`: slew-limited speed, lateral-acceleration-limited turning, braking before reversals, arrival braking curve and goal hold |
| [escape.py](../experiments/warp_ppo/escape.py) | `ContactEscape`: stall detection, obstacle marking, back-up and re-plan for a blind robot (S109) |
| [eval_nav.py](../experiments/warp_ppo/eval_nav.py) | Goal 1.2 m away behind 8 cm blocks; sources `blind`, `oracle` (perfect height map), `sensor` (simulated multizone ToF), `bump` (contact escape) |
| [odometry.py](../experiments/warp_ppo/odometry.py), [collect_odom.py](../experiments/warp_ppo/collect_odom.py), [eval_odom.py](../experiments/warp_ppo/eval_odom.py) | Estimator from IMU and wheel encoders (attitude filter, gyro bias at rest, calibrated encoder map, Kalman filter); ground-truth logs and offline evaluation (S108) |

The ToF configuration of the headline results is 18 x 14 zones over 55 x 42 deg, pitched -25 deg, 0.15 m above the base,
range noise 0.005 d, aimed at the planner target (S114). Navigation as reported in S121:

```bash
for PS in 0 1; do PP=""; SUF=""; [ $PS = 1 ] && PP="--pull" && SUF="_pp"
  PUSH_SCALE=$PS $PY eval_nav.py --run g39_goalcentre_s0 --source sensor --res 18x14 --fov 55,42 \
    --planner smooth --carrot 0.3 --w-max 2.0 --aim target $PP --out navS_tof18$SUF
done
PUSH_SCALE=0 $PY eval_nav.py --run g39_goalcentre_s0 --source oracle --planner smooth --carrot 0.3 --w-max 2.0 --out navS_oracle
```

`--record K` saves world K's trajectory as `<out>_traj.pkl` for `render_nav.py --traj <out>`. For runs trained with the
motion layer, the planner's command passes through it, as it would at deployment.

Odometry study (`--n` is the total number of robots; S108 logged 128):

```bash
$PY collect_odom.py --run g26_ret_rs025_mu01_s1 --where flat --n 128 --seconds 60 --out odom_flat
$PY collect_odom.py --run g26_ret_rs025_mu01_s1 --where tiles --n 128 --seconds 60 --out odom_tiles
$PY eval_odom.py --cal odom_flat --test odom_flat odom_tiles
```

## 6. Export and ROS 2

| File | Role |
|---|---|
| [record_parity.py](../experiments/warp_ppo/record_parity.py) | Records a short simulator rollout (observations, actions, torques) for verification and parity tests |
| [export_policy.py](../experiments/warp_ppo/export_policy.py) | Writes `actor.onnx`, `actor.ts`, `actor_weights.npz` and `bundle.json` (rates, observation layout, history, design table, LQR gains, motion-layer and goal-rule parameters, estimator calibration, source `args.json`) into `ros2/ballbot_control/config/bundle/<run>/` and checks them against the torch actor |
| [ros2/ballbot_control/](../ros2/ballbot_control/) | Pure-Python core (numpy), controller and estimator nodes, launch file, pytest suite |

```bash
$PY record_parity.py --run g39_goalcentre_s0 --n 3 --steps 600
$PY export_policy.py --run g39_goalcentre_s0
cd ../../ros2/ballbot_control && python -m pytest -q
```

The package was not built or run under ROS and not on hardware. Topics, safety behaviour, real-time notes and open issues
are in [ros2/README.md](../ros2/README.md).

## 7. Website and paper

| Command | Writes |
|---|---|
| `python site/build_data.py` | `site/data/study.json`, `site/figures/study/*.svg`, table fragments in the site pages, `paper_v2/tab_gen_*.tex`, `docs/research/site_crosscheck.md` |
| `python scripts/study/make_tracking_report.py` | `outputs/study/results/tracking_summary.csv`, `paper_v2/tab_tracking.tex`, paper figures |

Numbers without a data file are kept in `LOG_ONLY` inside [site/build_data.py](../site/build_data.py) with their log
entry. The site's own [reproduce page](../site/reproduce.html) covers the CPU study tools as well.
