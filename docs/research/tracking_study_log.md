# Velocity-tracking study log

Running record of the flat-ground command-tracking study: what was tried, why, what happened, and what it changed.
The paper draft is `paper_v2/main.tex`; its table and figures are regenerated from the benchmark CSVs with
`python scripts/study/make_tracking_report.py`.

Goal: a learned low-level controller that makes a small ballbot go where and how fast it is told, in any direction,
smoothly, while staying on the ball. Everything later (book climbing, navigation, character animation) builds on it.

## Protocol (fixed for every policy)

| Tool | What it measures |
|---|---|
| `scripts/study/eval_control.py` | 8 directions x speeds up to vmax, yaw +-1 (and up to wmax), rise / reversal times, arc, heading test (0.5 m/s from random start yaw), smoothness (action rate), max off-ball angle |
| `scripts/study/eval_falls.py` | 30 episodes of random training-distribution commands; falls and the command active at each fall |
| `scripts/study/render_control.py` | Video on a checkerboard floor: planned path (green), actual path (red), target point, off-ball angle; no time-limit resets; a fall ends the video with a banner; optional position guidance `--guide k`; saves `<video>_traj.npz` |
| `scripts/study/make_tracking_report.py` | `outputs/study/results/tracking_summary.csv`, `paper_v2/tab_tracking.tex`, paper figures |

All evaluations use the nominal robot (domain randomisation is switched off unless `--dr` is passed).

## Stages and decisions

**S0 baseline limits** (`ctrl_base`, torque 10 N m, damping 0.8, tilt 20 deg). Balances, barely moves (ratio 0.06 at 1 m/s).
Decision: give the policy more authority.

**S1 free limits** (`ctrl_free_*`, torque 30, damping 0.1, tilt 70). Looked like progress (reward up, speed 0.2).
A trace showed the policy kicks the ball out within 0.2-0.6 s and stands on its wheels; the tilt check never fires.
Decision: these results are void; add an explicit off-ball termination (`max_ball_offset_deg`, angle of the
ball-centre-to-body line from vertical) and log `off_ball_deg`. Lesson: check the constraint that defines the task.

**S2 on-ball** (`ballc_*`). Honest balance, ratio 0.21-0.33. Spin tracking appears for the first time.

**S3 smoothness** (`smooth_*`, `action_rate_coef`, soft `ball_offset_coef`). Jitter down 8-500x, speed down further.
Lesson: a strong smoothness penalty before the policy can move freezes it.

**S4 lean reward** (`lean_*`, `lean_coef`). Pays for leaning into the command while below the commanded speed.
Best 0.36. Probes: policies use 1-3 % of torque and lean ~1 deg; given a 2 m/s command, `lean_c` peaks at 1.21 m/s.
Lesson: the limit is learned caution, not physics. A hand-tuned cascaded PD controller (48 gain/sign combinations)
never balanced: tilt-only feedback lets the ball run away. Dropped; a proper LQR baseline is future work.

**S5 error reward** (`trk_a/b/c`, `cmd_reward: err`, vmax 1.5-2). Still 0.15-0.31. Yaw improved.
Fall probe: all S2-S5 policies fall in 20-30 of 30 random-command episodes; for `trk_a` 23 of 29 falls happen
under yaw commands above 1 rad/s (`trk_c`: 24 of 29). Training sampled +-3 rad/s.

**S6 spin cap** (`trk_d/e/f`, wmax 1). From scratch: ratio 0.91 at 1 m/s, 0.98 at 0.5 (`trk_f`); seed 1 reproduces
(0.95). Continuing timid policies reaches only 0.59, and one collapses into spinning (30/30 falls).
Lessons: infeasible commands in training suppress learning of feasible ones; initialise from scratch after a
distribution change if the previous policy learned to avoid motion.

**S7 refine** (`trk_g` = continue, `trk_h` = continue with 4x action-rate penalty). `trk_g`: action rate 0.147 to
0.013 with no loss at 0.5 m/s, yaw error 0.40 to 0.12. `trk_h`: fewest falls (3/30), about 10 % slower, completes the
full square course. Lesson: track first, then smooth.

**Video fix.** The 8 s episode limit was silently resetting the robot to the start in videos while the
commanded path kept going. Videos now disable the time limit, draw the whole planned path, stop on a real fall
with a banner, and report distance from the planned point.

**Heading.** After a 4 rad turn, world-frame commands are executed in the wrong direction. Every training
episode started at the same heading. Added `init_yaw_random` (uniform start yaw) and a heading test in the benchmark.

**Throughput.** Profiling showed the Python RK4 contact-frame stepper (4 forward passes per physics step) dominates
the step cost. Added `integrator: implicitfast` (one forward pass, same contact-frame fix): 1.10 vs 3.26 ms per
control step (3x). Validation: `trk_g` and `trk_h`, trained on RK4, give the same speed curve within 0.01 m/s, similar
smoothness and falls (trk_h: 5/30 vs 3/30) on implicitfast. All runs from S9 onward use it.

**Heading test** (0.5 m/s from 6 random start yaws x 4 directions): `trk_h` 0.31 m/s error, 21 % falls; `trk_g`
0.32 m/s, 38 % falls. Confirms the heading weakness seen in the path plot.

**First-principles model** (`scripts/study/ballbot_model.py`, expands the prior JHU report
`docs/research/papers/openballbot_rl_reproduce_report.pdf`, which stated M q'' + C q' + G = tau without deriving it).
Parameters are extracted from the XML: body 3.853 kg, COM 0.1306 m above ball centre, I_a 0.0366; ball 0.168 kg,
r 0.09. Omniwheel transmission W (wheel rate = W x ball-relative angular velocity); planar n = (3.61, -1.11, -2.50),
reflected rotor inertia J_eff = J_w |n|^2 = 0.1025 kg m^2 (190x the ball inertia). Planar Lagrangian with
q = (phi, theta) gives M(q), G(q), Coriolis, damping. Results: unstable pole 4.79 1/s; non-minimum-phase zero
z = sqrt(m_a g l / (I_a + m_a l^2 + m_a l r)) = 5.78 1/s (independent of drive); momentum identity (sum of the two
Lagrange equations has no torque/damping); lean law sin(theta) = kappa a, kappa = 0.176 rad per m/s^2 (10.1 deg).
Validation caveats: soft-contact creep at the ball-floor contact on instantaneous accelerations; wheel rates differ
from W by a constant 27-28 % (effective rolling radius), not speed-dependent slip.

**LQR baseline** (`ballbot_gym/controllers/lqr.py`, run name `lqr` in every benchmark). Designed only on the derived
(A, B): K = (0, 76.1, 4.33, 11.6). Needed a reference governor (a_max 0.5 m/s^2, because of the zero), the ball
rotation as state (using measured linear velocity destabilises it because of creep), and a slow outer integral on
measured velocity. Result: ratio 1.00-1.08 at 0.25-0.75 m/s, heading error 0.02 m/s, action rate 0.0002 (50x smoother
than PPO), but it falls in every 1 m/s trial and on the square course's turning leg.

**Theory vs learned** (`scripts/study/analyze_theory.py`). Momentum identity slope 0.93 / 0.88 / 0.89 (LQR, trk_g,
trk_h), R^2 0.82 / 0.60 / 0.60. Non-minimum-phase reversal in all three (1.7 / 0.3 / 1.1 cm backward first).
Lean law on quasi-static samples: 0.168 / 0.160 / 0.146 (95 / 91 / 83 % of kappa). The PPO policies rediscovered
the lean law from reward alone.

**1 m/s edge.** Both families fail near 1 m/s. LQR diagnostics: torque < 17 % of limit, wheel normal forces steady
17-18 N, no growing slip; failure coincides with a yaw spin-up and a growing body-ball oscillation. Hypothesis: 3-D
coupling (rotor gyroscopics, J_w w_w ~ 0.2 N m s per wheel at 40 rad/s). To test with a 3-D linearisation.

**Residual RL** (`residual_lqr: true`). First attempt (`res_a/b`: beta 0.3, random init, vmax 1.5) was worse than
the LQR alone (25-50 % falls at 0.5 m/s): the untrained residual destabilised the LQR (first episodes ~100 steps vs
~700 for LQR alone) and the commands included speeds the LQR cannot survive. `res_dr` (from res_a) was stopped.
Corrected `res_c`: zero-initialised policy output (`--zero-init-actor`, initial controller = LQR), beta 0.1,
log-std -2.5, vmax 1, wmax 1; first episodes ~700 steps. Running.

**Robustness benchmark** (`scripts/study/eval_push.py`; 0.5 m/s drive, one 50 ms push at t = 2 s, 10 trials per
force, any fall counts). Nominal: LQR falls 6/10 at 40 N, 9/10 at 50 N; trk_g 1/10, 5/10; trk_h and rob_a 0/10.
Randomised physics (no push): trk_g 7/10, trk_h 6/10, LQR 4/10, rob_a 0/10. Ablation on trk_g, one parameter at a
time: wheel damping 5/10, mass 1/10, friction / torque / latency / noise 0/10. Lesson: policies trained on the
nominal model are brittle to drivetrain friction; DR fixes it (rob_a) but makes the policy slow (0.35 at 1 m/s).
Note: an early version counted only falls after the push; fixed. Running >10 evaluation processes next to training
ran out of memory (OpenBLAS); use OPENBLAS_NUM_THREADS=1 and small batches.

**S8 spin curriculum** (`spin_a/b/c`, wmax 1.5, from `trk_g`, `trk_h`, `trk_f_s1`). Yaw at +-1 improves
(spin_a -0.92/+0.76, spin_b -0.89/+0.87, spin_c -0.95/+0.82) but speed drops (ratio at 1 m/s 0.87 / 0.70 / 0.75 vs
0.87 / 0.81 / 0.95 before) and random-command falls rise (19 / 13 / 22 of 30, on the harder distribution).
Heading weakness unchanged (0.31-0.34 m/s). Widening spin trades translation for yaw, as the S5 finding predicts.
The benchmark now also tests 1.5 rad/s spin.

**S9 robustness** (`rob_a`, `rob_b`, `rob_a_h`; `dr:` block in config). Mass +-15/25 %, wheel friction +-20 %,
damping 0.05-0.3, torque +-15/25 %, 20 % one-step action latency, sensor noise, pushes 5-20 / 10-35 N for 50 ms every
2-4 s, random start yaw. Queued after S8 (`scripts/study/run_queue_robust.sh`).

## Open problems

- 1 m/s is the stability edge (`trk_g` falls at the end of a sustained 1 m/s leg).
- Spin above 1 rad/s.
- Position drift: velocity tracking does not correct position; a naive position-feedback loop pushes speeds to the
  edge. Needs a learned or model-based path layer.
- Throughput: ~800 env steps/s per run on CPU. A GPU simulator port (MuJoCo Warp / MJX) would allow 10-100x more
  samples but needs the contact-frame friction fix reimplemented.

**S9 results.** `rob_b` (strong DR, 10-35 N pushes, from trk_g): 0/10 falls at every nominal push up to 50 N,
<= 4/10 under randomised physics, falls 8/30 on random commands, but ratio 0.27 at 1 m/s. `rob_a_h` (from trk_h):
ratio 0.29 and some grid falls. Fine-tuning a tracking policy under DR brings back the timid optimum.

**S11 (running).** From scratch under DR with the trk_f recipe (wmax 1, ent 0.005), 20M steps on implicitfast:
`robs_a` (full DR) and `robs_b` (damping narrowed to 0.07-0.15, a bench-measurable tolerance).

**S10 residual results.** `res_c` (zero-init, beta 0.1): keeps LQR strengths (heading err 0.019, 0 falls; act rate
0.0002; ratio 1.00-1.02 at <= 0.5 m/s; rise 0.90 s) and fixes yaw (+-1 rad/s: -1.01 / +0.93 vs LQR +-0.57, best of
all controllers); random-command falls 3/30. But falls 19 % at 0.75 m/s and 94 % at 1 m/s, push 6/10 at 30 N (LQR 0/10),
randomised physics 4-10/10. Interpretation: beta 0.1 is enough to shape yaw (LQR weak there), not to rescue balance
near the edge (LQR wrong there). Next: `res_d` = res_c continued with beta 0.2 + DR + pushes (running).

**res_d** (res_c + beta 0.2 + DR + pushes): worse. Overshoots (0.66 at 0.5 m/s cmd), 31 % falls at 0.5 m/s, 15/30
random-command falls, push 8/10 at 30 N. Heading err 0.13. Larger residual authority fights the LQR.

**robs_b mid-training** (10.5M of 20M, from scratch under DR): no grid falls, ratio 0.48 at 1 m/s (vs 0.27 for
rob_b fine-tuned), still improving. Supports "train from scratch under the harder distribution".

**S11 results.** `robs_a` (full DR, scratch, 20M): ratio ~0.35 at every speed, nominal pushes 0/10 up to 50 N, DR
pushes <= 2/10 at 50 N, random falls 5/30. `robs_b` (narrow damping): ratio ~0.50 at every speed, random falls 3/30,
DR pushes 1-5/10. A constant ratio = steady-state gain < 1, not a speed cap.

**S12 cascade** (`scripts/study/policy_io.py`, run name `<run>@ki=<k>`): outer integral c' = c + ki int(c - v),
saturated at the trained range 1.5 m/s with anti-windup. A cap at 1.0 was wrong (with inner gain 0.5 it capped the
achieved speed near 0.5). `robs_b@ki=1`: 0.25 / 0.51 / 0.73 / 0.79 at 0.25 / 0.5 / 0.75 / 1.0 m/s (falls only at 1.0,
12 %), heading err 0.06 (from 0.27), nominal pushes 0/10 up to 50 N, DR pushes 2-6/10, act rate 0.003. Square course:
0.5 m/s legs within 0.02-0.05 m/s, 1 m/s leg slow (0.47 short), fell on the turning leg after it.
`robs_a@ki=2`: saturates near 0.57 m/s, heading err 0.05, DR pushes <= 5/10 at 50 N.

**S13 (running).** `robs_c_s0/s1`: integrated velocity error as an observation (`obs_vint: true`), from scratch
under narrow-damping DR, 20M; `robs_b2`: robs_b continued 10M.

**S13 results.** `robs_c` (integrated error as observation): ratio 0.62 (s0) / 0.54 (s1), heading err 0.19 / 0.23,
random falls 6 / 4 of 30. With the outer integral (ki 1): robs_c_s0 0.25 / 0.51 / 0.76 at 0.25 / 0.5 / 0.75 m/s with
no falls, 0.93 at 1.0 (38 % falls), heading err 0.045 - most accurate robust controller. `robs_b2` (robs_b +10M):
smoother (act rate 0.0014), not faster.

**1 m/s edge vs rotor inertia** (`scripts/study/edge_armature.py`, model re-derived with `--armature`, env
`wheel_armature`). Continuous LQR with light rotors is sampled-data unstable (fast pole -135 1/s at 100 Hz) - added
`discrete_dt` (ZOH + discrete ARE) to the LQR. Discrete LQR: J_w 0.0005 / 0.002 / 0.005 all fine at 0.75 m/s, all
fall from 1 m/s, reaching 0.84 / 0.93 / 1.0 m/s. Lighter rotors -> slightly earlier edge: gyroscopic hypothesis
refuted. Remaining suspects: omniwheel contact model at high sliding speed (yaw spin-up), ball-floor contact.

**Perlin terrain, no terrain training** (`eval_control.py --perlin-amp`; 4 and 8 cm). At 0.5 m/s all but trk_h track
within 0.02. At 0.75 m/s: res_c 0 % falls on both, LQR 12-25 %, robs_b/c + integral 12-75 %, trk_h slows and falls in
heading test. Residual-on-LQR generalises best to rough ground.

**S14 (running).** Terrain training on random Perlin 0-10 cm: `res_t` (from res_c), `robs_t` (from robs_c),
`robs_t_scratch` (20M from scratch).

**S14 results.** Terrain training did not help: `res_t` ~ res_c (<= 0.5 m/s no falls up to 12 cm, heading-test falls
17 % at 12 cm); `robs_t` + integral degrades with amplitude (62-88 % falls at 12 cm).

**Step edges** (`scripts/study/eval_step.py`; raised floor at y = 1 m, 2 cm bevel; success = cross, stop, hold 2 s;
10 trials, random start heading). An early version counted success at crossing and used identical trials; fixed.
At 0.7 m/s: robs_c@ki1 10/10 at 1.5 and 2 cm, 3/10 at 3 cm; LQR 10/10 at 2 cm, 0 at 3 cm; res_c 7/10 -> 4/10 at
1.5 -> 2 cm; trk_h ~5/10 even flat (heading). At 0.4 m/s nobody climbs 2 cm. LQR speed sweep: climb window, minimum
speed ~0.3 / 0.4 / 0.5 / 0.5-0.6 m/s for 1 / 1.5 / 2 / 2.5 cm, failure above ~0.8 m/s; below sqrt(2gh) because the
motors add energy. Earlier book study (original envelope) never crossed 1 cm.

**S15 (running).** Step-climbing training on book terrain (step 0-4 cm at y = 1 m, 0.5 cm roughness), commands within
+-45 deg of +y at 0.3-1.0 m/s (`cmd_ang_range`, `cmd_vmin`): `climb_a` (from robs_c), `climb_b` (from res_c).
Also running: `robs_t_scratch` (20M, Perlin, from scratch).

**S15 results.** Step-trained `climb_a` (from robs_c) and `climb_b` (from res_c): at 0.7 m/s climb_a clears 2 cm in 8-9/10,
3 cm in 2/10, 4 cm 0/10 (no better than flat-trained robs_c@ki1: 10/10 at 2 cm, 3/10 at 3 cm); climb_b worse
(2/10 at 2 cm). The limit is physical, not learned.

**Traction hypothesis** (`eval_step.py --fric-scale`). Static estimate: horizontal force to roll the ball over a step
F = W tan(alpha), cos(alpha) = (r - h)/r; ~44 N at 3 cm, ~60 N at 4 cm, vs total wheel-ball friction ~52 N (3 wheels x
17-18 N x mu 1). Scaling wheel-ball cross-friction: robs_c@ki1 at 0.7 m/s, 3 cm: 1/4 (x1), 4/4 (x2), 4/4 (x4); 4 cm:
0/4, 1/4, 1/4. LQR 3 cm needs x4 (2/4 at 0.5 m/s). Supports traction as the limiting factor. Design implication for
the small robot: larger ball relative to step height, more wheel preload, higher-friction wheels. (4 trials per cell:
suggestive, not conclusive.)

## S16 Morphology design space and the interleaved hybrid controller (started 2026-11-04)

**Question.** Given what works where (LQR for accuracy/low speed, robust PPO for pushes, residual-on-LQR for rough
ground), can the pieces be combined in one trainable pipeline, and how do the *physical* design choices (scale, ball
mass, body mass, COM height, motor strength, rotor inertia) change both the achievable dynamics and how well a
controller can be learned?

**Tooling** (`scripts/study/design_space.py`). A design is `(s, kball, kbal, zb, J, tau, damp)`: uniform geometric
scale `s` (masses x s^3), ball-mass multiplier, ballast (body mass) multiplier, ballast height (moves COM and pitch
inertia), rotor inertia, per-wheel torque limit at s=1 (scaled by s^4, geometric similarity of m g l), damping (x s^2.5).
`make_variant_xml` rewrites `ballbot.xml` (meshes via `../meshes`); `analytic` extracts the derived planar model of
the variant and returns pole p, NMP zero z (closed form), Skogestad bound ||S||inf >= (z+p)/|z-p|, lean per
acceleration, torque authority tau_max/(m_a g l), LQR peak torque and linear recoverable tilt, acceleration limits.
`build` also writes the env yaml (Froude-scaled command range, push force scaled by weight) and a per-design LQR model
json (+ morphology context vector).

**Analytic findings** (`outputs/study/results/design_sweep.csv`, `paper_v2/figures/fig_design_analytic.pdf`).
- Nominal: p = 4.79 1/s, z = 5.78 1/s, z/p = 1.21, so ||S||inf >= 10.7 (poor robustness margin is structural).
- Lean law kappa is nearly design independent: 7.3-14.8 deg per m/s^2 over the whole sweep; heavier ball -> more lean
  (11.8 deg at 8x ball mass), higher COM -> less lean (7.3 deg at l = 38 cm).
- Scale: p ~ s^(-1/2) (Froude), kappa constant. A half-size robot must be controlled sqrt(2) faster.
- Sensitivity bound is worst for a HIGH COM (26.5 at l = 38 cm) and best for a LOW COM (9.0 at l = 8.7 cm); heavy
  body (4x ballast) also worsens it (14.3). z/p stays within 1.08-1.29 over the sweep, i.e. the robustness limit is
  a property of the topology, only mildly tunable by weight distribution.
- Torque authority tau_max/(m_a g l): LQR-recoverable linear tilt falls below 20 deg at authority ~ 6.9
  (tau = 6 N m per wheel at s = 1).

**Zero-shot LQR on each design** (model re-derived per variant, no training; `eval_design.py`). Works (no falls at 0.4 sqrt(s) m/s) on 9/13 designs (4/13 at 0.8 sqrt(s))
at 0.5 m/s; fails immediately on the half-scale robot (s = 0.5) and the 0.25x ball (ball mass 0.04 kg); heavy body
(4x) is marginal; light rotors need the discrete-time LQR. Validation of the variant sim vs model: ball rolling
acceleration agrees within ~10 % on all variants, body pitch acceleration is a factor 0.15-0.45 of the model, with the
ratio worst for light balls. Interpretation: the planar model is least faithful when the ball is much lighter than the
body (mass ratio > ~60), so model-based design degrades exactly where the sim contact is also stiffest.

**Learning runs.** `run_queue_design.sh`: from-scratch PPO (robust recipe, DR + weight-scaled pushes, Froude-scaled
commands, 5M steps) on 13 designs; evaluation by `eval_design.py` (speed ratio at {0.4,0.8,1.1} sqrt(s) m/s, push
falls at {0.5,1,1.5} x body weight).

**Interleaved hybrid controller** (env `hybrid_gate`, `morph_pool`). Policy outputs 3 wheel actions plus a gate logit;
applied action = (1-g) a_LQR + g a_pi with g = (1 + clip(x))/2 in [0, 1]. Observations add the LQR action and a
4-vector morphology context (log2 s, log mass ratio, l/r, log torque authority). Each reset draws a design from a pool
(model reload ~ms), so one policy sees a family of ballbots. g = 0 is the pure model-based controller, g = 1 pure
PPO, the residual controller is the special case a_pi = a_LQR + beta u. The output layer starts at zero with the
gate bias -0.8 (g = 0.1), so training starts from (almost) the LQR and the cascade integral can still wrap it.

**S16 literature sweep** (`docs/research/lit_ballbot_{classical,learning,hardware}.md`). Ballbot RL is thin: the base repo is
salehiac/OpenBallBot-RL (arXiv 2505.18417, sim only); the only sim-to-real ballbot RL paper found is asRoBallet
(arXiv 2604.24916): PPO beats LQR in sim (velocity MAE 4.0 vs 7.7 cm/s, station keeping 100 % vs 0 %), randomises floor
friction 0.6-1.2, motor Coulomb friction, gyro noise; torque-mode actuators train, velocity mode does not. No paper
found that gates between LQR and an RL policy (cautious: one search pass). Real ballbots run 160-200 Hz loops; body:ball
mass ratios 2-12 (ours 23); contact zenith 60-70 deg in the micro-ballbot paper/patent (ours 51).

**S16 tests of the literature hypotheses.**
- Control rate: LQR on nominal and half-scale robot at 100 / 250 / 500 Hz (`test_rate.py`): identical falls at 1.0 m/s
  and for s = 0.5. Rate is NOT the limit (consistent with delay bound exp(pT) ~ 1.1 at 10 ms). The LQR now takes its
  integrator step from the env's control period (`LQRPolicy.dt`).
- Friction: wheel-ball cross friction x2, x4 and roll friction x10 do not move the LQR 1 m/s limit or the s = 0.5
  failure. PPO tracks 1 m/s (0.91-0.95), so the limit belongs to the linear controller / planar model.
- Contact angle (`dz` design parameter rotates each wheel assembly about the ball centre; zenith exactly 51 + dz):
  LQR at zeta = 66 deg survives pushes of 1 x body weight 6/6, nominal 1/8, zeta = 41 deg 0/6 (6 trials each). The
  planar model predicts no effect on traction-limited torque (N ~ 1/cos zeta cancels the lever arm), so the model misses
  the mechanism. Reflected inertia drops 0.103 -> 0.043 kg m^2 and the sensitivity floor rises 10.7 -> 12.9.
- Realistic actuator (`env.motor`: first-order current lag, back-EMF torque-speed curve, Coulomb friction; off by
  default): with omega_nl = 150 rad/s, 3 N m stall, 50 mN m friction, 3 ms lag the nominal LQR loses 0.8 m/s tracking
  (falls 4/4) but pushes of 1 x weight are survived 6/6; i.e. controllers must be (re)trained with the actuator model.

**S16 design rules** are in `docs/research/design_rules.md` (`scripts/study/design_constraints.py`): structural floor
Ms >= (z+p)/(z-p) e^{pT}, traction limit mu N r_w (about 0.31 N m per wheel, 1.7 N m planar, 21 deg static recovery,
COM <= 0.18 m), torque authority, lean ceiling about 2 m/s^2, scaling laws, delay budget ln2/p = 145 ms.

**S16 training progress** (halfway, 2.5M of 5M steps): ep length / return: nom 407 / 669, s0.5 564 / 856 (learns although
LQR cannot), s1.5 82 / 111 (struggling), zhigh 557 / 898, kb4 407 / 679, tau3 683 / 1057.

## S17 Heading, look-ahead and perception (plan, 2026-11-04)

**Evidence in the repo.** The original side-mounted cameras mostly saw the ball itself and the frozen depth features were
ignored (blank depth changed nothing; `ballbot_v2_roadmap.md`). The ToF path exists (`ballbot_gym/sensors/tof.py`, 8x8
zones, 2 m range, noise and dropout) with side sensors aimed forward and down at (0.11, 0, 0.03) pitch -35 deg, which
only sees the floor about 0.2 m ahead (geometry estimate: (0.12 + 0.03) / tan 35 deg), i.e. 0.3 s of warning at 0.7 m/s.
Those perception runs used the old, weaker controller envelope; they were never repeated on the tracking-grade
controllers. No teacher-student distillation has been built.

**Hypotheses, in order of cost.**
1. *Heading dependence of traction.* The three wheels sit at 0/120/240 deg, so the traction-limited planar torque
   depends on the body heading relative to travel. Model: factor 5.55 to 6.25 (about +-6 percent, period 60 deg), so a
   weak effect expected. Sim test: `eval_step.py --init-yaws` (3 cm step, 0.7 m/s, robs_c@ki1, 16 trials over 8 headings).
2. *Look where you go.* Yaw the body toward the velocity direction (controller-level wrapper, no retraining) so that a
   forward-facing sensor sees the path. Measure tracking, pushes, steps with heading free vs aligned.
3. *Sensor placement.* Head-mounted forward sensor, pitched down 20 to 30 deg at ~0.4 m height sees about 0.7 to 1.1 m ahead
   (estimate), i.e. 1 to 1.5 s at 0.7 m/s. A look-ahead only helps if it changes behaviour: approach speed into the
   0.5 to 0.8 m/s climb window, approach direction, braking before a drop.
4. *Preview policy.* Privileged height-scan teacher as residual on robs_c for step approach; success vs step height with
   and without preview. Traction (not information) limits steps beyond ~3 cm, so expect gains mainly at 2 to 3 cm.
5. *Distil to a ToF/camera student* only if 4 shows a gain.

**S17 finding: observation quirk (from the in-browser port).** `ballbot_env.py` builds `motor_state` from
`qvel[joint.id]` instead of the DoF address, so the policy actually receives `[base vy, vz, roll rate] / 30`, not the three
wheel speeds. All trained policies learned with this input and the browser port only closes the loop when it reproduces it.
Consequences: (a) the policies do not use wheel encoders; (b) a hardware port must either feed the same signals or retrain
with true wheel speeds. Worth fixing and re-testing in the next training round (a possible accuracy gain, unmeasured).
**Live in-browser policy:** `site/live/FEASIBILITY.md`; MuJoCo WASM 3.13.0 reproduces the Python closed loop to <= 35 mm
over 8 s on flat ground (0.2 mm for a diagonal drive), contact-frame fix reproduced through the exported pipeline stages.

**S17 results (heading).** 3 cm step, robs_c@ki1, 0.7 m/s, start heading swept (4 trials each): success only at 0, 10, 20 and 140 deg
(4/4); 0/4 at 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 300, 320, 340 deg. Not 120-periodic (0 deg works, 120 and 240 do not),
and far larger than the planar traction model's +-6 percent. Mechanism not identified (candidates: which wheel leads into the
step, 3D force split between wheels). Test of a yaw hold before the climb: `eval_step.py --hold-yaw` (running).

**S17 results (design runs, 5M steps, single design each).** PPO from scratch with DR + weight-scaled pushes, evaluated at 0.4/0.8/1.1 x sqrt(s) m/s
without DR (`dz_ppoki.csv`, with outer integral): nominal ratio 0.75/0.65/0.36 with 0 falls at all speeds, push falls 0/0/12 percent at 0.5/1/1.5 x
weight (LQR on the same design: falls at 1.1 x speed everywhere, 88 percent push falls at 1 x weight). zhigh and kb4 similar. So PPO is
more robust and LQR more accurate at low speed; the learned policies track less accurately (ratio < 1) at the evaluated horizon.

**S17 experiment flaw found and fixed.** DR wheel damping was an absolute range (0.07 to 0.15) for every design. For scaled designs that
is far from the design's nominal damping (s = 0.5: nominal 0.018, trained at ~6x). The s = 0.5 policy survives with DR on (4/6 episodes
reach 600 steps) and falls within 25 to 77 steps with DR off. The single-design s0.5 and s1.5 results are therefore NOT comparable
to the others. `_randomize` now scales the range by the design's nominal damping (identical for the nominal robot); the morphology-pool
runs were restarted with the fix. LQR at s = 0.5 is only stable in a narrow band of simulator damping relative to the model damping
(0.018 to 0.05 with the damping-0.018 model; fails for matched damping models at 0.05, 0.1 and at 0), i.e. fragile, not a loop-rate or
friction issue. Single-design s0.5/s1.5 runs need re-running after the pool.

**S17 videos.** `render_cinematic.py` / `render_compare.py` (HUD cards, height-tinted floor with 1 cm contours, raking light, error-coloured
trail, minimap, time bar with event markers). Fixed: time-bar markers no longer reveal future events (pushes, waypoints, step, fall appear
when they happen). The cine push video (`push_robs_b`, my assumed push schedule) did not reproduce the fall of the earlier
`best_push_recover`; it is a different schedule and is labelled as such.

## S18 Pooled-morphology tournament (10M steps each, 12-design pool, DR scaled per design)

Eval: eval_design.py on 13 designs, 4 trials per speed, no DR. Files: outputs/study/results/pool_{hybrid,residual_ctx,ppo_ctx}.csv, dz_lqr.csv.

| controller | designs with 0 falls at 0.4 / 0.8 / 1.1 x sqrt(s) | mean speed ratio 0.4 / 0.8 | mean push-fall rate 0.5 / 1 / 1.5 x m g |
|---|---|---|---|
| LQR (zero-shot) | 11 / 5 / 0 of 15 rows | 0.98 / 1.04 | 0.30 / 0.65 / 0.94 |
| hybrid gate | 10 / 4 / 0 of 13 | 0.98 / 1.05 | 0.35 / 0.66 / 0.91 |
| residual on LQR | 9 / 0 / 0 of 13 | 1.13 / 1.25 | 0.34 / 0.48 / 0.82 |
| pure PPO + context | 8 / 9 / 8 of 13 | 0.35 / 0.30 | 0.25 / 0.37 / 0.62 |

Reading: the hybrid gate reproduces LQR (it did not beat it); residual overshoots speed; pure PPO is the only one that does not fall at speed
but tracks at about a third of the commanded speed. The half-scale design fails for all three. Note the LQR row count differs (15 vs 13).
Open: pure PPO needs the outer integral at eval, and its slowness is a tracking-reward issue, not a stability one.

S18b: pooled pure PPO re-evaluated with the outer integral (@ki=1.0), 12 designs (no s0.5): speed ratio rises from ~0.3 to 0.64-0.75 at 0.4 sqrt(s), 0.45-0.61 at 0.8, ~0.4 at 1.1;
zero falls at 1.1 on 8 of 12 designs (fails: tau3, tau6, s1.5, J0005 1 fall). Still under-tracks at speed; the weak-motor designs (tau3, tau6) and s1.5 fail.
File: outputs/study/results/pool_ppo_ctx_ki.csv.

S18c: hybrid gate check (scripts/study/gate_stats.py pool_hyb_s0 ...): mean gate g (0 = LQR, 1 = PPO branch) is 0.00 on nom, s1.5, zhigh, tau3, J0005 at 0, 0.4, 0.8, 1.1 x speed
(max 0.32; tau3 up to 0.27). The learned hybrid collapsed to pure LQR, which is why its table row equals the LQR row. Cause candidates: gate bias -0.8 start plus
the LQR's reward dominating at low speed; to try: gate entropy/KL term, bias 0, curriculum that forces PPO branch on LQR-failing designs, or train the PPO branch first and then the gate.
Also: cinematic clips (12) re-encoded into site/videos via scripts/site/add_cinematic.py; s0.5 and s1.5 single-design PPO relaunched after the DR fix.

## S19 Realistic sensing and actuation: SMC and MPC baselines (simulation only)

New code: ballbot_gym/controllers/realistic.py (SensorModel: tilt/gyro noise, gyro bias, encoder noise+quantisation, velocity noise, sensing delay;
SMCPolicy: LQR gain + switching term on the fast-mode sliding variable; MPCPolicy: condensed QP per plane via OSQP, hard torque limit, soft 12 deg tilt limit,
Smith-style delay compensation); LQRPolicy got `sensor` and `law()` hooks; scripts/study/eval_real.py. Levels: mild = tilt 0.003 rad, gyro 0.01 rad/s + 0.005 bias, enc 0.05 rad/s, delay 1 step (10 ms);
hard = 3x noise, delay 2 steps. Motor 1 = lag 3 ms, torque-speed curve, Coulomb friction. Nominal design, 4 headings x 4 trials per speed, 6 push trials. Files: outputs/study/results/real_*.csv.

Findings:
1. With no noise and no delay LQR, SMC and MPC give identical results (0.99 / 1.03 speed ratio at 0.4 / 0.8 sqrt(s), 4/4 falls at 1.1). The tilt/torque constraints in the MPC are never active
   and the SMC switching term is small, so on this plant the extra machinery buys nothing at nominal.
2. A single control step (10 ms) of sensing delay is what breaks the classical controllers at speed: 0.8 sqrt(s) falls 4/4 for LQR and MPC with delay and no noise,
   while the same noise with no delay is fine. Finer probe (LQR): delay 1 step falls 3/4 at 0.8, 0/4 at 0.7. So the speed limit drops from above 0.8 to about 0.75.
3. The MPC's delay compensation (rolling the measured state forward over the commands in flight) did NOT recover the lost speed (falls 4/4 at 0.8, 0/4 at 0.7).
   Not yet understood; candidates are the velocity reference governor/outer integral still using delayed velocity, and model error in the roll-forward.
4. The motor model alone cuts the 0.8 speed result to 4/4 falls for all three classical controllers but improves push survival at 1x weight (LQR 0/6 vs 3/6 in the ideal case); same pattern as the earlier motor result.
5. Single-design PPO + outer integral (trained with obs_noise 0.01, latency 0.2) is not affected by the realistic levels: speed ratio 0.75-0.81 / 0.57-0.68 at 0.4 / 0.8, 0 falls at every speed, in all six noise/motor conditions;
   push falls at 1x weight 0 to 0.17. It tracks less accurately than the LQR but is far more tolerant of delay.
Training with the realistic effects (motor, obs_noise 0.02, latency 0.5): dz_nom_real_s0 (5M steps) started.

## S20 Physics-engine fidelity and speed (side studies, simulation only)
Reports: experiments/mjx_port/REPORT.md (MuJoCo-Warp), experiments/isaac_port/REPORT.md (Isaac Sim 6.2 / PhysX); data in outputs/study/{mjx_port,isaac_port}; site page site/engines.html (16 clips).
- MuJoCo-Warp 3.17 (GPU, float32, our contact-frame fix as a custom kernel): LQR speed ratios within 0.01 of CPU; fall parity 94.6 % (speed) and 96.1 % (push) over 17 designs; about 260x one CPU process at 4096 worlds. Gap: trajectories 25-100x the CPU version spread; not bit-reproducible; untested with trained policies, sensors, terrain, motor model.
- Isaac Sim / PhysX: installs by pip (7.5 GB download, 16 GB venv at C:/Users/Owner/isaac_venv; remove with venv delete + pip cache purge). No anisotropic friction (ball-wheel pair lost in MJCF conversion), so a 16-roller ring per wheel was used; fragile. Nominal tracking and the contact-angle ranking reproduced; failure boundaries not (1.1x speed: MuJoCo falls, PhysX holds); 8 of 17 designs not finished; about 10x one MuJoCo process at 4096 envs.
- Decision: MuJoCo stays the study engine; MuJoCo-Warp is the candidate accelerator (needs CPU validation of any GPU-trained policy); PhysX only as a cross-check.

## S21: PPO trained entirely on MuJoCo-Warp (GPU), checked on the CPU engine

Code: `experiments/warp_ppo/train.py` (GPU-resident env + PPO), `eval_policy.py`, results in `outputs/study/warp_ppo/`.
Task as the CPU PPO+integral study (dz_nom, command tracking, 22-D obs, reward "err" with lean term). DR kept: obs noise, latency p=0.2, torque +-15 %, pushes.
NOT randomised on Warp (needs per-world model fields): mass, ball friction, wheel damping. Single seed.

- Throughput: about 140k env-steps/s end to end (physics + obs + reward + PPO update), 4096 worlds. 60M steps in 7 minutes. The CPU runs do 5M in hours.
- First run failed on CPU (all 28 runs fell). Cause was my port, not the engine: the CPU env's `motor_state` reads `qvel[joint.id]` (the known quirk) while my port used the wheel dofs. After copying the quirk exactly the policy transfers.
- Second run (`wp_nom_q_s0`, 60M steps), same protocol as eval_design:

| engine | ratio 0.4 / 0.8 / 1.1 | falls (of 4) at 0.4 / 0.8 / 1.1 | push falls 0.5x / 1x / 1.5x |
|---|---|---|---|
| CPU (MuJoCo 3.13, float64) | 0.70 / 0.76 / 0.65 | 0 / 0 / 0 | 0 / 0 / 0 (16 pushes each) |
| Warp (float32) | 0.74 / 0.71 / 0.61 | 0 / 0 / 0 | 0.25 / 0.19 / 0.25 |

- Reading: a policy trained only on the GPU engine balances on the CPU engine with no falls across the whole protocol. Speed ratios agree within 0.05. Warp is the harsher engine for pushes in this test (4/16 falls vs 0/16), the opposite of the LQR comparison; unexplained (single seed, 16 pushes, different push-direction sets).
- Not compared like-for-like with the CPU 5M-step runs (12x more samples here). Ratios are in the range of the CPU PPO+integral policy (0.64-0.75 at 0.4), with fewer falls.
- Caveats: one seed; no mass/friction/damping DR; flat floor only; no sensors beyond the 22-D proprioceptive vector.

### S21b: scale-up on Warp (full per-world DR, 3 seeds, 7 designs), each policy checked on both engines

Per-world randomisation added on the GPU (ball and base mass +-15 %, transverse friction +-20 %, wheel damping, torque, pushes, latency, obs noise); per-world physics verified to diverge under identical commands. 60M steps per run, about 7.5 min each. Evaluation: eval_design protocol, one policy per design, CPU = MuJoCo 3.13 float64 (4 trials per speed, 16 pushes per level), Warp = same protocol on GPU. Files: `outputs/study/warp_ppo/wpdr_*/eval_{cpu,warp}.json`.

| run | engine | speed ratio 0.4/0.8/1.1 | falls (of 4) | push falls 0.5/1/1.5x |
|---|---|---|---|---|
| nom s0 | CPU | 0.80/0.73/0.61 | 0/0/0 | 0/0/0 |
| nom s0 | Warp | 0.77/0.75/0.64 | 0/0/0 | 0/.12/.19 |
| nom s1 | CPU | 0.74/0.66/0.64 | 0/0/0 | 0/0/.06 |
| nom s1 | Warp | 0.69/0.71/0.66 | 1/0/1 | 0/.12/.25 |
| nom s2 | CPU | 0.75/0.69/0.65 | 0/0/0 | 0/0/0 |
| nom s2 | Warp | 0.78/0.73/0.66 | 1/0/0 | .19/.12/0 |
| tau3 | CPU | 0.25/0.21/0.20 | 0/0/0 | 0/.31/1.0 |
| tau3 | Warp | 0.31/0.24/0.22 | 0/0/0 | 0/.25/1.0 |
| tau6 | CPU | 0.41/0.32/0.32 | 0/0/0 | 0/0/.50 |
| tau6 | Warp | 0.50/0.39/0.33 | 0/0/0 | .06/.06/.38 |
| s0.5 | CPU | 0.65/0.66/0.63 | 0/0/1 | 0/0/0 |
| s0.5 | Warp | 0.73/0.70/0.69 | 1/0/0 | .06/.06/.19 |
| s1.5 | CPU | falls | 3/3/3 | 1.0/.94/1.0 |
| s1.5 | Warp | falls | 3/4/4 | .62/.94/1.0 |
| zhigh | CPU | 0.82/0.87/0.77 | 0/0/0 | 0/0/0 |
| zhigh | Warp | 0.85/0.84/0.79 | 0/0/0 | 0/0/0 |
| zlow | CPU | 0.77/0.71/0.63 | 0/0/0 | 0/0/0 |
| zlow | Warp | 0.72/0.72/0.63 | 1/0/0 | 0/0/.06 |

Reading: (1) a Warp-trained policy balances on the CPU engine for every design where it balances on Warp; the 1.5x design fails on both (it also failed with the earlier CPU-trained PPO, so it is a training difficulty for that design, not an engine gap). (2) Speed ratios agree within about 0.1 across engines; ranking of designs agrees (zhigh best, tau3 slowest). (3) Warp is the harsher engine for the same policy in falls and pushes by a few runs, consistently over designs; cause unknown (possible float32/solver effects, different push-direction samples). (4) The low-centre-of-mass design, which was fragile for the LQR, is held by the learned policy on both engines. (5) Caveats: one seed except nominal; 4 trials per speed cell so one fall is 25 %; no terrain, sensors or motor model on the GPU yet.

### S22: PhysX wheel inertia check and MuJoCo roller-ring wheel
See `experiments/isaac_port/RESEARCH_omniwheel.md` (follow-up checks). PhysX applies armature 0.005 and damping 0.1 correctly (accel 19.12 vs 19.14 rad/s^2), ruling out wheel inertia as the cause of "PhysX holds at 1.1x". A 12/24-roller MuJoCo ring wheel is fragile and discretisation dependent (0.8x: 2 of 48 configs hold; 1.1x: all fall or stall) and does not reproduce PhysX either, so the idealised anisotropic model remains the study reference and PhysX a cross-check.

### S23: GPU motor model
Realistic actuator (first-order lag, torque-speed curve, Coulomb friction) added as a Warp kernel inside the CUDA graph, evaluated every physics substep (`experiments/warp_ppo/train.py`, `--cfg nom_real`). Sanity check: with a 20 N m command wheel speed after 0.2 s is -195 rad/s ideal vs -24 rad/s with the motor model. Training and CPU re-evaluation of `wpreal_nom_s0/1` pending.

### S23 result: GPU motor model transfer (wpreal_nom_s0/s1)
Trained on Warp with lag, torque-speed curve, friction, stronger noise, 50% latency (60M steps each). Speed ratios at 0.4/0.8/1.1: CPU 0.41/0.34/0.33 (s0), 0.40/0.35/0.32 (s1); Warp 0.42/0.36/0.34 and 0.43/0.36/0.34. Zero speed falls on both engines. Push falls 0.5x/1x/1.5x: CPU 0/0/0.31 (s0), 0/0/0.5 (s1); Warp 0.06/0/0.5 and 0/0/0.56. Engines agree within about 0.03 on ratio. Comparison against CPU-trained dz_nom_real_s0 still to do.

### S24: terrain on MuJoCo-Warp (smoke test, side task)
`experiments/warp_ppo/check_terrain.py`: LQR at 0.4 m/s over a curb at y=0.6, CPU (1 run) vs Warp (256 worlds, start x spread +-0.4 m). The Warp hfield is one array shared by all worlds (hfield_data has no world axis), so per-world terrain would need either variety by start pose or several hfield geoms.
- Flat and 1 cm curb: no falls on either engine; pre-crossing ball-height RMS difference CPU vs Warp 2.4 mm, y 13 mm, tilt 0.9 deg.
- 3 cm curb: CPU falls at the crossing, Warp falls in 92.6% of worlds (the rest survive thanks to start offsets). Same qualitative outcome, so the hfield contact is active and consistent.
- Cost: with the host LQR in the loop 56k world-steps/s flat vs 16.6k with the curb (3.4x slower); 0.8% of worlds non-finite after toppling (needs the existing NaN auto-reset).
- Not yet tested: Perlin roughness (many cells), PPO training on terrain, ToF/height-scan sensor on GPU.

### S25: blind PPO trained on GPU terrain (wpterr_nom_s0)
`train.py --terrain`: shared stepped ridge (single steps 1, 2, 3 cm at |y| = 0.5, 1.1, 1.7 m, mirrored), random start segment/x/yaw, nominal design with the usual DR, 40M steps, 4096 worlds, 327 s, 42k..122k env-steps/s (flat was ~140k). Proprioceptive 22-D obs only (blind). Training fall rate 0.5% per step-batch at the end.
Eval (`eval_terrain.py`, Warp, 128 worlds, drive outward from y=0 at 0.3 m/s for 12 s, no DR): got past the 1 cm edge 87% (+y) / 77% (-y); past the 2 cm edge 20% / 17%; past the 3 cm edge 5% / 6%; only 2% / 17% still upright at the end. The same policy on a flat floor: 84% / 80% upright over 12 s (it was trained on 8 s episodes, so a 16-20% late fall rate on flat is itself a caveat). Reading: a blind policy handles about 1 cm steps and fails at 2 cm and above, consistent with the CPU perception study. No CPU cross-check of this policy yet (needs terrain in the CPU eval).

### S26: blind vs privileged height-scan vs depth (VL53L9-style), GPU, same ridge
Code: `train.py --terrain --sensor none|scan|depth [--mount head|cones]`, `eval_terrain.py`. Depth sensor via BVH `mjw.rays` (0.69 ms for 64 rays x 4096 worlds; 38x faster than the plain path), 18x14 zones, 55x42 deg, range 0.05-4 m, noise 0.5 % of range, 2 % dropout, head mount 0.15 m above base pitched -25 deg, yawed along the command (assumption). Teacher: 7x5 height grid ahead of the ball in the command frame. Learned MLP encoder of the sensor vector, joint PPO. 60M steps, 4096 worlds, seed 0, ~9 min each. Evaluation noise-free (no --dr), 128 worlds, 12 s outward drive at 0.3 m/s from y=0.
| policy | past 1 cm edge (+y/-y) | past 2 cm | past 3 cm | upright at end |
|---|---|---|---|---|
| blind | 0.59 / 0.84 | 0.16 / 0.38 | 0.02 / 0.08 | 0.23 / 0.19 |
| teacher scan | 0.88 / 0.88 | 0.16 / 0.23 | 0.09 / 0.06 | 0.00 / 0.02 |
| depth | 0.61 / 0.94 | 0.02 / 0.25 | 0.00 / 0.09 | 0.36 / 0.02 |
Result: no perception gain. Even the privileged teacher does not beat the blind policy at 2 and 3 cm, so the limit is not information at the sensor. Likely causes (not tested): step encounters are rare under random commands and random starts so the policy gets little signal; and traction. Salehi's design (Sec. III, Fig. 4): two 128x128 depth cameras aimed at the ball-ground contact, TinyAutoencoder pretrained then frozen, 20-D each; contact-zone sensing, not look-ahead. Cone axis as modelled hits the floor 0.22 m along the ray, not the ball. `--mount cones` implemented (rigid, outward, 25 deg down); the Salehi-style aim and frozen-autoencoder arms are not built yet. Single seed, one terrain.

### S27: edge-focused retrain; approach speed dominates (GPU, 60M steps each, seed 0)
`--edge-focus 0.7` (70% of episodes start 0.2-0.45 m before an edge, command drives across the ridge). Fraction past each edge (+y / -y), noise-free eval, 128 worlds, 12 s.
At 0.3 m/s command: blind 0.89/0.84, 0.25/0.38, 0.06/0.06 (1/2/3 cm edges); teacher 0.95/0.83, 0.20/0.25, 0.05/0.08; depth 0.97/0.98, 0.14/0.31, 0.00/0.08. No perception gain, and no gain from edge focus.
At 0.6 m/s command: blind (edge) 0.97/0.97, 0.97/0.97, 0.50/0.59; blind (random-start run) 0.97/0.91, 0.94/0.59, 0.31/0.16; teacher 0.95/0.97, 0.77/0.80, 0.28/0.27; depth 0.99-1.00/0.98, 0.97/0.98, 0.48/0.64.
Reading: the blind policy crosses 2 cm steps almost always at 0.6 m/s and fails at 0.3 m/s, so approach speed is the dominant factor (consistent with the 0.5-0.8 m/s climb window of S17); edge-focused training helps the blind policy at 3 cm (0.31/0.16 -> 0.50/0.59 at 0.6 m/s). The teacher is no better and in this run worse than blind; depth about equal to blind. The tracking reward penalises speed error, so a perceiving policy has no incentive to speed up before a step; a perception gain would need a reward or command interface that allows it (e.g. goal-directed reward, speed free). Single seed, noise-free eval.

### S28: generalist baseline on the tiled map (wpgen_none_s0)
Blind PPO, 22-D obs, DR on during training (no surface classes yet), 100M steps, 4096 worlds, 1480 s. Evaluation noise-free, 64 worlds per terrain type, random direction, 8 s, `eval_tiles.py`. Fall rate / crossed rate (crossed = left the 2.9 m tile square upright):
| type | 0.3 m/s | 0.6 m/s | 1.0 m/s |
|---|---|---|---|
| flat | 0.02/0.80 | 0.06/0.94 | 0.05/0.95 |
| step up (1-6 cm) | 0.56/0.05 | 0.52/0.48 | 0.48/0.52 |
| step down | 0.23/0.64 | 0.12/0.88 | 0.19/0.81 |
| small drop (3-8 cm) | 0.20/0.59 | 0.34/0.66 | 0.33/0.67 |
| large drop (10-20 cm) | 0.55/0.31 | 0.59/0.39 | 0.55/0.45 |
| ramp up (3-12 deg) | 0.09/0.36 | 0.05/0.75 | 0.05/0.81 |
| ramp down | 0.20/0.78 | 0.36/0.64 | 0.28/0.72 |
| stairs (1-4 cm risers) | 0.42/0.05 | 0.48/0.42 | 0.42/0.58 |
| bump (1-5 cm slab) | 0.64/0.02 | 0.77/0.22 | 0.67/0.33 |
| rough (0.5-3 cm) | 0.20/0.38 | 0.14/0.80 | 0.19/0.81 |
Reading: most of the failures are steps, bumps, stairs and large drops; faster is mostly better; flat is fine. Severity is mixed within a type, so per-severity breakdown is still to do. Caveat: the training fall rate on the tiled map includes truncated exits as non-falls; single seed.

### S29: literature review and ablation 1 launched
`docs/research/lit_rl_sota.md` (subagent, partly snippet-level evidence, tags [V]/[S]/[M] inside). Top points: asRoBallet (arXiv 2604.24916, to be verified) found motor Coulomb-friction dead zone the key real effect; cheapest next experiments are a privileged asymmetric critic, an observation/action history, RMA-style adaptation, CAPS-style smoothing, and a traversal reward with a terrain curriculum.
Code added: floor material classes per world (`--surfaces`: concrete, hardwood, carpet, grass, gravel, wet tile, rubber: friction and contact time constant, plausible not measured), `--hist K`, `--asym` (critic sees surface, mass, torque scale, damping, 34-D height scan). Ablation 1 running: base / asym / hist8 / both on tiles + surfaces + DR, 100M steps each, seed 0 (`run_abl1.sh`, ~35 min each).

### S30: GPU utilisation and Python overhead (measured, 2026-11-05)
One PPO process used 40-55 % of the GPU, 122 W of 360 W, 3.3-4.5 GB of 16 GB. Cause: ~29 host syncs per control step (boolean-mask indexing, int(mask.sum()), if x.any()) serialising launches, not compute. Changes in `train_dev.py`: command sampling, goal sampling, pushes, pulls, latency and the whole reset rewritten mask based (draw for all worlds, merge with where, in-place copy_ into tensors shared with Warp); rollout-loop logging and truncation bootstrap without syncs; TF32 and fused Adam; NaN guard (non-finite world counts as a fall, values sanitised). Measured with 4096 worlds, tiles + surfaces + designs + goals, GPU shared with another job: syncs per step 29 -> 1, env.step 40.7 -> 30.9 ms, reset(1 %) 14.6 -> 5.9 ms; end-to-end 12M-step PPO 52k -> 67k env-steps/s with the same learning curve (reward 0.353 vs 0.355, fall rate 5.6 vs 4.9 %). Two concurrent jobs: 105k alone, 78k + 74k = 152k together (+45 %), GPU 85 %. Three jobs (400M-step runs, 8192 worlds): ~57-67k each, ~185k aggregate, GPU 93 %, 11.6 GB. A first hist-16 run crashed with NaN observations before the guard was added; restarted. Env-step scaling with world count (step only): 8192 -> 184k env-steps/s. `run_queue.py` runs a queue of jobs with a concurrency limit. Shared GPU memory (system RAM over PCIe) and the video encode/decode engines cannot speed physics or PPO; NVENC is used to encode rendered videos.

### S31: goals, pulls, per-world designs, distillation, renderer (all in train_dev.py; old checkpoints need train.py)
- Goal mode: velocity command = saturating proportional rule on goal error (cap cmd_vmax, speed free), waypoint chains (0-3 follow-ups), goal vector + flag added to the observation (22 -> 25 inputs). Sustained pulls 3-12 N for 0.5-2 s.
- Designs per world: 13 compiled designs share a topology; 25 fields that differ are stacked and assigned per world (checked on five designs: ball radius 4.5/9/13.5 cm, torque limit 1.9/30/152 N m, ball rests correctly on the floor). Not supported with the motor model. `--design-cond` gives the policy a 6-D design vector; the critic always gets it.
- `distill.py`: DAgger from a privileged scan teacher to a depth + history student (wider network); runs end to end on a smoke test, not yet on a trained teacher.
- `render_policy.py`: Warp rollout of one world on a chosen tile, CPU MuJoCo renderer on the checkerboard floor, commanded vs actual path, NVENC encode. Smoke-tested.
- `eval_tiles2.py`: per-terrain and per-design evaluation for the velocity and goal tasks. Smoke-tested only (untrained checkpoints fall everywhere); metrics not yet validated on a competent policy.
- Ablation 1 (old train.py, tiles + surfaces + DR + pushes, 0.6 m/s, 64 worlds x 10 types, one seed): mean fall rate base 0.450, privileged critic 0.436, history-8 0.412 (standard error ~0.02 per cell, so no winner); flat falls 0.17 / 0.23 / 0.12. 'both' arm still training at the time of writing.
- Round 1 long runs (400M steps, 8192 worlds, tiles + surfaces + 13 designs + goals + pulls, width 512 x 3): gtea_s0 (scan teacher, design vector, asym), ggen_hist8dc_s0 (history 8, design vector, asym), ggen_hist16_s0 (history 16, asym, must infer design).

### S32: ablation 1 final and the memory lesson
Arm 'history 8 + privileged critic': mean fall 0.425, crossed 0.320 (flat 0.19, step up 0.55, stairs 0.53). All four arms: fall 0.450 / 0.436 / 0.412 / 0.425, crossed 0.452 / 0.353 / 0.369 / 0.320 (base / critic / history / both). Differences are within sampling noise on fall rate; the crossed rate is lowest for the arms with history or critic, so neither is shown to help in this one-seed, 100M-step test.
Operations: with the ablation arm plus three 8192-world jobs the dedicated GPU memory hit 16.3 GB, 0.9 GB spilled to system RAM and the new jobs dropped from ~60k to 3-6k env-steps/s (WDDM paging). Killing the heaviest job (history 16, ~5 GB) restored 73k + 79k env-steps/s with 9.1 GB used. A third 4096-world job brought the aggregate to ~150k (gtea 53k, hist8dc 62k, extra 34k) at 11.6 GB, 57 C, 152 W, so the GPU is now saturated for this workload (aggregate flat between 2 and 3 jobs). `gpu_guard.py` logs temperature/power/memory every 20 s and suspends jobs above 83 C or 15.2 GB; `run_queue.py` launches jobs only when memory is below a limit (10 GB) and supports after:NAME dependencies. Stage 2 queue (queue_gen2.txt): DAgger student from gtea_s0, history 16, teacher seed 1, no design vector, width 1024, pure tracking (goal-p 0).

## S33 First generalist result (ggen_hist8dc_s0, 400M steps, Warp GPU, DR on, surfaces on, pulls on, 13 designs, seed 0)

Training curve: fall rate per step plateaued at about 0.0025 from 236M to 400M steps (episode mean about 400 steps), reward flat at about 1.35. More steps of this recipe do not help.
Evaluation (eval_tiles2.py, 16 worlds per design and kind, 800 steps, vel task, engine Warp, 1 seed; eval_tiles2 metrics only smoke-validated):
| setting | speed | fall | success | track err |
|---|---|---|---|---|
| DR+surfaces+pull | 0.3 | 0.87 | 0.04 | 0.47 |
| DR+surfaces+pull | 0.6 | 0.90 | 0.08 | 1.00 |
| no DR, no pull | 0.6 | 0.38 | 0.33 | 0.41 |
| DR, no pull | 0.6 | 0.78 | 0.13 | 0.49 |
Reading: the policy is far from robust. DR (design/surface/friction spread) is the main cost, not pulls. Weak designs (nom, zlow, zeta41, zeta66) fall most. The earlier tiles-only blind policy had fall 0.41 to 0.45 without the design/DR spread, so the generalist recipe as run is a regression in capability per unit of difficulty.
Next: diagnose by cause (push, pull, drop, design) before spending more steps; try curriculum on DR range and push strength, smaller design set, and shorter evaluation of easier settings.

## S34 Fall breakdown (gdiag_s0, 120M steps, Warp, DR+surfaces+pulls, 13 designs, push ramp, seed 0)
Training falls per step by terrain kind: 0.0023 to 0.0040, near uniform (flat 0.0027, stairs 0.0032, large_drop 0.0040). Terrain is not the limiter.
By design (per step): s0.5 0.053, J0005 0.020, kb4 0.0065, bal025 0.0050, others 0.001 to 0.004. s0.5 and J0005 are effectively unlearnable (episodes of 19 and 50 steps) and take a large share of resets.
Action: dropped s0.5 and J0005 from the training set (11 designs), run gfix_s0. Eval at 0.3 m/s, no DR, nominal design: fall 0.13, track err 0.19.

## S35 Push-ramp run (gcur_s0, 300M steps, same setting as S33 plus push/pull strength ramp 0.2 to 1 over 60 percent)
Final training fall rate 0.0023 per step (S33 recipe 0.0025): no meaningful gain. Eval with DR, surfaces, pulls, 13 designs, 16 per cell, 800 steps, Warp, seed 0: see eval_gcur.txt (0.3 m/s fall 0.87, same as ggen_hist8dc_s0). A push-strength curriculum alone does not fix the plateau. Open question: are 5 to 20 N pushes every 2 to 4 s recoverable on tiles; compare with earlier flat-ground push results.

## S36 Correction: eval_tiles2.py design bug (affects every --dr number in S33 and S35)
Cause: eval_tiles2 assigns designs after env.reset via set_design, which did not update the per-world mass and damping copies used by DR. In --dr evals each world kept the mass of a random design. Designs that change mass (bal*, kb*) were hit hardest (bal4: 0.92 falls with DR vs 0.08 without). Training is unaffected (reset draws the design first). The S33 and S35 rows quoting 0.84 to 0.90 falls under DR are invalid. Fix: set_design now carries the mass and damping randomisation factor over to the new design.
Re-eval of gfix_s0 (Warp, 11 designs, 16 per cell, 800 steps, 0.3 m/s, DR + surfaces + pulls, seed 0):
- pushes off: per-design falls nom 0.15, zlow 0.18, bal4 0.14, J02 0.19, zeta41 0.21, kb025 0.27, zeta66 0.28, tau6 0.33, kb4 0.51, bal025 0.51, zhigh 0.54; per terrain flat 0.14, ramps 0.22, steps 0.26 to 0.38, large_drop 0.65.
- pushes at full strength (5 to 20 N every 2 to 4 s): nom 0.52, flat 0.53, large_drop 0.84; tau6 0.97 and bal025 0.99.
Reading: without pushes the generalist is decent; pushes are the main remaining cause of falls, and weak or light designs (tau6, bal025, kb4, zhigh) suffer most. The push force range is taken from the nominal design config for every design (heavier designs have scaled ranges in their own configs), so push harshness per design is not equalised yet.
Earlier ablation rows (push 0 / 0.5, per-term DR off) were also affected by this bug and are superseded.

## S37 Fixed-eval comparison of the gfix runs (Warp, 11 designs, DR + surfaces + pulls, 16 per cell, 800 steps, 0.3 m/s, seed 0, per-design push scaling)
| run | falls, pushes off | falls, pushes on |
|---|---|---|
| gfix_s0 (goal mix 0.4, hist 8) | 0.30 | 0.67 |
| gfix_track_s0 (goal 0, hist 8) | 0.36 | 0.69 |
| gfix_h16_s0 (goal 0.4, hist 16) | 0.38 | 0.70 |
History length, goal mix and tracking-only are within noise of each other. Pushes remain the main limiter. Next: g2_nom_s0 (nominal design only) to see whether the design spread or the push load is the cap; g2_* runs use per-design push scaling in training.

## S38 Nominal-design-only control (g2_nom_s0, 150M steps, Warp, tiles + surfaces + pulls + pushes, per-design push scaling, seed 0)
Training fall rate 0.0023, same as the 11-design runs (0.0022 to 0.0024). Eval (nominal design, 32 per terrain kind, 800 steps, 0.3 m/s, DR + surfaces + pulls): falls 0.36 with pushes off, 0.60 with pushes on; track err 0.18 / 0.30; flat 0.16 / 0.44, large_drop 0.78 / 0.91.
Reading: the design spread is not what caps robustness. Pushes (5 to 20 N every 2 to 4 s) plus the harder tile kinds are. The older CPU-trained flat policies took single 50 N pushes with 0/10 falls, so the Warp tile setting asks for much more (repeated pushes over 8 s, tile terrain, 0.3 m/s evaluation window with 800 steps); an equal-protocol Warp vs CPU push comparison is still open.

## S39 g2 runs (Warp, 11 designs, per-design push scaling, 300M steps, DR + surfaces + pulls, seed 0)
Eval as S37 (16 per cell, 800 steps, 0.3 m/s):
| run | falls pushes off | falls pushes on | track err off / on |
|---|---|---|---|
| g2_s0 (goal 0.4, hist 8, asym) | 0.32 | 0.65 | 0.22 / 0.29 |
| g2_track_s0 (goal 0, hist 8, asym) | 0.34 | 0.67 | 0.20 / 0.29 |
Same as the gfix runs (0.30 to 0.38 / 0.67 to 0.70): all blind recipes tried so far (history 8 or 16, goal mix or tracking only, 11 or 1 design, 512 wide) end at the same ceiling. Equal-protocol flat push comparison from S-earlier: Warp-trained policy 0 push falls on CPU, about 6 percent on Warp, so engine difference is small and the tile result is not a simulator artefact.
Ceiling reading: with pushes off, large drops (0.65 to 0.78) and steps/stairs (0.35 to 0.56) fall most, i.e. conditions a 9 cm ball cannot cross blind at speed (S27: approach speed decides crossing). Levers that can move this are foresight and speed adaptation (perception teacher g2_tea_s0 then a depth student), not more blind training.

## S40 Privileged-scan teacher vs blind (g2_tea_s0 vs g2_s0, same recipe except the 35-D height scan input; Warp, 11 designs, DR + surfaces + pulls, 16 per cell, 800 steps, 0.3 m/s, seed 0)
| run | falls pushes off | falls pushes on | track err off / on |
|---|---|---|---|
| g2_s0 (blind) | 0.32 | 0.65 | 0.22 / 0.29 |
| g2_tea_s0 (scan teacher) | 0.39 | 0.74 | 0.22 / 0.32 |
The scan does not help under the speed-tracking and goal-mix recipe (slightly worse, within noise to worse). Same as S26: a tracking reward gives foresight no value. Distillation of a depth student from this teacher (g2_stu_s0) was stopped: nothing to distil. The meaningful perception test is the goal-only task where speed is free: g2_goal_s0 (blind) vs g2_teagoal_s0 (scan), relaunched when a slot frees.

## S41 Wider network (g2_w1024_s0 vs g2_s0, 1024 vs 512 wide, same recipe; Warp, 11 designs, DR + surfaces + pulls, 16 per cell, 800 steps, 0.3 m/s, seed 0)
| run | falls pushes off | falls pushes on | track err off / on |
|---|---|---|---|
| g2_s0 (512) | 0.32 | 0.65 | 0.22 / 0.29 |
| g2_w1024_s0 (1024) | 0.38 | 0.68 | 0.18 / 0.27 |
Training fall rate lower (0.0020 vs 0.0022) but eval falls not better; tracking error slightly better. Capacity is not the limiter. All "pushes off" rows still include the sustained pulls (3 to 12 N, 0.5 to 2 s); the pull contribution is tested next.

## S42 Pulls, not pushes, cause the remaining falls (g2_s0; Warp, 11 designs, DR + surfaces, 16 per cell, 800 steps, 0.3 m/s, seed 0)
| setting | falls | track err |
|---|---|---|
| no pushes, no pulls | 0.31 | 0.22 |
| pushes only (5 to 20 N, 50 ms, scaled per design) | 0.36 | 0.22 |
| pushes + pulls (3 to 12 N for 0.5 to 2 s every 4 to 10 s) | 0.65 | 0.29 |
Short impulses cost +0.05 falls; sustained pulls cost +0.30. The 0.31 floor is the terrain (large drops, steps; S37-S39). The earlier "push-limited" reading (S37 to S41) was wrong: it conflated pushes and pulls. Next: pull-force curve, and decide a physically sensible pull range for training (a sustained 12 N is about a 24 degree lean against gravity).

## S43 Disturbance-load curve (g2_s0; Warp, 11 designs, DR + surfaces, 16 per cell, 800 steps, 0.3 m/s, seed 0; pushes and pulls scaled together)
| scale (pull max N, push max N) | falls | track err |
|---|---|---|
| 0 | 0.31 | 0.22 |
| 0.25 (3, 5) | 0.36 | 0.22 |
| 0.5 (6, 10) | 0.44 | 0.24 |
| 0.75 (9, 15) | 0.54 | 0.26 |
| 1.0 (12, 20) | 0.65 | 0.29 |
Graceful, roughly linear degradation: about +0.1 falls per +0.25 of the load, no sharp tolerance threshold. A pure LQR evaluated on the same tile and disturbance protocol is the missing calibration of what is achievable; not yet built for the Warp tile env.

## S44 Goal task (goal 1.1 m away, reach within 0.2 m, 800 steps; Warp, 11 designs, DR + surfaces, 16 per cell, seed 0, tile start)
| run | disturbances | falls | success | median time to goal |
|---|---|---|---|---|
| g2_goal_s0 (goal-only training) | pushes + pulls | 0.73 | 0.20 | 4.05 s |
| g2_goal_s0 | none | 0.41 | 0.41 | 3.78 s |
| g2_s0 (goal mix 0.4) | pushes + pulls | 0.72 | 0.16 | 4.53 s |
| g2_s0 | none | 0.42 | 0.32 | 4.29 s |
Training on goals only helps the goal task modestly (success 0.41 vs 0.32 undisturbed, 0.20 vs 0.16 disturbed; about 0.5 s faster) at the same fall rate. Per terrain, undisturbed success: flat 0.72, ramps down 0.65, small drops 0.69, but step_up 0.10, stairs 0.11, bump 0.05, large_drop 0.39. Steps and bumps are the barrier, as in S27. One seed, 16 per cell: differences of 0.1 are near the noise floor; g2_s1 will give a second seed.

## S45 Goal task, privileged scan vs blind (g2_teagoal_s0 vs g2_goal_s0, both goal-only, 300M steps; Warp, 11 designs, DR + surfaces, 16 per cell, 800 steps, seed 0)
| run | disturbances | falls | success | median time |
|---|---|---|---|---|
| g2_goal_s0 (blind) | none | 0.41 | 0.41 | 3.78 s |
| g2_teagoal_s0 (scan) | none | 0.39 | 0.34 | 5.03 s |
| g2_goal_s0 (blind) | pushes + pulls | 0.73 | 0.20 | 4.05 s |
| g2_teagoal_s0 (scan) | pushes + pulls | 0.72 | 0.16 | 4.83 s |
Undisturbed success on the hard kinds: step_up 0.10 blind vs 0.02 scan, stairs 0.11 vs 0.01, bump 0.05 vs 0.01. The privileged height scan does not help even when speed is free, and the scan policy is slower. Same result as S26 and S40. Reading: RL does not discover foresight use here. A step of 3 cm or more is only crossed at high approach speed (S27), so useful foresight would mean speeding up before the step, which plain PPO rewards rarely discover; it needs explicit structure (edge-focused starts, speed-up reward shaping, a hierarchical planner, or an imitation prior), not a bigger sensor.

## S46 Perception encoder pretraining, v1 (collect_percep.py + train_percep.py; Warp, g2_goal_s0 as driver policy, 2048 worlds, 11 designs, DR + surfaces + pushes + pulls, 606k frames, split by world 90/10, seed 0)
Input: noisy VL53L9-like 18x14 depth zones (+ proprio for v1); target: 7x5 height grid ahead (cm) and hazard heads (max rise / drop within 0.1 to 0.5 m centre column).
| model | R2 (height grid) | MAE | rise >= 2 cm F1 | drop >= 2 cm F1 |
|---|---|---|---|---|
| mean predictor | 0.00 | 3.87 cm | 0.00 | 0.00 |
| proprio only | 0.02 | 3.97 cm | 0.25 | 0.20 |
| depth + proprio (32-D latent) | 0.55 | 2.19 cm | 0.62 | 0.67 |
R2 by forward row (0.1, 0.3, ..., 1.3 m): 0.13, 0.42, 0.61, 0.69, 0.70, 0.68, 0.64. The nearest rows are the weakest: with the head mount pitched -25 degrees the floor within about 0.25 m of the ball is outside the zone grid (S26 geometry), and the ball occludes it. Hazard prevalence in the data: rise >= 2 cm 20 percent, drop >= 2 cm 17 percent of frames. Next: depth-only encoder with the policy's own layer shape, frozen into the policy (RL did not learn to use the sensor end to end, S26, S40, S45).
S46 addendum: the policy-shaped depth-only encoder (Linear 252-128-Tanh-Linear 32-Tanh, per-zone standardised input, no proprio) scores better than the wider depth+proprio MLP on the same validation worlds: R2 0.60 (rows 0.17, 0.47, 0.68, 0.74, 0.74, 0.71, 0.67), MAE 1.93 cm, rise >= 2 cm F1 0.70, drop >= 2 cm F1 0.73. It is saved as percep/enc_v1_net.pt and frozen into the policy for g3_enc_s0 (goal-only, 11 designs, depth sensor, hist 8, asym critic, seed 0; encoder and input statistics of the depth columns fixed).

## S47 Seed noise (g2_s1 vs g2_s0, same recipe, seeds 1 and 0; Warp, 11 designs, DR + surfaces, 16 per cell, 800 steps, 0.3 m/s, vel task)
| run | falls, pushes + pulls | falls, no disturbances | track err (pushes + pulls / none) |
|---|---|---|---|
| g2_s0 | 0.65 | 0.31 | 0.29 / 0.22 |
| g2_s1 | 0.68 | 0.37 | 0.31 / 0.21 |
Seed-to-seed spread is about 0.03 to 0.06 falls. Differences between recipes smaller than that (S37 to S41: history, width, goal mix, tracking-only) are therefore noise, consistent with "all blind recipes end at the same ceiling". g3_cross_s0 (blind, goal-only, 60 percent of goals across the terrain feature, --cross-focus) launched as the cheap exposure test for step crossing; g3_enc_s0 (frozen pretrained depth encoder) is the perception test.

## S48 Goal-only blind baseline, two seeds (g2_goal_s0 and g2_goal_s1; Warp, 11 designs, DR + surfaces, 16 per cell, goal 1.1 m, 800 steps)
| run | disturbances | falls | success | median time | step_up | stairs | bump | large_drop |
|---|---|---|---|---|---|---|---|---|
| g2_goal_s0 | none | 0.41 | 0.41 | 3.78 s | 0.10 | 0.11 | 0.05 | 0.39 |
| g2_goal_s1 | none | 0.38 | 0.43 | 3.97 s | 0.10 | 0.05 | 0.06 | 0.31 |
| g2_goal_s0 | pushes + pulls | 0.73 | 0.20 | 4.05 s | | | | |
| g2_goal_s1 | pushes + pulls | 0.71 | 0.20 | 4.06 s | | | | |
Reproduces across seeds (success 0.41 / 0.43 undisturbed, 0.20 / 0.20 disturbed). Goal-only training beats the goal mix (g2_s0 0.32 / 0.16) by about 0.10 undisturbed, larger than the seed spread, so that gain is real. Steps, stairs and bumps stay at 0.05 to 0.10 success for the blind policy on both seeds: that is the baseline the encoder (g3_enc_s0), cross-focus (g3_cross_s0) and combined (g3_enccross_s0) runs have to beat.

## S49 Frozen pretrained depth encoder (g3_enc_s0; Warp, 11 designs, DR + surfaces, goal task, 16 per cell, 800 steps, seed 0; encoder enc_v1_net.pt trained on 606k frames, R2 0.60, frozen, input statistics fixed)
| run | disturbances | falls | success | median time | step_up | stairs | bump |
|---|---|---|---|---|---|---|---|
| g3_enc_s0 | none | 0.41 | 0.41 | 3.76 s | 0.15 | 0.07 | 0.05 |
| g3_enc_s0 | pushes + pulls | 0.71 | 0.20 | 3.88 s | 0.07 | 0.09 | 0.07 |
Same as the blind goal-only baseline (S48: 0.41 / 0.43 undisturbed, 0.20 disturbed; step_up 0.10, stairs 0.05 to 0.11, bump 0.05 to 0.06). A pretrained frozen encoder that predicts the terrain ahead with R2 0.60 does not change step crossing. Third negative perception result (S26, S40/S45, S49). Perception is not the bottleneck of this task as posed.
Observation that points at the real gap: the goal task takes about 3.8 s for 1.1 m (about 0.3 m/s average) and the training log speed error stays near 0.6 m/s from early training to the end, i.e. velocity tracking itself is weak. Flat-ground tracking vs speed is measured next (clean eval), to compare with the CPU-trained policies (tracking-error ratio about 0.35, S-earlier).

## S50 Velocity tracking is sluggish on the tiles (g2_track_s0; Warp, nominal design, clean = no DR, no disturbances, flat tile, 32 per speed, 800 steps, seed 0)
| command | early + steady track err | steady-state err (after 3 s) | achieved speed / command | falls |
|---|---|---|---|---|
| 0.3 m/s | 0.146 | 0.129 | 0.58 | 0.00 |
| 0.6 m/s | 0.337 | n/a (robot leaves the 3 m tile before 3 s) | n/a | 0.11 |
| 0.9 m/s | 0.543 | n/a | n/a | 0.08 |
Error / command: 0.49 (0.3), 0.56 (0.6), 0.60 (0.9); the CPU-trained flat policies had a ratio of about 0.33 to 0.41 (S-earlier protocol, different transient handling). At 0.3 m/s the policy settles at only 58 percent of the commanded speed: balanced but sluggish. Training logs show speed error near 0.6 m/s from early training to the end for every recipe, so none of the recipe changes touched this.
Test started: reward weights (--rw-track, --rw-pen added to train_dev.py). g4_pen_s0 = penalties on lean, action rate and action size cut to 0.25x (vel-only, 11 designs, 150M steps); compare steady-state speed ratio with g2_track_s0 (0.58 at 0.3 m/s). Next: --rw-track 2, and an arena large enough to measure steady-state at 0.6 and 0.9 m/s.

## S51 Cross-focus goals (g3_cross_s0: goal-only, blind, 60 percent of goals 1.1 to 1.3 m from the tile centre along a tile axis; Warp, 11 designs, DR + surfaces, goal task, 16 per cell (176 per kind), 800 steps, seed 0)
| run | disturbances | falls | success | median time | step_up | stairs | bump | ramp_up | flat |
|---|---|---|---|---|---|---|---|---|---|
| g3_cross_s0 | none | 0.48 | 0.40 | 3.35 s | 0.15 | 0.17 | 0.10 | 0.26 | 0.63 |
| g2_goal_s0/s1 (blind baseline, S48) | none | 0.41 / 0.38 | 0.41 / 0.43 | 3.78 / 3.97 s | 0.10 / 0.10 | 0.11 / 0.05 | 0.05 / 0.06 | 0.14 / 0.20 | 0.72 / 0.74 |
| g3_cross_s0 | pushes + pulls | 0.77 | 0.18 | 3.41 s | 0.11 | 0.14 | 0.06 | 0.18 | 0.24 |
Cross-focus makes the policy faster (0.4 to 0.6 s quicker, 3.35 s vs 3.8 to 4.0) and a bit better on the hard kinds (step_up 0.15 vs 0.10, stairs 0.17 vs 0.05 to 0.11, bump 0.10 vs 0.05 to 0.06, ramp_up 0.26 vs 0.14 to 0.20; about 1.5 to 2 standard errors each, standard error about 0.03 per kind) at the price of more falls (0.48 vs 0.38 to 0.41) and lower success on flat ground (0.63 vs 0.72 to 0.74): faster and riskier. Overall success is unchanged (0.40 vs 0.41 to 0.43). Modest and suggestive, not decisive; one seed. The mechanism (exposure to crossing) works in the expected direction.

## S52 Tracking reward weight fixes the sluggish speed (Warp, nominal design, clean = no DR, no disturbances, flat tile, 32 worlds, 800 steps, 0.3 m/s command, steady state = after 3 s, seed 0; all runs vel-only, 11 training designs)
| run | training steps at eval | steady-state err | achieved speed / command | falls |
|---|---|---|---|---|
| g2_track_s0 (baseline, S50) | 300M | 0.129 | 0.58 | 0.00 |
| g4_pen_s0 (lean, action-rate, action-size penalties x 0.25) | about 110M | 0.127 | 0.59 | 0.00 |
| g4_trk_s0 (tracking reward x 2, --rw-track 2) | about 45M | 0.043 | 0.92 | 0.00 |
Doubling the weight on the velocity-tracking reward cuts the steady-state error by two thirds and brings the achieved speed to 92 percent of the command, after a fraction of the training. Lighter penalties do nothing. The sluggishness was a reward-balance effect, not a physical limit. Training logs for runs with the old reward showed speed error stuck near 0.6 m/s for every recipe; that was the symptom. g4_pen_s0 stopped (question answered). A repeat eval of the g2_track_s0 checkpoint returned nan (a few worlds fell, steady-state counter empty), so the S50 value is used. Next: full 300M runs with --rw-track 2 (goal-only + cross focus, and vel-only), evaluation at 0.6 and 0.9 m/s with a larger arena, and under disturbances.

## S53 CORRECTION of S52: clean steady-state tracking benchmark on a perfectly flat world (TILES_FLAT=1 EVAL_NOEXIT=1, Warp, nominal design, no DR, no disturbances, 32 worlds, 800 steps, steady state = after 3 s, seed 0, 300M-step g2_track_s0 and 150M-step g4_trk_s0)
| command | run | steady-state err | achieved speed / command | falls |
|---|---|---|---|---|
| 0.3 | g2_track_s0 | 0.130 | 0.58 | 0.00 |
| 0.3 | g4_trk_s0 | nan (one world fell, empty window) | nan | 0.02 |
| 0.6 | g2_track_s0 | 0.308 | 0.49 | 0.12 |
| 0.6 | g4_trk_s0 | 0.247 | 0.59 | 0.12 |
| 0.9 | g2_track_s0 | 0.502 | 0.45 | 0.14 |
| 0.9 | g4_trk_s0 | 0.413 | 0.54 | 0.14 |
The S52 claim (error cut by two thirds, 92 percent of command) was an artefact: that measurement used a 3 to 4 s window cut by the tile exit, on a 45M-step checkpoint. On the clean benchmark, doubling the tracking reward gives a modest gain (steady-state error 10 to 20 percent lower, achieved speed 5 to 10 points higher), and both policies still reach only about half of the commanded speed at 0.6 and 0.9 m/s. Even on a flat world with no disturbances they fall 12 to 14 percent of the time at 0.6 and 0.9 m/s. The CPU-trained flat policies had an error / command ratio of about 0.35; here it is 0.43 (0.3 m/s) to 0.55 (0.9 m/s) for the better run. Next: --rw-track 4 (g6_trk4_s0, vel-only 150M), and a pure LQR on this same flat benchmark as the calibration of what is achievable.

## S54 Goal-task summary of the goal-only variants (Warp, 11 designs, DR + surfaces, goal 1.1 m, 800 steps, 16 per cell (176 per kind), seed 0 unless noted; undisturbed / pushes + pulls)
| run | falls | success | median time | step_up | stairs | bump | ramp_up |
|---|---|---|---|---|---|---|---|
| g2_goal blind (2 seeds) | 0.38 to 0.41 / 0.71 to 0.73 | 0.41 to 0.43 / 0.20 | 3.8 to 4.0 s | 0.10 | 0.05 to 0.11 | 0.05 to 0.06 | 0.14 to 0.20 |
| g3_enc (frozen depth encoder) | 0.41 / 0.71 | 0.41 / 0.20 | 3.76 s | 0.15 | 0.07 | 0.05 | 0.14 |
| g3_cross (60 percent cross-focus goals) | 0.48 / 0.77 | 0.40 / 0.18 | 3.35 s | 0.15 | 0.17 | 0.10 | 0.26 |
| g3_enccross (encoder + cross-focus) | 0.38 / 0.69 | 0.42 / 0.22 | 3.63 s | 0.15 | 0.08 | 0.06 | 0.12 |
Overall success is flat across all four goal-only variants (0.40 to 0.43 undisturbed, 0.18 to 0.22 disturbed). Step up improves from 0.10 to 0.15 for the three non-baseline variants, stairs and bumps only for cross-focus without the encoder, and the encoder adds nothing on top of cross-focus. The unit of the problem is stable: every variant ends with about 40 percent of the goal trials failing by falling even undisturbed, dominated by large drops (0.7) and steps (0.5 to 0.6). Next reading: the g5 runs (tracking reward x 2 on top of cross-focus) and the tracking-weight sweep (g6_trk4_s0).

## S55 Tracking-weight curve on the clean flat benchmark (TILES_FLAT=1 EVAL_NOEXIT=1; Warp, nominal design, no DR, no disturbances, 32 worlds, 800 steps, steady state after 3 s, seed 0; vel-only runs, 11 training designs)
| weight | run (steps) | 0.3 m/s err / achieved | 0.6 m/s err / achieved / falls | 0.9 m/s err / achieved / falls |
|---|---|---|---|---|
| x1 | g2_track_s0 (300M) | 0.130 / 0.58 | 0.308 / 0.49 / 0.12 | 0.502 / 0.45 / 0.14 |
| x2 | g4_trk_s0 (150M) | nan | 0.247 / 0.59 / 0.12 | 0.413 / 0.54 / 0.14 |
| x4 | g6_trk4_s0 (150M) | 0.105 / 0.66 | 0.237 / 0.61 / 0.03 | 0.386 / 0.58 / 0.11 |
Diminishing returns: x2 and x4 gain 10 to 20 percent in error over x1, the second doubling adds about 5 percent; at x4 the policy reaches 58 to 66 percent of the commanded speed. The reward weight lever is nearly used up. Context from the earlier study (S-earlier, 13 designs, CPU protocol): LQR tracks at ratio 1.00 to 1.02 up to 0.5 m/s but falls at speed; the earlier pure PPO tracked at about a third of the command and did not fall. So the learned policy is now at about 60 percent and robust, and the precision gap at low speed (0.66 vs LQR 1.0 at 0.3 m/s) is real and not a physical limit. Next structured fix: use LQR as an imitation prior at low speed (auxiliary behaviour-cloning loss on the LQR action while the command is below 0.5 m/s), keep PPO for robustness at speed.

## S56 LQR calibration on the Warp flat benchmark (lqr_gpu.py, batched GPU LQR: discrete LQR per design at the 10 ms control period, velocity reference governor a_max 0.5 m/s^2, integral gain 1.0, yaw gain 5; Warp, nominal design, no DR, TILES_FLAT=1 EVAL_NOEXIT=1, 32 worlds, 800 steps, steady state after 3 s, seed 7)
| command | steady-state err | achieved speed / command | falls |
|---|---|---|---|
| 0.3 m/s | 0.019 | 1.02 | 0.12 |
| 0.6 m/s | 0.023 | 1.00 | 0.22 |
| 0.9 m/s | 0.024 | 1.00 | 0.97 |
The learned policies (S55) track at error 0.105 to 0.39 and 58 to 66 percent of the command on the same benchmark: LQR is 5 to 10 times more accurate, so the roughly 60 percent ceiling is a policy deficiency, not a physical limit of the robot. LQR is fragile (12 to 22 percent falls at 0.3 to 0.6 m/s even undisturbed, 97 percent at 0.9 m/s), the learned policy robust and imprecise. Complementary, as in the earlier study (S-earlier: gate collapsed to LQR; residual with large authority fought the LQR). Next: LQR as a base controller with a bounded learned residual, trained by PPO (--lqr-base S, residual scale S), using the best reward weights (tracking x2 to x4) so the policy can be penalised for the residual size and rewarded for tracking and survival.

## S57 Cross-focus + tracking reward x2 on the goal task (g5_goal_s0; goal-only, 60 percent cross-focus, --rw-track 2, 300M steps; Warp, 11 designs, DR + surfaces, goal 1.1 m, 800 steps, 16 per cell (176 per kind), seed 0)
| run | disturbances | falls | success | median time | step_up | stairs | bump | ramp_up | flat |
|---|---|---|---|---|---|---|---|---|---|
| g5_goal_s0 | none | 0.44 | 0.38 | 3.92 s | 0.14 | 0.09 | 0.07 | 0.14 | 0.65 |
| g3_cross_s0 (S51) | none | 0.48 | 0.40 | 3.35 s | 0.15 | 0.17 | 0.10 | 0.26 | 0.63 |
| g5_goal_s0 | pushes + pulls | 0.75 | 0.17 | 3.89 s | | | | | |
Adding the doubled tracking reward on top of cross-focus does not improve the goal task: success 0.38 vs 0.40, time 3.92 s vs 3.35 s, stairs 0.09 vs 0.17, ramps 0.14 vs 0.26. The better velocity tracking measured on the flat benchmark (S55) does not translate into a better goal-chasing policy on the tiles; within the one-seed noise (standard error about 0.03 per kind) the two are equal or the doubled weight is slightly worse. Every pure-learning goal-only variant (S48, S49, S51, S54, S57) therefore sits in the same band: undisturbed success 0.38 to 0.43, disturbed 0.17 to 0.22. The remaining untested structural lever is the LQR base with a learned residual (g7_lqrres_s0, running).

## S58 LQR + learned residual, early checkpoint (g7_lqrres_s0, residual authority 0.5, tracking x2, half goal tasks, cross-focus, about 93M of 200M steps; Warp, nominal design, clean, TILES_FLAT=1 EVAL_NOEXIT=1, 32 worlds, 800 steps, steady state after 3 s, seed 0)
| command | steady-state err | achieved / command | falls |
|---|---|---|---|
| 0.3 m/s | 0.019 | 1.03 | 0.16 |
| 0.6 m/s | 0.025 | 1.02 | 0.20 |
| 0.9 m/s | 0.053 | 1.03 | 0.48 |
Reference rows on the same benchmark: LQR alone 0.019 to 0.024, ratio 1.00 to 1.02, falls 0.12 / 0.22 / 0.97 (S56); best pure-learning policy (g6_trk4_s0, S55) error 0.105 / 0.237 / 0.386, ratio 0.66 / 0.61 / 0.58, falls 0.00 / 0.03 / 0.11.
Reading: the residual policy keeps LQR's tracking precision at every speed (error 0.02 to 0.05, ratio about 1.0) and already halves LQR's falls at 0.9 m/s (0.97 to 0.48), but still falls more at low speed than the pure-learning policy (0.16 / 0.20 vs 0.00 / 0.03). It is the first policy with accurate tracking at 0.9 m/s (error 0.05 against 0.39 for the best pure-learning policy). Training is still in progress (93M of 200M); the open question is whether the residual learns to remove the low-speed falls.

## S59 Clean flat-world tracking table, NaN-fixed (TILES_FLAT=1 EVAL_NOEXIT=1; Warp, nominal design, no DR, no disturbances, 32 worlds, 800 steps, steady state after 3 s, seed 0). Supersedes the nan rows in S53 and S55 (cause: a fallen world carried non-finite velocities and 0 x nan stayed nan; fixed with torch.where in eval_tiles2.py).
| run | tracking weight, task mix | 0.3 m/s err / achieved / falls | 0.6 m/s | 0.9 m/s |
|---|---|---|---|---|
| g2_track_s0 | x1, vel-only | 0.126 / 0.59 / 0.02 | 0.303 / 0.50 / 0.09 | 0.503 / 0.44 / 0.12 |
| g4_trk_s0 | x2, vel-only | 0.108 / 0.65 / 0.00 | 0.247 / 0.59 / 0.09 | 0.413 / 0.54 / 0.11 |
| g6_trk4_s0 | x4, vel-only | 0.109 / 0.65 / 0.00 | 0.240 / 0.61 / 0.09 | 0.400 / 0.56 / 0.12 |
| g5_mix_s0 | x2, half goals + cross-focus | 0.114 / 0.63 / 0.00 | 0.281 / 0.54 / 0.12 | 0.467 / 0.48 / 0.08 |
| LQR alone (S56) | n/a | 0.019 / 1.02 / 0.12 | 0.023 / 1.00 / 0.22 | 0.024 / 1.00 / 0.97 |
Weight x2 and x4 give the same result within noise (x4 is not better than x2); both beat x1 by about 15 percent in error. Mixing in goal tasks costs about the same as going back toward x1. No pure-learning variant gets above 0.65 of the commanded speed at 0.3 m/s or above 0.61 at 0.6 m/s. LQR-based policies (S58) are the only ones that reach 1.0.

## S60 LQR + learned residual, final (g7_lqrres_s0: LQR base, residual authority 0.5, tracking x2, half goal tasks, cross-focus 0.6, 11 designs, 200M steps; Warp; seed 0; protocols as in S59, S54, S37)
| measure | g7_lqrres_s0 | best pure learning (run) |
|---|---|---|
| flat world, clean, err / achieved / falls at 0.3 | 0.019 / 1.02 / 0.05 | 0.108 / 0.65 / 0.00 (g4_trk_s0) |
| at 0.6 | 0.022 / 1.01 / 0.11 | 0.240 / 0.61 / 0.09 (g6_trk4_s0) |
| at 0.9 | 0.049 / 1.02 / 0.27 | 0.400 / 0.56 / 0.12 (g6_trk4_s0) |
| goal task, undisturbed: falls / success / median time | 0.53 / 0.47 / 2.23 s | 0.38 to 0.48 / 0.38 to 0.43 / 3.4 to 4.0 s |
| goal task, undisturbed: step_up / stairs / bump / ramp_up | 0.25 / 0.34 / 0.13 / 0.49 | 0.10 to 0.18 / 0.05 to 0.17 / 0.05 to 0.10 / 0.14 to 0.26 |
| goal task, pushes + pulls: falls / success | 0.82 / 0.18 | 0.69 to 0.77 / 0.17 to 0.22 |
| vel 0.3 on tiles, DR + surfaces: falls undisturbed / disturbed | 0.63 / 0.83 | 0.38 / 0.69 (g5_mix_s0) |
| vel 0.3 on tiles: steady-state err / achieved, undisturbed | 0.103 / 0.93 | 0.20 total err (g5_mix_s0) |
First policy that is both accurate (error 2 to 5 percent of the command up to 0.9 m/s) and goal-directed (success 0.47, 40 percent faster, step_up 0.25, stairs 0.34, ramp_up 0.49), at the price of more falls on tiles (0.63 vs 0.38 on the velocity task) and weaker disturbance robustness (0.82 vs 0.69 to 0.77 goal falls). The early checkpoint (S58, 93M) had flat falls 0.16 / 0.20 / 0.48: training the residual further cut them to 0.05 / 0.11 / 0.27. One seed; g7_lqrres_s1 is the replicate. Next: robustness of the residual (longer training, more authority only where LQR fails, fall-penalty weight, disturbance-heavy curriculum), then renders.

## S61 Residual authority 1.0 fails (g7_lqrres1_s0 vs g7_lqrres_s0; same recipe, residual authority rs = 1.0 vs 0.5; 200M steps; Warp; evaluation protocols as in S60)
| run | flat benchmark falls at 0.3 / 0.6 / 0.9 | goal task, undisturbed falls / success | goal task, pushes + pulls falls / success | training fall rate at the end |
|---|---|---|---|---|
| g7_lqrres_s0 (rs 0.5) | 0.05 / 0.11 / 0.27 | 0.53 / 0.47 | 0.82 / 0.18 | 0.0044 |
| g7_lqrres1_s0 (rs 1.0) | 1.00 / 1.00 / 1.00 | 0.89 / 0.11 | 0.96 / 0.04 | 0.0164 |
With full authority the learned residual can override the stabilising LQR action (the sum is clipped to [-1, 1] and the policy explores with noise of the same size), and the robot falls everywhere even on flat ground. Authority must stay well below 1. Next: rs 0.25 (g8_lqr025_s0), and g8_lqrfp_s0 (rs 0.5, fall penalty x 3) already running. A photo-finish for 0.5 vs 0.25 is the open question; seed-1 replicate g7_lqrres_s1 (rs 0.5) running.

## S62 LQR + residual, two seeds (g7_lqrres_s0 and g7_lqrres_s1; residual authority 0.5, tracking x2, half goals, cross-focus 0.6, 11 designs, 200M steps; Warp; protocols as in S60)
| measure | seed 0 | seed 1 | best pure learning |
|---|---|---|---|
| flat world, clean: err at 0.3 / 0.6 / 0.9 | 0.019 / 0.022 / 0.049 | 0.017 / 0.027 / 0.039 | 0.108 / 0.240 / 0.400 |
| achieved / command | 1.02 / 1.01 / 1.02 | 1.01 / 1.01 / 1.02 | 0.65 / 0.61 / 0.56 |
| flat world falls at 0.3 / 0.6 / 0.9 | 0.05 / 0.11 / 0.27 | 0.06 / 0.16 / 0.16 | 0.00 / 0.09 / 0.12 |
| goal, undisturbed: success / median time / falls | 0.47 / 2.23 s / 0.53 | 0.45 / 2.21 s / 0.55 | 0.38 to 0.43 / 3.4 to 4.0 s / 0.38 to 0.48 |
| goal, undisturbed: step_up / stairs / ramp_up | 0.25 / 0.34 / 0.49 | 0.30 / 0.29 / 0.51 | 0.10 to 0.18 / 0.05 to 0.17 / 0.14 to 0.26 |
| goal, pushes + pulls: success / falls | 0.18 / 0.82 | 0.19 / 0.81 | 0.17 to 0.22 / 0.69 to 0.77 |
| vel 0.3 on tiles: falls undisturbed / disturbed | 0.63 / 0.83 | 0.64 / 0.83 | 0.38 / 0.69 |
| vel 0.3 on tiles: achieved / command (steady state) | 0.93 | 0.93 | about 0.6 |
The two seeds agree to within noise on every measure. The LQR-residual advantages (precision 2 to 4 percent of the command up to 0.9 m/s, goal success +0.05, 40 percent faster, steps / stairs / ramps 1.5 to 3 times better) and its deficits (falls on tiles 0.63 vs 0.38, falls under disturbance 0.82 vs 0.69 to 0.77) are both reproducible. Open variants running: authority 0.25 (g8_lqr025_s0), fall penalty x 3 (g8_lqrfp_s0), soft LQR base (g8_lqrsoft_s0). The 0.9 m/s flat falls differ between seeds (0.27 vs 0.16): seed noise at 32 worlds per cell is about 0.06.

## S63 Residual variants against the 0.5-authority baseline (all: LQR base, tracking x2, half goal tasks, cross-focus 0.6, 11 designs, 200M steps, seed 0 unless noted; Warp; protocols as in S60)
| variant | flat falls at 0.3 / 0.6 / 0.9 | flat err at 0.3 / 0.6 / 0.9 | goal undisturbed: falls / success / time | goal pushes + pulls: falls / success | vel 0.3 on tiles: falls undisturbed / disturbed |
|---|---|---|---|---|---|
| rs 0.5 (g7_lqrres_s0 and _s1, S62) | 0.05 to 0.06 / 0.11 to 0.16 / 0.16 to 0.27 | 0.017 to 0.019 / 0.022 to 0.027 / 0.039 to 0.049 | 0.53 to 0.55 / 0.45 to 0.47 / 2.2 s | 0.81 to 0.82 / 0.18 to 0.19 | 0.63 to 0.64 / 0.83 |
| rs 0.25 (g8_lqr025_s0) | 0.14 / 0.14 / 0.83 | 0.021 / 0.029 / 0.085 | 0.62 / 0.38 / 2.11 s | 0.85 / 0.15 | 0.66 / 0.85 |
| rs 0.5 with fall penalty 6 instead of 2 (g8_lqrfp_s0) | 0.11 / 0.11 / 0.31 | 0.022 / 0.027 / 0.045 | 0.54 / 0.46 / 2.20 s | 0.84 / 0.16 | 0.65 / 0.84 |
Neither variant reduces the falls. Lower authority (0.25) is worse (0.83 flat falls at 0.9 m/s): too little learned correction to cover the LQR failure modes; tripling the fall penalty changes nothing measurable (the policy already avoids falls where it can; the residual cannot rescue states the LQR base drives into). So the fall penalty and the authority are not the levers; candidates still running: soft LQR base (g8_lqrsoft_s0, control cost x3, reference acceleration 0.3), wide domain randomisation curriculum (g9_dr_s0 and g9_dr_s1, ranges up to 1.75 x).

## S64 Baseline: LQR + residual under widened domain randomisation (g7_lqrres_s0 and g7_lqrres_s1, trained at the standard randomisation ranges; Warp, 11 designs, DR + surfaces, goal task, 16 worlds per cell (176 per kind), 800 steps, seeds 0 and 1; DR_SCALE multiplies mass +-15 %, friction +-20 %, torque +-15 %, damping spread, latency probability 20 % and sensor noise 0.01 together)
| DR scale | undisturbed: falls / success / median time | pushes + pulls: falls / success |
|---|---|---|
| 1.0 | 0.55 / 0.45 / 2.23 s (s0), 0.52 / 0.48 / 2.21 s (s1) | 0.83 / 0.17 (s0), 0.82 / 0.18 (s1) |
| 1.75 | 0.69 / 0.31 / 2.20 s (s0), 0.70 / 0.30 / 2.18 s (s1) | 0.90 / 0.10 (both) |
| 2.5 | 0.82 / 0.18 / 2.14 s (s0), 0.81 / 0.19 / 2.14 s (s1) | 0.95 / 0.05 (both) |
Success drops by about one third at 1.75 x and by about 60 percent at 2.5 x, for both seeds (agreeing within 0.03). The policy is not robust to wider parameter variation than it was trained on; g9_dr_s0 and g9_dr_s1 (same recipe, ranges ramped to 1.75 x during training, 200M steps) are the test of whether training over the wider envelope closes this. The g6_trk4_s0 rows of the same chain are invalid for the goal task (a velocity-only policy never trained on goals) and are not used.

## S65 Corrections found by the site and paper cross-check (docs/research/site_crosscheck.md), and an evaluation-file fix
- World counts: the flat benchmark rows in S55, S59, S60, S62 use 64 worlds per row (two nominal-design copies of 32: `--eval-designs nom,nom --n 32`), not 32. The GPU LQR calibration (S56) really used 32.
- S51 and S54: g2_goal_s0 ramp_up success is 0.18 (not 0.14).
- Wording: "LQR is 5 to 10 times more accurate than the learned policies" is 6 to 17 times; "40 percent faster" is 42 to 43 percent; "within 2 to 5 percent of the command" is 4 to 6 percent (steady-state error / command).
- Evaluation files used to be overwritten silently (the JSON did not record PUSH_SCALE, DR_SCALE, TILES_FLAT, EVAL_NOEXIT, and a later evaluation reused the file name; for example the g7_lqrres_s0 goal-task files hold the DR scale 2.5 runs after the S64 chain ran, and the full-load g2_s0 file holds the 0.75 run). Fixed in eval_tiles2.py: every setting that changes the result is now part of the file name (`_ps`, `_drs`, `_flat`, `_noexit`, `_off-`) and stored in `env_settings` inside the JSON. `experiments/warp_ppo/eval_battery.py` runs the standard battery (flat benchmark, goal task and velocity-on-tiles at the chosen DR scales, pushes + pulls on and off) with every setting explicit and writes `<run>/battery.json`. Numbers in this log are from the printed outputs at the time and stay valid; the site reads a frozen snapshot copy of the files (`site/data/eval_snapshot/`).
- Thin data noted by the cross-check, to be closed: single-seed rows on the hard terrain kinds, training and evaluation on the same 64 tiles (`tile_seed` 0) and 11 designs (no held-out terrain or design test), no CPU re-evaluation of any tile policy, no LQR-alone row on the tile and disturbance protocols, only two points of the disturbance load curve for the residual policy.

## S66 Soft LQR base (g8_lqrsoft_s0: LQR control cost x3 (r = 3), reference acceleration limit 0.3 m/s^2 instead of 0.5, residual authority 0.5, tracking x2, half goal tasks, cross-focus 0.6, 11 designs, 200M steps, seed 0; Warp; evaluated with eval_battery.py, files in outputs/study/warp_ppo/g8_lqrsoft_s0/)
| measure | soft LQR base | standard LQR base (g7_lqrres_s0 / _s1, S62) |
|---|---|---|
| flat world, clean, 64 worlds: err at 0.3 / 0.6 / 0.9 | 0.041 / 0.048 / 0.067 | 0.017 to 0.019 / 0.022 to 0.027 / 0.039 to 0.049 |
| achieved / command | 0.91 / 0.96 / 0.99 | 1.01 to 1.02 |
| flat falls at 0.3 / 0.6 / 0.9 | 0.05 / 0.16 / 0.28 | 0.05 to 0.06 / 0.11 to 0.16 / 0.16 to 0.27 |
| goal, undisturbed: falls / success / median time | 0.49 / 0.51 / 2.94 s | 0.53 to 0.55 / 0.45 to 0.47 / 2.2 s |
| goal, pushes + pulls: falls / success | 0.77 / 0.22 | 0.81 to 0.83 / 0.17 to 0.19 |
| vel 0.3 on tiles: falls undisturbed / disturbed | 0.65 / 0.82 | 0.63 to 0.64 / 0.83 |
The soft base gives the best goal success so far (0.51 undisturbed, 0.22 with pushes and pulls) at the price of slightly lower tracking precision (error 0.04 to 0.07 instead of 0.02 to 0.05) and a slower approach (2.9 s instead of 2.2 s, as expected from the lower reference acceleration). Flat falls and velocity-task falls are unchanged, so it does not remove the main failure mode but it moves the goal-task trade-off: a few points more success for less speed. One seed (differences of 0.03 to 0.05 are near the seed spread).

## S67 Wide randomisation training does not buy robustness to wide randomisation (g9_dr_s0 and g9_dr_s1: LQR base, residual 0.5, tracking x2, half goals, cross-focus 0.6, 11 designs, all randomisation ranges ramped to 1.75 x over the first 50 percent of 200M steps; evaluated with eval_battery.py at DR scales 1, 1.75, 2.5; Warp; goal task 16 worlds per design and kind, 176 per kind; seeds 0 and 1; baseline = g7_lqrres_s0/_s1 trained at the standard ranges, S64)
| goal task | DR scale | trained 1.75 x (s0 / s1): falls / success | trained standard (S64, s0 / s1): falls / success |
|---|---|---|---|
| undisturbed | 1.0 | 0.53 / 0.47, 0.52 / 0.47 | 0.55 / 0.45, 0.52 / 0.48 |
| undisturbed | 1.75 | 0.68 / 0.32, 0.66 / 0.34 | 0.69 / 0.31, 0.70 / 0.30 |
| undisturbed | 2.5 | 0.80 / 0.20, 0.80 / 0.20 | 0.82 / 0.18, 0.81 / 0.19 |
| pushes + pulls | 1.0 | 0.84 / 0.16, 0.83 / 0.17 | 0.83 / 0.17, 0.82 / 0.18 |
| pushes + pulls | 1.75 | 0.90 / 0.10 (both) | 0.90 / 0.10 (both) |
| pushes + pulls | 2.5 | 0.94 / 0.06 (both) | 0.95 / 0.05 (both) |
Flat benchmark (clean, nominal, 64 worlds): trained 1.75 x s0 / s1: err 0.020 / 0.020, 0.031 / 0.032, 0.065 / 0.062 at 0.3 / 0.6 / 0.9 m/s; achieved 1.00 to 1.03; falls 0.12 / 0.11, 0.12 / 0.12, 0.41 / 0.25. Velocity 0.3 m/s on tiles, undisturbed: falls 0.64 to 0.65 / 0.75 to 0.76 / 0.83 to 0.85 and error 0.12 / 0.16 to 0.17 / 0.22 to 0.23 at scale 1 / 1.75 / 2.5.
Reading: a policy trained over 1.75 x and evaluated at 1.75 x (success 0.32 to 0.34) is no better than the standard-trained policy evaluated at 1.75 x (0.30 to 0.31), and the same at 2.5 x. There is no generalisation gap for wider training to close; the wider parameter spread is intrinsically harder for this controller (LQR base designed for the nominal model of each design plus a learned residual limited to authority 0.5). Next: single-factor ablation at scale 2.5 to find which randomisation term does the damage (mass, friction, torque, damping, latency, sensor noise), then target that term.

## S68 Which randomisation term hurts: mass (leave-one-out ablation, g9_dr_s0 trained over 1.75 x; Warp, goal task undisturbed (no pushes, no pulls), DR + surfaces, DR scale 2.5, 16 worlds per design and kind (176 per kind), 800 steps, seed 0; ablate_dr.py, files outputs/study/warp_ppo/g9_dr_s0/ablate_dr_2.5.json)
| setting | falls | success | median time |
|---|---|---|---|
| all randomisation on (scale 2.5) | 0.80 | 0.20 | 2.10 s |
| no mass randomisation | 0.57 | 0.43 | 2.15 s |
| no friction randomisation | 0.78 | 0.22 | 2.10 s |
| no torque randomisation | 0.79 | 0.21 | 2.10 s |
| no damping randomisation | 0.77 | 0.23 | 2.13 s |
| no latency randomisation | 0.76 | 0.24 | 2.12 s |
| no sensor noise | 0.78 | 0.22 | 2.12 s |
| all off (ceiling) | 0.42 | 0.57 | 2.24 s |
Removing the mass randomisation alone recovers success from 0.20 to 0.43 (about 60 percent of the gap to the all-off ceiling 0.57); removing any other single term moves success by less than 0.04. Mass randomisation scales the mass of the base and of the ball by (1 + scale x U(-0.15, 0.15)), so up to +-37 percent at scale 2.5, with the inertia and the LQR gains unchanged. The LQR base is designed on the nominal model of each design; a mass error detunes it, and a learned residual with authority 0.5 cannot repair that. This explains S67 (wide training did not help: the residual learns to compensate friction, torque, damping, latency and noise but not mass). Single seed (0), one policy; the all-off ceiling (0.57) is also above the standard-scale value (0.47), consistent with mass being the main loss at scale 1 as well.

## S69 CORRECTION: stale body_subtreemass under mass randomisation invalidated every domain-randomised result before this entry
Cause. The GPU engine finds the centre of mass of every body subtree by dividing mass-weighted positions by the stored `body_subtreemass`, and uses that point as the reference of the dynamics. `train_dev.py` randomised `body_mass` (ball and base) per world but never updated `body_subtreemass`, so after any mass change the reference point was wrong by a fraction of the robot's distance from the world origin (training and evaluation tiles lie up to 12 m from it). The error is zero at the nominal mass, which is why every run without mass randomisation, and every evaluation with randomisation off, is unaffected.
Evidence. LQR alone, Warp, flat clean nominal, 64 worlds, falls at 0.3 / 0.6 m/s: ball mass fixed to 0.63, 0.8, 1.2, 1.37, 2.0 times nominal: 0.88 / 1.00, 0.62 / 0.94, 0.80 / 1.00, 1.00 / 1.00, 0.92 / 1.00 (nominal 1.0: 0.12 / 0.16). Base mass only at scale 2.5 (+-37 %): 0.14 / 0.23 (the base subtree masses are dominated by the unchanged children); ball mass only: 0.41 / 0.81. Scaling the ball inertia with the mass changed nothing (0.53 / 0.84 vs 0.53 / 0.88); the linear planar model with the nominal discrete gain stays stable from 0.5 to 3 times the ball mass and from 0.5 to 2 times the base mass (largest closed-loop modulus 0.955 to 0.972); the integral gain (0.3, 3), the control cost (R x3) and the reference acceleration (0.3, 0.15) did not remove it. Recomputing the constants for a 1.2 x ball (only `body_subtreemass` changes) cut the falls from 0.80 / 1.00 to 0.28 / 0.27. Fix: `_subtree()` recomputes the subtree masses from the per-world masses (single-design and batched-design paths) at every reset. After the fix LQR alone under +-37 % mass randomisation: ball only 0.17 / 0.23 (error 0.019 / 0.023, achieved 1.02 / 1.00), both masses with two designs 0.17 / 0.17.
Superseded (all domain-randomised on the old engine, training or evaluation): every result that used `--dr` (all goal-task and tile-velocity evaluations, S37 onward including S39 to S48, S51, S54, S57, S60, S62 to S67; S68 conclusion "mass is the driver" is an artefact); all policies trained before this entry (g2 to g11) were trained with the artefact. Not affected: results with randomisation off (flat-world tracking benchmark S55, S59 and the flat rows of S60, S62, S66, S67; LQR calibration S56; the encoder pretraining data are affected only through the policy that collected them). The training runs g10_softdr_s0, g11_dr25_s0 and g9_dr_s2 were stopped. Re-run plan on the fixed engine: g12_* (LQR + residual 0.5 at standard randomisation, 1.75 x, soft base at 1.75 x, pure-learning baseline), then the evaluation battery, the randomisation ablation, and re-evaluation of the older policies for reference.

## S70 CORRECTION 2 of the stale subtree-mass artefact: the evaluation path (set_design) and the first corrected result
The first fix (S69, `_subtree()` in `reset`) made the training physics consistent, but `GpuBallbot.set_design` (used by `eval_tiles2.py` and `render_policy.py` to assign designs after the reset) copied each design's nominal `body_subtreemass` over the corrected value, so every randomised evaluation still carried the artefact. Fixed: `set_design` recomputes the subtree masses from the randomised masses. Every evaluation file written before this entry with randomisation on is invalid, including the first g12 battery (`g12_res_s0`: goal success 0.39 at scale 1, 0.12 at 2.5) and the ablation in the old `ablate_dr_2.5.json` (all on 0.11, no mass 0.47). The training runs themselves (g12_*) used the correct `reset` path and are valid.
Corrected ablation (g12_res_s0: LQR base, residual 0.5, tracking x2, half goals, cross-focus 0.6, 11 designs, 200M steps, trained at the standard randomisation ranges; Warp; goal task undisturbed (no pushes, no pulls), DR + surfaces, DR scale 2.5, 176 worlds per kind, seed 0; ablate_dr.py):
| setting | falls | success |
|---|---|---|
| all randomisation on (scale 2.5) | 0.54 | 0.46 |
| no mass | 0.53 | 0.47 |
| no friction | 0.51 | 0.49 |
| no torque | 0.52 | 0.48 |
| no damping | 0.51 | 0.49 |
| no latency | 0.49 | 0.50 |
| no sensor noise | 0.53 | 0.46 |
| all off (ceiling) | 0.40 | 0.60 |
Reading: on the corrected engine no single randomisation term dominates (removing any one moves success by 0.00 to 0.04); the whole scale 2.5 randomisation (mass +-37 %, friction, torque, damping, latency, noise) costs 0.14 success (0.60 to 0.46). A policy trained at the standard ranges is already robust to randomisation well beyond them. S67 and S68 (wide training does not help; mass is the driver) were artefacts of the evaluation path and are void. Next: full evaluation batteries for g12_res_s0, g12_dr175_s0, g12_soft175_s0 on the corrected path (eval_battery.py at DR scales 1, 1.75, 2.5), the pure-learning baseline g12_pure_s0 when it finishes, second seeds g12_res_s1, g12_dr175_s1.

## S71 First full evaluation on the corrected engine: LQR + residual (g12_res_s0: LQR base, residual authority 0.5, tracking x2, half goal tasks, cross-focus 0.6, 11 designs, trained at the standard randomisation ranges, 200M steps, seed 0; Warp; eval_battery.py after the S69 and S70 fixes; goal task 176 worlds per kind (16 per design), flat benchmark 64 worlds, seed 0)
| setting | goal task falls / success / median time | step_up / stairs / bump / ramp_up success | vel 0.3 on tiles: falls / error |
|---|---|---|---|
| DR scale 1, no pushes or pulls | 0.42 / 0.58 / 2.61 s | 0.33 / 0.38 / 0.17 / 0.74 | 0.55 / 0.112 |
| DR scale 1.75, no pushes or pulls | 0.48 / 0.52 / 2.57 s | 0.24 / 0.37 / 0.19 / 0.60 | 0.57 / 0.111 |
| DR scale 2.5, no pushes or pulls | 0.55 / 0.45 / 2.49 s | 0.27 / 0.32 / 0.12 / 0.50 | 0.66 / 0.115 |
| DR scale 1, pushes + pulls | 0.75 / 0.25 / 2.40 s | 0.15 / 0.22 / 0.11 / 0.30 | 0.72 / 0.218 |
| DR scale 1.75, pushes + pulls | 0.77 / 0.23 / 2.39 s | 0.15 / 0.18 / 0.06 / 0.34 | 0.75 / 0.225 |
| DR scale 2.5, pushes + pulls | 0.82 / 0.18 / 2.30 s | 0.10 / 0.15 / 0.09 / 0.24 | 0.80 / 0.241 |
Flat world, clean, nominal design, no randomisation (64 worlds): error 0.035 / 0.056 / 0.067, achieved speed 0.93 / 0.95 / 0.98 of the command, falls 0.03 / 0.11 / 0.20 at 0.3 / 0.6 / 0.9 m/s.
Reading: on the corrected engine the LQR + residual policy reaches goal success 0.58 undisturbed (0.25 with pushes and pulls), crosses ramps 0.74, stairs 0.38, steps 0.33, in 2.6 s, and degrades gracefully when every randomisation range is widened (0.58, 0.52, 0.45 at 1, 1.75, 2.5 times the standard ranges). Tracking stays at 93 to 98 percent of the command. All earlier goal-task numbers with randomisation on (S37 to S67) are superseded by these. Pending on the corrected engine: g12_dr175_s0 and g12_soft175_s0 batteries (running), the pure-learning baseline g12_pure_s0 (300M steps), second seeds g12_res_s1 and g12_dr175_s1, held-out terrain seeds and held-out designs.

## S72 Corrected-engine batteries for the three LQR-residual recipes (seed 0, 200M steps, 11 designs, goal task 176 worlds per kind, flat benchmark 64 worlds, eval_battery.py after S69 and S70; recipes: A = LQR base, residual 0.5, trained at the standard ranges; B = same, randomisation ranges ramped to 1.75 x during training; C = soft LQR base (R x3, reference acceleration 0.3) trained with the 1.75 x ramp)
| measure | A: g12_res_s0 | B: g12_dr175_s0 | C: g12_soft175_s0 |
|---|---|---|---|
| goal, undisturbed, DR scale 1 / 1.75 / 2.5: success | 0.58 / 0.52 / 0.45 | 0.58 / 0.54 / 0.45 | 0.63 / 0.57 / 0.49 |
| goal, undisturbed: falls | 0.42 / 0.48 / 0.55 | 0.42 / 0.46 / 0.55 | 0.37 / 0.43 / 0.51 |
| goal, pushes + pulls, scale 1 / 1.75 / 2.5: success | 0.25 / 0.23 / 0.18 | 0.29 / 0.25 / 0.19 | 0.33 / 0.29 / 0.23 |
| goal, median time to the goal, scale 1 undisturbed | 2.61 s | 2.44 s | 3.06 s |
| goal, scale 1 undisturbed: step_up / stairs / bump / ramp_up | 0.33 / 0.38 / 0.17 / 0.74 | 0.31 / 0.44 / 0.26 / 0.65 | 0.35 / 0.39 / 0.16 / 0.75 |
| flat, clean, err at 0.3 / 0.6 / 0.9 | 0.035 / 0.056 / 0.067 | 0.030 / 0.048 / 0.062 | 0.031 / 0.067 / 0.099 |
| flat, achieved / command | 0.93 / 0.95 / 0.98 | 0.96 / 0.98 / 1.00 | 0.93 / 0.93 / 0.96 |
| flat falls at 0.3 / 0.6 / 0.9 | 0.03 / 0.11 / 0.20 | 0.06 / 0.19 / 0.20 | 0.08 / 0.14 / 0.16 |
| vel 0.3 on tiles: falls undisturbed (scale 1 / 2.5), disturbed (scale 1) | 0.55 / 0.66, 0.72 | 0.54 / 0.62, 0.71 | 0.55 / 0.65, 0.71 |
Reading: training over wider randomisation (B) changes almost nothing on the corrected engine (differences of 0.00 to 0.04, within seed noise): the standard-range policy (A) is already robust. The soft LQR base (C) gives the best goal success (0.63 undisturbed, 0.33 with pushes and pulls) at the cost of a slower approach (3.1 s) and slightly less accurate tracking at 0.9 m/s (0.099 vs 0.062 to 0.067). No recipe fixes the velocity task on tiles (falls 0.55 undisturbed, 0.71 disturbed at 0.3 m/s): its falls come from the hard terrain kinds and disturbances, not from the parameters. One seed each; seed-1 replicates g12_res_s1 and g12_dr175_s1 and the pure-learning baseline g12_pure_s0 pending.

## S73 Seed replicate of recipe A on the corrected engine (g12_res_s0 and g12_res_s1: LQR base, residual 0.5, tracking x2, half goal tasks, cross-focus 0.6, 11 designs, trained at the standard randomisation ranges, 200M steps; Warp; eval_battery.py; goal task 176 worlds per kind, flat benchmark 64 worlds)
| measure | seed 0 | seed 1 |
|---|---|---|
| goal, undisturbed, DR scale 1 / 1.75 / 2.5: success (falls) | 0.58 (0.42) / 0.52 (0.48) / 0.45 (0.55) | 0.58 (0.42) / 0.54 (0.46) / 0.44 (0.56) |
| goal, pushes + pulls, scale 1 / 1.75 / 2.5: success (falls) | 0.25 (0.75) / 0.23 (0.77) / 0.18 (0.82) | 0.27 (0.73) / 0.25 (0.75) / 0.19 (0.81) |
| goal, median time, scale 1 undisturbed | 2.61 s | 2.43 s |
| goal, scale 1 undisturbed: step_up / stairs / bump / ramp_up | 0.33 / 0.38 / 0.17 / 0.74 | 0.28 / 0.39 / 0.20 / 0.69 |
| flat, clean: err at 0.3 / 0.6 / 0.9 | 0.035 / 0.056 / 0.067 | 0.027 / 0.040 / 0.050 |
| flat: achieved / command | 0.93 / 0.95 / 0.98 | 0.97 / 0.98 / 0.99 |
| flat falls at 0.3 / 0.6 / 0.9 | 0.03 / 0.11 / 0.20 | 0.05 / 0.11 / 0.34 |
| vel 0.3 on tiles: falls (error) undisturbed scale 1 / 2.5, disturbed scale 1 | 0.55 (0.112) / 0.66 (0.115), 0.72 (0.218) | 0.55 (0.104) / 0.65 (0.111), 0.74 (0.205) |
The two seeds agree: goal success within 0.00 to 0.02 at every randomisation scale, with and without disturbances, per-terrain success within 0.05, velocity-task falls identical (0.55). The flat 0.9 m/s falls differ (0.20 vs 0.34; 64 worlds, binomial standard error about 0.05). Seed spread of the headline goal-task numbers on the corrected engine: about 0.02. Pending: second seed for the other two recipes and the pure-learning baseline.

## S74 Corrected-engine batteries: DR-1.75 recipe seed 1, soft recipe seeds 0 and 1, authority 0.75, pure learning (all engine: MuJoCo-Warp float32, GPU, 100 Hz, stale-subtree-mass fix in place)
Protocol: eval_battery.py, DR on (scale 1 / 1.75 / 2.5), surfaces on, sensor noise on, 16 worlds per design and terrain kind, 11 designs. "Disturbed" = pushes + pulls on. Flat rows: nominal design, DR off, noise off, 64 worlds. Runs: g12_dr175 (LQR base 0.5 authority, R 1, a_max 0.5, trained at DR scale 1.75), g12_soft175 (R x3, a_max 0.3), g13_softrs075 (soft, authority 0.75), g12_pure (no LQR, same training otherwise, 300M). Seeds 0 and 1 where stated; 200M steps each.
| run | goal success, undisturbed 1 / 1.75 / 2.5 | goal success, disturbed 1 / 1.75 / 2.5 | median goal time | flat err 0.3 / 0.6 / 0.9 | vel 0.3 on tiles: falls undisturbed / disturbed (scale 1) |
|---|---|---|---|---|---|
| g12_dr175 s0 | 0.58 / 0.54 / 0.45 | 0.29 / 0.25 / 0.19 | 2.4 s | 0.03 / 0.05 / 0.06 | 0.54 / 0.71 |
| g12_dr175 s1 | 0.60 / 0.56 / 0.47 | 0.30 / 0.25 / 0.19 | 2.3 s | 0.02 / 0.03 / 0.04 | 0.53 / 0.71 |
| g12_soft175 s0 | 0.63 / 0.57 / 0.49 | 0.33 / 0.29 / 0.23 | 3.0 s | 0.03 / 0.07 / 0.10 | 0.55 / 0.71 |
| g12_soft175 s1 | 0.62 / 0.58 / 0.52 | 0.34 / 0.29 / 0.24 | 3.0 s | 0.05 / 0.09 / 0.13 | 0.53 / 0.73 |
| g13_softrs075 s0 | 0.58 / 0.53 / 0.45 | 0.32 / 0.26 / 0.21 | 3.3 s | 0.06 / 0.16 / 0.19 | 0.54 / 0.70 |
| g12_pure s0 | 0.44 / 0.43 / 0.37 | 0.26 / 0.23 / 0.18 | 3.1 s | 0.13 / 0.25 / 0.40 | 0.32 / 0.54 |
Findings. (1) Seed replicates agree to about 0.02 (dr175, soft175). (2) The soft base gives a small, consistent gain in goal success (about +0.03 to +0.05 undisturbed, +0.04 disturbed at scale 1) at the cost of speed (3.0 s against 2.4 s) and a little flat accuracy. (3) Authority 0.75 is no better than 0.5 and tracks worse on the flat floor. (4) Pure learning tracks poorly on the flat floor (achieved/command about 0.6) and reaches the goal less often undisturbed (0.44 against 0.58-0.63); under disturbance all recipes are close (0.26 to 0.34). Its lower velocity-task fall rate comes with slow, inaccurate tracking, so it is not a robustness advantage. (5) Velocity task on tiles and disturbance robustness (goal success about 0.3 with pushes and pulls) are unchanged by any recipe so far. Pure learning is one seed.

## S75 Started: disturbance-focused variants of the 1.0-R recipe (running, not yet evaluated)
g14_fp5 (fall penalty 5.0 instead of 2.0), g14_softfp5 (soft base plus fall penalty 5.0), g14_pramp (push and pull strength ramped from 0.2 to 1 over the first half of training). Seed 0, 200M steps, same 11 designs and DR scale 1.75. Each is a one-variable change against g12_dr175 or g12_soft175; results go in the next entry once the batteries have run.

## S76 Disturbance-focused variants and held-out terrain (engine: MuJoCo-Warp float32 GPU, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 16 worlds per design and terrain kind, seed 0 per variant; disturbed = pushes + pulls)
Each variant is a one-variable change against g12_dr175 (LQR base, R 1, a_max 0.5) or g12_soft175 (R x3, a_max 0.3), 200M steps.
| run | change | goal success undisturbed 1 / 1.75 / 2.5 | goal success disturbed 1 / 1.75 / 2.5 | flat err 0.3 / 0.6 / 0.9 | vel 0.3 on tiles: falls undisturbed / disturbed (scale 1) |
|---|---|---|---|---|---|
| g14_fp5 | fall penalty 5.0 (was 2.0) | 0.57 / 0.53 / 0.46 | 0.27 / 0.25 / 0.18 | 0.03 / 0.05 / 0.05 | 0.54 / 0.73 |
| g14_softfp5 | soft base + fall penalty 5.0 | 0.63 / 0.59 / 0.50 | 0.33 / 0.27 / 0.22 | 0.03 / 0.06 / 0.10 | 0.54 / 0.71 |
| g14_pramp | push and pull strength ramped 0.2 to 1 over first half | 0.57 / 0.52 / 0.47 | 0.25 / 0.21 / 0.16 | 0.03 / 0.05 / 0.07 | 0.54 / 0.73 |
Reference (S74): g12_dr175 0.58-0.60 / 0.54-0.56 / 0.45-0.47 undisturbed, 0.29-0.30 / 0.25 / 0.19 disturbed; g12_soft175 0.62-0.63 / 0.57-0.58 / 0.49-0.52 undisturbed, 0.33-0.34 / 0.29 / 0.23-0.24 disturbed. Findings: a higher fall penalty changes nothing outside the seed spread (about 0.02), on either base; the push curriculum is slightly worse disturbed (0.25 against 0.29). Disturbance robustness (goal success about 0.3 with pushes and pulls) and the tile velocity task (falls 0.54 / 0.72) are not reward-weighting or curriculum problems with this recipe.
Held-out terrain (tile seed 1, trained on seed 0), g12_dr175, DR scale 1, goal task, 176 worlds per terrain kind (1760 per cell):
| seed | undisturbed success | disturbed success | median time |
|---|---|---|---|
| 0 | 0.52 | 0.26 | 2.4 s / 2.4 s |
| 1 | 0.53 | 0.26 | 2.3 s / 2.2 s |
Against tile seed 0 (0.58-0.60 and 0.29-0.30) the held-out terrain costs about 0.06 undisturbed and 0.03-0.04 disturbed: the policy mostly transfers to unseen terrain layouts, with a small drop. File names carry the tile seed (_ts1).

## S77 Started: perception arm on the corrected engine (running, not yet evaluated)
g15_edge_depth_s0 (depth sensor 18x14, head mount) and g15_edge_blind_s0 (blind control), both g12_dr175 recipe plus edge-focused starts (--edge-focus 0.5), seed 0, 200M steps. The control separates the effect of edge starts from the effect of the sensor. No pretrained encoder (the earlier percep_v1 data came from a pre-fix policy and engine).

## S78 Perception arm on the corrected engine: depth sensor against blind control, both with edge-focused starts (engine: MuJoCo-Warp float32 GPU, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind, seed 0 per arm, one run each, 200M steps; disturbed = pushes + pulls)
Both arms are the g12_dr175 recipe plus --edge-focus 0.5. The depth arm adds the 18x14 head-mounted depth sensor (noisy, with dropout); the blind arm is identical without it.
| measure | depth | blind | g12_dr175 s0 / s1 (no edge starts, S74) |
|---|---|---|---|
| goal success undisturbed, scale 1 / 1.75 / 2.5 | 0.60 / 0.53 / 0.45 | 0.60 / 0.54 / 0.47 | 0.58-0.60 / 0.54-0.56 / 0.45-0.47 |
| goal success disturbed, scale 1 / 1.75 / 2.5 | 0.29 / 0.26 / 0.18 | 0.28 / 0.23 / 0.19 | 0.29-0.30 / 0.25 / 0.19 |
| median goal time (scale 1) | 2.36 s / 2.22 s | 2.30 s / 2.22 s | 2.3-2.4 s |
| flat err 0.3 / 0.6 / 0.9 (DR off, noise off, 64 worlds) | 0.020 / 0.032 / 0.057 | 0.021 / 0.035 / 0.056 | 0.02-0.03 / 0.03-0.05 / 0.04-0.06 |
| vel 0.3 on tiles: falls undisturbed / disturbed (scale 1) | 0.56 / 0.73 | 0.53 / 0.73 | 0.53-0.54 / 0.71 |
Finding: the depth sensor gives no gain over the blind control on any measure (differences within about 0.03, the size of the seed and evaluation spread; velocity-task falls undisturbed are 0.56 against 0.53, slightly worse with depth), and edge-focused starts alone do not move the baseline. Foresight from this sensor, as posed (reward unchanged, sensor features learned end to end), does not help. This confirms the earlier perception findings (S26, S40, S45, S49) on the corrected engine. One seed per arm. The depth battery was run twice by accident (the second run overwrote battery.json); the values above are from the file, and the two runs differed by up to 0.02 in goal success, which is a direct measure of evaluation noise at this world count.

## S79 Why the robot falls, and a Rezero-style ball retainer (engine: MuJoCo-Warp float32 GPU; goal task, DR on at scale 1, surfaces on, noise on, 1760 worlds = 16 per design and terrain kind, policy g12_dr175_s0 seed 0 unless stated)
Fall diagnosis (fall_cause.py). The model has two free bodies, the base (with the wheels) and the ball, held together only by gravity and the three wheel-ball contacts. Per world and step we track tilt, the offset angle (ball centre to base origin against vertical, the termination quantity, limit 30 deg) and the misalignment between the base axis and that offset direction (about 0.1 to 0.3 deg while the wheels sit on the ball). "Derailed" = misalignment above 5 deg.
| condition | fall rate | falls that end by the offset limit (tilt below 70) | derail before tilt 25 deg | median tilt when derailing | wheels at torque limit, last 1 s |
|---|---|---|---|---|---|
| undisturbed | 0.43 | 0.95 | 0.78 | 14.7 deg | 0.7 % |
| pushes + pulls | 0.72 | 0.97 | 0.49 | 15.4 deg | 0.4 % |
| pushes + pulls, LQR alone | 0.85 | 0.98 | (not separated) | | 0.4 % |
Finding: the robot does not fall because the motors run out of torque (saturation below 1 %) and rarely because it tips past 70 deg. In nearly all falls the body derails off the ball, usually at moderate tilt (median about 15 deg); survivors reach about 12 deg (90th percentile 17 deg, undisturbed) without derailing. Pulls are active in 58 % of disturbed falls, pushes in 3 %.
Retainer design (retainer.py, after the arrestors of ETH Zurich's Rezero ballbot, which per its project coverage hold the ball against the wheels; we found no design details, so this is our own design). Three passive spherical arrestors (radius 8 mm) on the base, one per wheel on the azimuth opposite the wheel, at polar angle phi from the top of the ball. Each is placed to just touch the ball at the LOADED seat (the model is settled under gravity first: the reset pose has the wheels 10 mm inside the ball, the loaded seat is 1 cm higher, with 2.2 mm wheel penetration), so the arrestors carry no force while the robot is seated and act only when the body starts to lift or slide. Contact forces on a sphere pass through its centre, so the arrestors add no torque about the ball centre and do not change the planar model the LQR uses. Frictionless (ideal ball-transfer unit) by default; massless (real units weigh about 10 g). Retainer runs start from the seated pose. Two implementation pitfalls were found and fixed before any result was kept: the settle must start with the ball resting on the floor (a drop wedged the wheels into the ball) and must run on a flat copy of the model (the first design's live model already carries the tile terrain). The Warp sphere / height-field pair produced contacts with 6 cm clearance (an artefact), so arrestor-floor contact is not simulated; floor clearance is checked geometrically instead: the bottom of an arrestor sits 6.5 cm above the floor at phi 100 (touches only beyond 57 deg tilt), 3.3 cm at phi 120, 1.2 cm at 135 deg (diametrically opposite the wheel contact; touches at about 21 deg tilt, so that placement is not buildable on this ball).
Zero-shot (policy trained without arrestors, evaluated with them):
| arrestor mount | goal success undisturbed | goal success pushes + pulls | misalignment at fall |
|---|---|---|---|
| none | 0.57 | 0.28 | 12 to 19 deg |
| 135 deg (opposite the wheel) | 0.66 | 0.37 | 4 to 7 deg |
| 120 deg | 0.69 | 0.39 | 3 to 5 deg |
| 110 deg | 0.72 | 0.40 | 3 to 4 deg |
| 100 deg | 0.75 | 0.45 | 2 to 3 deg |
| 95 deg | 0.75 | 0.43 | 2 to 3 deg |
| 90 deg (equator) | 0.76 | 0.44 | 2 to 3 deg |
Arrestors raise goal success by about 0.17 undisturbed and 0.16 with pushes and pulls before any retraining, the first change in this study that moves disturbance robustness. Mounts between 90 and 100 deg are equal within noise; 100 deg is chosen because only a mount below the equator can resist the body lifting off. Roller friction (repeated on the fixed settle, 100 deg mount): sliding friction 0 / 0.05 / 0.2 gives goal success 0.75 / 0.74 / 0.76 undisturbed and 0.45 / 0.42 / 0.42 with pushes and pulls, the same within noise: seated arrestors carry no load, so their drag barely acts. With the arrestors the remaining falls are mostly genuine tipping (the offset limit then acts as a 30 deg tilt limit). One seed of one policy, zero-shot; training with arrestors is running (g16_ret100_s0, s1).

## S80 Training with the ball retainer, and the second pure-learning seed (engine: MuJoCo-Warp float32 GPU, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind; flat rows: nominal design, DR off, noise off, 64 worlds; disturbed = pushes + pulls)
g16_ret100_s0 / s1: the g12_dr175 recipe (LQR base, authority 0.5, tracking x2, trained at DR scale 1.75, 200M steps) trained and evaluated with the 100 deg arrestors of S79 (RETAINER=0,0,0,100). Reference rows from S74 and S79.
| policy | arrestors | goal success undisturbed 1 / 1.75 / 2.5 | goal success disturbed 1 / 1.75 / 2.5 | vel 0.3 on tiles falls undisturbed / disturbed (scale 1) | flat err 0.3 / 0.6 / 0.9 | flat achieved / command |
|---|---|---|---|---|---|---|
| g12_dr175 s0, s1 | none | 0.58-0.60 / 0.54-0.56 / 0.45-0.47 | 0.29-0.30 / 0.25 / 0.19 | 0.53-0.54 / 0.71 | 0.02-0.03 / 0.03-0.05 / 0.04-0.06 | 0.96-1.00 |
| g12_dr175 s0 | added at evaluation (zero-shot) | 0.75 (scale 1) | 0.45 (scale 1) | 0.36 / 0.53 | 0.03 / 0.05 / 0.07 | 0.96-0.99 |
| g16_ret100 s0 | trained with | 0.71 / 0.70 / 0.62 | 0.46 / 0.42 / 0.35 | 0.31 / 0.49 | 0.06 / 0.15 / 0.19 | 0.80-0.86 |
| g16_ret100 s1 | trained with | 0.76 / 0.73 / 0.67 | 0.49 / 0.44 / 0.37 | 0.32 / 0.50 | 0.05 / 0.11 / 0.12 | 0.86-0.90 |
| g12_pure s0 | none | 0.44 / 0.43 / 0.37 | 0.26 / 0.23 / 0.18 | 0.32 / 0.54 | 0.13 / 0.25 / 0.40 | 0.57-0.60 |
| g12_pure s1 | none | 0.51 / 0.44 / 0.39 | 0.27 / 0.23 / 0.19 | 0.38 / 0.57 | 0.11 / 0.25 / 0.42 | 0.54-0.66 |
Findings. (1) The retainer is the first change that improves every robustness measure at once: goal success +0.15 undisturbed and +0.17 to +0.19 with pushes and pulls at scale 1, and the tile velocity task, unchanged by every earlier recipe, falls from 0.53 to 0.31 undisturbed and from 0.71 to 0.49 disturbed. Both seeds agree within 0.05. (2) The gain grows with randomisation: at scale 2.5 undisturbed success is 0.62-0.67 against 0.45-0.47, disturbed 0.35-0.37 against 0.19. (3) Most of the gain is mechanical: the policy trained without arrestors, given them at evaluation, reaches the same goal success at scale 1 (0.75 / 0.45) with unchanged tracking; training with them adds a little on the velocity task (0.31 against 0.36 undisturbed). (4) Training with the arrestors costs flat tracking accuracy (achieved speed 80 to 90 percent of the command against 96 to 100), on both seeds; the arrestors themselves do not (zero-shot row). A tracking-weight test (g17_ret100_trk3, tracking x3) is running. (5) The second pure-learning seed confirms S74: pure learning tracks at 54 to 66 percent of the command and reaches the goal less often undisturbed (0.44 to 0.51 against 0.58 to 0.60 for the LQR residual); under disturbance all no-arrestor recipes are close (0.26 to 0.30). A pure-learning run with arrestors (g16_pure_ret100_s0, 300M) is running.
S80 addendum, held-out terrain (tile seed 1, trained on seed 0), goal task, DR scale 1, 176 worlds per terrain kind: g16_ret100 s0 / s1 reach 0.66 / 0.68 undisturbed and 0.42 / 0.43 with pushes and pulls (training map: 0.71-0.76 and 0.46-0.49). Without arrestors, g12_dr175 on the same held-out map: 0.52-0.53 and 0.26 (S76). The retainer gain holds on unseen terrain (+0.14 undisturbed, +0.16 disturbed).

## S81 Retainer with pure learning, and the tracking cost (engine: MuJoCo-Warp float32 GPU, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind; flat rows: nominal design, DR off, noise off, 64 worlds; disturbed = pushes + pulls; one seed per row unless stated)
| policy | arrestors | goal success undisturbed 1 / 1.75 / 2.5 | goal success disturbed 1 / 1.75 / 2.5 | goal falls undisturbed / disturbed (scale 1) | vel 0.3 on tiles falls undisturbed / disturbed (scale 1) | flat err 0.3 / 0.6 / 0.9 | flat achieved / command |
|---|---|---|---|---|---|---|---|
| g12_pure s0 | none | 0.44 / 0.43 / 0.37 | 0.26 / 0.23 / 0.18 | 0.32 / 0.62 | 0.32 / 0.54 | 0.13 / 0.25 / 0.40 | 0.57-0.60 |
| g12_pure s0 | added at evaluation | 0.54 (scale 1) | 0.35 (scale 1) | 0.19 / 0.47 | | | |
| g16_pure_ret100 s0 | trained with | 0.70 / 0.67 / 0.64 | 0.50 / 0.45 / 0.40 | 0.14 / 0.40 | 0.16 / 0.34 | 0.09 / 0.20 / 0.32 | 0.66-0.76 |
| g16_ret100 s0, s1 (LQR residual) | trained with | 0.71-0.76 / 0.70-0.73 / 0.62-0.67 | 0.46-0.49 / 0.42-0.44 / 0.35-0.37 | 0.24-0.28 / 0.51-0.53 | 0.31-0.32 / 0.49-0.50 | 0.05-0.06 / 0.11-0.15 / 0.12-0.19 | 0.80-0.90 |
| g17_ret100_trk3 s0 (LQR residual, tracking x3) | trained with | 0.71 / 0.68 / 0.62 | 0.49 / 0.43 / 0.36 | 0.28 / 0.51 | 0.32 / 0.46 | 0.04 / 0.17 / 0.19 | 0.76-0.89 |
| g12_dr175 s0 (LQR residual) | added at evaluation | 0.75 (scale 1) | 0.45 (scale 1) | | 0.36 / 0.53 | 0.03 / 0.05 / 0.07 | 0.96-0.99 |
Findings. (1) The retainer helps pure learning more than the LQR residual: goal success 0.44 to 0.70 undisturbed, 0.26 to 0.50 disturbed, and the fewest falls of any controller on both tasks (goal 0.14 / 0.40, tile velocity 0.16 / 0.34). (2) For pure learning, training with the arrestors matters: adding them at evaluation gives only part of the gain (0.54 / 0.35), unlike the LQR residual, where the zero-shot gain is the whole gain. (3) A stronger tracking weight (x3) does not recover the tracking accuracy lost when the residual is trained with arrestors (achieved 0.76 to 0.89 of the command), so the loss is not a reward-weighting problem. (4) Trade-off today: the most robust controller is pure learning trained with arrestors (loose tracking, 66 to 76 percent of the command); the most accurate robust one is the residual trained without arrestors and run with them (96 to 99 percent, goal 0.75 / 0.45). Running: g16_pure_ret100_s1 (second seed), g18_ret_rs025_s0 (authority 0.25 with arrestors), g18_ret_soft_s0 (soft LQR base with arrestors).

## S82 Tracking accuracy against robustness with the retainer: learned authority and base controller (engine: MuJoCo-Warp float32 GPU, all trained and evaluated with the 100 deg arrestors, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind; flat rows: nominal design, DR off, noise off, 64 worlds; disturbed = pushes + pulls; seed 0 unless stated)
| controller | flat achieved / command | flat err 0.9 | goal success undisturbed 1 / 1.75 / 2.5 | goal success disturbed 1 / 1.75 / 2.5 | vel 0.3 tiles falls undisturbed / disturbed (scale 1) |
|---|---|---|---|---|---|
| LQR + residual, authority 0.25 (g18_ret_rs025) | 1.00-1.01 | 0.043 | 0.71 / 0.68 / 0.60 | 0.42 / 0.36 / 0.29 | 0.34 / 0.55 |
| LQR + residual, authority 0.5 (g16_ret100, s0 and s1) | 0.80-0.90 | 0.12-0.19 | 0.71-0.76 / 0.70-0.73 / 0.62-0.67 | 0.46-0.49 / 0.42-0.44 / 0.35-0.37 | 0.31-0.32 / 0.49-0.50 |
| soft LQR (R x3, a_max 0.3) + residual 0.5 (g18_ret_soft) | 0.68-0.77 | 0.27 | 0.75 / 0.75 / 0.66 | 0.51 / 0.47 / 0.39 | 0.29 / 0.49 |
| pure learning (g16_pure_ret100, s0 and s1) | 0.66-0.83 | 0.30-0.32 | 0.70-0.74 / 0.67-0.73 / 0.64-0.67 | 0.50 / 0.45-0.46 / 0.40 | 0.16-0.19 / 0.34-0.38 |
Held-out terrain (tile seed 1, goal, scale 1): pure learning with arrestors 0.66 / 0.71 undisturbed and 0.46 / 0.47 disturbed (s0 / s1), against 0.42-0.43 disturbed for the residual at authority 0.5 and 0.26 without arrestors.
Finding: with the retainer there is a clear trade-off between speed-tracking accuracy and disturbance robustness, set by how much the learned part may override the firm LQR. A small residual (authority 0.25) keeps the LQR's tracking exactly (achieved 1.00 of the command, error 0.04 m/s at 0.9) and still gains most of the robustness (goal 0.71 / 0.42 against 0.58-0.60 / 0.29-0.30 without arrestors); looser controllers (authority 0.5, soft LQR, pure learning) gain a further 0.05 to 0.10 under disturbance and at wide randomisation but reach only 66 to 90 percent of the commanded speed. The tracking loss comes from the controller, not the arrestors (S80 zero-shot row: 0.96-0.99). Running: authority 0.25 seed 1, authority 0.35 and 0.15, to map the curve and confirm the authority 0.25 point.

## S83 The authority curve with the retainer, and the recommended controller (engine: MuJoCo-Warp float32 GPU, all trained and evaluated with the 100 deg arrestors, LQR base R 1, a_max 0.5, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind; flat rows: nominal design, DR off, noise off, 64 worlds; disturbed = pushes + pulls)
| learned authority | runs | flat achieved / command | flat err 0.3 / 0.6 / 0.9 | goal success undisturbed 1 / 1.75 / 2.5 | goal success disturbed 1 / 1.75 / 2.5 | median goal time |
|---|---|---|---|---|---|---|
| 0.15 | g18_ret_rs015 s0 | 1.01-1.02 | 0.024 / 0.026 / 0.046 | 0.69 / 0.65 / 0.56 | 0.35 / 0.31 / 0.22 | 2.2 s |
| 0.25 | g18_ret_rs025 s0 | 1.00-1.01 | 0.018 / 0.029 / 0.043 | 0.71 / 0.68 / 0.60 | 0.42 / 0.36 / 0.29 | 2.4 s |
| 0.25 | g18_ret_rs025 s1 | 0.99-1.01 | 0.031 / 0.038 / 0.039 | 0.76 / 0.72 / 0.66 | 0.43 / 0.39 / 0.32 | 2.5 s |
| 0.35 | g18_ret_rs035 s0 | 0.92-0.97 | 0.034 / 0.058 / 0.071 | 0.72 / 0.71 / 0.63 | 0.45 / 0.40 / 0.32 | 2.7 s |
| 0.5 | g16_ret100 s0, s1 | 0.80-0.90 | 0.05-0.06 / 0.11-0.15 / 0.12-0.19 | 0.71-0.76 / 0.70-0.73 / 0.62-0.67 | 0.46-0.49 / 0.42-0.44 / 0.35-0.37 | 2.7 s |
Finding: tracking accuracy falls as the learned authority rises above 0.25, and disturbance robustness rises with it but saturates; below 0.25 robustness drops (0.35 disturbed at authority 0.15) with no tracking gain. Authority 0.25 with the arrestors is the knee and our recommended controller for the goal (robust, goal-directed, velocity-tracking): exact speed tracking (achieved 0.99 to 1.01 of the command, error at most 0.04 m/s up to 0.9 m/s), goal success 0.71-0.76 undisturbed and 0.42-0.43 with pushes and pulls (no arrestors: 0.58-0.60 and 0.29-0.30), on two seeds. A third seed (g18_ret_rs025_s2) and a 400M-step run (g19_ret_rs025_400m_s0) are running. Known weakness (single clip, not a statistic): at 0.6 m/s up an 11 deg ramp the controller lags the command (0.40 m/s at 2.8 s) and falls at 5.6 s.
S83 addendum, where the recommended controller still falls (g18_ret_rs025 s0 / s1, velocity 0.3 m/s on tiles, undisturbed, DR scale 1, 176 worlds per kind): flat 0.08 / 0.07, ramp up 0.12 / 0.14, ramp down 0.20 / 0.26, rough 0.17 / 0.16, step down 0.16 / 0.16, small drop 0.18 / 0.17, stairs 0.48 / 0.52, step up 0.61 / 0.62, large drop 0.64 / 0.66, bump 0.77 / 0.74. The remaining falls are concentrated on discrete obstacles (bumps, step ups, large drops, stairs); ramps at 0.3 m/s are mostly handled. The 11 deg ramp clip at 0.6 m/s (S83) is a single run at twice the speed and is not representative of ramps at 0.3 m/s. This is where foresight (an explicit obstacle-aware speed layer) is still needed.
S83 addendum, held-out robot designs (never trained on: s1.5 = robot scaled 1.5x, tau3 = lower wheel torque limit; goal task, DR scale 1, 160 worlds per design = 16 per terrain kind, seed 1; files eval2_goal_goal_dr_surf*_des-s1.5-tau3.json):
| policy | undisturbed overall (s1.5 / tau3) | pushes + pulls overall (s1.5 / tau3) |
|---|---|---|
| g18_ret_rs025 s1 (recommended, arrestors) | 0.49 (0.68 / 0.30) | 0.42 (0.64 / 0.20) |
| g12_dr175 s1 (no arrestors) | 0.44 (0.70 / 0.18) | 0.33 (0.59 / 0.08) |
The larger robot transfers well with or without arrestors (0.64 to 0.70); the low-torque design is the hard case (0.08 to 0.30), where the arrestors help most (0.20 against 0.08 disturbed). Held-out designs cost more than held-out terrain: overall success drops from 0.76 / 0.43 (training designs) to 0.49 / 0.42, almost all of it on tau3. One seed per policy, two held-out designs.

## S84 Depth sensor on the recommended controller, and three seeds of the recommended controller (engine: MuJoCo-Warp float32 GPU, arrestors at 100 deg, LQR + residual authority 0.25, DR on at scales 1 / 1.75 / 2.5, surfaces on, noise on, 176 worlds per terrain kind; flat: nominal design, DR off, noise off, 64 worlds; disturbed = pushes + pulls)
| run | sensor | flat achieved / command | goal undisturbed 1 / 1.75 / 2.5 | goal disturbed 1 / 1.75 / 2.5 | vel 0.3 tiles falls undisturbed / disturbed (scale 1) |
|---|---|---|---|---|---|
| g18_ret_rs025 s0 | none | 1.00-1.01 | 0.71 / 0.68 / 0.60 | 0.42 / 0.36 / 0.29 | 0.34 / 0.55 |
| g18_ret_rs025 s1 | none | 0.99-1.01 | 0.76 / 0.72 / 0.66 | 0.43 / 0.39 / 0.32 | |
| g18_ret_rs025 s2 | none | 0.90-0.99 | 0.76 / 0.71 / 0.63 | 0.44 / 0.41 / 0.32 | |
| g20_ret_rs025_depth s0 | depth 18x14, edge-focused starts | 0.96-0.98 | 0.74 / 0.69 / 0.63 | 0.43 / 0.38 / 0.31 | 0.35 / 0.54 |
Per obstacle, vel 0.3 undisturbed falls (depth / blind s0 / blind s1): step up 0.55 / 0.61 / 0.62, large drop 0.68 / 0.64 / 0.66, bump 0.75 / 0.77 / 0.74, stairs 0.51 / 0.48 / 0.52, flat 0.08 / 0.08 / 0.07.
Findings. (1) Three seeds of the recommended controller agree: goal success 0.71 to 0.76 undisturbed and 0.42 to 0.44 with pushes and pulls (no arrestors: 0.58-0.60 and 0.29-0.30). (2) The depth sensor adds nothing on top of the arrestors, overall or on the obstacles where the remaining falls are (differences within seed spread, except a possible 0.06 gain on step up). This repeats S78 with the retainer: foresight from a raw depth sensor, learned end to end with an unchanged reward, does not reduce obstacle falls. One seed for the depth arm.
S84 addendum, training length: g19_ret_rs025_400m_s0 (the recommended controller trained 400M steps instead of 200M, seed 0): goal success 0.75 / 0.70 / 0.64 undisturbed and 0.43 / 0.36 / 0.30 with pushes and pulls at DR scales 1 / 1.75 / 2.5; tile velocity falls 0.38 undisturbed, 0.55 disturbed; flat achieved 0.89-1.00 of the command, error 0.06-0.07 m/s. Within the spread of the three 200M seeds on every measure: twice the training gives no gain, 200M steps is enough for this recipe.
S84 addendum, held-out terrain for the recommended controller (tile seed 1, trained on seed 0, goal, DR scale 1, 176 worlds per terrain kind): g18_ret_rs025 s0 / s1 reach 0.66 / 0.69 undisturbed and 0.38 / 0.39 with pushes and pulls (training map 0.71-0.76 and 0.42-0.43; no arrestors on the same held-out map 0.52-0.53 and 0.26). The gain over no arrestors holds on unseen terrain (+0.15 undisturbed, +0.12 disturbed), with the usual small held-out drop.

## S85 Generalisation to unseen robot designs: LQR base against pure learning, and training on the low-torque design (engine: MuJoCo-Warp float32 GPU, arrestors at 100 deg, goal task, DR scale 1, surfaces on, noise on, 160 worlds per design = 16 per terrain kind; s1.5 = robot scaled 1.5x, tau3 = lower wheel torque limit)
| policy | trained on tau3 | undisturbed: s1.5 / tau3 | pushes + pulls: s1.5 / tau3 |
|---|---|---|---|
| g18_ret_rs025 s1 (LQR + residual 0.25) | no | 0.68 / 0.30 | 0.64 / 0.20 |
| g16_pure_ret100 s1 (pure learning) | no | 0.28 / 0.00 | 0.24 / 0.00 |
| g12_dr175 s1 (LQR + residual 0.5, no arrestors) | no | 0.70 / 0.18 | 0.59 / 0.08 |
| g21_ret_rs025_tau3 s0 (LQR + residual 0.25) | yes (12 designs) | 0.83 / 0.28 | 0.81 / 0.16 |
Findings. (1) The LQR base is what carries a policy to robot designs it never saw: pure learning with arrestors, the most robust controller on its training designs (S81), collapses on unseen designs (0.28 on the larger robot, 0.00 on the low-torque one), while the LQR-based controllers keep 0.6 to 0.7 on the larger robot. For a real robot whose parameters are only approximately known, this decides for the LQR + residual structure. (2) The low-torque design is hard for every controller (at most 0.30) and adding it to training does not improve it (0.28 trained against 0.30 held out): the limit is the motor, not the policy. Training on 12 designs did raise the held-out larger robot to 0.83 (one seed; second seed running). Values from eval2_goal_goal_dr_surf[_pull]_des-s1.5-tau3.json in each run folder.
S85 addendum, second seed of training on 12 designs (g21_ret_rs025_tau3_s1): held-out larger robot s1.5 0.80 undisturbed and 0.63 with pushes and pulls, tau3 0.28 / 0.15 (seed 0: 0.83 / 0.81 and 0.28 / 0.16). Both seeds raise the unseen larger robot undisturbed (0.80-0.83 against 0.68 for the 11-design controller); with pushes and pulls the gain is seed dependent (0.63 to 0.81 against 0.64). The low-torque design stays at about 0.28 / 0.15 on both seeds, confirming the motor limit. On its 12 training designs the controller keeps 0.67-0.71 undisturbed and 0.40-0.41 disturbed at DR scale 1, with flat tracking 0.96-1.00 of the command.

## S86 How obstacle falls depend on approach speed (recommended controller g18_ret_rs025_s1, arrestors on; velocity task on tiles, undisturbed, DR scale 1, surfaces on, noise on, 176 worlds per terrain kind; evaluation only)
| commanded speed (m/s) | all tiles | flat | ramp up | stairs | step up | bump | large drop |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.17 | 0.12 | 0.13 | 0.21 | 0.16 | 0.16 | 0.40 |
| 0.20 | 0.26 | 0.08 | 0.16 | 0.32 | 0.41 | 0.44 | 0.62 |
| 0.30 | 0.35 | 0.07 | 0.14 | 0.52 | 0.62 | 0.74 | 0.66 |
| 0.45 | 0.40 | 0.12 | 0.14 | 0.51 | 0.59 | 0.77 | 0.77 |
Finding: obstacle falls are strongly speed dependent. Crossing at 0.1 m/s instead of 0.3 cuts step-up falls from 0.62 to 0.16, bump falls from 0.74 to 0.16 and stairs from 0.52 to 0.21; large drops improve least (0.66 to 0.40). Flat and ramps are speed independent in this range. So the current controller can cross most obstacles if it approaches slowly: an obstacle-aware speed layer (slow down only near a feature, keep the commanded speed elsewhere) could remove most of the remaining terrain falls without retraining the balance controller. This bounds the expected gain of that experiment; it is not the experiment itself. One seed.

## S87 Correction to S86: obstacle falls do not depend on approach speed; they depend on obstacle size (engine: MuJoCo-Warp float32 GPU; g18_ret_rs025_s1, arrestors on; velocity task on tiles, undisturbed, DR scale 1, surfaces on, noise on, 176 worlds per kind; eval_governor.py with --source none)
S86 is withdrawn. Its episodes were 8 s long, and terrain features start 0.6 m or more from the tile centre, so at 0.1 m/s many robots never reached the feature; the low fall rates at low speed were an exposure artefact. With 20 s episodes (every robot reaches the feature):
| commanded speed | all tiles | step up | bump | large drop | stairs | flat |
|---|---|---|---|---|---|---|
| 0.1 m/s | 0.42 | 0.64 | 0.64 | 0.80 | 0.55 | 0.15 |
| 0.2 m/s | 0.41 | 0.66 | 0.80 | 0.73 | 0.56 | 0.08 |
| 0.3 m/s | 0.37 | 0.64 | 0.74 | 0.66 | 0.52 | 0.10 |
A rule-based speed governor driven by the privileged height map (slow to 0.1 m/s near a discontinuity, governor.py) accordingly did not help: falls 0.39 -> 0.37 on the 15 s velocity task, with slower progress (eval file gov_vel03_ps0_oracle.json).
Falls by feature size (0.3 m/s, 20 s, 352 worlds per kind; lower / upper half of the drawn severity range):
| feature | range | split | falls, smaller half | falls, larger half |
|---|---|---|---|---|
| step up | 1.1-5.9 cm | 2.3 cm | 0.40 | 0.95 |
| bump | 1.4-3.6 cm | 2.4 cm | 0.59 | 0.90 |
| stairs (riser) | 1.2-3.2 cm | 1.4 cm | 0.27 | 0.70 |
| large drop | 10.9-18.3 cm | 15.1 cm | 0.61 | 0.77 |
| small drop | 3.1-5.8 cm | 4.4 cm | 0.23 | 0.34 |
| step down | 1.2-5.5 cm | 2.6 cm | 0.12 | 0.21 |
| ramp up / down | 3.5-11.9 deg | | 0.12-0.15 / 0.28-0.29 | |
Finding: the remaining terrain falls are set by feature size, not speed. Up-steps and bumps above about 2.4 cm (about a quarter of the 9 cm ball radius) are nearly always fatal for this controller at any speed; ledges of 11 cm and more mostly are. Terrain awareness can therefore help in two ways only: (a) a climbing technique the balance policy learns when it can see the feature (tested next with the privileged height scan as policy input, the upper bound for any sensor), and (b) refusing features it cannot cross (stop before them: fewer falls, no crossing). Tile features enclose the start point, so routing around them is not possible on this map.
S87 addendum, the refusal (stop-before-obstacle) governor is not evaluable on this map: with the privileged map, refusing up-steps above 2.4 cm and ledges above 8 cm within 0.5 m stopped 94 % of worlds, including on flat tiles, because a robot approaching its tile exit sees the border of the neighbouring tile (tile bases differ in height) and stops before it can leave, which the protocol counts as failure. Stopped robots also keep falling over 20 s (flat 0.20, other kinds 0.28 to 0.41), a long-horizon standing-balance weakness under DR and latency noted for later. The refusal idea is dropped for this map; the learned climbing test (privileged height scan as policy input: g22_ret_rs025_scan_s0, g22_ret_rs05_scan_s0, g22_pure_ret_scan_s0, arrestors on, 200M steps) is running.
S85 is suspended pending re-evaluation: the held-out-design evaluations loaded only s1.5 and tau3, and the 6-D design vector given to design-conditioned policies was computed relative to the first loaded design instead of the nominal design used in training (s1.5 was presented as nominal-sized; tau3 was described relative to s1.5). Fixed in train_dev.py (reference is always 'nom'; training runs, which load 'nom' first, are unaffected; verified that the vectors now match a nominal-first list). The S85 table and its conclusions (in particular "pure learning collapses on unseen designs") must not be used until the re-run is logged. The LQR part of the LQR-based controllers always used the correct per-design gains.
The S83 held-out-design addendum (g18_ret_rs025_s1 and g12_dr175_s1 on s1.5 / tau3) has the same design-vector fault and is suspended with S85; both are in the queued re-run.

## S85-R Re-run of the held-out-design evaluation with the corrected design vector (replaces S85 and the S83 held-out-design addendum)
Engine: MuJoCo-Warp float32 GPU; goal task, DR scale 1, surfaces on, noise on, 160 worlds per design (16 per terrain kind); s1.5 = robot scaled 1.5x (ball radius 13.5 cm), tau3 = wheel torque limit one tenth of nominal; files eval2_goal_goal_dr_surf[_pull]_des-s1.5-tau3.json (re-written after the fix).
| policy | arrestors | trained on tau3 | undisturbed: s1.5 / tau3 | pushes + pulls: s1.5 / tau3 |
|---|---|---|---|---|
| g18_ret_rs025 s1 (LQR + residual 0.25) | yes | no | 0.71 / 0.30 | 0.66 / 0.15 |
| g12_dr175 s1 (LQR + residual 0.5) | no | no | 0.69 / 0.15 | 0.61 / 0.12 |
| g16_pure_ret100 s1 (pure learning) | yes | no | 0.09 / 0.00 | 0.15 / 0.00 |
| g21_ret_rs025_tau3 s0 (LQR + residual 0.25, 12 designs) | yes | yes | 0.89 / 0.29 | 0.76 / 0.17 |
| g21_ret_rs025_tau3 s1 | yes | yes | 0.75 / 0.26 | 0.66 / 0.21 |
Findings (same as the suspended S85, now on correct inputs). (1) The LQR base carries a policy to an unseen robot: the LQR-based controllers keep 0.61 to 0.71 on the larger robot; pure learning with arrestors, the most robust controller on its training designs (S81), collapses (0.09 to 0.15). (2) The tenth-torque design is motor limited for every controller (at most 0.30) and training on it does not help (0.26 to 0.29 trained against 0.30 held out). (3) Training on 12 designs raises the unseen larger robot undisturbed on both seeds (0.75 and 0.89 against 0.71); with pushes and pulls the gain is seed dependent (0.66 and 0.76 against 0.66). One seed for each 11-design policy.

## S88 Ball size against step height (test of the traction limit), and terrain information for the balance policy (engine: MuJoCo-Warp float32 GPU, arrestors on, DR scale 1, surfaces on, noise on)
Traction model (quasi-static, simplified). A driven ball of radius r pressed against a step edge of height h must lift off the floor and pivot on the edge; the contact normal at the edge makes angle theta with the vertical, cos theta = 1 - h/r, and rolling over needs a tangential-to-normal force ratio of tan theta, which friction can supply only if tan theta < mu. With mu about 1 the limit is theta = 45 deg, h < r (1 - cos 45 deg) = 0.29 r: 2.6 cm for the 9 cm ball, 4.0 cm for the 13.5 cm ball of design s1.5. (Floor friction varies with the surface class, about 0.4 to 1.2, so the limit varies per world.)
Test: g18_ret_rs025_s1 (trained on the 9 cm-ball designs) on the velocity task at 0.3 m/s, 20 s episodes, 64 worlds per terrain kind and design copy (640 per kind), undisturbed, eval_governor.py --eval-designs nom,nom vs s1.5,s1.5:
| falls | nominal (9 cm ball) | s1.5 (13.5 cm ball, unseen design) |
|---|---|---|
| step up 1.1-2.3 cm / 2.3-5.9 cm | 0.40 / 0.98 | 0.26 / 0.80 |
| bump 1.4-2.4 cm / 2.4-3.6 cm | 0.52 / 0.94 | 0.33 / 0.62 |
| stairs riser up to 1.4 / above | 0.17 / 0.67 | 0.19 / 0.64 |
| large drop 10.9-15.1 / 15.1-18.3 cm | 0.64 / 0.83 | 0.62 / 0.72 |
The larger ball crosses up-steps and bumps the small ball cannot, as the traction limit predicts (the large-step half spans 2.3 to 5.9 cm, so about half of it is below the 4.0 cm limit of the larger ball), even though the controller never trained on that design. Drops, which are not a traction problem, change little. For hardware this means the climbable step height is set mainly by ball radius and ball-floor friction.
Terrain information for the balance policy (privileged height scan 7 x 5 ahead as policy input, the upper bound for any terrain sensor; arrestors on, 200M steps, seed 0):
| run | goal undisturbed / disturbed (scale 1) | vel 0.3 falls undisturbed | step up | bump | stairs | large drop |
|---|---|---|---|---|---|---|
| g22_ret_rs025_scan (residual 0.25 + scan) | 0.68 / 0.41 | 0.36 | 0.59 | 0.78 | 0.51 | 0.70 |
| g18_ret_rs025 s0 (blind) | 0.71 / 0.42 | 0.34 | 0.61 | 0.77 | 0.48 | 0.64 |
| g22_ret_rs05_scan (residual 0.5 + scan) | 0.74 / 0.53 | 0.32 | 0.55 | 0.71 | 0.45 | 0.63 |
| g16_ret100 s0 / s1 (blind) | 0.71-0.76 / 0.46-0.49 | 0.31-0.32 | 0.53-0.54 | 0.56-0.65 | 0.42-0.43 | 0.68 |
Finding so far: even a perfect height map ahead does not reduce obstacle falls for the LQR-based controllers (per-obstacle differences within or against the blind seeds); a possible small gain under pushes and pulls for residual 0.5 (0.53 against 0.46-0.49) is one seed. Together with S87 and the traction model, the obstacle falls are a physical limit of ball size and friction, not missing foresight. Pure learning with the scan (g22_pure_ret_scan_s0) is still being evaluated.
S88 addendum, pure learning on obstacles and the scan for pure learning. On the 8 s velocity battery, pure learning with arrestors falls much less on obstacles than the LQR-based controllers (step up 0.24-0.31 against 0.61-0.62, bump 0.20-0.30 against 0.74-0.77), and adding the privileged scan (g22_pure_ret_scan_s0) changes nothing (step up 0.29, bump 0.24, large drop 0.38; goal 0.69 / 0.50 at scale 1 against 0.70-0.74 / 0.50 blind). The 20 s by-size test explains it (velocity 0.3 m/s, 640 worlds per kind, undisturbed; falls / crossing success on the larger half of each feature range):
| feature (larger half) | pure learning g16_pure_ret100_s1 | LQR + residual 0.25 g18_ret_rs025_s1 |
|---|---|---|
| step up 2.3-5.9 cm | falls 0.69, crossed 0.02 | falls 0.98, crossed 0.02 |
| bump 2.4-3.6 cm | falls 0.30, crossed 0.02 | falls 0.94, crossed 0.06 |
| stairs above 1.4 cm riser | falls 0.38, crossed 0.03 | falls 0.67, crossed 0.33 |
| large drop 15.1-18.3 cm | falls 0.58, crossed 0.39 | falls 0.83, crossed 0.17 |
Neither controller crosses large up-steps or bumps (2 to 6 percent), consistent with the traction limit. Pure learning falls less because it stalls in front of them and keeps balancing (a refusal behaviour it learned blind, by feel), not because it climbs. So terrain sensing cannot raise crossing of these features on this ball; what it could add is earlier, deliberate refusal or a route around, which these enclosed tiles do not allow.

## S89 Terrain-aware navigation around unclimbable obstacles (engine: MuJoCo-Warp float32 GPU; balance controller g18_ret_rs025_s1 frozen, arrestors on; DR on at scale 1, surfaces on, noise on, undisturbed; 48 worlds per design, 11 designs = 528 worlds; 20 s episodes; eval_nav.py)
Map: every tile is a 'blocks' tile (tiles.py): flat ground, four 8 cm blocks (3 times the traction limit of the 9 cm ball) across the tile axes 0.65 to 0.85 m from the centre, 0.3 to 0.5 m wide, plus up to two 24 cm square blocks between the axes. Start at the tile centre, goal 1.2 m away along a random axis, so the straight line is blocked and the robot must go around a block end (gaps between neighbouring axis blocks at least 0.7 m). A first version with blocks up to 0.8 m wide nearly closed the ring (gaps about 0.5 m, less than the robot width plus margin) and was replaced before any result was kept.
Navigation layer above the balance controller: wavefront planner on a 64 x 64 grid of 5 cm cells around the tile (obstacle = ground more than 3 cm above the start ground, inflated by 0.22 m), distance-to-goal field recomputed every 0.2 s, the robot steers toward the lowest-cost of 24 points 0.15 m around it at min(0.3, 1.5 x goal distance) m/s, Froude scaled. (A reactive corridor rule was tried first and stalled in local minima beside the blocks: 0.56 to 0.59 of worlds stuck.)
| hazard source | falls | reached goal | stuck | rolled onto a block |
|---|---|---|---|---|
| blind (straight to the goal) | 0.91 | 0.09 | 0.00 | 0.74 |
| privileged height map (upper bound) | 0.09 | 0.91 | 0.00 | 0.00 |
Finding: where obstacles can be avoided, terrain awareness is decisive: falls drop tenfold (0.91 to 0.09) and goal success rises from 0.09 to 0.91 with the same balance controller. The remaining 9 percent falls are the balance controller's own base rate on flat ground over 20 s (S87 flat 0.08 to 0.15). Simulated sensor configurations are being evaluated next with the same planner.
S89 continued, simulated sensor configurations (same map, planner and controller; a sensor's noisy ranges, range-proportional noise 0.5 %, 2 % dropout, are turned into 3-D hit points with the known sensor pose; hits more than 3 cm above the ground under the ball are kept in a per-world memory of 600 points that the planner rasterises; head mount = gimbal-stabilised, yawed to the command direction, 15 cm above the base origin unless stated):
| hazard source | falls | reached goal | stuck | rolled onto a block |
|---|---|---|---|---|
| blind | 0.91 | 0.09 | 0.00 | 0.74 |
| privileged height map | 0.09 | 0.91 | 0.00 | 0.00 |
| ToF 18 x 14 zones, 55 x 42 deg, head, pitch -25 deg | 0.11 | 0.85 | 0.04 | 0.02 |
| depth camera 32 x 24, 87 x 58 deg, head, pitch -25 deg | 0.13 | 0.83 | 0.05 | 0.02 |
| depth camera 32 x 24, 87 x 58 deg, head, pitch -40 deg | 0.15 | 0.51 | 0.34 | 0.00 |
| ToF 8 x 8, 45 x 45 deg, head, pitch -25 deg | 0.14 | 0.73 | 0.13 | 0.02 |
| ToF 18 x 14, mounted at the base origin (up 0 m), pitch -10 deg | 0.11 | 0.78 | 0.11 | 0.02 |
| two ToF 18 x 14 on the body cones (rigid with the body, aimed outward and 25 deg down) | 0.35 | 0.58 | 0.07 | 0.25 |
With pushes and pulls (same map, pushes 5-20 N and pulls 3-12 N as in training):
| hazard source | falls | reached goal | stuck | rolled onto a block |
|---|---|---|---|---|
| blind | 0.86 | 0.14 | 0.00 | 0.44 |
| privileged height map | 0.41 | 0.58 | 0.01 | 0.13 |
| ToF 18 x 14, head, pitch -25 deg | 0.52 | 0.45 | 0.03 | 0.15 |
Findings. (1) A modest forward-looking time-of-flight sensor (18 x 14 zones) gets within 6 points of the perfect map (0.85 against 0.91 success, falls 0.11 against 0.09); a denser, wider depth camera is no better at the same mount. (2) Where the sensor looks matters more than its resolution: pitching the camera to -40 deg (it sees mostly the near ground, so blocks enter view late and the memory map stays incomplete) leaves a third of the worlds stuck; an 8 x 8 sensor or a low mount loses 7 to 12 points; the two body cones, aimed outward rather than along the motion and tilting with the body, are worst (0.58 success, a quarter of the worlds roll onto a block). (3) Under pushes and pulls even the perfect map falls 0.41: a push can carry the robot into a block before the planner's 0.22 m margin recovers, so disturbance rejection near obstacles is the next limit (a larger margin or a push-aware planner). One controller seed, one planner setting.
S89 continued, a wider planner margin does not fix the disturbed case: inflating obstacles by 0.30 m or 0.38 m instead of 0.22 m closes most of the 0.7 m gaps between neighbouring blocks, so worlds get stuck instead (ToF undisturbed: success 0.85 -> 0.11 -> 0.00, stuck 0.04 -> 0.70 -> 0.71; with pushes and pulls the perfect map at 0.38 m: success 0.00, falls 0.57). (The perfect-map run at 0.30 m with pushes crashed and was not repeated.) The disturbed-case limit is pushes carrying the robot into blocks inside the planner's margin; addressing it needs disturbance rejection near obstacles (for example a policy trained on this map with the planner in the loop), not a larger margin on a map with 0.7 m gaps.

## S90 Smooth motion to goals, and training with the planner in the loop (engine: MuJoCo-Warp float32 GPU; blocks map of S89; DR scale 1, surfaces on, noise on, undisturbed; 528 worlds; balance controller g18_ret_rs025_s1 frozen unless stated)
The S89 wavefront planner chose the best of 24 neighbours 0.15 m away every step, so its heading jumped (commanded heading-rate RMS 14 rad/s; the ToF clip shows a zig-zag commanded path). nav.py replaces it with a smooth follower on the same distance field: carrot point 0.3 m down the field, commanded heading turning at most 2 rad/s, speed lowered with heading error and changed at most 0.6 m/s^2. Two faults found and fixed while tuning, before any result was kept: (a) a robot inside the inflated margin saw only blocked cells around it, its carrot never left it and it stopped for good (0.10 to 0.19 of worlds stuck, many then falling while standing); the descent now starts from the best point on rings 0.1 to 0.3 m around it; (b) a larger margin (0.26 or 0.28 m) closes the 0.7 m gaps on the 5 cm grid (0.78 to 0.94 stuck), so the margin stays 0.22 m.
| planner (privileged map) | reached goal | falls | stuck | rolled onto a block | heading-rate RMS (rad/s) | path / straight line |
|---|---|---|---|---|---|---|
| S89 wavefront, best neighbour | 0.91 | 0.09 | 0.00 | 0.01 | 14.4 | 1.85 |
| smooth follower (carrot 0.3 m, 2 rad/s) | 0.86 | 0.14 | 0.00 | 0.04 | 1.7 | 1.89 |
| smooth follower (carrot 0.2 m, 3 rad/s) | 0.85 | 0.15 | 0.00 | 0.03 | 2.5 | 1.92 |
The smooth follower cuts the commanded heading rate about 8-fold for a few points of success: rate-limited turns clip block corners slightly more often.
Training with the planner in the loop (train_dev.py --nav-frac 0.3): 30 percent of worlds start on blocks tiles with a goal 1.2 m behind a block and follow the smooth planner's commands (privileged map; one distance field precomputed per blocks tile and goal axis, since the map is static); reaching the goal ends the world's episode without penalty; the other 70 percent train as before on the ten terrain kinds. Recipe otherwise identical to the recommended controller (LQR + residual 0.25, arrestors, 11 designs, DR scale 1.75, 200M steps). Runs g23_nav_rs025_s0 and s1 started; evaluations queued: the navigation course with the privileged map and the ToF 18 x 14 sensor, with and without pushes and pulls, against the frozen controller under the same smooth planner, plus the standard battery.
S90 continued, smooth following with a real sensor (frozen g18_ret_rs025_s1, ToF 18 x 14 head, undisturbed, 528 worlds). With the smooth follower the ToF result fell from 0.85 (S89 wavefront) to 0.72, with 0.16 of worlds rolling onto a block. Cause: the head gimbal yaws to the command direction, and a rate-limited command sweeps the field of view slowly, so blocks to the side are seen late. Tried and rejected: a proximity slow-down near seen obstacles (0.3 / 0.5 m range: success 0.44 / 0.26, stuck 0.28 / 0.46, because every route passes within 0.5 m of a block and the robot crawls); faster turning, 4 rad/s (0.74); replanning every 0.04 s (0.69, the noisy map flickers). Adopted: the planner aims the sensor at its target heading (eval hook env.yaw_override; on hardware the gimbal is commanded by the navigation layer):
| smooth follower, ToF 18 x 14 | reached goal | falls | stuck | rolled onto a block | heading-rate RMS |
|---|---|---|---|---|---|
| sensor along the command | 0.72 | 0.22 | 0.05 | 0.16 | 1.8 |
| sensor aimed at the target, 2 rad/s | 0.78 | 0.16 | 0.05 | 0.03 | 1.8 |
| sensor aimed at the target, 4 rad/s | 0.76 | 0.17 | 0.07 | 0.04 | 3.3 |
| S89 wavefront (jerky), for reference | 0.85 | 0.11 | 0.04 | 0.02 | about 14 |
Aiming the sensor removes most block contacts (0.16 to 0.03) and keeps the motion smooth; about 7 points of success remain against the jerky planner, mostly slow or stuck approaches near blocks at the 20 s limit.

## S91 Training with the planner in the loop: a null result (engine: MuJoCo-Warp float32 GPU; blocks map; smooth follower; DR scale 1, surfaces on, noise on; 528 worlds; 20 s)
g23_nav_rs025_s0 and s1 (S90 recipe: 30 percent of training worlds on blocks tiles following the smooth planner, otherwise the recommended controller's recipe) against the frozen recommended controller g18_ret_rs025_s1, same planner:
| hazard source | frozen: reached / fell / on block | planner-trained s0 | planner-trained s1 |
|---|---|---|---|
| perfect map, undisturbed | 0.86 / 0.14 / 0.04 | 0.85 / 0.13 / 0.01 | 0.85 / 0.14 / 0.02 |
| perfect map, pushes + pulls | 0.55 / 0.44 / 0.20 | 0.55 / 0.43 / 0.13 | 0.57 / 0.42 / 0.13 |
| ToF 18 x 14, undisturbed | 0.72 / 0.22 / 0.16 | 0.68 / 0.25 / 0.15 | 0.71 / 0.23 / 0.17 |
| ToF 18 x 14, pushes + pulls | 0.41 / 0.56 / 0.22 | 0.43 / 0.53 / 0.18 | 0.46 / 0.51 / 0.21 |
Standard battery of s0: goal 0.71 / 0.43 at scale 1, flat achieved 0.85 to 0.99 of the command, the same as the recommended controller. Finding: training with the planner in the loop neither helps nor hurts (differences within seed noise, except a consistent 0.07 fewer block contacts under pushes, which does not turn into fewer falls). The frozen controller already follows the planner's smooth commands; the disturbed-case falls are push recoveries the balance controller loses, with or without obstacle experience. Two seeds.

## S92 Correction: flat-benchmark falls were mostly map-edge drop-offs (affects the "flat falls" column of every battery since S59; tracking error and achieved speed are unaffected)
The flat benchmark (TILES_FLAT=1, EVAL_NOEXIT=1) placed its 64 worlds on the tile positions of the normal protocol, which include the border tiles of the 24 m height field (tile centres at 10.5 m, 1.5 m from the edge). With no tile-exit truncation a robot driving outward leaves the height field and is counted as a fall. Measured with a new per-fall position record (eval_tiles2.py, fall_at_map_edge = fell within 0.6 m of the border): at 0.9 m/s for 8 s, 97 percent of the recorded flat falls were at the edge (g18_ret_rs025_s1: 0.25 recorded, about 0.01 real); at 0.3 m/s for 20 s over 11 designs, 74 to 76 percent (0.33 recorded, about 0.08 real). This also explains why flat falls appeared to rise with commanded speed. Fixed: the flat benchmark now uses interior tiles only (at least 7.5 m to the border); the flat rows of every battery.json are being re-run with --flat-only, and the falls_at_map_edge field is kept. The steady-state error and achieved-speed metrics were computed over surviving worlds and are unchanged by this. Flat-ground falls reported earlier in this log (S59 onward, "flat falls" columns, and the 8 to 15 percent base rate quoted in S87 and S89) are superseded by the re-run values.
Standing-still check (speed 0, 20 s, 11 designs, flat world): falls 0.06 with DR and 0.09 without, carried by kb4 (ball mass x4: 0.21 and 0.53) and kb025 (ball mass x0.25: 0.24 and 0.16); nominal 0.00. These are real: a standing robot cannot reach the map edge.
S92 addendum, the real flat-ground base rate (interior tiles, 20 s, 11 designs x 32 worlds, DR scale 1, surfaces on, noise on, no pushes, g18_ret_rs025_s1): driving at 0.3 m/s falls 0.095 overall (nominal 0.03, zhigh 0.03, zlow 0.02, J02 0.00, tau6 0.01, kb4 0.12, bal4 0.18, bal025 0.10, kb025 0.48); standing still falls 0.055 (nominal 0.00, kb4 0.18, kb025 0.22, all others at most 0.04). No fall was at the map edge. So the long-horizon weakness is design specific: the light-ball design kb025 (ball mass x0.25) and the heavy-ball or heavy-ballast designs kb4 and bal4 carry nearly all flat-ground falls; the nominal robot falls 0 to 3 percent in 20 s.

## S93 The long-horizon base fall rate is in the LQR base, on the extreme-ball designs; and the scale study starts (engine: MuJoCo-Warp float32 GPU; flat interior tiles, 20 s, 11 designs x 32 worlds, surfaces on, noise on, no pushes)
LQR alone (residual zeroed, LQR_ONLY=1, arrestors on, DR scale 1 / 0), standing: falls 0.22 / 0.23 overall; kb025 (ball mass x0.25) 0.79 / 1.00, kb4 (ball mass x4) 0.68 / 0.83, zeta66 0.33 / 0.32, bal025 0.27 / 0.24, zhigh 0.18 / 0.09, nominal 0.03 / 0.00. With the learned residual (S92 addendum): 0.055 overall, kb025 0.22, kb4 0.18. So the per-design LQR gains (planar model, Q = diag(1e-6, 50, 1, 1), R = 1, integral gain 1) do not hold a standing robot for 20 s when the ball is much lighter or heavier than the body model assumes, and the residual, trained on 8 s episodes (max_steps 800), only partly repairs it. Two fixes started: (1) g24_ret_rs025_ep20_s0: the recommended recipe trained on 20 s episodes (--ep-steps 2000), so the residual sees the long horizon; (2) an LQR gain sweep (R in 1, 3, 10; integral gain 1, 0.3, 0) on the five weak designs, standing 20 s (lqr_gpu.py, running; note lqr_gpu places worlds on all flat tiles, so only its standing rows are free of the S92 edge artefact).
Scale study (the user's small-to-large ballbot plan): uniform geometric scales s = 0.75, 1.5, 2.0 built with scripts/study/design_space.py (ball radius 6.75, 13.5, 18 cm; masses x s^3; Froude-scaled commands; arrestors re-fitted per model). LQR alone balances all three standing (falls 0.00 over 10 s). g24_scale_rs025_s0: the recommended recipe on the 11 designs plus s0.75, s1.5, s2.0 (14 designs, 200M steps). Planned evaluation: crossing success against step, bump and drop height per scale, compared with the traction limit h_max = 0.29 r (1.95, 2.6, 3.9, 5.2 cm), and the held-out s0.5 (4.5 cm ball).
S93 addendum, LQR gain sweep (lqr_gpu.py, standing 20 s, 32 worlds per design, DR on, no arrestors; falls):
| R, integral gain | nom | kb025 | kb4 | zeta66 | bal025 |
|---|---|---|---|---|---|
| 1, 1.0 (current) | 0.03 | 1.00 | 0.44 | 0.12 | 0.25 |
| 1, 0.3 | 0.03 | 1.00 | 0.47 | 0.12 | 0.31 |
| 1, 0 | 0.28 | 1.00 | 0.88 | 0.50 | 0.75 |
| 3, 1.0 | 0.00 | 1.00 | 0.53 | 0.16 | 0.12 |
| 10, 1.0 | 0.00 | 1.00 | 0.41 | 0.12 | 0.22 |
| 10, 0 | 0.41 | 1.00 | 0.94 | 0.44 | 0.88 |
Findings: the velocity integral term is essential for long-horizon standing on every design (without it even the nominal robot drifts and falls, 0.28 to 0.41); no setting of R or the integral gain rescues kb025 (ball mass x0.25, a 42 g ball under a 3.9 kg body: falls 1.00 throughout) or brings kb4 (ball mass x4) below 0.41. The planar model behind the gains does not describe these extreme ball-mass designs well enough for a 20 s hold. Decision: keep R 1, integral gain 1; rely on the long-episode residual (g24_ret_rs025_ep20_s0) for kb4 and bal025, and treat kb025 as outside the realistic design envelope (it stays in training as a stressor, but it is reported separately).
S92 addendum, re-run complete: the flat rows of all 34 batteries were re-measured on interior tiles (eval_battery.py --flat-only). Corrected flat falls are 0.00 to 0.02 for every LQR-based controller at every speed (nominal design, 64 worlds, 8 s); no fall at the map edge. The earlier 0.05 to 0.34 flat fall values were the artefact.
S92 addendum, LQR alone re-measured on interior tiles (lqr_gpu.py, nominal design, DR off, 64 worlds, 8 s): 0.3 m/s falls 0.00 (steady-state error 0.014 m/s, achieved 1.00), 0.6 m/s falls 0.00 (error 0.017, achieved 1.00), 0.9 m/s falls 0.89 and 0.91 in two samples (the saved sample is outputs/study/warp_ppo/lqr_flat_interior.log: error 0.026 over the surviving worlds, falls 0.91). The earlier S56 flat values for the LQR alone (0.12 / 0.22 / 0.97) were edge drop-offs at 0.3 and 0.6; the 0.9 m/s failure is real (the LQR's speed wall near 1 m/s, where the learned residual is what makes 0.9 m/s trackable: residual controllers fall 0.00 to 0.02 there).

## S94 [VOID as a controller result, see S98: the 20 s standing falls below were float32 solver blow-ups of the 0.001 wheel-ball friction; driving numbers stand] Long-horizon standing: seed spread, the 20 s-episode run, and a standing-focused run (engine: MuJoCo-Warp float32 GPU; flat interior tiles, 20 s, 11 designs x 32 worlds, DR scale 1, surfaces on, noise on, no pushes)
| policy | standing 20 s: all / kb4 / bal025 / kb025 / zeta66 / nom | driving 0.3 m/s 20 s: all / kb4 / kb025 / nom |
|---|---|---|
| g18_ret_rs025 s0 (8 s episodes) | 0.25 / 0.79 / 0.51 / 0.31 / 0.43 / 0.15 | 0.09 / 0.17 / 0.47 / 0.01 |
| g18_ret_rs025 s1 (8 s episodes) | 0.06 / 0.18 / 0.01 / 0.22 / 0.09 / 0.00 | 0.10 / 0.12 / 0.48 / 0.03 |
| g24_ret_rs025_ep20 s0 (20 s episodes) | 0.21 / 0.95 / 0.52 / 0.26 / 0.29 / 0.01 | 0.09 / 0.13 / 0.49 / 0.00 |
Findings. (1) Standing still for 20 s has a large seed spread (0.06 to 0.25 overall; kb4 0.18 to 0.79), while driving at 0.3 m/s is reproducible (0.09 to 0.10). The S92 addendum base rate (0.055, seed 1) was the best seed; the earlier statement that the nominal robot "falls 0 percent standing" holds for seed 1 only (seed 0: 0.15). (2) The 20 s-episode run is inside that spread: longer episodes alone do not fix long-horizon standing, and its standard battery is unchanged (goal 0.74 / 0.43, flat 0.96 to 1.02, falls 0.00 to 0.02). (3) Standing is the weak mode because training rarely asks for it: a resampled command is zero with probability 0.1 and lasts 3 s, so a 20 s hold is never practised. Started g25_ret_rs025_stand_s0: 20 s episodes and zero-command probability 0.35 (--zero-p, new CLI), otherwise the recommended recipe. The scale run s0 battery (14 designs): goal 0.72 / 0.47 at scale 1, flat achieved 0.91 to 0.98, falls 0.00 to 0.03; its step-size curves per ball size are running.

## S96 [VOID as a controller result, see S98: standing numbers are solver blow-ups; the battery and driving rows stand] Standing-focused training, seed 0 (engine: MuJoCo-Warp float32 GPU; g25_ret_rs025_stand_s0 = recommended recipe with 20 s episodes and zero-command probability 0.35, 200M steps; evaluation as S94: flat interior tiles, 20 s, 11 designs x 32 worlds, DR scale 1, surfaces on, noise on, no pushes; standard battery with DR 1 / 1.75 / 2.5, pushes and pulls on and off)
| policy | standing 20 s: all / kb4 / bal025 / kb025 / zeta66 / nom | driving 0.3 m/s 20 s: all / kb4 / kb025 / nom |
|---|---|---|
| g18_ret_rs025 s0, s1 (S94, 8 s episodes, zero-p 0.1) | 0.25, 0.06 / 0.79, 0.18 / 0.51, 0.01 / 0.31, 0.22 / 0.43, 0.09 / 0.15, 0.00 | 0.09, 0.10 / 0.17, 0.12 / 0.47, 0.48 / 0.01, 0.03 |
| g25_ret_rs025_stand s0 (20 s episodes, zero-p 0.35) | 0.10 / 0.30 / 0.18 / 0.33 / 0.15 / 0.01 | 0.10 / 0.13 / 0.53 / 0.02 |
Standard battery (goal success at DR 1 none / pushes+pulls, DR 2.5 pushes+pulls; vel fall DR 1 none / pushes+pulls; flat falls 0.3 / 0.6 / 0.9): g25 s0 0.74 / 0.44 / 0.30; 0.37 / 0.55; 0.02 / 0.00 / 0.00. Standard seeds s0 s1 s2: 0.71 0.76 0.76 / 0.42 0.43 0.44 / 0.29 0.32 0.32; 0.34 0.35 0.36 / 0.55 0.56 0.55; 0.00.
Findings. (1) The standing-focused recipe costs nothing on the standard battery: every cell is inside the three-seed spread of the recommended recipe. (2) Standing 20 s overall 0.10 sits between the two standard seeds (0.06, 0.25); kb4 improves from 0.79 / 0.18 to 0.30 and bal025 from 0.51 / 0.01 to 0.18, but the standard seed 1 is already as good or better on four of the five listed designs, so one seed cannot separate the recipe from seed luck. (3) kb025 (quarter ball mass) stays the hard design, standing 0.33 and driving 0.53 (the standard seeds: 0.47, 0.48): this is the design-limited mode that S93 showed LQR gains cannot fix, and more standing practice does not fix it either. Verdict: not a fix on this evidence; seed 1 (g25_ret_rs025_stand_s1, running) decides, and the standard seed 2 standing test is added to widen the baseline. Simulation only.
S96 addendum, standard seed 2 standing (g18_ret_rs025_s2, same protocol, files eval2_vel_0.0_flat_noexit / eval2_vel_0.3_flat_noexit): standing 20 s all 0.49; nom 0.54, zhigh 0.62, zlow 0.48, kb4 1.00, bal4 0.15, J02 0.13, tau6 0.04, bal025 0.88, kb025 0.31, zeta41 0.53, zeta66 0.74. Driving 0.3 m/s all 0.11; nom 0.01, kb4 0.15, kb025 0.56. So the recommended recipe's 20 s standing fall rate over three seeds is 0.06 / 0.25 / 0.49 (nominal robot 0.00 / 0.15 / 0.54) while its driving rate is 0.09 / 0.10 / 0.11: the standing mode is unpractised in training (8 s episodes, zero command with probability 0.1) and whether a seed happens to stand is luck. This hardens the S94 diagnosis and raises the bar for the fix: the standing-focused recipe must bring all three of its seeds under the best standard seed. g25 seed 2 launched (g25_ret_rs025_stand_s2) so the comparison is three seeds against three. The standard battery of seed 2 is unaffected (S94 / Table above), so this weakness is invisible to the 8 s battery: a long-horizon standing test (20 s, zero command) is now part of the protocol for any recipe claim.
S96 addendum, standing-focused seed 1 (g25_ret_rs025_stand_s1, same protocol): standing 20 s all 0.30; nom 0.16, zhigh 0.18, zlow 0.07, kb4 0.95, bal4 0.09, J02 0.07, tau6 0.01, bal025 0.96, kb025 0.23, zeta41 0.08, zeta66 0.52. Driving 0.3 m/s all 0.09 (nom 0.01, kb4 0.15, kb025 0.48); flat battery achieved 0.88 / 0.98 / 1.00, falls 0.00. Verdict: the standing-focused recipe is NOT the fix. Its two seeds stand at 0.10 and 0.30 against the standard recipe's 0.06 / 0.25 / 0.49: the same seed lottery, and the same two designs (kb4, four times ball mass; bal025, quarter base mass) go to 0.95 or better in the bad seeds while driving on them is fine (0.15). Practising the zero command 3.5 times more often did not teach those designs to stand, so the cause is not exposure. Since the LQR alone also fails standing on kb4 and kb025 at every gain tried (S93 addendum), the suspect is the LQR inside the controller: on the extreme ball- and base-mass designs its planar model is wrong enough that standing is only as good as the residual's ability to cancel it, which a 0.25 authority may not allow. Diagnostic queued (q28): the same 20 s standing test for pure learning with arrestors (g16_pure_ret100 s0, s1, no LQR) and for the authority variants (rs 0.15, 0.35). If pure learning stands on kb4 and bal025, the LQR is the cause and the fix is design-aware LQR gains or more authority at zero command, not more training.

## S99 The recommended recipe retrained on the corrected contact model (engine: MuJoCo-Warp float32 GPU; g26_ret_rs025_mu01 s0, s1, s2 = g18_ret_rs025 recipe with wheel-ball free-rolling friction 0.01, 200M steps each; evaluation at 0.01; standing and driving as S94 protocol: flat interior tiles, 20 s, 11 designs x 32 worlds, DR 1, surfaces on, noise on, arrestors on, no pushes; eval_tiles2 now reports the non-finite fraction separately)
| policy | standing 20 s: all / kb025 / bal4 / every other design | driving 0.3 m/s 20 s: all / kb025 / bal4 / others | non-finite |
|---|---|---|---|
| g26 s0 | 0.028 / 0.27 / 0.04 / 0.00 | 0.062 / 0.48 / 0.20 / 0.00 | 0.000 |
| g18 s1, s2 evaluated at 0.01 (S98 addendum, trained at 0.001) | 0.027, 0.039 / 0.24, 0.29 / 0.05, 0.13 / 0.00 to 0.02 | 0.063, 0.075 / 0.50, 0.54 / 0.18, 0.23 / 0.00 to 0.03 | not recorded |
Training fall rate at the end: 0.0019 to 0.0021 against 0.0021 to 0.0023 for the g18 seeds (the blow-ups were a small part of training's fall count). Seed 0 flat tracking: achieved 0.90 / 0.96 / 0.99 at 0.3 / 0.6 / 0.9 m/s, falls 0.00. Reading so far: training on the corrected model reproduces what the old policies already showed on it; the two real weak designs (kb025 light ball unseating within a second, bal4 heavy base in driving) are unchanged, so they are properties of the recipe at these mass extremes, not of training noise. Batteries and seeds 1 and 2 pending.
S99 addendum, seed 0 battery and seed 1 standing. g26 s0 standard battery (goal success DR 1 none / P, 1.75 none / P, 2.5 none / P; vel 0.3 fall same order; flat achieved 0.3 / 0.6 / 0.9): 0.74 0.45 0.72 0.41 0.68 0.33; 0.34 0.53 0.36 0.55 0.38 0.61; 0.90 0.96 0.99. The g18 seeds at 0.001: goal 0.71 to 0.76 / 0.42 to 0.44 / 0.68 to 0.72 / 0.36 to 0.41 / 0.60 to 0.66 / 0.29 to 0.32; vel 0.34 to 0.36 / 0.55 to 0.56 / 0.38 to 0.40 / 0.58 to 0.60 / 0.44 to 0.46 / 0.64 to 0.67. Goal success at or above the g18 range in every cell, velocity falls 0.02 to 0.06 lower (as at 0.01 evaluation of the old policy), flat tracking at the low end of the range (0.90 at 0.3 m/s, g18 0.90 to 0.99). g26 s1 standing 20 s: 0.030 (kb025 0.28, bal4 0.06, every other design 0.00), non-finite 0.000. Ball-size study at 0.01, 2x robot: overall 0.18 to 0.13, large-step 0.71 to 0.56, large-bump 0.55 to 0.42 (the stalled-at-the-riser blow-ups were largest for the big robots); seed 1 re-evaluation running, chapter 9 to be regenerated from the stepsize_mu01_* files.
S99 addendum, the ball-size study re-evaluated at friction 0.01 (g24_scale_rs025 s0 and s1, policies trained at 0.001; files stepsize_mu01_{s0.75,nom,s1.5,s2.0}.json; S95 protocol; overall fall 0.001 to 0.01, seed 0 | seed 1): 6.75 cm ball 0.51 to 0.46 | 0.48 to 0.46; 9 cm 0.33 to 0.31 | 0.35 to 0.34; 13.5 cm 0.26 to 0.20 | 0.26 to 0.21; 18 cm 0.18 to 0.13 | 0.18 to 0.13. Large-half step-up fall: 0.98 to 1.00 | 1.00 to 1.00; 0.96 to 0.93 | 0.98 to 0.95; 0.69 to 0.56 | 0.73 to 0.65; 0.71 to 0.56 | 0.65 to 0.60. Large-half bump: 1.00 to 1.00 | 1.00 to 0.98; 0.97 to 0.95 | 0.98 to 0.95; 0.56 to 0.48 | 0.53 to 0.52; 0.55 to 0.42 | 0.45 to 0.41. The small robots are unchanged (their falls are traction, not blow-ups), the two large robots lose 0.05 to 0.06 of overall falls and up to 0.15 at the large steps, where a stalled robot sits almost still at the riser, the blow-up condition. The size ordering is unchanged; the pooled 50 percent fall point moves up from 0.15 to 0.26 r (mean 0.21 r) to 0.17 to 0.29 r (mean 0.24 r), so the design rule tightens from "ball radius 3.4 to 4.8 times the tallest step" to "3.4 to 4.2 times" (upper value 1 / 0.24; the lower value, the friction-1 traction limit 0.29 r, is unchanged); figure outputs/study/warp_ppo/review/6_ballsize_vs_step_mu01.png is the 0.01 chart (two seeds, spread bars). Chapter 9 of the site and paper to be regenerated from these files; the scale batteries at 0.01 are running.
S99 addendum, seed 2 standing and driving (g26_ret_rs025_mu01_s2, same protocol): standing 20 s 0.025 (kb025 0.24, bal4 0.04, every other design 0.00), driving 0.3 m/s 0.064 (kb025 0.49, bal4 0.21, others 0.00), non-finite 0.000. Three seeds trained on the corrected model therefore stand at 0.028 / 0.030 / 0.025 and drive at 0.062 / 0.068 / 0.064, with kb025 0.24 to 0.28 standing and 0.48 to 0.53 driving, bal4 0.04 to 0.06 and 0.20 to 0.22: a seed spread of 0.005 where the old numbers spread 0.43.
S99 addendum, batteries. g26 s1 (goal DR 1 none / P, 1.75 none / P, 2.5 none / P; vel fall same order; flat achieved 0.3 / 0.6 / 0.9): 0.74 0.46 0.72 0.42 0.69 0.36; 0.34 0.51 0.36 0.55 0.39 0.60; 0.85 0.95 0.99. Seeds 0 and 1 agree within 0.03 on every cell; both sit at or above the g18 range on goal success and 0.02 to 0.07 below it on velocity falls; flat achieved speed at 0.3 m/s is the one cell at the low end (0.85 to 0.90 against g18's 0.90 to 0.99; late tracking error 0.041 to 0.045 against 0.024 to 0.045), within the old seed range but worth watching when the seed 2 battery lands. Scale policy g24_scale_rs025_s0 at 0.001 vs 0.01 (battery.json vs battery_mu0.01.json): goal 0.72 0.47 0.68 0.42 0.62 0.36 vs 0.74 0.47 0.71 0.43 0.66 0.38; vel 0.33 0.52 0.37 0.56 0.44 0.63 vs 0.31 0.50 0.32 0.55 0.37 0.58; flat 0.92 0.91 0.98 vs 0.91 0.92 0.98. Same picture as the recommended policy: goals unchanged or up to 0.04 better, velocity falls 0.01 to 0.07 lower, tracking unchanged. Scale seed 1 (g24_scale_rs025_s1) at 0.001 vs 0.01: goal 0.69 0.43 0.64 0.40 0.60 0.34 vs 0.70 0.44 0.67 0.39 0.63 0.35; vel 0.34 0.53 0.38 0.56 0.46 0.62 vs 0.31 0.50 0.32 0.53 0.37 0.59; flat 0.86 0.93 0.99 vs 0.89 0.93 0.99. Same direction and size.
S99 close-out. g26 s2 battery: goal 0.73 0.45 0.72 0.41 0.68 0.34; vel 0.34 0.53 0.34 0.56 0.39 0.62; flat achieved 0.93 0.98 1.00 (falls 0.00). Three seeds of the recommended recipe on the corrected contact model (friction 0.01), ranges: goal success 0.73 to 0.74 / 0.45 to 0.46 / 0.72 / 0.41 to 0.42 / 0.68 to 0.69 / 0.33 to 0.36; velocity-task falls 0.34 / 0.51 to 0.53 / 0.34 to 0.36 / 0.55 to 0.56 / 0.38 to 0.39 / 0.60 to 0.62; flat achieved 0.85 to 0.93 / 0.95 to 0.98 / 0.99 to 1.00 with falls 0.00; standing 20 s 0.025 to 0.030 and driving 0.062 to 0.068 with non-finite 0.000; every seed within 0.03 of the others on every cell. Decision: g26_ret_rs025_mu01 s0, s1, s2 replace g18_ret_rs025 s0, s1, s2 as the recommended checkpoints; the recipe is unchanged (LQR + residual 0.25, arrestors, 11 designs, DR 1.75, 8 s episodes), only the contact model is corrected. The g18 numbers remain valid comparisons for every 8 s result and for driving; their standing numbers are void (S98). Simulation only. Remaining re-measurements on the corrected model: S87 speed test and S88 by-size test (running), the scale recipe retrained at 0.01 (g27_scale_rs025_mu01_s0, training).
S99 addendum, the S88 by-size test at friction 0.01 (g18_ret_rs025_s1, files bysize_mu01_nom.json and bysize_mu01_s1.5.json, S88 protocol; falls on the smaller / larger half of each feature range, 0.001 (S88) then 0.01): nominal 9 cm ball: step up 0.40 / 0.98 then 0.30 / 0.96; bump 0.52 / 0.94 then 0.58 / 0.97; stairs 0.17 / 0.67 then 0.16 / 0.62; large drop 0.64 / 0.83 then 0.55 / 0.72. s1.5 13.5 cm ball (unseen design): step up 0.26 / 0.80 then 0.21 / 0.80; bump 0.33 / 0.62 then 0.34 / 0.64; stairs 0.19 / 0.64 then 0.17 / 0.66; large drop 0.62 / 0.72 then 0.64 / 0.81. Every cell within 0.11 and the pattern unchanged: the larger ball crosses up-steps and bumps the small one cannot, drops change little. The traction reading of S88 and Rule 1 of the bounds page stand on the corrected model.

## S100 Speed does matter after all: the S87 speed test on the corrected contact model (engine: MuJoCo-Warp float32 GPU; g18_ret_rs025_s1 at free-rolling friction 0.01, files g18_ret_rs025_s1/speed_mu01_{0.1,0.2,0.3,0.6,0.9}.json; S87 protocol: velocity task on tiles, 20 s, 176 worlds per kind, undisturbed, DR 1, surfaces on, noise on, arrestors on; falls)
| commanded speed | all tiles | step up | bump | large drop | stairs | flat |
|---|---|---|---|---|---|---|
| 0.1 m/s (S87 at 0.001: 0.42 / 0.64 / 0.64 / 0.80 / 0.55 / 0.15) | 0.28 | 0.46 | 0.54 | 0.74 | 0.43 | 0.06 |
| 0.2 m/s (S87: 0.41 / 0.66 / 0.80 / 0.73 / 0.56 / 0.08) | 0.34 | 0.61 | 0.73 | 0.72 | 0.48 | 0.05 |
| 0.3 m/s (S87: 0.37 / 0.64 / 0.74 / 0.66 / 0.52 / 0.10) | 0.35 | 0.62 | 0.74 | 0.66 | 0.49 | 0.06 |
| 0.6 m/s | 0.40 | 0.54 | 0.66 | 0.77 | 0.51 | 0.11 |
| 0.9 m/s | 0.51 | 0.62 | 0.66 | 0.83 | 0.62 | 0.23 |
Findings. (1) The S87 conclusion "obstacle falls do not depend on approach speed" was partly the blow-up: a robot creeping at 0.1 m/s spends most of the episode nearly stationary, which is the blow-up condition, so its falls were inflated most (0.42 at 0.001, 0.28 at 0.01; flat 0.15 to 0.06, step up 0.64 to 0.46). On the corrected model falls rise monotonically with commanded speed, 0.28 at 0.1 m/s to 0.51 at 0.9 m/s, and every robot still reaches its feature in 20 s (2 m at 0.1 m/s against features starting 0.6 m out). (2) The size dependence stands (S88 addendum above; the large-half falls stay near 1 for the 9 cm ball) and dominates: at 0.1 m/s a step in the larger half still fells most robots. (3) Consequence: the terrain-aware speed layer, dropped after S87 (governor falls 0.39 to 0.37 at 0.001), has to be re-tested on the corrected model, because slowing down before a feature is now expected to buy something on the order of 0.07 (0.35 to 0.28) on the features that are crossable at all. Re-test queued: the privileged-height-map governor of S87 at 0.01. S87 is marked "speed part superseded by S100"; its exposure correction of S86 (20 s episodes) stands. Simulation only.
S100 addendum, the speed governor on the corrected model (g18_ret_rs025_s1 at 0.01, S87 governor protocol: velocity task 0.3 m/s, 15 s, 176 worlds per kind, undisturbed, DR 1, surfaces on, noise on, arrestors on; governor.py "slow" mode driven by the privileged height map, slowing to v_slow near a discontinuity; files gov_mu01_vel03_ps0_{none,oracle,oracle_v005}.json):
| speed layer | falls all | step up | bump | stairs | large drop | flat | mean speed scale | progress in 15 s |
|---|---|---|---|---|---|---|---|---|
| none (S87 at 0.001: 0.395) | 0.36 | 0.64 | 0.73 | 0.48 | 0.73 | 0.05 | 1.00 | 1.35 m |
| oracle map, slow to 0.1 m/s (S87 at 0.001: 0.368) | 0.26 | 0.47 | 0.47 | 0.30 | 0.74 | 0.05 | 0.50 | 1.24 m |
| oracle map, slow to 0.05 m/s | 0.18 | 0.28 | 0.18 | 0.15 | 0.69 | 0.06 | 0.36 | 1.07 m |
Findings. (1) On the corrected model the speed governor halves obstacle falls (0.36 to 0.18) for a 21 percent loss of progress; bumps fall from 0.73 to 0.18, up-steps 0.64 to 0.28, stairs 0.48 to 0.15. Drops are untouched (0.69 to 0.74): a drop is not a traction event and speed does not help it. (2) At 0.001 the same governor looked useless (0.395 to 0.368) because the slowed robots blew up; the terrain-aware speed layer, dropped after S87, is back on the roadmap as a real gain. (3) This is the privileged upper bound (perfect height map); the practical question is how much a ToF or depth sensor recovers of it, which is the next test, together with the governor on the recommended g26 checkpoints and the "refuse" mode. Simulation only.
S100 addendum, the governor on a corrected-model checkpoint and the refuse mode (same protocol). g26_ret_rs025_mu01_s1 (trained at 0.01): no governor 0.41 (step up 0.66, bump 0.81, large drop 0.69, stairs 0.61, flat 0.10, progress 1.31 m); oracle slow to 0.1 m/s 0.37 (0.60 / 0.51 / 0.75 / 0.47 / 0.12, scale 0.55, 1.13 m); slow to 0.05 m/s 0.345 (0.45 / 0.31 / 0.72 / 0.36 / 0.11, scale 0.45, 0.95 m). So this seed is weaker on obstacles than g18 s1 at the same friction (0.41 against 0.36 without a governor) and gains less from slowing (0.41 to 0.345 against 0.36 to 0.18): the governor's benefit is policy-dependent, and one seed each cannot say whether the 0.01-trained policies systematically gain less. Running: the same test on g26 s0, s2 and g18 s0, s2. Refuse mode (g18 s1, oracle, refuse instead of slow): falls 0.08 but 93 percent of the worlds refuse, progress 0.89 m; a refusing robot does not fall, which says nothing about crossing, so refuse stays a navigation primitive (stop and re-plan) rather than a speed layer.

## S123 Step and bump height against falls for the recommended controller (engine: MuJoCo-Warp float32 GPU; g39_goalcentre_s0, s1, s2, nominal design (9 cm ball), motion layer on (eval_governor.py now applies the motion layer, goal-centre and dead-band settings of the run; before this, sweeps of motion-layer runs with that script had no layer), velocity task 0.3 m/s, 2000 steps, undisturbed, DR scale 1, surfaces on, noise on, 640 worlds per seed over the ten tile kinds (64 per kind), falls binned by the per-world feature height; files g39_goalcentre_s{0,1,2}/stepsize_nom_mref.json)
| feature height | step up: falls, 3 seeds pooled | bump: falls, 3 seeds pooled | reference, g24 s0 nominal ball (S95 addendum, no motion layer) step up / bump |
|---|---|---|---|
| 1 to 2 cm | 0.24 (n = 165) | 0.11 (n = 63) | 0.29 / 0.48 |
| 2 to 3 cm | 0.63 (n = 54) | 0.64 (n = 129) | 0.50 / 0.77 |
| 3 to 4 cm | 0.91 (n = 54) | 0.96 (n = 192) | 0.89 / 0.97 |
| 4 to 5 cm | 0.98 (n = 54) | none drawn | 1.00 / none |
| 5 to 6 cm | 0.98 (n = 57) | none drawn | 1.00 / none |
Other tile kinds, falls over the same runs (n = 384 each): flat 0.02, ramp up 0.04, rough 0.09, step down 0.10, small drop 0.12, ramp down 0.16, stairs 0.44, large drop 0.67. Per seed the step-up bins agree within 0.1 to 0.2 (2 to 3 cm: 0.78 / 0.56 / 0.56).
Findings. (1) The step limit of the recommended controller is the same as that of the S95 recipe: the 50 percent fall point is near 2.3 cm for step ups, close to the predicted traction limit 0.29 r = 2.6 cm of the 9 cm ball, and nothing above 4 cm is crossed. (2) Bumps below 2 cm are now crossed far more often (falls 0.11 against 0.48), plausibly because the motion layer approaches them with a gentle command ramp; above 3 cm they remain fatal (0.96). (3) Flat ground, ramps (up to 11 degrees, 18 cm of rise) and rough ground are essentially solved (falls 0.02 to 0.09); step downs and small drops 0.10 to 0.12; stairs, large drops and the step band above 2.5 cm are the failures, set by the ball size (S95), not by the controller recipe. Simulation only.

## S121 Recommendation: S113 recipe with goal-centre logic (g39), standard 80 ms history (engine: MuJoCo-Warp float32 GPU; navigation as S114 with eval_nav.py positions from env.ball_xy(), 528 worlds, DR on, noise on, motion layer in the loop; batteries and goal test as S113 and S118)
| run | ToF success / stuck / fall | ToF with pushes and pulls success / fall | oracle success / stuck / fall | goal test hold error | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P |
|---|---|---|---|---|---|
| g34 s0 (S113, origin logic) | 0.877 / 0.044 / 0.080 | 0.557 / 0.413 | 0.938 / 0.002 / 0.061 | 12.6 cm | 0.79 0.51 0.77 0.47 0.69 0.39 |
| g39 s0 (goal centre) | 0.866 / 0.077 / 0.057 | 0.544 / 0.413 | 0.953 / 0.001 / 0.045 | 3.6 cm | 0.75 0.49 0.73 0.44 0.70 0.37 |
| g40 s0 (goal centre + 240 ms history) | 0.824 / 0.112 / 0.064 | 0.453 / 0.472 | 0.907 / 0.028 / 0.064 | 4.9 cm | 0.79 0.55 0.77 0.48 0.74 0.42 |
g40 third seed (g40_hist24_gc_s2): goal test falls 0.066, hold error 5.5 cm, 0.06 reversals, weighted designs 0.22, battery goal success 0.80 0.52 0.77 0.47 0.73 0.40, velocity falls 0.27 0.48 0.30 0.53 0.36 0.60, flat 0.9 m/s 1.026: the S120 battery and goal-test gains of the 240 ms history reproduce over three seeds, but the disturbed-navigation loss (S120 addendum) stands.
Decision. For a robot whose job is to navigate among obstacles and people, g39 is the recommended controller: it keeps g34's navigation (ToF 0.87 against 0.88 under the stricter centre criterion, with pushes 0.54 against 0.56, falls lower: 0.057 against 0.080 without pushes, 0.045 against 0.061 with the perfect map, the best perfect-map result of the study at 0.953) and adds the precise, calm goal hold of S118 (3.6 to 4.9 cm over three seeds, against 11 to 14 cm), at battery robustness within the g34 range. g40 (with the 240 ms history) is the choice where flat-ground goal accuracy and calm falls matter more than disturbed navigation. Navigation on g39 seeds 1 and 2 (same protocol): ToF success / stuck / fall 0.867 / 0.076 / 0.057 and 0.892 / 0.047 / 0.061; with pushes and pulls success / fall 0.545 / 0.405 and 0.487 / 0.460; oracle 0.949 / 0.006 / 0.045 and 0.938 / 0.008 / 0.055. Over three seeds g39 reaches 0.87 to 0.89 with ToF (g34 s0 0.877), 0.49 to 0.55 under pushes (g34 s0 0.557, single seed) and 0.94 to 0.95 with the perfect map; the recommendation stands, with the pushed-navigation spread (0.49 to 0.55) noted.
S122 (running): the adaptive-authority hold dead band still read the ball body-origin speed (qvel[9:11]) in g39; the ROS 2 parity work flagged that the deployment core feeds it the ball-centre speed instead. train_dev.py --db-centre (default off) now uses the true ball-centre velocity, ball_vxy() = v_origin + (R omega_body) x (R offset); check while rolling at 0.84 m/s (16 worlds, three designs): ball_vxy matches the finite-difference velocity of the ball geom to 5 mm/s median, the origin velocity is off by 0.27 to 0.35 m/s median. Evaluation and render scripts re-apply db_centre from args.json. Training: g41_gc_dbc_s0 and s1 (g39 recipe plus --db-centre). Result, seed 0 (g41_gc_dbc_s0): the training process was killed by a session restart at iteration 760, 199.5M of 200M steps (the last 0.25 percent of the learning-rate schedule missing); the saved checkpoint was evaluated as it stands. Goal test (eval_motion.py, metrics at the true centre, 528 worlds): falls 0.085, hold error 4.3 cm, hold speed 0.027 m/s, 0.11 reversals, weighted designs 0.31, against g39 s0 / s1 / s2 0.070 / 0.076 / 0.093, 3.6 / 4.6 / 4.9 cm, 0.025 / 0.035 / 0.026 m/s, 0.19 / 0.07 / 0.15, 0.26 / 0.28 / 0.33. Battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P = 0.77 0.50 0.75 0.46 0.71 0.39 (g39: 0.73 to 0.77, 0.49 to 0.50, 0.70 to 0.74, 0.44 to 0.45, 0.64 to 0.70, 0.37 to 0.39), velocity falls 0.32 0.50 0.34 0.53 0.37 0.58, flat 0.9 m/s 1.070. So with one (nearly complete) seed the dead-band fix is indistinguishable from g39 on every number, as expected; its value is consistency between training and deployment (the ROS 2 core feeds the dead band the ball-centre speed), not a measured gain. Seeds 1 and 2 are retraining from scratch (the interrupted runs had reached 96.5 and 88.6 percent) and a navigation test of s0 is running; the controller bundle stays g39 until three seeds agree. Navigation of g41 s0 (S121 protocol, ToF without pushes / with pushes and pulls / perfect map): success 0.860 / 0.528 / 0.932, stuck 0.066 / n.a. / 0.007, fall 0.074 / 0.430 / 0.061, heading-rate RMS 1.28 / 1.28 / 1.21; g39 over three seeds 0.865 to 0.892 / 0.487 to 0.545 / 0.938 to 0.953, falls 0.057 to 0.061 / 0.405 to 0.460 / 0.045 to 0.055. Within the g39 spread everywhere (slightly more falls undisturbed, 0.074 against 0.057 to 0.061, one seed). Three-seed summary (seeds 1 and 2 trained to the full 200M steps after a restart killed the first attempts; s0 is the 199.5M checkpoint). Goal test (528 worlds, metrics at the true centre): g41 s0 / s1 / s2 falls 0.085 / 0.078 / 0.091, hold error 4.3 / 3.6 / 4.8 cm, hold speed 0.027 / 0.025 / 0.028 m/s, reversals 0.11 / 0.18 / 0.12, weighted designs 0.31 / 0.29 / 0.33; g39 s0 / s1 / s2: 0.070 / 0.076 / 0.093, 3.6 / 4.6 / 4.9 cm, 0.025 / 0.035 / 0.026, 0.19 / 0.07 / 0.15, 0.26 / 0.28 / 0.33. Battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P: g41 s1 0.77 0.50 0.73 0.45 0.68 0.40, s2 0.76 0.50 0.72 0.46 0.68 0.38 (s0 0.77 0.50 0.75 0.46 0.71 0.39); g39 0.73 to 0.77, 0.49 to 0.50, 0.70 to 0.74, 0.44 to 0.45, 0.64 to 0.70, 0.37 to 0.39. Velocity falls g41 s1 0.49 0.32 0.52 0.34 0.58 0.38, s2 0.48 0.30 0.52 0.32 0.58 0.35 (g39 0.47 to 0.50, 0.31 to 0.32, 0.52 to 0.53, 0.32 to 0.34, 0.57 to 0.59, 0.36 to 0.38); flat 0.9 m/s achieved 1.070 / 1.071 / 1.082 (g39 1.060 to 1.064: a 0.01 to 0.02 larger overshoot). Verdict: the dead band on the ball-centre speed leaves every robustness and precision number inside the g39 seed spread (a hair more overshoot at 0.9 m/s); its value is that training and the ROS 2 core now use the same signal. g39 stays the exported bundle; g41 is an equivalent alternative and the better choice if the dead-band consistency matters in hardware tests. Navigation of g41 s1 / s2 (S121 protocol, ToF without pushes / with pushes and pulls / perfect map): success 0.894 / 0.545 / 0.951 and 0.879 / 0.509 / 0.943, stuck 0.055 and 0.074 undisturbed, falls 0.051 / 0.398 / 0.044 and 0.047 / 0.434 / 0.051, heading-rate RMS 1.28 to 1.29. Over three seeds g41 gives ToF 0.86 to 0.89, with pushes 0.51 to 0.55, perfect map 0.93 to 0.95, falls undisturbed 0.047 to 0.074; g39 gives 0.87 to 0.89, 0.49 to 0.55, 0.94 to 0.95, 0.057 to 0.061: the same, so the verdict above stands. Next: export g39 as the ROS 2 controller bundle (the goal-centre policy also removes the ROS core's need to dead-reckon the ball orientation, S118 note). Simulation only.

## S120 240 ms history plus goal-centre logic: the best controller so far (one seed) (engine: MuJoCo-Warp float32 GPU; g40_hist24_gc_s0 = S113 recipe (g34) with --hist 24 (S117) and --goal-centre (S118), 200M steps, final training fall rate 0.0017; protocol as S113; goal-test metrics at the true ball centre for every run, see S118)
| run | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P | vel falls (same order) | goal test falls / hold error / hold speed / reversals / median overshoot | weighted designs | flat 0.9 m/s achieved |
|---|---|---|---|---|---|
| g34 s0 (S113) | 0.79 0.51 0.77 0.47 0.69 0.39 | 0.32 0.48 0.33 0.52 0.37 0.60 | 0.098 / 12.6 cm / 0.058 / 0.62 / 20.5 cm | 0.32 | 1.063 |
| g38 hist 24 s1 (S117) | 0.81 0.51 0.78 0.46 0.73 0.39 | 0.26 0.50 0.29 0.51 0.33 0.60 | 0.059 / 8.5 cm / 0.047 / 2.10 / 11.4 cm | 0.22 | 1.047 |
| g39 goal centre s0 (S118) | 0.75 0.49 0.73 0.44 0.70 0.37 | 0.32 0.50 0.34 0.53 0.38 0.59 | 0.070 / 3.6 cm / 0.025 / 0.19 / 9.9 cm | 0.26 | 1.063 |
| g40 both s0 | 0.79 0.55 0.77 0.48 0.74 0.42 | 0.27 0.47 0.28 0.51 0.34 0.59 | 0.062 / 4.9 cm / 0.031 / 0.01 / 10.0 cm | 0.23 | 1.038 |
Findings. The two changes combine without conflict: g40 keeps the lower falls of the longer history (velocity falls without pushes 0.27 / 0.28 / 0.34, goal-test falls 0.062, weighted designs 0.23) and the precise, calm goal holding of the centre logic (4.9 cm hold error, no hunting: 0.01 reversals per world), and it has the best pushed-goal success of the study (0.55 / 0.48 / 0.42 against 0.49 to 0.53 / 0.44 to 0.48 / 0.37 to 0.39 for every earlier seed) and the least speed overshoot at 0.9 m/s. Its battery goal success is judged at the true centre (stricter than the origin used for g34 and g38). One seed; g40_hist24_gc_s1 is training. If it agrees, the recommendation becomes g40 (S113 recipe + 240 ms history + goal-centre logic). Simulation only.
S120 addendum, seed 1 and navigation. g40_hist24_gc_s1: goal test falls 0.062, hold error 6.2 cm, hold speed 0.034 m/s, 0.01 reversals, weighted designs 0.23; battery goal success 0.79 0.51 0.76 0.46 0.72 0.39, velocity falls 0.29 0.48 0.30 0.53 0.34 0.60, flat 0.9 m/s 1.045. So the calm-condition gains and the precise hold reproduce, while s0's best-ever pushed-goal success (0.55) does not (s1 0.51, the usual level). Navigation (eval_nav.py as S114; eval_nav.py now takes the robot position from env.ball_xy(), i.e. the true centre for goal-centre runs and the origin otherwise, recorded as position_point; NAV_ORIGIN=1 forces the origin): ToF without pushes, success / stuck / fall: g34 s0 0.877 / 0.044 / 0.080, g32 s1 0.852 / 0.070 / 0.078, g38 hist24 s0 0.854 / 0.083 / 0.063, g40 s0 at the origin 0.847 / 0.097 / 0.057, g40 s0 at the centre 0.824 / 0.112 / 0.064; oracle map g40 s0 0.907 / 0.028 / 0.064 (g34 0.938). ToF with pushes and pulls, success / fall: g34 0.557 / 0.413, g32 0.544 / 0.400, g38 0.485 / 0.449, g40 at the origin 0.447 / 0.491, g40 at the centre 0.453 / 0.472. Reading: the 240 ms history makes planner-driven navigation more hesitant near obstacles (stuck 0.08 to 0.10 against 0.04 to 0.07) and clearly worse under pushes (success 0.45 to 0.49 against 0.54 to 0.56, falls up 0.04 to 0.08), in both runs that have it, while the goal-centre logic only makes the arrival criterion about 0.02 stricter. So the history is a trade: calmer and fewer falls in the goal and velocity batteries, worse disturbed navigation. The recommendation does not move to g40 yet; g39 (goal-centre logic with the standard 80 ms history) is the candidate that should keep g34's navigation and add the precise hold, and its navigation test is running.

## S119 Design sweep: which robot properties the controller needs (engine: MuJoCo-Warp float32 GPU; tool scripts/study/robot_spec.py (new): a spec in physical units (ball radius and mass, body mass and COM height above the ball centre, per-wheel motor torque limit, rotor inertia, wheel damping, wheel-ball contact angle) is mapped onto the parametric generator of design_space.py, checked with the analytic margins of design_constraints.py and, with --eval, simulated with the recommended controller g34_mref_rsad_db_s0 (design-conditioned; each spec is an unseen design); one property varied at a time around specs/nominal.yaml, which reproduces dz_nom (ball and body mass factors 1.000, COM height within 0.3 mm); 256 worlds per case, DR on (standard ranges), noise on; cases on a fully flat map except terrain: goal hold without disturbance, with the short pushes, with half-strength sustained pulls (1.5 to 6 N), driving at 0.3 m/s, and goals on the standard terrain tiles; files outputs/study/specs/ota_*.json, ota_sweep.csv (physics), ota_sim.csv (simulation))
| variation (nominal) | values | falls: still / pushes / half pulls / drive / terrain goals |
|---|---|---|
| nominal | ball 9 cm 0.168 kg, body 3.85 kg COM 0.131 m, torque 30, rotor 0.005, contact 51 deg | 0.007 / 0.007 / 0.017 / 0.001 / 0.138 |
| ball mass (0.168 kg) | 0.042 / 0.084 / 0.34 / 0.67 | 0.41 / 0.41 / 0.46 / 0.39 / 0.53; 0.052 / 0.052 / 0.070 / 0.006 / 0.19; 0.000 / 0.000 / 0.012 / 0.000 / 0.13; 0.000 / 0.000 / 0.013 / 0.000 / 0.14 |
| body COM height (0.131 m) | 0.08 / 0.10 / 0.17 / 0.22 | 0.037 / 0.031 / 0.051 / 0.010 / 0.20; 0.020 / 0.021 / 0.031 / 0.004 / 0.17; 0.004 / 0.004 / 0.018 / 0.000 / 0.13; 0.003 / 0.001 / 0.013 / 0.000 / 0.125 |
| body mass (3.85 kg, COM held) | 2.0 / 3.0 / 5.0 / 6.5 | 0.002 / 0.003 / 0.243 / 0.000 / 0.14; 0.004 / 0.007 / 0.034 / 0.000 / 0.14; 0.009 / 0.013 / 0.019 / 0.001 / 0.15; 0.021 / 0.018 / 0.020 / 0.001 / 0.15 |
| motor torque limit per wheel (30 N m) | 3 / 6 / 10 / 15 | 0.48 / 0.48 / 0.51 / 0.001 / 0.54; 0.13 / 0.13 / 0.12 / 0.000 / 0.23; 0.044 / 0.052 / 0.067 / 0.000 / 0.18; 0.009 / 0.011 / 0.041 / 0.000 / 0.145 |
| rotor inertia (0.005 kg m^2) | 0.001 / 0.02 | 0.43 / 0.44 / 0.47 / 0.37 / 0.54; 0.000 / 0.000 / 0.014 / 0.000 / 0.109 |
| contact angle (51 deg) | 41 / 61 | 0.015 / 0.015 / 0.034 / 0.004 / 0.14; 0.025 / 0.028 / 0.031 / 0.006 / 0.17 |
| ball radius (9 cm, masses held) | 7 / 12 cm | 0.11 / 0.11 / 0.11 / 0.025 / 0.26; 0.70 / 0.70 / 0.71 / 0.76 / 0.73 |
Torque actually used (ROS 2 parity rollout of g34 s0, 33 worlds, all 11 trained designs, DR and pushes on, 6 s): per-wheel |torque| median 0.73 N m, p90 2.6, p99 8.1, peaks to the 30 N m limit during recoveries; the actuators are direct (gear 1), so these are torques at the wheel.
Findings. (1) Ball mass is the strongest single property and the planar analytic model misses it entirely (its margins hardly move): below about 2 percent of the body mass the robot falls even when standing (0.41 at 1.1 percent), 4.4 percent (nominal) is good, 9 to 17 percent is better still (no falls without pulls). (2) A higher body COM is easier, monotonically over 0.9 to 2.4 ball radii (still-falls 0.037 to 0.003, terrain 0.20 to 0.125); the analytic traction margin flags 2.4 radii as infeasible because it asks for recovery from a 15 deg tilt, which the controller never reaches, so that constraint is too conservative. (3) Body mass barely matters for balance and terrain but sets the sustained-pull resistance (half pulls: 0.243 at 2 kg, 0.034 at 3 kg, about 0.02 from 3.85 kg up), as the static estimate predicts (capacity about m_a g sin 5 deg: 1.7 N at 2 kg, 3.3 N nominal); above about 5 kg still-falls rise slightly (0.021 at 6.5 kg) with the same motors. (4) Motor torque: at least half the nominal limit (15 N m per wheel) is needed for nominal-level robustness; a third gives 6 times more still-falls and terrain 0.18; at 10 to 20 percent the robot cannot hold itself (0.13 to 0.48), while driving at 0.3 m/s still works. (5) Rotor inertia: 4 times nominal is the best variant of the sweep (terrain 0.109); one fifth fails badly (0.43). (6) Contact angle: 51 deg is best; plus or minus 10 deg costs a little. (7) Caveat, extrapolation: the controller was trained on 11 designs that never change the ball size or lower the rotor inertia; the 12 cm ball (0.70 falls) contradicts the physics, which predicts an easier robot (unstable pole 3.8 against 4.8), so the ball-size and low-rotor-inertia rows measure the controller's training range as much as the robot; LQR-only check (eval_tiles2.py goal task, flat map, no disturbances, 256 worlds, DR on, noise on; full controller / LQR base alone): 12 cm ball 0.69 / 0.022 falls, 7 cm ball 0.11 / 0.28. So the 12 cm ball is physically easy and fails only because the learned residual has never seen another ball size, while the 7 cm ball is genuinely harder (the residual already helps it). Same check for the other weak variants: rotor inertia 0.001 0.45 / 0.49, motor torque 3 0.48 / 0.46, i.e. these fail equally with the LQR alone, so they are physical limits of the build, not training gaps. Guidelines derived from this entry: docs/research/design_guidelines.md. Simulation only; there is no physical robot.

## S118 Goal logic measured a point that swings around the ball (found by the ROS 2 parity work; engine: MuJoCo-Warp float32 GPU)
The goal rule, the goal observation, the motion layer's arrival and hold loop, the navigation goal test and the goal metrics of eval_motion.py all used qpos[10:12], the ball body origin. In ballbot.xml the ball geom sits at (0, 0, -0.14) m in the ball body frame, so the origin is 0.14 m from the ball centre (more for scaled designs) and rotates with the ball: upright at reset, it swings horizontally by up to the full 0.14 m after a quarter turn (14 cm of rolling for the nominal 9 cm ball). Check (train_dev.py, 16 worlds, three designs, ball rolled by wheel commands): the origin was 0.13 to 0.18 m from the true centre (MuJoCo geom_xpos) within 1.2 s; the new ball_xy() matches the true centre to 2 to 5 mm. Consequences: the goal the robot "holds" is a point on a 14 cm circle around its contact point, which depends on how far the ball has rolled; arrival, hold error (6 to 8 cm RMS in S110 to S117) and the goal hunting in some seeds are plausibly partly this artefact, and the 0.2 m arrival and 5 cm stop radii were effectively blurred by up to 0.14 m. Velocity tracking, the battery's velocity tests and balance are not affected (they do not use the goal position). Fix: train_dev.py --goal-centre (default off, so every earlier run is reproduced unchanged) moves all goal logic to the true centre; the evaluation and render scripts re-apply it from args.json; eval_motion.py now measures the goal metrics (arrival, overshoot, hold error) at the true ball centre for every run (field goal_metric_point), so the S110 to S117 hold numbers are on the old basis and will be re-measured. eval_nav.py's planner still uses the origin for its own position (not changed). Running: g39_goalcentre_s0 (S113 recipe g34 plus --goal-centre), to compare with the three g34 seeds; then the 240 ms history (S117) combined with the fix. RESULT (eval_motion.py goal test, metrics at the true ball centre for all runs, 528 worlds, flat interior tiles, DR on, noise on, motion layer): g34 s0 / s1 / s2 (old origin logic): falls 0.098 / 0.089 / 0.095, hold error 12.6 / 11.4 / 13.6 cm RMS, hold speed 0.058 / 0.062 / 0.060 m/s, reversals 0.62 / 1.05 / 0.54, median overshoot 20 / 16 / 23 cm; g39_goalcentre_s0 (fix): falls 0.070, hold error 3.6 cm, hold speed 0.025 m/s, 0.20 reversals, overshoot 9.9 cm, arrived 0.94. So the old policies really held the ball 11 to 14 cm (RMS) from the goal (the 6 to 8 cm of S110 to S117 was measured at the swinging origin), and training on the true centre cuts the real hold error by a factor of 3 to 4, halves the hold speed and removes the hunting. g38_hist24_s1 (old logic, 240 ms history, second seed): falls 0.059 (s0 0.064), hold error 8.5 cm at the centre. One seed of g39; g40_hist24_gc_s0 (240 ms history plus the fix) is training. Battery (eval_battery.py as S113): g39 s0 goal success DR1 none / P, 1.75 none / P, 2.5 none / P = 0.75 0.49 0.73 0.44 0.70 0.37, velocity falls 0.32 0.50 0.34 0.53 0.38 0.59, flat 0.3 / 0.9 m/s achieved 0.996 / 1.063 (g34 three seeds: 0.77 to 0.79, 0.51 to 0.52, 0.73 to 0.77, 0.45 to 0.47, 0.68 to 0.69, 0.39; velocity falls unchanged). The battery's goal success is now judged at the true centre for g39 (the env rule follows goal_centre) but at the origin for g34, whose swinging point can enter the 0.2 m arrival circle early; the 0.02 to 0.03 lower success of g39 is therefore partly a stricter measurement and is within the seed spread. Velocity tracking and falls are unchanged, as expected. Second seed (g39_goalcentre_s1): goal test falls 0.076, hold error 4.6 cm, hold speed 0.035 m/s, 0.07 reversals; battery goal success 0.73 0.49 0.70 0.44 0.64 0.38, velocity falls 0.31 0.47 0.33 0.52 0.37 0.57. The precise, calm hold reproduces (3.6 / 4.6 cm against 11 to 14 cm); terrain-goal success without pushes is 0.02 to 0.05 below g34 under the stricter centre criterion. Third seed (g39_goalcentre_s2): goal test falls 0.093, hold error 4.9 cm, hold speed 0.026 m/s, 0.15 reversals; battery goal success 0.77 0.50 0.74 0.45 0.69 0.39, velocity falls 0.31 0.48 0.32 0.52 0.36 0.59, i.e. within the g34 range under the stricter criterion. Over three seeds the goal-centre logic gives a 3.6 to 4.9 cm hold (g34: 11.4 to 13.6 cm) at unchanged robustness. S117 addendum, second seed of the 240 ms history (g38_hist24_s1, old goal logic): battery goal success 0.81 0.51 0.78 0.46 0.73 0.39, velocity falls 0.26 0.50 0.29 0.51 0.33 0.60, flat 0.9 m/s achieved 1.047, goal test falls 0.059 (metric at the centre: hold error 8.5 cm, 2.1 reversals). Both seeds agree: lower falls without pushes than every 200M-step seed (velocity 0.26 to 0.28 at DR1 against 0.31 to 0.32), equal with pushes. The ROS 2 core (ros2/ballbot_control) reproduces the trained behaviour and therefore tracks the origin by dead-reckoning the ball orientation (track_point: ball_body_origin); a retrained goal-centre policy removes that need. Simulation only.
S118 note, other findings of the ROS 2 parity work (ros2/README.md): (1) observation entries 9 to 11, meant as motor states, are base y and z velocity and body x rate divided by the torque scale tau, not wheel rates; the trained policies depend on them, so the deployment core reproduces them, but a future retraining should use wheel rates, which a real robot measures directly. (2) The parity test (14,155 control ticks, 33 worlds, all 11 designs, DR and pushes on, noise and latency off) reproduces the simulator's observations, policy actions and wheel torques to float32 rounding (max torque difference 1.5e-3 N m); the exported actor (TorchScript, ONNX, numpy) matches the torch actor to below 5e-7. (3) Not verified: any ROS build or run, any hardware.

## S117 A longer observation history (240 ms) improves calm-condition robustness; it does not lift the pull limit (engine: MuJoCo-Warp float32 GPU; g38_hist24_s0 and g38_hist48_s0 = S113 recipe (g34) with --hist 24 and --hist 48 instead of 8, i.e. 240 and 480 ms of (observation, action) history; 200M steps, final training fall rate 0.0016 / 0.0016, about 45k and 40k env-steps/s (the larger networks need about 6 GB of GPU memory each); protocol as S113; pull sweep as the S116 addendum, flat map, 352 worlds)
| run | goal test falls / reversals / hold speed / weighted designs | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P | vel falls (same order) | flat 0.9 m/s achieved | flat-map falls: none / pulls 1.5-6 N / pulls 3-12 N |
|---|---|---|---|---|---|
| g34 s0 (hist 8) | 0.104 / 0.67 / 0.058 / 0.32 | 0.79 0.51 0.77 0.47 0.69 0.39 | 0.32 0.48 0.33 0.52 0.37 0.60 | 1.063 | 0.074 / 0.139 / 0.361 |
| g38 hist 24 s0 | 0.064 / 1.00 / 0.041 / 0.22 | 0.81 0.53 0.77 0.47 0.72 0.39 | 0.28 0.48 0.30 0.51 0.35 0.60 | 1.048 | 0.055 / 0.118 / 0.353 |
| g38 hist 48 s0 | 0.061 / 2.30 / 0.064 / 0.22 | 0.81 0.51 0.77 0.47 0.74 0.40 | 0.28 0.51 0.30 0.54 0.36 0.61 | 1.035 | 0.043 / 0.116 / 0.351 |
Findings. (1) The hypothesis that a longer history lets the policy estimate and lean against a sustained pull is rejected: at 3 to 12 N the falls stay at 0.35 for all three history lengths, matching the static estimate (pull capacity about m_a g sin 5 deg = 3.3 N for the nominal robot, robot_spec.py). (2) But the 240 ms history is the first change since S112 that improves the calm-condition numbers without a cost elsewhere: flat goal-test falls 0.064 against 0.081 to 0.110 for eight 200M-step seeds of the S112/S113 recipe, the three hard designs 0.22 against 0.28 to 0.35, a calm hold (0.041 m/s, 1.0 reversals), battery goal success without pushes equal or better in every cell, driving falls without pushes 0.28 to 0.35 against 0.32 to 0.37, and less overshoot at 0.9 m/s (1.048 against 1.063). Unlike the 400M-step run (S115) it does not lose terrain crossings. (3) 480 ms adds nothing more and brings goal hunting back (2.3 reversals). One seed each; g38_hist24_s1 is training to confirm before the recommendation changes. Simulation only.

## S116 Stronger training pushes do not help; the sustained pulls, not the pushes, cause the pushed-battery falls (engine: MuJoCo-Warp float32 GPU; g37_push15_s0 and g37_push20_s0 = S113 recipe (g34) trained with pushes and pulls 1.5 and 2 times the evaluation strength (new --push-train, default 1 keeps all earlier runs identical); g37_push15_s1 stopped at 96M steps once s0 was clearly negative; protocol as S113; disturbance model: a 5-step base push every 2 to 4 s with the per-design push_force range, plus with --pull a sustained pull of 3 to 12 N for 0.5 to 2 s every 4 to 10 s, both multiplied by PUSH_SCALE)
| run | goal test falls / weighted designs | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P |
|---|---|---|
| g34 s0 (trained at 1x) | 0.104 / 0.32 | 0.79 0.51 0.77 0.47 0.69 0.39 |
| g37 push 1.5 s0 | 0.136 / 0.37 | 0.79 0.49 0.76 0.43 0.70 0.36 |
| g37 push 2.0 s0 | 0.119 / 0.37 | 0.79 0.49 0.76 0.45 0.72 0.36 |
Findings. (1) Training on harder disturbances lowers pushed-battery success by 0.02 to 0.04 and raises flat goal-test falls; no gain anywhere. (2) Where the pushed falls come from (g34 s0, eval_tiles2.py goal task on a fully flat map, TILES_FLAT=1, 352 worlds, DR on, noise on): no disturbances fall 0.074; short pushes only 0.078; pushes plus sustained pulls 0.361. On the tiled battery the pulled falls are spread over every tile kind (flat 0.36, steps and bumps 0.59 to 0.63). So the robot already handles the short pushes completely, and essentially all of the about 0.5 pushed-battery gap is the sustained 3 to 12 N pulls. A sustained horizontal force F needs a steady lean of about atan(F / m g) plus the acceleration to recover the drift, which for this robot's mass is likely near its tilt and traction limits at the top of the range; the next test sweeps the pull strength to find where it stops coping. Simulation only.
S116 addendum, pull-strength sweep (g34 s0, same flat-map protocol, pushes on, PUSH_SCALE scales both pushes and pulls; the pushes alone are harmless at 1x, so the change is the pull). Falls: pulls 0.75 to 3 N 0.078 (as without pulls), 1.5 to 6 N 0.139, 2.25 to 9 N 0.221, 3 to 12 N 0.361. The controller absorbs sustained pulls up to about 3 N completely; beyond that falls rise roughly in proportion to the pull. Per design at full strength (file g34_mref_rsad_db_s0/eval2_goal_goal_dr_surf_pull_flat_noexit.json, 32 worlds per design; corrected 2026-11-10, the first version of this sentence misread the file): zhigh 0.17, bal4 0.18, nom 0.22, kb4 0.24, zeta41 0.25, zeta66 0.25, J02 0.33, zlow 0.34, tau6 (weak motors) 0.47, kb025 (light ball) 0.59, bal025 (light body, 0.25 x ballast) 0.93. Body mass dominates: the light body almost always falls and the heavy body is among the best, as the static estimate F ~ m_a g sin(theta) predicts (pull capacity grows with the mass that leans against the pull); a higher COM helps slightly (zhigh best, zlow worse); weak motors and a light ball are next worst. A lean-and-torque limit, not a learning gap. For the site and paper: "absorbs short pushes fully and sustained pulls up to about 3 N; at 3 to 12 N sustained pulls 36 percent fall on flat ground". Hypothesis: a steady pull needs integral action (a steady lean against an unseen constant force); the LQR base has no integrator and the policy sees only 8 frames (80 ms) of history, too short to estimate a constant force. Running: g38_hist24_s0 (S113 recipe with --hist 24, 240 ms). Video c13_push_pull_hold (g34 s0, flat goal hold, pushes and pulls, seed 0 of four quick checks of which three survived 12 s). render_policy.py now draws the applied push and pull force as a red arrow (captured before the physics step; reading the force buffer after the step returned zeros) and takes --pull.

## S115 Twice the training (400M steps) on the full deployment recipe: fewer flat-goal falls, worse terrain goals, goal hunting returns (engine: MuJoCo-Warp float32 GPU; g36_deploy_400m s0, s1 = S112 recipe with the S113 dead band (--rs-adapt 0.15,0.3,0.05), hard-design weighting (--design-w bal4:2,tau6:2,kb025:2) and 400M instead of 200M steps; final training fall rate 0.0018 / 0.0016; protocol as S113 and S114; s2 still training)
| run | goal test falls / reversals / hold speed / weighted designs | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P | vel falls (same order) | ToF nav success / fall / heading-rate RMS, with P success |
|---|---|---|---|---|
| g34 s0 (200M, S113) | 0.104 / 0.67 / 0.058 / 0.32 | 0.79 0.51 0.77 0.47 0.69 0.39 | 0.32 0.48 0.33 0.52 0.37 0.60 | 0.877 / 0.080 / 1.31, 0.557 |
| g36 s0 | 0.068 / 3.73 / 0.076 / 0.23 | 0.74 0.47 0.71 0.42 0.65 0.35 | 0.32 0.49 0.34 0.53 0.39 0.60 | 0.879 / 0.057 / 1.30, 0.540 |
| g36 s1 | 0.072 / 2.52 / 0.067 / 0.25 | 0.73 0.48 0.69 0.46 0.65 0.38 | 0.32 0.49 0.32 0.51 0.37 0.59 | not run |
| g36 s2 | 0.064 / 3.36 / 0.073 / 0.22 | 0.72 0.45 0.69 0.41 0.64 0.34 | 0.32 0.49 0.34 0.53 0.38 0.59 | not run |
Findings. (1) The longer run is better on the flat goal test (falls 0.068 / 0.072 against 0.081 to 0.110 for eight 200M seeds; the three hard designs 0.23 / 0.25 against 0.28 to 0.35) and falls less in ToF navigation (0.057 against 0.070 to 0.080), but it is worse on the battery's terrain goals in every cell (no pushes 0.73 / 0.74 against 0.75 to 0.81 at DR1, 0.65 / 0.65 against 0.67 to 0.72 at DR 2.5; with pushes 0.47 / 0.48 against 0.50 to 0.53), i.e. it trades crossing features for flat-ground arrival. (2) Goal hunting returns in both seeds (2.5 to 3.7 reversals, hold speed 0.067 to 0.076 m/s) despite the dead band. Deployment-time settings do not remove it (g36 s0, eval_motion.py with RS_ADAPT: dead band 0.10 m/s: falls 0.085, 3.77 reversals, 0.077 m/s; 0.20 m/s: 0.083, 3.83, 0.077 m/s; fixed authority 0.15: 0.165, 1.89, 0.063 m/s), so the hunting is the residual's own behaviour near the goal, not the authority switch, and lowering authority buys calm with falls. Decision: training longer does not pay; the recommended recipe is the 200M-step S112 recipe with the S113 dead band (g34), with the hard-design weighting optional. eval_motion.py now accepts RS_ADAPT to override the trained authority setting and records it. Simulation only.
S115 addendum, blind navigation and the contact escape on the recommended controller (g34_mref_rsad_db_s0, eval_nav.py as S109: no terrain sensing, 40 s, 528 worlds, DR on, noise on, no pushes, motion layer in the loop; files navS_blind_40s.json, navS_bump_v15_40s.json). Blind straight to the goal: success 0.017, fall 0.983 (g26 in S109: 0.078 / 0.922). With the S109 escape (stall detection, obstacle memory, back up and re-plan, v_max 0.15 m/s): success 0.634, fall 0.261, stuck 0.104 (g26: 0.547 / 0.237 / 0.216). So the escape layer works better on the new controller (+0.09 success, half the stuck robots), while a blind robot that never backs off falls almost always; detection still comes late (falls 0.26), as noted in S109.

## S114 Terrain navigation with the motion-layer recipe: more goals, a quarter less wobble (engine: MuJoCo-Warp float32 GPU; eval_nav.py, smooth wavefront planner, carrot 0.3 m, turn limit 2 rad/s, v_max 0.3 m/s, 11 designs x 48 worlds = 528, 20 s, DR on, noise on, arrestors on, surfaces on; sources: ToF 18 x 14 zones, 55 x 42 deg, pitch -25 deg, 0.15 m up, range noise 0.005 d, aim at the target, or a perfect height map (oracle); pushes and pulls off (PUSH_SCALE 0) unless marked P (PUSH_SCALE 1, --pull); files <run>/navS_*.json. New in eval_nav.py: for runs trained with the motion layer the planner's command now passes through motion_ref.py (arrival braking and goal hold) as at deployment; recorded as motion_ref in each file, NO_MREF=1 turns it off)
| run (motion layer in the loop) | ToF: success / fall / stuck / heading-rate RMS / command change | ToF P: success / fall | oracle: success / fall / stuck / heading-rate RMS |
|---|---|---|---|
| g26_ret_rs025_mu01_s1 (S104 headline, no layer) | 0.845 / 0.072 / 0.083 / 1.74 / 0.50 | 0.511 / 0.432 | 0.886 / 0.072 / 0.042 / 1.69 |
| g32_mref_rsad_s1 (S112 recipe, layer) | 0.852 / 0.078 / 0.070 / 1.29 / 0.40 | 0.544 / 0.400 | 0.926 / 0.064 / 0.009 / 1.23 |
| g34_mref_rsad_db_s0 (S113 dead band, layer) | 0.877 / 0.080 / 0.044 / 1.31 / 0.40 | 0.557 / 0.413 | 0.938 / 0.061 / 0.002 / 1.23 |
Findings. (1) With the motion-layer recipe the planner-driven path wobbles a quarter less (heading-rate RMS 1.29 to 1.31 against 1.74; command change 0.40 against 0.50), the direct answer to the oscillation seen in the S104 navigation videos. (2) Goal success rises slightly with ToF (0.85 / 0.88 against 0.845) and more clearly with the perfect map (0.93 / 0.94 against 0.89), mostly because fewer robots get stuck (oracle stuck 0.002 to 0.009 against 0.042); falls are unchanged (0.06 to 0.08). Under pushes and pulls success is 0.54 / 0.56 against 0.51, falls 0.40 / 0.41 against 0.43: pushes near blocks remain the hard case. (3) The ToF-to-oracle gap is 0.06 for both new checkpoints (0.04 for g26), so perception, not control, now limits ToF navigation. (4) Caveat: this compares checkpoint plus layer against checkpoint without; the layer's own share is not separated. Running next: g34 s0 with NO_MREF=1 (same checkpoint, planner commands raw) to split the gain. One seed per recipe; the differences in success (0.01 to 0.05) are within the seed spread seen elsewhere, the wobble reduction (25 percent) is not. Simulation only.
S114 addendum, splitting the gain (g34_mref_rsad_db_s0, same protocol, NO_MREF=1 so the planner's command goes to the policy raw; files navS_*_nomref.json): ToF success 0.881, fall 0.070, stuck 0.049, heading-rate RMS 1.79, command change 0.51; oracle success 0.915, fall 0.083, stuck 0.002, heading-rate RMS 1.74. So the success gain over g26 belongs to the checkpoint (0.88 with or without the layer), and the whole wobble reduction belongs to the motion layer (1.79 without, 1.31 with; g26 1.74). With the perfect map the layer adds 0.02 success and cuts falls from 0.083 to 0.061. Recommendation for deployment: keep the motion layer between planner and controller; it costs nothing measurable and removes a quarter of the heading oscillation.
S114 addendum, the layer's speed cost and its limits (g34_mref_rsad_db_s0, ToF as above, MREF_KW sets the MotionRef limits, files navS_tof18_mk*.json). Median time to goal: no layer 8.9 s, default layer 9.2 s; so the layer costs about 0.3 s per route on average (the 8.5 against 11.6 s of the c12 video pair is one route, not typical). Success / fall / stuck / heading-rate RMS / median time: default (accelerate 0.4, brake 0.5 m/s^2) 0.877 / 0.080 / 0.044 / 1.31 / 9.2 s; accelerate 0.6, brake 0.8: 0.890 / 0.070 / 0.040 / 1.30 / 9.1 s; accelerate 0.8, brake 1.0, lateral 0.5: 0.884 / 0.080 / 0.036 / 1.37 / 9.2 s; accelerate 0.6, brake 0.8, lateral 0.5, turn rate 2.0 rad/s: 0.877 / 0.072 / 0.051 / 1.72 / 9.0 s. The smoothing comes from the turn-rate limit (1.5 rad/s default): raising it to 2.0 brings the wobble back to the no-layer level (1.72 against 1.79), while faster acceleration and braking cost nothing. For navigation, accelerate 0.6 / brake 0.8 with the default turn limits is the best setting found (differences in success and falls are within noise). Check on a second policy and under pushes: g32_mref_rsad_s1 without pushes 0.866 / 0.068 / 0.066 / 1.29 / 9.3 s (default layer 0.852 / 0.078 / 0.070 / 1.29); g34 s0 with pushes and pulls 0.513 / 0.455 (default layer 0.557 / 0.413). The faster setting is no better without pushes and somewhat worse with them (falls +0.04, perhaps because harder braking adds to a push); the default limits stay.
S114 addendum, hard-design weighting, two seeds (g35_mref_rsad_hardw s0, s1 = S112 recipe with --design-w bal4:2,tau6:2,kb025:2, so these three designs get 43 instead of 27 percent of the training worlds; protocol as S113). Battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P: s0 0.81 0.53 0.78 0.45 0.72 0.39; s1 0.79 0.52 0.77 0.45 0.72 0.39 (g32 and g34, five seeds: 0.75 to 0.79, 0.50 to 0.52, 0.72 to 0.77, 0.44 to 0.47, 0.67 to 0.69, 0.38 to 0.39). Velocity falls unchanged (within 0.03). Goal test falls 0.091 / 0.108; the three weighted designs fall 0.285 / 0.292 on average against 0.29 to 0.35 for the unweighted runs, the other eight designs 0.018 / 0.039 against 0.005 to 0.039. So more practice does not fix the hard designs in the flat goal test; it gives a small but consistent rise in goal success without pushes at high DR (2.5: 0.72 / 0.72 against 0.67 to 0.69) at no measured cost. Seed 2 (g35_mref_rsad_hardw_s2) withdraws that rise: battery goal success 0.77 0.52 0.73 0.49 0.67 0.40, velocity falls 0.31 0.46 0.33 0.50 0.35 0.56, goal test falls 0.081 (weighted designs 0.278, others 0.008), 0.69 reversals, hold speed 0.044 m/s. Over three seeds the weighting is within the seed spread of the unweighted recipe on every battery cell (no-push DR 2.5: 0.72 / 0.72 / 0.67 against 0.67 to 0.69; with pushes slightly better at 1.75 and 2.5, by 0.01 to 0.04), and the weighted designs fall 0.28 to 0.29 against 0.29 to 0.35. Verdict: harmless, at most a marginal gain; kept in the deployment candidate (g36) because it costs nothing. Running: seed 2, and an LQR-only goal test on these designs to see whether their falls are a physical limit of the design rather than of the policy.
S114 addendum, where the hard designs fall (g34_mref_rsad_db_s0, eval_motion.py goal with the motion layer, 528 worlds, fully flat map TILES_FLAT=1 so no terrain is involved; files motion_goal_ref_allflat*.json). Overall falls: full controller 0.068 (0.104 on the tiled map), LQR alone 0.324, full controller with gentler braking (0.3 m/s^2) 0.062. Per design, full controller / LQR alone: kb025 0.38 / 0.96, bal4 0.23 / 0.38, tau6 0.12 / 0.15, every other design at most 0.02 / 0.17 to 0.35. So (1) on flat ground the learned residual is what makes goal arrival work at all (0.32 to 0.07); (2) the light ball (kb025, 0.25 x ball mass) is nearly impossible for the linear controller and the residual saves 60 percent of it; (3) the heavy-body (bal4, 4 x ballast) and weak-motor (tau6, 6 N m torque limit) designs are close to their LQR level, i.e. near the physical limit of those builds for this manoeuvre; (4) a third of tau6's tiled-map falls (0.29 to 0.12) and a tenth of the others' come from terrain. Gentler braking helps slightly (0.068 to 0.062), within noise. Conclusion: the hard designs are a design-space boundary more than a training gap; for a future robot this argues for a ball that is not too light relative to the body, motors with torque margin and a body that is not top-heavy, rather than more training (naming corrected 2026-11-10: kb025 is the light-ball design, not a soft ball; tau6 is the weak-motor design, not a slow one; the simulation has no ball-stiffness parameter).

## S113 Goal dead band and tracking weight on top of adaptive authority: no measurable gain; the S112 goal hunting was one seed (engine: MuJoCo-Warp float32 GPU; all runs = motion-layer recipe at friction 0.01, 11 designs, DR 1.75 in training, 200M steps; g33_mref_rsad_trk3_s0 = S112 recipe with tracking reward weight 3; g34_mref_rsad_db s0, s1 = S112 recipe with --rs-adapt 0.15,0.3,0.05, i.e. the near-zero-command term only raises authority while the ball moves faster than 0.05 m/s; final training fall rate 0.0016 / 0.0018 / 0.0015; evaluations as S111 and S112: eval_motion.py on flat interior tiles, 11 designs x 48 worlds, 15 s, DR on, noise on, arrestors on, no pushes, and the standard battery, eval_battery.py, DR on, surfaces on, noise on, 16 worlds per design and condition)
| run | goal test falls / reversals / hold error / hold speed | curve falls / speed along | zigzag falls | battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P | vel falls DR1 none / P, 1.75 none / P, 2.5 none / P | flat achieved 0.9 |
|---|---|---|---|---|---|---|
| g32 s0 (S112) | 0.083 / 4.94 / 8.0 cm / 0.096 m/s | 0.034 / 0.75 | 0.30 | 0.76 0.50 0.73 0.44 0.68 0.39 | 0.32 0.48 0.32 0.52 0.38 0.58 | 1.07 |
| g32 s1 (S112) | 0.104 / 0.57 / 7.1 cm / 0.063 m/s | 0.040 / 0.77 | 0.32 | 0.75 0.52 0.72 0.46 0.67 0.38 | 0.32 0.48 0.33 0.51 0.37 0.58 | 1.06 |
| g33 s0 (tracking 3) | 0.100 / 0.99 / 6.9 cm / 0.068 m/s | 0.040 / 0.80 | 0.31 | 0.77 0.53 0.75 0.45 0.70 0.41 | 0.32 0.48 0.34 0.52 0.36 0.57 | 1.06 |
| g34 s0 (dead band) | 0.104 / 0.67 / 7.1 cm / 0.058 m/s | 0.047 / 0.78 | 0.33 | 0.79 0.51 0.77 0.47 0.69 0.39 | 0.32 0.48 0.33 0.52 0.37 0.60 | 1.06 |
| g34 s1 (dead band) | 0.091 / 0.86 / 6.7 cm / 0.065 m/s | 0.049 / 0.73 | 0.28 | 0.77 0.51 0.73 0.45 0.68 0.39 | 0.32 0.48 0.34 0.53 0.36 0.58 | 1.06 |
Findings. (1) The goal hunting that motivated the dead band (S112, g32 s0: 4.94 command reversals per world, hold speed 0.096 m/s) is a property of that one seed: g32 s1 already reverses 0.57 times and holds at 0.063 m/s, and every later run lies between 0.57 and 0.99 reversals. The S112 statement "the switching rule needs a dead band at the goal" is withdrawn as a general claim. (2) The dead band (g34, two seeds) gives the calmest holds (0.058 / 0.065 m/s), but within the seed spread of g32 s1 (0.063); tracking weight 3 (g33, one seed) gives the best curve tracking (0.80 of the commanded speed against 0.73 to 0.78) but by less than the spread across g34's own seeds. (3) Robustness is unchanged: battery goal success and velocity falls agree within 0.03 across all five checkpoints at every DR scale, with and without pushes. (4) Zigzag (60 degree switches every 2 s) costs 0.28 to 0.33 of the robots in every variant, worst on the light ball (kb025, 0.25 x ball mass, 0.56 to 0.65). WITHDRAWN as a recipe limit, see the zigzag addendum below: it is an exposure artefact of the test. Decision: the recommended recipe stays S112 adaptive authority (0.15, 0.3); the dead band is harmless and may be kept for deployment, where it removes a seed-dependent failure mode at no measured cost. Running: g34 s2 and g33 s1, so that both variants have at least two seeds. Simulation only.
S113 note, rendering artefacts and a solver buffer. (a) Showcase renders (render_policy.py, one world) of 8 s velocity clips fell on every terrain tile. CORRECTED 2026-11-10: the first explanation (driving off the map edge) was wrong; the map spans +-10.5 m. Each tile has a flat pad at its centre and the feature starts about 0.55 to 0.65 m past the centre, followed by further features; the height profiles along the clip paths show: 3 cm step up at +0.65 m, where the robot stalls at 3.0 s and falls at 3.75 s (a genuine failure, consistent with the battery's step-up falls of about 0.4); 2.6 cm step down at +0.55 m, crossed cleanly, then a 2.5 cm drop at +1.55 m, crossed, then a 5 cm step up at +2.35 m, where it falls; 11 deg ramp from +0.55 to +1.45 m (18 cm rise), climbed completely by 4.5 s, then a 33 cm drop at the ramp's far end, where it falls. So the 8 s clips ran into the next, harder feature. The clip c11_step3cm_nom (cut at 3.4 s) shows the onset of a failure, not a climb, and is withdrawn as a showcase. (b) render_policy.py now re-applies rs_adapt and the motion layer from args.json, like the evaluation scripts (it did not before, so renders of S110 to S113 checkpoints would have shown them without the layer they were trained with). (c) Training logs carry occasional "nefc overflow, increase njmax beyond 64" warnings (298 in g34 s0 over 200M steps, plus 2 narrowphase overflows), i.e. some constraint rows are dropped for some worlds on some steps. The per-world constraint buffer is now settable (NJMAX, default 64 unchanged). Running: the g34 s0 battery again with NJMAX=256 and NCONMAX=32 (g34_mref_rsad_db_s0_nj256) to measure whether the dropped rows change any reported number. Result: no. Battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P = 0.78 0.51 0.76 0.46 0.70 0.39 with the large buffers against 0.79 0.51 0.77 0.47 0.69 0.39 at the defaults; velocity falls 0.32 0.49 0.33 0.52 0.39 0.59 against 0.32 0.48 0.33 0.52 0.37 0.60. All twelve cells agree within 0.02, inside the 16-worlds-per-design sampling noise, so the overflow warnings do not affect any reported number and the defaults stay.
S113 addendum, g33 seed 1 (g33_mref_rsad_trk3_s1, same recipe and protocol): goal test falls 0.081, 1.38 reversals, hold speed 0.060 m/s; curve falls 0.036, speed along 0.77 (seed 0: 0.80); battery goal success DR1 none / P, 1.75 none / P, 2.5 none / P = 0.77 0.51 0.73 0.45 0.67 0.38; velocity falls 0.30 0.48 0.33 0.53 0.37 0.59. Tracking weight 3 therefore does not reproducibly improve curve tracking (0.80 / 0.77 against 0.75 / 0.77 for g32), and robustness again agrees within 0.03. Both seeds of g33 and g34 confirm the S113 decision. g34 seed 2 (g34_mref_rsad_db_s2, same protocol): goal test falls 0.110, 0.48 reversals, hold error 5.9 cm, hold speed 0.052 m/s; curve falls 0.036, speed along 0.80; battery goal success 0.78 0.52 0.75 0.45 0.69 0.39; velocity falls 0.31 0.48 0.32 0.51 0.37 0.57. Over three seeds the dead band holds at 0.058 / 0.065 / 0.052 m/s with 0.67 / 0.86 / 0.48 reversals, against 0.096 / 0.063 m/s and 4.94 / 0.57 reversals for g32: it removes the seed-dependent hunting reliably, at no cost in the battery (all cells within 0.03).
S113 zigzag addendum, the zigzag falls are terrain, not turning (g32_mref_rsad_s1, eval_motion.py zigzag, 528 worlds, otherwise as above). Changing the motion layer's lateral-acceleration limit does nothing (falls 0.37 / 0.31 / 0.32 / 0.32 at a_lat 0.15 / 0.2 / 0.35 default / 0.5), and raw unshaped commands fall 0.29. The test starts each robot on a flat interior tile but lets it run 15 s; the zigzag's net forward travel (about 2 m) carries it onto neighbouring terrain tiles, while the curve test (a 0.5 m circle) stays on the start tile and falls 0.03 to 0.05. On a fully flat map (TILES_FLAT=1) the same zigzag falls 0.051. So turning sharply is not a weakness of the recipe; the S110 to S113 zigzag column measures terrain exposure and should be read as such (the same exposure trap as S103). The goal test is not affected: on the flat map g32 s1 falls 0.093 (0.104 on tiles), arrives 0.93, 0.63 reversals, hold speed 0.066 m/s, so the goal-test falls of about 0.1 are genuine and concentrated on bal4, tau6 and kb025 (S112, S113 tables). eval_motion.py should default to the flat map for the curve and zigzag scenarios; until then, results carry the map in their file name (motion_*_allflat.json).

## S112 Adaptive residual authority (engine: MuJoCo-Warp float32 GPU; g32_mref_rsad s0, s1 = motion-layer recipe with --rs-adapt 0.15,0.3: authority 0.15 while cruising, rising at once to 0.3 when the command changes faster than 0.3 m/s per s, nears zero (under 0.1 m/s) or the body tilts beyond 4 deg, decaying over about 0.5 s; final training fall rate 0.0016 for both, the lowest of the study; same evaluations as S111)
| run | goal test falls / reversals / hold error / hold speed | curve: falls / speed along / sway | zigzag falls | standing / driving 20 s | battery goal DR1 none / P, 1.75 none / P, 2.5 none / P | vel falls DR1 none / P, 1.75 none / P, 2.5 none / P | flat achieved 0.3 / 0.6 / 0.9 |
|---|---|---|---|---|---|---|---|
| g32 s0 | 0.08 / 4.94 / 8.0 cm / 0.096 m/s | 0.034 / 0.75 / 2.0 | 0.30 | 0.018 / 0.057 | 0.76 0.50 0.73 0.44 0.68 0.39 | 0.32 0.48 0.32 0.52 0.38 0.58 | 0.98 / 1.03 / 1.07 |
| g32 s1 | 0.10 / 0.57 / 7.1 cm / 0.063 m/s | 0.040 / 0.77 / 1.9 | 0.32 | 0.014 / 0.057 | 0.75 0.52 0.72 0.46 0.67 0.38 | 0.32 0.48 0.33 0.51 0.37 0.58 | 0.97 / 1.02 / 1.06 |
| g29 s0 to s2 (fixed 0.25, S111) | 0.11 to 0.12 / 0.38 to 0.71 / 5.4 to 6.0 cm / 0.047 to 0.050 | 0.04 to 0.06 / 0.65 to 0.79 / 1.8 to 2.0 | 0.22 to 0.34 | 0.015 to 0.023 / 0.055 to 0.061 | 0.76-0.78 0.49-0.51 0.73-0.75 0.44-0.46 0.67-0.70 0.39-0.40 | 0.32-0.34 0.49-0.52 0.34 0.54-0.55 0.37-0.39 0.59-0.61 | 0.83-0.97 / 0.94-0.98 / 0.99-1.01 |
Findings. (1) Adaptive authority is the most robust variant so far: fewest falls at goals (0.08 to 0.10) and on curves (0.034 to 0.040), the best standing (0.014 to 0.018), the fewest velocity-task falls in every battery cell (0.48 to 0.52 with pushes and pulls against 0.49 to 0.55), and goal success equal to g29 (with pushes and pulls 0.50 to 0.52 at DR 1). (2) It does not buy g30's tracking: curve speed 0.75 to 0.77 (g29 range), and flat speed overshoots at 0.6 and 0.9 m/s (1.02 to 1.07; the higher authority is switched on by the start-up acceleration and the residual then pushes beyond the command). (3) One seed hunts at the goal (s0: 4.94 command reversals per world, hold speed 0.096 m/s, hold error 8 cm), the switch near zero command giving the residual room to fidget; the other does not (0.57). So the switching rule needs a dead band at the goal (keep authority high only while the robot is still decelerating, then drop it once inside the hold radius). Next: g33 (adaptive authority + tracking reward 3, training) and, if it keeps the robustness, a hold dead band. Simulation only, two seeds.
S112 addendum, the tracking reward over two seeds and the dead band. g31_mref_trk3_s1 (tracking reward 3, seed 1): goal falls 0.11, hold error 5.4 cm, curve speed along 0.75, sway 1.9, zigzag falls 0.31, standing 0.020, driving 0.059; battery goal 0.78 0.51 0.76 0.46 0.71 0.38, vel falls 0.31 0.51 0.34 0.53 0.38 0.60, flat 0.94 / 0.98 / 1.00. With seed 0 (curve 0.81, flat 0.99 to 1.03) the tracking reward sits inside the g29 seed spread on every measure (curve 0.65 to 0.79): no clear effect from raising the tracking weight from 2 to 3, withdrawn as a lever. The hold dead band is implemented as a third --rs-adapt value (0.15,0.3,0.05: the near-zero-command term raises authority only while the ball still moves faster than 0.05 m/s); g34_mref_rsad_db s0 and s1 are training, g33 (adaptive + tracking 3) continues for completeness.

## S111 The tracking-robustness dial, three seeds of the motion-layer recipe, and two ways to move along it (engine: MuJoCo-Warp float32 GPU; all runs = recommended recipe at friction 0.01 trained and evaluated with the motion layer; g29_mref s0 to s2 (residual authority 0.25), g30_mref_rs015_s0 (authority 0.15), g31_mref_trk3_s0 (authority 0.25, tracking reward weight 3 instead of 2); eval_motion.py as S106 and the standard battery, 20 s standing and driving as S94)
| run | goal test falls / hold error | curve: speed along / sway rad/s | zigzag falls | standing / driving 20 s | battery goal DR1 none / P, 1.75 none / P, 2.5 none / P | flat achieved 0.3 / 0.6 / 0.9 |
|---|---|---|---|---|---|---|
| g29 s0 | 0.11 / 6.0 cm | 0.65 / 2.0 | 0.22 | 0.023 / 0.061 | 0.76 0.49 0.73 0.44 0.67 0.39 | 0.83 / 0.94 / 0.99 |
| g29 s1 | 0.12 / 5.9 cm | 0.75 / 1.9 | 0.32 | 0.015 / 0.058 | 0.78 0.49 0.73 0.45 0.68 0.39 | 0.95 / 0.98 / 1.01 |
| g29 s2 | 0.11 / 5.4 cm | 0.79 / 1.8 | 0.34 | 0.021 / 0.055 | 0.78 0.51 0.75 0.46 0.70 0.40 | 0.97 / 0.98 / 1.01 |
| g31 tracking x3 | 0.10 / 5.5 cm | 0.81 / 1.8 | 0.38 | 0.022 / 0.050 | 0.77 0.49 0.75 0.44 0.69 0.37 | 0.99 / 1.00 / 1.03 |
| g30 authority 0.15 | 0.30 / 9.2 cm | 0.88 / 1.7 | 0.42 | 0.045 / 0.064 | 0.80 0.45 0.76 0.40 0.70 0.32 | 1.01 / 1.01 / 1.02 |
| g26 s0 to s2 (raw commands, previous recommendation) | 0.15 / 7.5 cm (s1) | 0.62 / 2.2 (s1) | 0.21 (s1) | 0.025 to 0.030 / 0.062 to 0.068 | 0.73-0.74 0.45-0.46 0.72 0.41-0.42 0.68-0.69 0.33-0.36 | 0.85-0.93 / 0.95-0.98 / 0.99-1.00 |
Findings. (1) Three seeds confirm S110: the motion-layer recipe (g29) is equal or better than g26 on every battery cell (with pushes and pulls by 0.03 to 0.06), stands better (0.015 to 0.023) and arrives and holds better (falls 0.11 to 0.12, hold error 5.4 to 6.0 cm). It becomes the recommended recipe; its curve tracking spreads over seeds (0.65 to 0.79 of the speed). (2) Residual authority is a dial between tracking and recovery: at 0.15 the robot tracks speed exactly on flat ground (1.01 to 1.02), follows the curve best (0.88, sway 1.7) and reaches the most goals undisturbed (0.80), but falls three times as often at goals (0.30), more in zigzags (0.42) and standing (0.045), and reaches fewer goals under pushes and pulls (0.45 / 0.40 / 0.32). (3) A stronger tracking reward at authority 0.25 (g31) moves along the dial cheaply: curve 0.81, flat 0.99 to 1.03, goal falls 0.10, standing 0.022, driving 0.050 (the best), battery inside the g29 range; the price is the zigzag (0.38). (4) Sharp 60 degree command switches remain the hardest case for every learned variant (0.22 to 0.42 falls); the motion layer limits the turn rate but the policies still struggle with the repeated reversals of lateral lean. Next: adaptive authority (g32, 0.15 cruising up to 0.3 on command changes, near-zero commands and tilt; two seeds training), which aims for g30's tracking with g29's recovery. Simulation only; single seeds for g30 and g31.

## S110 Training with the motion-reference layer, and where the wobble comes from (engine: MuJoCo-Warp float32 GPU; g29_mref s0, s1 = recommended recipe at friction 0.01 trained with --motion-ref (commands shaped by motion_ref.py, half of the velocity commands curving at up to 1 rad/s), 200M steps, final training fall rate 0.0018 / 0.0016; eval_motion.py as S106: flat interior tiles, 11 designs x 48 worlds, 15 s, DR on, noise on, arrestors on, no pushes; files g29_mref_s{0,1}/motion_*.json and g26_ret_rs025_mu01_s1/motion_*_lqronly.json)
| controller | goal: falls / reversals / overshoot / hold error / hold speed | curve 0.6 rad/s: falls / actual heading-rate RMS / speed along / sideways | zigzag: falls / speed along |
|---|---|---|---|
| g26 s1, raw commands (current) | 0.15 / 1.19 / 0.19 m / 0.075 m / 0.077 m/s | 0.05 / 2.2 rad/s / 0.62 / 0.050 | 0.21 / 0.41 |
| g29 s0, motion layer | 0.11 / 0.38 / 0.22 m / 0.060 m / 0.047 m/s | 0.05 / 2.0 / 0.65 / 0.044 | 0.22 / 0.32 |
| g29 s1, motion layer | 0.12 / 0.39 / 0.21 m / 0.059 m / 0.049 m/s | 0.06 / 1.9 / 0.75 / 0.060 | 0.32 / 0.35 |
| g29 s0 / s1, raw commands | 0.33 / 0.27 falls | 0.06 / 0.08 falls | 0.23 / 0.31 falls |
| g26 s1, LQR alone (residual zeroed), raw commands | 0.53 / 3.40 / 0.40 m / 0.26 m / 0.41 m/s | 0.11 / 1.5 / 0.98 / 0.035 | 0.55 / 0.52 |
Findings. (1) Training with the layer helps arrival and hold: falls on the goal test 0.15 to 0.11 or 0.12, command reversals 1.19 to 0.38, hold error 7.5 to 6.0 cm, hold speed 0.077 to 0.048 m/s; the new seeds depend on the layer (without it they fall 0.27 to 0.33 at goals). Overshoot and zigzags do not improve. (2) The curve test locates the wobble: the LQR alone tracks a 0.6 rad/s, 0.3 m/s curve at 0.98 of the commanded speed with the least sway (heading-rate RMS 1.5, sideways 0.035 m/s), while every learned policy makes only 0.62 to 0.75 of the speed with more sway (1.9 to 2.2 rad/s). The learned residual buys robustness (the LQR alone falls 0.11 on the curve, mostly the extreme designs, and 0.53 at goals) but costs tracking: it holds the robot back and adds sway. (3) So the fix is in the residual's objective, not in the command layer: reward tracking of the shaped command more tightly (speed along the path and heading rate), or cap the residual's authority in the direction of travel while leaving it free across it, and use the LQR's own velocity loop for the nominal motion. Next experiment: the recommended recipe with a curve-tracking reward and the motion layer (g30), measured on these three tests. Note: the standard battery and standing tests of g29 were first run without the layer (unfair to a run trained with it); eval_tiles2.py now applies the layer for runs whose args.json has motion_ref, and they are re-running. Simulation only.
S110 addendum, the fair standard evaluation of g29 (with the layer; files g29_mref_s{0,1}/battery.json and eval2_vel_{0.0,0.3}_dr_surf_ps0_flat_noexit.json). Battery, goal success at DR 1 none / P, 1.75 none / P, 2.5 none / P: g29 s0 0.76 0.49 0.73 0.44 0.67 0.39, s1 0.78 0.49 0.73 0.45 0.68 0.39, against g26 s0 to s2 0.73 to 0.74 / 0.45 to 0.46 / 0.72 / 0.41 to 0.42 / 0.68 to 0.69 / 0.33 to 0.36: equal undisturbed and 0.03 to 0.06 better with pushes and pulls in every cell. Velocity-task falls: s0 0.34 0.50 0.34 0.55 0.37 0.61, s1 0.32 0.52 0.34 0.54 0.39 0.59, the g26 range. Flat achieved speed 0.3 / 0.6 / 0.9 m/s: s0 0.83 / 0.94 / 0.99, s1 0.95 / 0.98 / 1.01 (g26 0.85 to 0.93 / 0.95 to 0.98 / 0.99 to 1.00), falls 0.00. Standing 20 s: 0.023 / 0.015 (g26 0.025 to 0.030), kb025 0.20 / 0.13 (g26 0.24 to 0.28), bal4 0.05 / 0.03; driving 0.3 m/s 20 s: 0.061 / 0.058 (g26 0.062 to 0.068). Reading: training with the motion layer costs nothing and gains a little everywhere it can be measured (robustness to pushes, standing, goal arrival and hold), with the curve-tracking gap of S110 still open. Two seeds; a third and the tracking fixes (g30 authority 0.15, g31 tracking reward 3) are training.

## S109 A blind ballbot that finds obstacles by getting stuck, remembers them and re-plans (engine: MuJoCo-Warp float32 GPU; g26_ret_rs025_mu01_s1 at 0.01; S104 navigation protocol: blocks map, 8 cm blocks, goal 1.2 m behind a block, 528 worlds, undisturbed; experiments/warp_ppo/escape.py through eval_nav.py --source bump: stall detection (under 30 percent of the commanded speed for 1 s, no 0.1 m approach to the goal in 4 s, or tilt above 15 deg), a 0.3 m wide segment of obstacle points marked 0.2 m ahead (widened 0.2 m per repeat), 1 s back-up at 0.15 m/s, re-plan on the smooth wavefront planner, give up after 6 failures; files g26_ret_rs025_mu01_s1/navS_{blind,blind_40s,bump_v30,bump_v15,bump_v15_40s}.json)
| controller | episode | reached / fell / stuck / onto a block | median time | escape events (per world) |
|---|---|---|---|---|
| blind, straight at the goal | 20 s | 0.09 / 0.91 / 0.00 / 0.75 | | |
| blind, straight at the goal | 40 s | 0.08 / 0.92 / 0.00 / 0.75 | 6.5 s | |
| blind + escape, 0.3 m/s | 20 s | 0.54 / 0.35 / 0.11 / 0.50 | 8.9 s | 1.2 |
| blind + escape, 0.15 m/s | 20 s | 0.46 / 0.21 / 0.33 / 0.50 | 13.1 s | 1.5 |
| blind + escape, 0.15 m/s | 40 s | 0.55 / 0.24 / 0.22 / 0.55 | 14.7 s | 1.7 |
| ToF 18 x 14 + planner (S104) | 20 s | 0.84 / 0.07 / 0.08 / 0.04 | 10.8 s | |
Findings. (1) With no terrain sensing at all, bump-remember-replan raises goals reached behind a block from 0.08 to 0.54 or 0.55 and cuts falls from 0.92 to 0.24 to 0.35, with no new learned policy: the same planner and balance controller, a stall detector and a contact layer in the map (the Nav2 pattern of a bump costmap layer plus BackUp recovery). (2) Slower approach trades falls for time: at 0.15 m/s falls drop to 0.21 to 0.24, and with 40 s most of the robots that were still working in 20 s finish. (3) The weak point is detection: half the robots partly roll onto a block before the stall triggers (onto a block 0.50 to 0.55), and those are where the falls come from. Earlier detection is the next step (the effort or residual spike as the ball meets the block face, rather than a 1 s slow-speed window). (4) The motion around an escape is jerky (heading-rate RMS of the command 8 to 10 rad/s against about 1.7 for the sensor-guided planner): back-up and re-plan switch the command abruptly, the same tracking problem as S105 and S106. A ToF sensor remains far better (0.84); the escape is the fallback for blind spots and sensor failure. Simulation only.
S108 addendum, terrain (odom_tiles.npz: the same driving on the full tiled map with steps, drops, ramps, stairs, bumps; 128 robots, 23 upright at 60 s, so 10 and 30 s are the reliable horizons; file odom/odom_results_tiles.json): realistic sensors, calibrated on flat ground: position error 0.06 m after 10 s (1.7 m driven, 3.6 percent; p90 0.18 m) and 0.20 m after 30 s (5.0 m, 3.8 percent; p90 0.43 m), against 2.4 and 2.0 percent on flat ground; even ideal sensors give 3.5 and 3.2 percent here, so terrain adds about 1.5 percent of distance through slip and through the ball rolling over edges that the flat-ground calibration does not model. Uncalibrated: 12 percent over the first 1.7 m.

## S108 How well the ballbot tracks its own position from wheel encoders and an IMU (simulation study; logs from collect_odom.py: g26_ret_rs025_mu01_s1 at 0.01, design nom, all-flat map, commands capped at 0.4 m/s (resampled every 3 s, half the robots chasing goals), DR on, noise on, arrestors on, no pushes, 128 robots x 60 s, 108 upright at the end; estimator experiments/warp_ppo/odometry.py, evaluation eval_odom.py; calibration fitted on robots 0 to 63, results on robots 64 to 127; file outputs/study/warp_ppo/odom/odom_results_flat.json)
Kinematics first. The simulator logs the ball BODY origin, which sits 14 cm above the ball centre in this model; the centre (ball_p + R_ball @ (0, 0, -0.14)) rolls cleanly (median slip 2 cm/s at a median 0.46 m/s), so the encoders can see the ball's motion. The planar model's wheel matrix W is off: measured wheel rates are about 0.65 times W times the ball's rotation relative to the body, with a wide scatter, so the estimator uses a calibration matrix fitted from data (as a real robot would) rather than W.
| sensors | position error after 10 / 30 / 60 s, median (p90) | as percent of distance driven (median) | heading error after 60 s (median) |
|---|---|---|---|
| ideal (true wheel and body rates, calibrated) | 0.04 (0.09) / 0.06 (0.13) / 0.07 (0.13) m | 2.4 / 0.9 / 0.6 | 0.05 deg |
| realistic: 4096-count wheel encoders, MEMS gyro (bias sd 0.005 rad/s, random walk, noise) and accelerometer, calibrated | 0.05 (0.10) / 0.10 (0.26) / 0.22 (0.58) m | 2.4 / 2.0 / 1.8 | 11.8 deg |
| realistic, ball radius 2 percent wrong | 0.04 / 0.11 / 0.22 m | 2.1 / 1.8 / 1.8 | 11.8 deg |
| realistic, uncalibrated (planar model W) | 0.25 / 0.34 / 0.39 m | 13.3 / 6.0 / 3.2 | 11.8 deg |
Median distance driven: 1.8 m at 10 s, 5.8 m at 30 s, 12.7 m at 60 s. Findings. (1) With calibrated encoders and a realistic IMU the robot knows where it is to about 2 percent of distance driven: 5 cm after 2 m, 22 cm after 13 m (p90 58 cm). (2) Heading is the error that grows: yaw has no absolute reference, the gyro bias is only re-estimated while the robot stands still (rare in this driving), and 12 degrees of yaw error after a minute turns into most of the position error. A 2 percent ball-radius error changes almost nothing by comparison. (3) Calibration matters most: the planar model's W gives 13 percent error over the first 2 m. Consequences for the stack: wheel-plus-IMU odometry is good for local motion, escape manoeuvres and short routes (a few metres), and needs an absolute heading or position correction for longer missions: a better gyro, stop-and-recalibrate pauses, or landmark or map matching from the ToF or a camera (in ROS 2: robot_localization fusing odometry with a localiser). Not yet measured: the terrain log (slip on steps and drops; running), and the error of the velocity estimate the controller itself would use. Simulation only; the sensor noise values are typical MEMS figures, not a specific part.

## S107 Approach speed against obstacle height, exposure-controlled (engine: MuJoCo-Warp float32 GPU; g26_ret_rs025_mu01_s1 at 0.01; velocity task on the tiles, 40 s so every robot reaches its feature, 176 worlds per kind, undisturbed, DR 1, surfaces on, noise on, arrestors on; files g26_ret_rs025_mu01_s1/speed40_{0.1,0.2,0.3,0.45,0.6,0.9}.json; falls, per-world height bins with at least 8 worlds)
| speed m/s | all | step up 1-2 / 2-3 / 3-4 / 4-5 cm | bump 1-2 / 2-3 / 3-4 cm | stairs | large drop | flat | tile exit reached |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.36 | 0.25 / 0.58 / 0.88 / 1.00 | 0.24 / 0.66 / 0.92 | 0.55 | 0.74 | 0.06 | 0.64 |
| 0.2 | 0.37 | 0.29 / 0.81 / 1.00 / 1.00 | 0.24 / 0.69 / 0.95 | 0.55 | 0.73 | 0.12 | 0.63 |
| 0.3 | 0.37 | 0.25 / 0.77 / 0.96 / 1.00 | 0.31 / 0.71 / 0.97 | 0.52 | 0.67 | 0.07 | 0.63 |
| 0.45 | 0.39 | 0.23 / 0.73 / 0.88 / 1.00 | 0.28 / 0.78 / 0.94 | 0.49 | 0.74 | 0.09 | 0.61 |
| 0.6 | 0.40 | 0.17 / 0.73 / 0.88 / 0.96 | 0.17 / 0.66 / 0.97 | 0.51 | 0.69 | 0.15 | 0.60 |
| 0.9 | 0.50 | 0.39 / 0.50 / 0.73 / 1.00 | 0.21 / 0.58 / 0.91 | 0.55 | 0.82 | 0.28 | 0.50 |
Findings. Once every robot reaches its feature, approach speed between 0.1 and 0.6 m/s changes overall falls by only 0.04 (0.36 to 0.40); the S100 rise to 0.51 at 0.9 m/s is mostly flat-ground and drop falls at that speed (flat 0.28), not obstacles. Per height there are two small windows: creeping (0.1 m/s) is best on 2 to 3 cm steps (0.58 against 0.73 to 0.81), and charging (0.9 m/s) is best on 2 to 4 cm steps and bumps (0.50 and 0.73 against 0.58 to 1.00) but costs far more on flat ground and drops. Nothing crosses a step above 4 cm at any speed. So momentum is real but narrow: worth at most a per-feature choice between creep and charge on 2 to 4 cm steps, inside the traction band, and only if the flat-ground cost of high speed is fixed first. The decisive levers remain ball size (traction) and routing. Bins hold 8 to about 60 worlds; differences under about 0.1 are within noise. Simulation only.

## S106 The motion-reference layer on an unchanged policy does not help (engine: MuJoCo-Warp float32 GPU; g26_ret_rs025_mu01_s1 at 0.01; eval_motion.py: flat interior tiles, 11 designs x 48 worlds, 15 s, DR on, noise on, arrestors on, no pushes; raw = commands as today, ref = through motion_ref.py; files g26_ret_rs025_mu01_s1/motion_{goal,curve,zigzag}_{raw,ref}.json)
| scenario | mode | falls | command heading-rate RMS | actual heading-rate RMS | speed along desired / desired | sideways speed | command reversals per world | goal: arrived / overshoot median / hold error RMS / hold speed RMS |
|---|---|---|---|---|---|---|---|---|
| goal 1.2 m | raw | 0.15 | 9.4 rad/s | 3.6 | 0.10 | 0.023 | 1.19 | 0.89 / 0.19 m / 0.075 m / 0.077 m/s |
| goal 1.2 m | ref | 0.31 | 0.8 | 3.0 | 0.08 | 0.035 | 0.19 | 0.82 / 0.31 m / 0.106 m / 0.089 m/s |
| curve 0.6 rad/s | raw | 0.05 | 0.6 | 2.2 | 0.62 | 0.050 | 0 | |
| curve 0.6 rad/s | ref | 0.05 | 0.6 | 2.2 | 0.61 | 0.050 | 0 | |
| zigzag 60 deg / 2 s | raw | 0.21 | 13.6 | 3.5 | 0.41 | 0.076 | 5.49 | |
| zigzag 60 deg / 2 s | ref | 0.24 | 1.1 | 3.3 | 0.29 | 0.095 | 0 | |
Findings. (1) The layer does what it is built to do to the command (heading-rate RMS of the command 9.4 to 0.8 and 13.6 to 1.1 rad/s, reversals 1.19 to 0.19 and 5.49 to 0), but the robot does not follow better: on the goal test falls double (0.15 to 0.31; nominal design 0.06 to 0.35) and overshoot grows; on the curve nothing changes; on the zigzag progress drops. The policy was trained on the raw goal rule and on piecewise-constant commands, so slow ramps and a gentle position-hold loop are commands it has never seen; the residual reacts badly to them. (2) The balance layer's own tracking is the limit, as S105 said: on a steady 0.6 rad/s curve at 0.3 m/s the robot makes 0.62 of the desired speed along the path, slides 0.05 m/s sideways and turns its velocity at 2.2 rad/s RMS against 0.6 commanded, with no command shaping involved. (3) Even the raw flat goal test falls 0.15 in 15 s, concentrated on kb025 (0.67), tau6 (0.31) and bal4 (0.27): arriving and stopping is harder than cruising (S99: 0.06 driving, 0.03 standing). Consequence: command shaping has to be trained with, not bolted on. g29_mref_s0 (the recommended recipe trained with --motion-ref, including curving commands) is the real test and starts after the approach-speed sweep. Simulation only.

## S105 Why the paths oscillate and the robot hunts at the goal (diagnosis, no fix yet; g26_ret_rs025_mu01_s1 at 0.01)
Recorded ToF navigation run (rec_tof18_smooth_traj.pkl, one world, 9.5 s, reached the goal): the commanded heading turns smoothly (heading-rate RMS 1.6 rad/s, no reversal above 90 degrees in any 0.5 s) but the ball's actual velocity heading swings at 8.1 rad/s RMS, with a dominant sideways sway at 0.32 Hz; along the command the robot makes 0.16 m/s of a 0.24 m/s command (66 percent) and slides 0.075 m/s sideways. Tilt stays small (mean 2.0, max 5.4 deg). So the oscillation is in the balance layer's tracking of a continuously changing command, not in the planner. Three causes, from the code: (1) training commands are piecewise constant (resampled every 3 s) or a straight goal direction, so a continuously turning command from the follower is outside what the residual was trained on; (2) a ballbot must lean before it can accelerate (non-minimum-phase), and the LQR reference governor (a_max 0.5 m/s squared) plus the velocity integral (ki 1) lag the command and then overshoot, a slow surge-and-sway; (3) at the goal, update_goal_cmd is a saturating proportional rule on position (v = 1.5 d toward the goal, zero inside 5 cm) and the LQR's position weight is 1e-6, so nothing holds position: an overshoot flips the command 180 degrees, and the on/off switch at 5 cm chatters, so the robot hunts around a missed goal. Plan: a motion-reference layer between planner and balance (a time-parametrised path with velocity and acceleration feed-forward, a braking profile v = sqrt(2 a d) into the goal, and a position-hold mode at arrival), and training commands that include smooth curves and decelerations to a stop. Momentum (same session, S100 per-world data, g18 s1 at 0.01): step-up falls by height 1-2 / 2-3 / 3-4 cm are 0.19 / 0.27 / 0.58 at 0.1 m/s, 0.25 / 0.73 / 0.88 at 0.3, 0.16 / 0.58 / 0.81 at 0.6, 0.40 / 0.50 / 0.69 at 0.9; bumps 1-2 cm 0.28 / 0.31 / 0.10 / 0.31. Both creeping and charging beat the 0.3 m/s cruise on mid-size steps, so a per-feature speed choice (approach speed as a function of feature type and height, chosen from a measured table or learned) is a real option for the motion-reference layer; it has to be measured with 40 s exposure-controlled episodes (S103 lesson).

## S104 Navigation on the recommended checkpoint and the corrected contact model (engine: MuJoCo-Warp float32 GPU; g26_ret_rs025_mu01_s1 at its own friction 0.01; S90 protocol: blocks map, 8 cm blocks, goal 1.2 m away behind a block, 48 worlds per design (528), 20 s, wavefront planner with the smooth follower (carrot 0.3 m, 2 rad/s), sensor aimed at the planner target; files g26_ret_rs025_mu01_s1/navS_{oracle,tof18}{,_pp}.json)
| hazard source | reached / fell / stuck / onto a block, undisturbed | median time | heading-rate RMS | reached / fell / stuck / onto a block, pushes and pulls |
|---|---|---|---|---|
| perfect map | 0.89 / 0.07 / 0.04 / 0.01 | 9.5 s | 1.69 rad/s | 0.57 / 0.39 / 0.04 / 0.16 |
| ToF 18 x 14 | 0.84 / 0.07 / 0.08 / 0.04 | 10.8 s | 1.74 rad/s | 0.51 / 0.43 / 0.06 / 0.17 |
Blind (eval_nav --source blind, straight at the goal, file navS_blind.json): reached 0.09, fell 0.91, onto a block 0.75, the same as S89 with the old checkpoint (0.09). Against S90 (g18 s1 at 0.001, smooth follower): ToF reached 0.78 there and 0.84 here, perfect map 0.91 there and 0.89 here; motion is as smooth (heading-rate RMS about 1.7 to 1.8 rad/s against about 14 for the jerky planner of S89). So routing around unclimbable blocks with the ToF head sensor works on the recommended controller and the corrected model: 0.84 of goals behind a block, within 0.05 of a perfect map. Under pushes and pulls about half the goals are reached; the falls there are mostly pushes near or onto a block (onto a block 0.16 to 0.17), the remaining hard case for navigation. Simulation only.

## S103 A ToF-driven speed layer matches the perfect-map one (engine: MuJoCo-Warp float32 GPU; recommended checkpoints g26_ret_rs025_mu01 s0, s1, s2 at their own friction 0.01; S101 governor protocol: velocity task 0.3 m/s, 15 s, 176 worlds per kind, undisturbed, DR 1, surfaces on, noise on, arrestors on; governor.py slow mode to 0.05 m/s)
Estimator (experiments/warp_ppo/train_gov_est.py, data from collect_percep.py with g26 s1 driving the tiles, 2048 worlds, every 6th step, split by world 4:1): noisy 18x14 head ToF zones (the VL53L9-like sensor with range noise and dropout) plus the 25-D proprioception, MLP 512 x 2, output the 7x5 height grid ahead that the governor reads from the privileged map (percep/tof18_gov.pt). Validation MSE 0.38 against 1.43 for predicting the mean (grid in units of 6 cm); on the governor's own decision (any feature above 1 cm within 0.5 m ahead, centre columns) precision 0.73, recall 0.85, base rate 0.58.
| seed | falls none / oracle map / ToF | progress in 15 s none / oracle / ToF | step up (none / oracle / ToF) | bump | stairs | large drop | flat |
|---|---|---|---|---|---|---|---|
| s0 | 0.38 / 0.16 / 0.16 | 1.35 / 1.04 / 0.99 m | 0.62 / 0.30 / 0.21 | 0.78 / 0.11 / 0.43 | 0.56 / 0.11 / 0.13 | 0.73 / 0.59 / 0.36 | 0.09 / 0.07 / 0.06 |
| s1 | 0.39 / 0.15 / 0.15 | 1.34 / 1.02 / 0.96 m | 0.65 / 0.26 / 0.19 | 0.80 / 0.10 / 0.39 | 0.56 / 0.11 / 0.12 | 0.68 / 0.53 / 0.28 | 0.09 / 0.09 / 0.08 |
| s2 | 0.38 / 0.17 / 0.17 | 1.34 / 1.05 / 1.02 m | 0.64 / 0.31 / 0.26 | 0.74 / 0.11 / 0.43 | 0.53 / 0.14 / 0.15 | 0.73 / 0.71 / 0.36 | 0.09 / 0.06 / 0.07 |
Findings. (1) The deployable sensor keeps the whole gain: falls 0.38 to 0.39 without a speed layer, 0.15 to 0.17 with the ToF-driven layer, identical to the perfect-map layer on every seed, for 4 to 6 percent less progress than the oracle (the false alarms slow it a little more often). (2) The two differ by feature: the ToF layer is worse on bumps (0.39 to 0.43 against 0.10 to 0.11; a 1.4 to 3.6 cm bump is about one zone tall at 0.5 m and easy to miss) and clearly better on large drops (0.28 to 0.36 against 0.53 to 0.71) and up-steps (0.19 to 0.26 against 0.26 to 0.31). The drop result says the oracle grid used here (rows 0.1 to 0.5 m, centre columns) reacts too late to a drop edge, while the ToF estimate, trained on the full grid, flags the edge earlier; a larger look-ahead for the oracle is the obvious fix and makes the oracle a better upper bound. (3) For the robot: an 18x14 ToF on the head, an estimator of this size and the rule-based governor are enough to halve obstacle falls on the recommended controller. Next: the same on the goal task with pushes and pulls (the realistic use), and a bump-specific fix (finer zones near the ground or a longer look-back of recent frames). Simulation only.
S103 addendum, CAUTION, possible exposure artefact (2026-11-09 08:56). Goal task, g26 s1 at 0.01, 1.1 m goal, undisturbed (eval_governor.py --task goal; files gov_goal_ps0_*.json): 15 s episodes: no layer falls 0.26, success 0.74; ToF layer (slow to 0.05) falls 0.13, success 0.34, neither 0.53 (median time 5.7 s). 30 s episodes: no layer 0.24 / 0.76; ToF layer 0.23 / 0.62, neither 0.15 (median time 12.9 s). ToF layer slowing to 0.1 m/s, 15 s: 0.19 / 0.58. With pushes and pulls, 15 s: no layer 0.55 / 0.45 (s1), 0.53 / 0.47 (s0); ToF layer 0.47 / 0.27, 0.47 / 0.28. s0 undisturbed 15 s: 0.25 / 0.75 against 0.13 / 0.36. Reading: given enough time to reach the goal, the slowed robot falls as often as the unslowed one (0.23 against 0.24) and succeeds less; the 15 s halving was mostly robots that had not yet reached a feature, the same exposure artefact as S86. The velocity-task halving of S100, S101 addendum and S103 (15 s, mean progress 0.96 to 1.07 m with the layer against 1.34 m without, features from 0.6 m out) may carry the same artefact. Re-test running with 40 s episodes (q55: g26 s1, no layer, oracle and ToF layers). Until it lands, every "the speed layer halves obstacle falls" statement is suspended; the site must not state it.
S103 addendum, the 40 s re-test: the halving was an exposure artefact. g26 s1, velocity task 0.3 m/s, 40 s (4000 steps), otherwise the S101 governor protocol (files gov_vel03_ps0_{none,oracle_v005}_40s.json): no layer falls 0.38 (step up 0.62, bump 0.79, stairs 0.56, large drop 0.72, flat 0.11; tile exit reached 0.62, progress 1.35 m); privileged-map layer slowing to 0.05 m/s falls 0.33 (0.61 / 0.72 / 0.43 / 0.74 / 0.07; exit 0.58, progress 1.31 m). Given time to reach the feature, the slowed robot falls almost as often: the real effect of the speed layer is 0.05 overall, almost all of it on stairs (0.56 to 0.43) and bumps (0.79 to 0.72); step-ups and drops are unchanged. The goal task says the same (30 s: no layer 0.24 falls / 0.76 success; ToF layer to 0.05 m/s 0.23 / 0.62; ToF layer to 0.1 m/s 0.25 / 0.74). Withdrawn: the "halves obstacle falls" result of S100, S101 and S103 (all 15 s episodes, in which the slowed robots covered 0.96 to 1.07 m against 1.34 m with features starting 0.6 m out); the S101 and S102 per-seed governor tables are valid as 15 s measurements only. Stands: falls rise with commanded speed (S100 table, 20 s, every speed reaches its feature), the ToF estimator's accuracy (precision 0.73, recall 0.85), and the S102 equality of the two checkpoint families (same 15 s protocol on both sides). Meaning: obstacle falls are set by obstacle size against ball size (traction, S88, S95, S99), and slowing down before a feature the robot can cross at all buys little; the tools that work are routing around what cannot be crossed (S89, S90: the ToF sensor and a planner) and a larger ball. This is S86's lesson a second time: a slowed robot must be given the time to meet the obstacle. ToF-driven layer, same 40 s protocol (gov_vel03_ps0_tof18_v005_40s.json): falls 0.35 (step up 0.61, bump 0.78, stairs 0.47, large drop 0.77, flat 0.09; exit 0.54, progress 1.28 m). So none / perfect map / ToF: 0.38 / 0.33 / 0.35 overall, stairs 0.56 / 0.43 / 0.47, bumps 0.79 / 0.72 / 0.78, step ups, drops and flat within 0.05. Final: the speed layer is worth 0.03 to 0.05 of obstacle falls, mostly on stairs, at a cost of 0.04 to 0.08 in tile exits reached; not worth a hardware sensor on its own. Navigation (routing around blocks) is being re-measured on the recommended checkpoint at its own friction (q56: blind, perfect map, ToF; with and without pushes).

## S102 S101's family gap was an evaluation bug: eval_governor.py did not apply the run's free-rolling friction (2026-11-09 07:45)
eval_tiles2.py re-applies `pair_mu1` from a run's args.json (added with S98), but eval_governor.py, fall_cause.py, render_policy.py, eval_nav.py, render_nav.py and nan_hunt.py did not. Every eval_governor.py result on a run trained at 0.01 was therefore evaluated at the xml value 0.001, unless PAIR_MU1 was exported by hand: that covers the g26 and g28 rows of the S101 governor table, the g27 step-size curves, and the "policy against model" test (whose "at 0.01" comparator was itself at 0.001, so both sides were the same model). Found by the behaviour comparison (q48), which exported PAIR_MU1=0.01 explicitly: g26 s1 with the slow-to-0.05 governor falls 0.16 there (0.345 in S101), g18 s1 0.17; without the governor g26 s1 falls 0.37 (0.41 in S101), g18 s1 0.35. So on the same contact model the two families are equal within 0.02, with and without the speed layer, exactly as on the 8 s battery and in standing. Withdrawn: the S101 family gap, the "accidental curriculum" hypothesis and its stall-termination test (g28 was evaluated at 0.001 too), the S101 addendum on g27 versus g24, the two-family recommendation. Stands: the governor's gain itself (measured on g18 at an explicit 0.01) and everything evaluated with eval_tiles2.py (batteries, standing, driving, flat tracking) or with PAIR_MU1 exported (the g18 and g24 re-checks of S99 and S100). Fix: all six scripts now re-apply pair_mu1 from args.json when PAIR_MU1 is not set, and eval_governor.py records it in its output. Re-runs of every affected result are running (q49: g26 and g28 governor rows; q50: g27 step-size curves and the g26 slow-to-0.1 row). The behaviour metrics (q48), all at an explicit 0.01: g18 s1 / g26 s1 without governor: falls 0.35 / 0.37, mean speed along the command 0.22 / 0.20 m/s, time creeping 14 / 15 percent, peak tilt 15.0 / 16.1 deg, residual RMS 0.27 / 0.36; with governor: 0.17 / 0.16, 0.09 / 0.09 m/s, 44 / 46 percent, 9.1 / 7.8 deg, residual RMS 0.17 / 0.25. Every fall in both ends by the body leaving the ball (offset at the 30 deg limit). The one real difference is that the g26 residual works harder (RMS 0.25 to 0.36 against 0.17 to 0.27) for the same result. Lesson for lesson 8: the same artefact bit twice, the second time through an evaluation script that had not been taught the new contact model; every evaluation path must read the run's own model parameters from one place.
S102 addendum, the re-runs at the correct friction (files now record pair_mu1 0.01; S101 governor protocol): falls without / with the slow-to-0.05 governor, progress in 15 s: g26 s0 0.38 / 0.16 (1.35 / 1.04 m), s1 0.39 / 0.15 (1.34 / 1.02 m), s2 0.38 / 0.17 (1.34 / 1.05 m); g18 s0 to s2 (S101, explicit 0.01) 0.36 to 0.38 / 0.18 to 0.19. Equal within 0.02 on every seed: the speed layer halves obstacle falls for every policy of the recommended recipe on the corrected model, at a 22 to 24 percent loss of progress. g28 (stall termination, no longer of interest) s0 0.37 / 0.20. g27 scale retrain step-size curves at 0.01 (overall, large-step, large-bump falls): 6.75 cm 0.46 / 0.98 / 1.00, 9 cm 0.33 / 0.98 / 0.92, 13.5 cm 0.22 / 0.55 / 0.66, 18 cm 0.14 / 0.64 / 0.39, against g24 at 0.01 (S99 addendum) 0.46 / 1.00 / 1.00, 0.31 / 0.93 / 0.95, 0.20 / 0.56 / 0.48, 0.13 / 0.56 / 0.42: overall equal within 0.02 at every size, single cells within 0.18 (one seed of g27). The corrected recipe is complete and needs no curriculum change; next is the sensor-driven speed layer.

## S101 [WITHDRAWN as a result, see S102: the g26 / g27 / g28 rows were evaluated at friction 0.001 by a script bug] The speed layer across six seeds, and a surprise: policies trained at friction 0.01 are worse on obstacles than policies trained at 0.001 and evaluated at 0.01 (engine: MuJoCo-Warp float32 GPU; all evaluation at free-rolling friction 0.01; S87 governor protocol: velocity task 0.3 m/s, 15 s, 176 worlds per kind, undisturbed, DR 1, surfaces on, noise on, arrestors on; governor.py slow mode on the privileged height map, v_slow 0.05 m/s; files gov_mu01_vel03_ps0_{none,oracle_v005}.json under g18_* and gov_vel03_ps0_{none,oracle_v005}.json under g26_*)
| policy (training friction) | no governor: all / step up / bump / stairs / large drop / flat / progress | governor: all / step up / bump / stairs / large drop / flat / progress |
|---|---|---|
| g18 s0 (0.001) | 0.36 / 0.62 / 0.74 / 0.49 / 0.73 / 0.08 / 1.36 m | 0.19 / 0.34 / 0.15 / 0.18 / 0.72 / 0.07 / 1.06 m |
| g18 s1 (0.001) | 0.36 / 0.64 / 0.73 / 0.48 / 0.73 / 0.05 / 1.35 m | 0.18 / 0.28 / 0.18 / 0.15 / 0.69 / 0.06 / 1.07 m |
| g18 s2 (0.001) | 0.38 / 0.62 / 0.74 / 0.53 / 0.74 / 0.10 / 1.35 m | 0.18 / 0.31 / 0.14 / 0.10 / 0.74 / 0.07 / 1.03 m |
| g26 s0 (0.01) | 0.40 / 0.68 / 0.78 / n.r. / 0.72 / 0.10 / 1.32 m | 0.32 / 0.42 / 0.35 / n.r. / 0.73 / 0.09 / 0.97 m |
| g26 s1 (0.01) | 0.41 / 0.66 / 0.81 / 0.61 / 0.69 / 0.10 / 1.31 m | 0.34 / 0.45 / 0.31 / 0.36 / 0.72 / 0.11 / 0.95 m |
| g26 s2 (0.01) | 0.39 / 0.62 / 0.75 / n.r. / 0.78 / 0.09 / 1.31 m | 0.35 / 0.48 / 0.35 / n.r. / 0.73 / 0.11 / 0.99 m |
(n.r. = not read out in this pass; the files hold it.)
Findings. (1) The speed layer is a real gain on every seed of both families: the three g18 seeds go from 0.36 to 0.38 down to 0.18 to 0.19 (halved; bumps 0.73 to 0.14 to 0.18, stairs 0.49 to 0.53 down to 0.10 to 0.18) for a 22 percent loss of progress; the three g26 seeds go from 0.39 to 0.41 down to 0.32 to 0.35 (a fifth off) for a 26 percent loss. Drops are untouched in every row (0.69 to 0.78). (2) The surprise: the policies trained on the corrected model (g26) are worse on this 15 s obstacle test than the old policies evaluated on the same corrected model (0.39 to 0.41 against 0.36 to 0.38 without a governor), and gain far less from slowing (0.32 to 0.35 against 0.18 to 0.19). The 8 s battery does not show this (g26 is at or above g18 on every cell, S99) and standing is equal, so it is specific to long encounters with features at low speed. Three seeds each and tight within-family agreement (0.02) make it a systematic difference, not seed noise. Cause unknown; candidates: the g26 residual learned a different approach behaviour at the rollers' higher free-rolling resistance, or the training-time blow-ups at 0.001 acted as an accidental penalty on creeping into features that shaped the g18 residual usefully. Test queued: g26 s0 on the same protocol at friction 0.001 (does the gap follow the policy or the model?). (3) Consequence for the recommendation (S99): g26 remains the clean re-baseline on the battery and the only family with a reproducible standing number, but for driving over features with a speed layer the g18 checkpoints evaluated at 0.01 are better by 0.14 to 0.17. Until the cause is known the recommended deployment is g18 s1 or s2 at friction 0.01 for obstacle courses with a speed layer, and the site must present both families with this table rather than a single winner. Simulation only.
S101 addendum, the gap follows the policy, not the model: g26 s0 evaluated at friction 0.001 (files g26_ret_rs025_mu01_s0/gov_mu0.001_vel03_ps0_{none,oracle_v005}.json) gives 0.40 without the governor and 0.32 with it (step up 0.65 / 0.47, bump 0.79 / 0.32, stairs 0.55 / 0.30, large drop 0.72 / 0.73, flat 0.10 / 0.10, progress 1.32 / 0.97 m), the same as at 0.01 (0.40 / 0.32). So evaluating either family at either friction changes nothing; what differs is what the two families learned. The g18 residual, trained on a model where creeping into a feature often ended in a non-finite "fall", behaves better when slowed before features than the g26 residual, which trained without that accidental penalty. That makes the training-time blow-up an unintended curriculum, and it means the corrected model needs its own way to teach slow feature approaches (for example a training-time speed layer or an explicit slow-approach curriculum) before g26 can replace g18 on feature-rich terrain. Open; S101 stays as logged.
S101 addendum, the same gap in the scale family (g27_scale_rs025_mu01_s0, the 14-design scale recipe trained at 0.01, against g24_scale_rs025_s0 trained at 0.001, both evaluated at 0.01 on the S95 protocol; files g27_scale_rs025_mu01_s0/stepsize_*.json and g24_scale_rs025_s0/stepsize_mu01_*.json; overall fall, large-step fall, large-bump fall, flat fall): 6.75 cm ball g27 0.50 / 1.00 / 0.98 / 0.19 against g24 0.46 / 1.00 / 1.00 / 0.12; 9 cm 0.36 / 0.98 / 0.97 / 0.05 against 0.31 / 0.93 / 0.95 / 0.02; 13.5 cm 0.25 / 0.67 / 0.50 / 0.02 against 0.20 / 0.56 / 0.48 / 0.00; 18 cm 0.18 / 0.58 / 0.44 / 0.01 against 0.13 / 0.56 / 0.42 / 0.00. The policy trained at 0.01 is 0.04 to 0.05 worse overall at every size, mostly through more flat and small-feature falls on the 20 s test, the same direction as S101 in a second recipe and seed. The accidental-curriculum hypothesis now gets a direct test: train the recommended recipe at 0.01 with a deliberate stall termination (a world whose commanded speed exceeds 0.15 m/s but whose achieved speed along the command stays under a fifth of it for 1 s is ended with a penalty), which reproduces on purpose what the blow-up did by accident. If that run recovers the g18 obstacle numbers, the mechanism is confirmed and the corrected recipe is complete.
S101 addendum, g27 standard battery (g27_scale_rs025_mu01_s0/battery.json against g24_scale_rs025_s0/battery_mu0.01.json, both at 0.01; goal success and vel fall in the usual six-cell order): goal 0.72 0.46 0.70 0.42 0.65 0.37 against 0.74 0.47 0.71 0.43 0.66 0.38; vel 0.30 0.51 0.33 0.53 0.37 0.59 against 0.31 0.50 0.32 0.55 0.37 0.58. Equal within 0.02 on every 8 s cell, exactly as g26 against g18 (S99): the family gap exists only on the 20 s feature encounters. Flat rows (re-run under the standard file name after a queue-script PUSH_SCALE export had mis-named them): achieved 0.86 / 0.94 / 0.99 at 0.3 / 0.6 / 0.9 m/s, late error 0.051 / 0.062 / 0.052, falls 0.00 (g24 s0 at 0.01: 0.91 / 0.92 / 0.98).
S101 addendum, the stall-termination test, seed 0 (g28_ret_rs025_mu01_stall_s0: recommended recipe at friction 0.01 plus --stall-pen 1.0, an episode ends with penalty 1.0 when the robot is commanded above 0.15 m/s but makes under a fifth of it along the command for 1 s; 200M steps, training fall rate at the end 0.0019; S101 governor protocol; files gov_vel03_ps0_{none,oracle_v005}.json): without governor 0.38 (step up 0.60, bump 0.79, stairs 0.51, large drop 0.64, flat 0.14, progress 1.33 m); with the slow-to-0.05 governor 0.36 (0.48 / 0.36 / 0.36 / 0.72 / 0.11, speed scale 0.43, progress 1.00 m). Standing 20 s: kb025 0.29, bal4 0.08, every other design 0.00; driving 0.3 m/s: kb025 0.53, bal4 0.18, others 0.00. Reading: the deliberate stall termination does not reproduce what the g18 family has (g18 with governor 0.18 to 0.19; g26 0.32 to 0.35; g28 s0 0.36). The accidental-penalty hypothesis is not supported by this seed; seed 1 agrees (g28 s1: 0.40 without, 0.35 with the governor) and so does seed 2 (0.39 / 0.34), so across three seeds the stall-termination recipe sits at 0.38 to 0.40 without and 0.34 to 0.36 with the governor, inside the g26 range; the hypothesis is rejected, and if they agree the cause of the family gap is still unknown and the next step is a behavioural comparison (approach speed, time spent at the riser, tilt at the feature) of g18 against g26 under the governor rather than another training variant. The standing "falls" were a simulator blow-up: anisotropic wheel-ball friction 0.001 in float32 (engine: MuJoCo-Warp float32 GPU; diagnostic experiments/warp_ppo/nan_hunt.py, LQR alone on g18_ret_rs025_s2's designs, zero command, 10 s, 11 designs x 10 tile kinds x 8 worlds = 880 worlds, DR 1, surfaces on, arrestors on, no pushes; metric: fraction of worlds whose state becomes non-finite)
How it was found. fall_cause.py gained a --stand mode (zero command instead of the goal task). On the worst standard seed (g18 s2, standing fall 0.49 in S96) 91 percent of the standing "falls" were neither a tip (tilt > 25 deg) nor an unseating (wheel-ball misalignment > 5 deg): the traces show robots upright at 1 to 2 deg of tilt whose qpos becomes NaN in a single 10 ms control step, with |qvel| 0.02 and |qacc| 0.3 the step before. The training termination counts a non-finite state as a fall, so these entered every 20 s standing number (S94, S96, S97) and, at a much lower rate, training itself.
What was ruled out (same test, non-finite fraction; baseline 0.127): per-world contact buffer 16 to 64 (0.126); arrestors off (0.133); the degenerate-tangent branch of the wheel contact-frame kernel guarded (0.117); warm start off (0.151); CG solver (0.130); 300 Newton iterations with 200 line-search iterations (0.133); no domain randomisation and no floor material classes (0.163); plane floor instead of the height field (0.228, so the height field is not the cause either). Disabling the wheel contact-frame fix removed the NaN but the robot cannot balance without it, so that is not a control.
What it is. The wheel-ball contact pairs are anisotropic: friction "0.001 1.0" (free rolling along the roller axis, grip across it) with an elliptic cone; the contact-frame fix is what aligns the 0.001 direction with the roller axis. A 1000 to 1 friction ratio in a float32 elliptic-cone Newton solve is ill-conditioned, and it blows up preferentially on the designs where the ball is heavy relative to the body (kb4 0.69, bal025 0.28, zeta66 0.25 per 10 s; nom 0.00). Raising the free-rolling coefficient to 0.05 (PAIR_MU1 knob in common.load_cpu_model, applied to the CPU model so it survives the per-design and per-world friction batching) gives 0 non-finite worlds of 880, and the remaining falls (63, 7 percent) are the real ones (kb025 unseating within a second of the start, S97).
Consequences. (1) Every 20 s standing number in S94, S96 and S97 is void as a controller result; the design-level statements there (heavy-ball designs fail for pure learning too; light ball fails only with the LQR) must be re-measured. The driving numbers are unaffected because the blow-up needs a near-stationary wheel-ball contact. (2) The standing-focused recipe (g25) was an answer to an artefact; the recipe comparison is withdrawn. (3) Training ran with this contact model throughout: the non-finite terminations were rare during motion (training fall rate 0.002 includes them) but the free-rolling coefficient is a physical parameter of the omniwheel rollers, and 0.001 was never measured. The next decision is the value to adopt (smallest that is stable, or a measured roller value), after which the standard battery is re-run on the recommended seeds to see whether anything else moves. Simulation only. Open question: whether MuJoCo CPU (float64) shows the same instability at 0.001.
S98 addendum, stability boundary (same test): free-rolling coefficient 0.001 non-finite 0.127; 0.002 0.001 (1 world of 880); 0.005, 0.01, 0.02, 0.05 all 0.000. Real falls in the same runs 64 to 69 of 880 at every value, so the coefficient does not change the fall count once the blow-up is gone. Decision proposed: adopt 0.01 (ten times the xml value, still "free rolling"; rolling resistance of real omniwheel rollers is of this order and was never measured here) as the model default for all future training and evaluation, and re-run the standard battery and the 20 s standing test of the recommended policy (trained at 0.001) at 0.01 to see what, if anything, moves.
S98 addendum, the recommended policy evaluated at free-rolling friction 0.01 (PAIR_MU1=0.01, files eval2_vel_0.0_mu0.01_flat_noexit.json and eval2_vel_0.3_mu0.01_flat_noexit.json; S94 protocol: flat interior tiles, 20 s, 11 designs x 32 worlds, DR 1, surfaces on, noise on, arrestors on, no pushes):
| policy (trained at 0.001) | standing 20 s at 0.001 (S94 / S96) | standing 20 s at 0.01: all / kb4 / bal025 / kb025 / bal4 | driving 0.3 m/s 20 s at 0.01: all / kb025 / bal4 / nom |
|---|---|---|---|
| g18_ret_rs025 s1 | 0.06 | 0.027 / 0.00 / 0.00 / 0.24 / 0.05 | 0.063 / 0.50 / 0.18 / 0.00 |
| g18_ret_rs025 s2 | 0.49 | 0.039 / 0.00 / 0.00 / 0.29 / 0.13 | 0.075 / 0.54 / 0.23 / 0.02 |
The standing seed lottery is gone (0.027 and 0.039 against 0.06 and 0.49), the heavy-ball designs kb4 and bal025 stand at 0.00, and driving improves slightly (0.063 / 0.075 against 0.09 to 0.11) because the blow-ups were in those numbers too. What remains is real and design-specific: the quarter-mass ball (kb025) unseats within the first second in a quarter of the standing worlds and half of the driving worlds, and the four-times base mass (bal4) falls in a fifth of the driving worlds. Those two are the controller's genuine weak designs and the only standing or flat-driving numbers that survive this correction. Decision: free-rolling friction 0.01 is the model default from here (train_dev --pair-mu1, saved in args.json and re-applied by the evaluation scripts); g26_ret_rs025_mu01 seeds 0 and 1 are training with it. The standard battery of g18 s1 at 0.01 follows. Tracked cleanup: S94, S96 and S97 standing numbers are void and marked so; the site must not state any standing result until the g26 seeds are evaluated.
S98 addendum, standard battery of g18_ret_rs025_s1 at free-rolling friction 0.01 against its own battery at 0.001 (battery_mu0.01.json vs battery.json; same protocol, 16 worlds per kind and design, DR 1 / 1.75 / 2.5, pushes and pulls on and off; flat interior tiles 64 worlds):
| metric (0.001 / 0.01) | DR 1 none | DR 1 P | DR 1.75 none | DR 1.75 P | DR 2.5 none | DR 2.5 P |
|---|---|---|---|---|---|---|
| goal success | 0.76 / 0.77 | 0.43 / 0.44 | 0.72 / 0.72 | 0.39 / 0.39 | 0.66 / 0.68 | 0.32 / 0.32 |
| vel 0.3 fall | 0.35 / 0.33 | 0.56 / 0.52 | 0.39 / 0.34 | 0.59 / 0.57 | 0.46 / 0.39 | 0.65 / 0.63 |
Flat tracking (0.3 / 0.6 / 0.9 m/s): achieved 0.98 / 0.98 / 1.00 at 0.001 and 0.98 / 0.99 / 1.00 at 0.01, late error 0.024 / 0.033 / 0.034 against 0.024 / 0.032 / 0.034, falls 0.00 everywhere. Goal success and flat tracking are unchanged to within 0.02; the velocity-task fall rates drop by 0.02 to 0.07 because the blow-ups were inside those 8 s numbers as well (more so at higher DR scale, where the heavy-ball draws are more extreme). Conclusion: the recommended policy transfers to the corrected contact model without loss, and every obstacle result on the site stands; only the standing claims were artefacts. eval_tiles2 now records a separate `nonfinite` fraction in every cell (new runs only).

## S97 [VOID as a controller result, see S98: every standing fall rate here except kb025's unseating is a solver blow-up] Standing still for 20 s: which designs fall, and whether the LQR is to blame (engine: MuJoCo-Warp float32 GPU; flat interior tiles, zero command, 20 s, 11 designs x 32 worlds, DR scale 1, surfaces on, noise on, arrestors on, no pushes; files eval2_vel_0.0_flat_noexit.json per run; falls)
| controller | seed | all | kb4 | bal025 | kb025 | zeta66 | nom |
|---|---|---|---|---|---|---|---|
| pure learning, arrestors (g16_pure_ret100) | s0 | 0.29 | 0.70 | 0.68 | 0.00 | 0.42 | 0.27 |
| pure learning, arrestors | s1 | 0.22 | 0.70 | 0.74 | 0.00 | 0.19 | 0.13 |
| LQR + residual 0.15 (g18_ret_rs015) | s0 | 0.07 | 0.11 | 0.13 | 0.40 | 0.07 | 0.00 |
| LQR + residual 0.25 (g18_ret_rs025, recommended) | s0 / s1 / s2 | 0.25 / 0.06 / 0.49 | 0.79 / 0.18 / 1.00 | 0.51 / 0.01 / 0.88 | 0.31 / 0.22 / 0.31 | 0.43 / 0.09 / 0.74 | 0.15 / 0.00 / 0.54 |
| LQR + residual 0.25, standing-focused (g25) | s0 / s1 | 0.10 / 0.30 | 0.30 / 0.95 | 0.18 / 0.96 | 0.33 / 0.23 | 0.15 / 0.52 | 0.01 / 0.16 |
| LQR + residual 0.35 (g18_ret_rs035) | s0 | 0.10 | 0.34 | 0.24 | 0.21 | 0.19 | 0.01 |
| LQR + residual 0.5 (g16_ret100) | s0 / s1 | 0.17 / 0.09 | 0.68 / 0.30 | 0.19 / 0.09 | 0.20 / 0.12 | 0.33 / 0.05 | 0.04 / 0.00 |
Driving at 0.3 m/s for 20 s is 0.09 to 0.11 for every LQR-containing policy measured (S94, S96), so everything here is specific to standing.
Findings. (1) Two separate problems. kb4 (four times ball mass) and bal025 (quarter base mass), the two heavy-ball-relative-to-body designs, fall for pure learning as well (0.70, 0.68 to 0.74 on both seeds), so the LQR is not their cause; they are hard to stand for every learned controller and only the low-authority 0.15 controller held them (0.11, 0.13; one seed). kb025 (quarter ball mass) is the reverse: pure learning stands it perfectly on both seeds (0.00) while every LQR-containing controller fails 0.12 to 0.40, so for the light ball the LQR's planar model is the cause. (2) Seed spread dominates the recipe effect for every LQR-containing controller (0.06 to 0.49 across the three standard seeds; 0.10 and 0.30 for the standing-focused pair), so none of the single-seed rows above can be ranked; the only robust statements are the two design-level ones in (1), which hold on every seed. (3) The weakness is invisible to the standard 8 s battery and to driving tests; the 20 s zero-command test is the one that sees it (protocol rule, S96). Next: fall_cause in standing mode on kb4 and bal025 to classify the heavy-ball standing falls (slow derail, tip, or torque saturation), then one of: residual authority that shrinks with the command (0.15 at zero command, 0.25 when driving), or a design-aware LQR that uses the sampled ball mass. Simulation only; worlds per cell 32, so a single design cell has a resolution of about 0.03.

## S95 Ball size against obstacle size: the scale study, first pass (engine: MuJoCo-Warp float32 GPU; policy g24_scale_rs025_s0 = recommended recipe trained on 14 designs including s0.75, s1.5, s2.0; arrestors on; velocity task 0.3 m/s Froude-scaled, 20 s, 640 worlds per terrain kind per design, DR scale 1, surfaces on, noise on, no pushes; falls on the smaller / larger half of each feature's drawn range; absolute feature sizes are the same for every robot)
| robot scale | ball radius | predicted h_max 0.29 r | step up 1.1-2.3 / 2.3-5.9 cm | bump 1.4-2.4 / 2.4-3.6 cm | stairs riser below / above 1.4 cm | large drop 10.9-15.1 / 15.1-18.3 cm | all tiles |
|---|---|---|---|---|---|---|---|
| 0.5 (held out, not trained) | 4.5 cm | 1.3 cm | 0.90 / 1.00 | 0.98 / 1.00 | 0.94 / 1.00 | 0.98 / 0.98 | 0.95 |
| 0.75 | 6.75 cm | 2.0 cm | 0.49 / 0.98 | 0.81 / 1.00 | 0.38 / 0.88 | 0.84 / 0.92 | 0.51 |
| 1.0 | 9 cm | 2.6 cm | 0.34 / 0.96 | 0.67 / 0.97 | 0.17 / 0.70 | 0.66 / 0.73 | 0.33 |
| 1.5 | 13.5 cm | 3.9 cm | 0.30 / 0.69 | 0.42 / 0.56 | 0.20 / 0.38 | 0.41 / 0.56 | 0.26 |
| 2.0 | 18 cm | 5.2 cm | 0.32 / 0.71 | 0.25 / 0.55 | 0.27 / 0.33 | 0.20 / 0.16 | 0.18 |
(Table values re-read from the current stepsize_*.json files, which were regenerated with per-world heights after the first pass; the first-pass values differed by up to 0.17 on single cells because the evaluation draws fresh start positions and the half-range split is a per-sample median.)
Findings. (1) Falls on up-steps and bumps fall monotonically with ball radius, in step with the traction limit: the 2.3 to 5.9 cm step band is fatal for the 6.75 cm ball (1.00), mostly fatal for 9 cm (0.95), and survivable for 13.5 and 18 cm (0.69, 0.58), whose limits (3.9, 5.2 cm) cover most of that band. (2) Drops improve with size beyond what traction explains (0.86 to 0.19 for the larger drops): a bigger robot has more time, in its own Froude time scale, to recover from a 15 cm ledge, which is a smaller fraction of its size. (3) The held-out 0.5x robot (4.5 cm ball) fails almost everywhere, including flat ground: the policy does not transfer that far down, consistent with s0.5 having been unlearnable in S44 era training; small ballbots need their own training and are the hardest case. (4) The scale policy keeps the recommended controller's standard results on the 14 designs (goal 0.72 / 0.47, flat achieved 0.91 to 0.98). Finer crossing-against-height curves (per-world feature heights, four bins or more) and the second seed are being computed.
S95 addendum, falls against step height in 1 cm bins (same runs, per-world feature heights; n per bin 18 to 64):
| ball radius (predicted limit) | step up 1-2 cm | 2-3 | 3-4 | 4-5 | 5-6 | bump 1-2 | 2-3 | 3-4 |
|---|---|---|---|---|---|---|---|---|
| 6.75 cm (2.0 cm) | 0.38 | 0.83 | 0.94 | 1.00 | 1.00 | 0.52 | 0.95 | 1.00 |
| 9 cm (2.6 cm) | 0.29 | 0.50 | 0.89 | 1.00 | 1.00 | 0.48 | 0.77 | 0.97 |
| 13.5 cm (3.9 cm) | 0.31 | 0.28 | 0.56 | 0.72 | 0.79 | 0.33 | 0.47 | 0.56 |
The 50 percent fall point sits near the predicted traction limit for each ball (between 1 and 3 cm for 6.75 cm, 2 to 3 cm for 9 cm, 3 to 4 cm for 13.5 cm), and the floor of about 0.3 at the smallest steps is the controller's general fall rate on terrain over 20 s rather than a traction effect. 18 cm ball and seed 1 pending.
S95 addendum, 18 cm ball and the chart: with per-world heights for all four trained sizes (outputs/study/warp_ppo/review/6_ballsize_vs_step.png, 0.5 cm bins): step-up falls for the 18 cm ball are 0.26 to 0.36 below 3 cm, 0.50 at 3 to 3.5 cm, 0.78 at 4.5 to 5 cm and 0.84 at 5.5 to 6 cm; bumps 0.21 to 0.33 below 2.5 cm, 0.50 at 3 to 3.5 cm, 0.64 at 3.5 to 4 cm. The rise begins below the nominal limit 0.29 r (5.2 cm for 18 cm) because floor friction is randomised per world (surface classes, about 0.4 to 1.2): at mu 0.4 the limit is only 0.07 r, so the measured curve is the traction limit averaged over the friction distribution, which shifts the 50 percent point to roughly 0.18 to 0.22 r for every size. Clips: videos/c7_step4cm_s075 (6.75 cm ball, 5 cm step, falls at 6.6 s) and c7_step4cm_s2 (18 cm ball, same step, crosses).
S95 addendum, seed 1 (g24_scale_rs025_s1, identical recipe and protocol, files g24_scale_rs025_s1/stepsize_*.json): overall fall seed 0 | seed 1: 6.75 cm 0.51 | 0.48; 9 cm 0.33 | 0.35; 13.5 cm 0.26 | 0.26; 18 cm 0.18 | 0.18. Large-half step-up fall (above 2.35 cm): 0.98 | 1.00; 0.96 | 0.98; 0.69 | 0.73; 0.71 | 0.65. Large-half bump fall: 1.00 | 1.00; 0.97 | 0.98; 0.56 | 0.53; 0.55 | 0.45. Every cell agrees within 0.10 and the overall rates within 0.03, so the size ordering and the traction reading are not a seed artefact. Figure 6_ballsize_vs_step.png is now the mean of both seeds with seed-spread bars. Pooling both seeds' per-world arrays, the measured 50 percent fall point spans 0.16 to 0.26 r across sizes and features (mean 0.21 r), slightly wider than the 0.18 to 0.22 r read from seed 0 alone; the design rule (ball radius 3.4 times the tallest step at friction near 1, 4.8 times when friction is unknown) is unchanged. The scale study conclusions stand on 2 seeds; the standing weakness result (S94) and this one are the only two-seed checks of the scale family.
