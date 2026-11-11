# Literature review: RL state of the art for a robust, goal-directed ballbot

Scope: recent (2022 to mid 2026) RL results relevant to a ballbot trained with GPU PPO (4096 worlds, 100 Hz,
22-D proprio obs, 256-256 tanh MLP, no history). Written 2026-11-05.

## Evidence legend and honesty statement

- [V] = I fetched the source page (arXiv HTML/abstract) in this session and the claim is read from it.
  Note: for asRoBallet and FastTD3 the fetch returned full-text extractions; for FastSAC-humanoid only the abstract.
- [S] = seen only in a search-result snippet or third-party summary; the primary text was not read.
- [M] = from my background knowledge of the field, NOT checked against a source in this session. Treat as a lead.
- Search was limited (about 17 queries, 2 full-text fetches). Several requested topics (action chunking, temporal
  ensembles, ADR for legged robots, residual-over-LQR, PPO adaptive-KL schedules, ANYmal-parkour details, wheel-legged
  jumping) returned nothing citable and are marked [M] or "not found". Do not cite [M] items in the paper without
  checking the original. Where a number is given it is from the cited source; GPU-hour estimates are my own guesses.
- Citation hygiene: the arXiv IDs below came from search results; verify the 2026 IDs (asRoBallet 2604.24916,
  SSR 2605.30770) before citing, since I could not cross-check them against a publisher record.

## 0. Most relevant single paper: asRoBallet (RSS 2026)

asRoBallet, "Closing the Sim2Real Gap via Friction-Aware RL for Underactuated Spherical Dynamics",
arXiv 2604.24916, https://arxiv.org/abs/2604.24916 , html https://arxiv.org/html/2604.24916 [V].

Read from full text:
- Real ETH-type omniwheel ballbot, MuJoCo sim modelling discrete rollers, zero-shot transfer. SB3 PPO, 8 envs x 2048
  steps (16,384 samples/update), minibatch 512, 10 epochs, MLP 2x64 tanh, separate actor/critic, 4e6 steps (~2.5 h CPU).
- Obs (16-D velocity tracking): body-frame velocity, command, tilt, CoM position in body frame, gyro, previous action.
  Only a_{t-1} as history. No explicit latency randomisation (measured link latency ~1.2 ms).
- Randomisation: ball-floor tangential friction U(0.6,1.2), motor Coulomb friction U(0.08,0.12) Nm, gyro noise
  N(0,0.009). The 80-120 mNm Coulomb friction produces a wheel torque dead zone and station-keeping limit cycles.
- A training-only "virtual ball joint" (low-damped joint between torso and ball) was needed for training to converge
  (authors attribute it to a sparse reward landscape); removing it after convergence did not help.
- Sim results (MAE cm/s velocity; success): RL 3.8-6.6 vs LQR 7.5-23.1; LQR station keeping fails (0 to 18 percent)
  under random friction/orientation/arm pose where RL gets 100 percent. RL station-keeping range 4.3-5.0 cm.
- Caveat: these are the authors' sim comparisons; hardware numbers I did not extract.

What to try here: (i) their obs contains CoM position in body frame, which gives the policy a built-in handle on
mass-distribution variation; our design randomisation (scale, masses) likely needs the same. (ii) Dead zone is a
dominant real effect; a motor-model with Coulomb friction + dead zone in sim is directly on the critical path
(configs dz_* already exist in this repo; compare against their ranges). (iii) Their policy is tiny (64x64) and trained
on only 4e6 steps, a reminder that capacity is probably not our bottleneck.

Other ballbot RL papers:
- Salehi, "Reinforcement Learning for Ballbot Navigation in Uneven Terrain", arXiv 2505.18417 (v2 Jan 2026),
  https://arxiv.org/html/2505.18417v2 [S]. MuJoCo, PPO, 500 Hz, 4-5 h of data, simulation only, no hardware.
  Repo: https://github.com/salehiac/OpenBallBot-RL . Useful as baseline and observation/reward conditioning reference.
- "Learning Ball-balancing Robot Through Deep RL", arXiv 2208.10142 [S: title only]. DDPG, CoppeliaSim, sim only.

## (a) PPO recipe for massively parallel GPU locomotion

1. Batch size is the first-order PPO knob in the massively parallel regime. Rudin et al., "Learning to Walk in Minutes",
   CoRL 2021 / PMLR 164, arXiv 2109.11978, https://arxiv.org/abs/2109.11978 [S]: batch size B (envs x steps/iter) is
   "crucial", millions of samples per update; large batches help learning dynamics but not wall clock. Flat in under 4
   min, rough in ~20 min on ANYmal [S]. Try: with 4096 worlds sweep horizon 24 / 48 / 96 (batch 98k / 196k / 393k) with
   minibatch count fixed at 4. Note asRoBallet used only 16k per update and still worked for a flat task, so the right B
   is task-dependent (steps/terrain are harder than flat). [V for asRoBallet numbers]
2. Adaptive learning rate by KL (rsl_rl / legged_gym default style: target KL ~0.01, lr multiplied/divided by 1.5), advantage
   normalisation per minibatch, empirical running obs normalisation, init action std ~1.0 with learned state-independent log
   std, entropy coefficient ~0.005-0.01, value-loss clipping, gamma 0.99, lambda 0.95, clip 0.2, 5 epochs, 4 minibatches.
   [M]. I found no paper that ablates these; the Rudin paper is the closest [S]. Action: confirm which of these the
   current train_ppo.py has (see scripts/study/train_ppo.py) and add the missing ones; an A/B on adaptive-lr vs
   fixed-lr is cheap and often decides stability on a 4096-world run.
3. Asymmetric actor-critic (privileged critic) is standard in recent humanoid PPO: Booster Gym arXiv 2506.15132 [S],
   SSR arXiv 2605.30770 [S] (critic gets base velocity, contacts, height map), contrastive-representation sim-to-real
   arXiv 2509.12858 [S]. For wheel-legged climbing Chamorro et al. (Ascento, arXiv 2402.06143) used privileged
   terrain info in an asymmetric AC [S]. Action: give the critic true ball slip, ball-floor friction, true masses/COM,
   motor lag and command, terrain height ahead; keep the actor realistic. This is a free win vs the "teacher gave no
   gain" result because the critic does not change deployment observability.
4. Symmetry: mirror augmentation / symmetric PPO objective (Hakoda et al., Frontiers, [S]; humanoid papers
   2508.01247 [S]). The ballbot has a 3-fold wheel symmetry plus x/y reflection symmetry in the plane: augment each batch
   by reflecting (x,y) and by 120-degree rotations of the world frame (needs rotating obs, actions, and terrain scan).
   Expected benefit: sample efficiency and removal of directional bias in tracking error. [M for the effect size]
5. Bootstrapping on time-outs: truncation (episode time limit) must bootstrap V(s_T) rather than treat as terminal;
   mistakes here cause bias in speed-tracking tasks with fixed-length episodes. [M] Check the env's `truncated` handling.
6. Observation/reward normalisation: FastTD3 text did not mention obs normalisation [V (absence)], PPO stacks
   typically use running mean/std obs norm [M]. Reward-scale mismatch matters: FastTD3 shows the same reward produced
   different gaits for PPO and TD3 and reward tuning for each algorithm changed results [V]. Do not transfer reward
   weights across algorithms.

## (b) Off-policy at scale: FastTD3, FastSAC, PQN, SimbaV2

1. FastTD3 (Seo et al., arXiv 2505.22642, https://arxiv.org/html/2505.22642v3) [V]. Recipe: thousands of envs, batch
   32,768, UTD of 2-8 updates per 128-4096 env steps, distributional critic (v_min/v_max), clipped double Q (min works
   best without LayerNorm), 3-layer plain MLPs (critic 1024-512-256, actor 512-256-128), per-env Gaussian noise with
   sigma in [sigma_min, sigma_max], sigma_max 0.4, replay N x num_envs on GPU. Reported faster than PPO in IsaacLab and
   MuJoCo Playground, especially rough terrain with domain randomisation; the advantage is wall-clock, not universal
   [V/S]. PPO won on gait naturalness with the PPO-tuned reward [V]. Sim-to-real to Booster T1 [S].
2. FastSAC / sim-to-real humanoid in 15 minutes (Seo et al., arXiv 2512.01996, https://arxiv.org/abs/2512.01996)
   [V abstract; S for details]. Large batches (up to 8K), joint-limit-aware action bounds, <10 reward terms; FastSAC/FastTD3
   competitive with or better than PPO on motion tracking, deployed on real G1. Worth noting for us: the "joint-limit-aware
   action bounds" analogue is torque saturation of the omniwheel motors.
3. SimbaV2 (Lee et al., arXiv 2502.15280, https://arxiv.org/html/2502.15280v2) [S]: hyperspherical (L2) normalisation of
   features and weights with distributional critic on SAC; SOTA on 57 continuous tasks across 4 suites; no weight decay or
   resets needed. Not tested by me on locomotion; benchmark evidence is on standard suites, not rough-terrain robots.
4. PQN (Gallici et al., ICLR 2025, arXiv 2407.04811, https://arxiv.org/html/2407.04811v2) [S]: LayerNorm makes online
   TD stable without target nets or replay; but it is a discrete-action method (Atari/Craftax/SMAX); applicability to a
   continuous 3-wheel torque action is unclear. Low priority.
5. When do off-policy methods beat PPO here? Evidence supports: rough terrain with heavy randomisation, wall-clock
   to a given return, and high-UTD sample efficiency. Evidence against switching now: our remaining problems (step
   crossing at low speed, speed-tracking vs perception) look like reward/observability issues, not optimiser issues. I
   would run FastTD3 as a one-off sanity comparison (experiment 9), not a rewrite. Open risk: off-policy with
   non-stationary curricula and per-env randomisation is touchier [M].

## (c) Teacher-student / privileged learning

1. RMA (Kumar et al., RSS 2021, arXiv 2107.04034, https://arxiv.org/abs/2107.04034) [S]: base policy conditioned on a
   learned extrinsics vector z; adaptation module regresses z from recent state-action history by supervised learning;
   async rates. Older than the 2022 window but the template for everything below. Try: z-dim 8-16 from the privileged
   physical params (mass, COM offset, friction, motor gain/lag, dead zone, wheel inertia); history 50 steps at 100 Hz
   (0.5 s) by a 1D conv.
2. Extreme Parkour (Cheng et al., arXiv 2309.14341, https://arxiv.org/abs/2309.14341) [S]: phase 1 teacher with
   privileged "scandots" (ground-truth terrain samples) in sim; phase 2 distils to a depth+proprio student by supervised
   action matching; unified inner-product reward so one net does diverse skills. I did not read the loss details.
3. ANYmal Parkour (Hoeller et al., Science Robotics 2024, arXiv 2306.14874, https://arxiv.org/html/2306.14874v1) [S]:
   modular perception / locomotion / navigation, up to 2 m/s on hardware. Relevant structural idea: a high-level
   navigation policy selects among locomotion skills; for us a "step-approach" skill vs "balance" skill.
4. Why our height-scan teacher may have given no gain: with a reward that only pays speed tracking there is no incentive
   to exploit terrain foresight; a teacher needs to be rewarded for traversal (e.g. terrain-level progress, no-fall) and
   evaluated on terrain, not on flat speed error. This is an inference from our finding plus the above papers, not a
   published result. Test: train with terrain-conditioned curriculum and measure step success at 0.2 / 0.4 / 0.6 m/s.
5. DAgger vs behaviour cloning for distillation: DAgger-style on-policy student rollouts is the usual fix for covariate
   shift [M]; the parkour papers use on-policy student rollouts with teacher labels [S].

## (d) History, recurrence, transformers, state estimation

1. Radosavovic et al., "Real-World Humanoid Locomotion with RL", arXiv 2303.03381 (Science Robotics 2024),
   https://arxiv.org/abs/2303.03381 [S]: causal transformer over history of proprio obs and actions, trained by
   PPO on randomised sim, zero-shot, in-context adaptation. A 16-step history in a small MLP or 1D-conv is the cheap
   version; I expect most of the benefit for dead zone / friction / latency identification [M].
2. Chamorro et al. 2024 (Ascento) [S] and RMA [S] both say the policy needs access to what the last actions did; our current
   policy has a_{t-1} at best (asRoBallet too [V]).
3. OpenAI ADR paper found memory-augmented (LSTM) policies show emergent meta-learning under heavy randomisation
   (arXiv 1910.07113, https://arxiv.org/abs/1910.07113) [S]. Lesson: memory pays off when randomisation is wide
   (our scale/mass/grip/surface goals).
4. State estimation: the ball's slip and ball-floor velocity are not observable from IMU + encoders alone; a learned
   estimator (regress ball velocity, slip, friction coefficient from 0.5 s history, auxiliary loss on the policy trunk)
   is the cheap way to add privileged information. [M] (no ballbot source found).
5. Cost warning: GRU/transformer in PPO with 4096 envs increases step cost and complicates minibatching (needs sequence
   chunks). Prefer stacked-history MLP / conv first.

## (e) Action representation and smoothness

1. CAPS (Mysore et al., ICRA 2021, arXiv 2012.06644, https://arxiv.org/abs/2012.06644) [S]: temporal term
   ||a_t - a_{t+1}|| plus spatial term (action similarity under Gaussian state perturbation) as loss additions; smoother
   control, ~80 percent power reduction on a quadrotor. The Lipschitz framing I could not confirm from snippets. Try:
   lambda_T ~ 0.01 to 0.1, lambda_S ~ 0.01 to 0.1, sweep, check torque-rate and dead-zone jitter. The asRoBallet limit cycle
   is exactly the failure CAPS-like penalties plus a deadzone-aware motor model should target. [V for asRoBallet issue]
2. Low-pass filtered actions / action-rate penalties are the dominant practical alternative in legged RL [M]. Cheap:
   a = alpha*a_prev + (1-alpha)*pi with alpha learned or fixed 0.5-0.8. Randomise alpha across training for robustness.
3. Residual over a model-based controller (LQR/MPC): I found no citable paper for the ballbot case. asRoBallet
   reports RL beats their LQR in sim, particularly under friction/orientation/arm-pose variation (LQR station keeping
   0-18 percent success there) [V], which argues against LQR as a hard base; as a residual with a learned gate or
   curriculum it can still help early learning (this repo has controllers/lqr.py). Treat as a hypothesis: expect faster
   convergence and a better station-keeping floor, with a risk of capping peak performance.
4. Action chunking / temporal ensembles (ACT-style): developed for imitation learning manipulation [M]; not shown for
   100 Hz balance RL. Not recommended; balance is high-bandwidth. A lower-rate policy (50 Hz) with held output is the
   sensible variant to test because it mirrors real latency.
5. Joint-limit-aware action bounds from the FastSAC humanoid paper [S]: bound normalised torque by motor limits
   including back-EMF speed derating so the policy does not command impossible torque. Direct fit for omniwheel motors.

## (f) Domain randomisation, adaptation, curricula, goal-conditioning

1. DR vs SysID vs adaptation: sim-to-real best practice combines measured system identification for the nominal
   model (motor constants, dead zone, friction; asRoBallet measured Coulomb friction at 80-120 mNm [V]) with DR around it,
   plus history/adaptation for residual mismatch [M synthesis].
2. Hwangbo et al. 2019 (Science Robotics, arXiv 1901.08652, https://arxiv.org/abs/1901.08652) [S]: learned actuator network
   (MLP 3x32 on recent position error and velocity histories) was necessary; ideal-actuator training failed. Try: fit a
   small actuator net on the real motor data once available; until then randomise lag, dead zone, Coulomb friction, torque
   constant and back-EMF.
3. ADR (OpenAI, arXiv 1910.07113) [S]: automatically widen each randomisation range when performance on its boundary
   exceeds a threshold; improved sim-to-real. For legged robots I found no source in this session. Try per-parameter ADR
   on mass, friction, motor gain, lag, with a boundary-evaluation fraction of 10-20 percent of envs.
4. Terrain curriculum: Rudin et al. game-inspired curriculum [S]: promote an env's terrain level when it traverses far
   enough, demote when it covers under half the commanded distance; no tuning needed. Directly applicable to step height
   (0 to 5+ cm), ramp angle, roughness.
5. Goal-conditioned / multi-task: ANYmal parkour (separate skills + navigation policy) [S]; Extreme Parkour (single
   network, unified reward) [S]. For the ballbot: one network conditioned on command type (velocity | goal position)
   with a one-hot mode and shared observation, mixing 70/30; an easy first step, no source tested this on a ballbot.
6. Push/pull robustness: random impulses to the torso with magnitude curriculum plus critic seeing the push force [M];
   evaluate with held-out impulse set.

## (g) Wheeled / balancing / wheel-legged sim-to-real

1. asRoBallet: see section 0 [V].
2. Chamorro et al., "Learning Quadrupedal... Ascento stairs" (arXiv 2402.06143, https://arxiv.org/abs/2402.06143)
   [S]: wheel-legged balancing robot learned blind 15 cm stair climbing in real world via step reflexes with privileged
   terrain info in an asymmetric AC. This is the nearest published analogue for "blind step crossing needs reflex learning".
   Lesson to test here: add a contact/impulse-triggered reflex term or a privileged-critic that sees step edge distance.
3. CTBC (arXiv 2509.02986, https://arxiv.org/abs/2509.02986) [S]: contact-triggered blind climbing for wheeled bipeds with
   instruction learning; 20 cm stairs on Tron1; trained in Isaac Gym, cross-checked in MuJoCo. Contact-triggered behaviour
   switching might transfer in spirit; a ballbot has no legs, so only the trigger-and-boost idea carries over.
4. Ballbot-specific hardware dead zone and friction dominate sim-to-real error [V]; wheel-legged papers mostly deal
   with leg mechanics, so transfer is partial.
5. I did not find ballbot papers on perception-guided step crossing; this project is likely in unexplored territory,
   and published "2 cm step needs 0.6 m/s approach" style numbers have no counterpart that I can cite.

## (h) Morphology-conditioned / universal controllers

1. MetaMorph (Gupta et al., ICLR 2022, arXiv 2203.11931, https://arxiv.org/abs/2203.11931) [S]: transformer over a modular
   body graph, pretrain across morphologies, zero-shot to unseen designs, few-shot transfer. Not directly usable for a
   fixed ballbot topology, but the idea of design parameters as tokens is.
2. For ours (scale, masses, torque, wheel inertia, grip): give the policy (actor and critic) a normalised design vector
   (log scale, mass fractions, torque limit, wheel inertia, friction) as input, or feed it to the critic only and let
   the history encoder (d) infer it for the actor. Start with the critic-only version [M]. Published evidence of
   design-conditioned vs blind-robust policies for balancing robots: not found.
3. Scale generalisation needs scale-invariant normalisation of obs and torque (normalise by sqrt(g/L) and m g L); cheap
   and principled [M].

## Interpretation of our two observed findings (hypotheses, not literature results)

- Blind crossing needs ~0.6 m/s approach: consistent with a policy that has no memory of contact and no way to sense the
  step; adding history (d), privileged critic (a3), and rewarding traversal (c4) all address this.
- Height-scan teacher no gain: reward only measures speed error, no value in foresight (c4). Also a ballbot's step
  crossing is a transient (momentum) event that a 256-256 MLP without history cannot time.

## Ranked experiments (GPU hours are my guesses; assume ~1 GPU-hour per 4096-world run of the current length;
rescale to your measured throughput)

Metric definitions: TE = RMS velocity tracking error (m/s) on a fixed 50-command grid (|v| 0.2-1.5 m/s), 5 seeds;
SC(h,v) = step-crossing success rate for step height h at approach speed v over 500 episodes; PUSH = max survived
impulse (N s) at 90 percent survival; DESIGN = worst-quartile TE across 200 held-out robot designs; SURF = worst surface TE.

1. Asymmetric critic with privileged physics + terrain (a3). Cost ~6 GPU h (3 seeds x 2 variants). Metric: TE, PUSH,
   DESIGN vs baseline; expect lower variance. Pass if DESIGN worst-quartile TE falls by >10 percent.
2. Observation history (16 steps stacked, or 1D-conv) with action history (d1, c1, section 0). Cost ~8 GPU h.
   Metric: SC(2 cm, 0.2 m/s) and DESIGN, SURF. Baseline reference: current 0.6 m/s requirement.
3. RMA-style two-stage (privileged z, then adaptation module) (c1). Cost ~12 GPU h (teacher + distill + eval). Metric:
   DESIGN and PUSH vs experiment 2; distillation gap = TE(student) - TE(teacher) under 5 percent.
4. Motor model fidelity: Coulomb friction + dead zone + lag randomised (g, f2; asRoBallet ranges 80-120 mNm [V]) plus
   torque-bound derating. Cost ~5 GPU h. Metric: station-keeping position std (cm) and limit-cycle amplitude with the
   dead-zone model ON in eval; compare to asRoBallet 4.3-5.0 cm [V].
5. CAPS + action filter (e1, e2). Cost ~6 GPU h (small grid on lambda). Metric: mean |delta action|/step, TE not worse by
   more than 5 percent, station-keeping jitter amplitude.
6. Terrain-progress reward + game-style curriculum on step height (c4, f4). Cost ~8 GPU h. Metric: SC(h,v) surface for
   h = 1..5 cm, v = 0.2..1.0 m/s; success at 0.3 m/s, 3 cm is the target.
7. PPO hygiene sweep: batch size (horizon 24/48/96), adaptive KL lr vs fixed, entropy/std schedule, time-out bootstrap check
   (a1, a2, a5). Cost ~10 GPU h. Metric: TE at fixed wall-clock budget and seed variance (std over 5 seeds).
8. Symmetry augmentation (3-fold + mirror) (a4). Cost ~6 GPU h. Metric: directional bias = max over heading of TE minus
   mean, and sample efficiency to TE under 0.1 m/s.
9. FastTD3 baseline (b1). Cost ~10-15 GPU h including porting effort; reward re-tuning required [V]. Metric: TE and
   SC at matched wall-clock; do only after 1-8 saturate.
10. Per-parameter ADR on mass/friction/motor gain/lag, plus goal-conditioned mode input (f3, f5). Cost ~15 GPU h. Metric: DESIGN
    and SURF worst-case, and range widths reached; goal-reaching success at 1 m, 3 m.

Notable omissions I did not evaluate: LSTM/GRU policies (d5), residual LQR (e3, no source), action chunking (e4).
Order rationale: 1, 2, 4 attack observability and sim fidelity, which asRoBallet and Ascento both point to; 7 is cheap
insurance; 9 and 10 are expensive and speculative.

## Source list (all URLs from search results; verification status in text)

- https://arxiv.org/abs/2604.24916 (asRoBallet) [V]
- https://arxiv.org/html/2505.18417v2 (Salehi) [S]
- https://arxiv.org/abs/2208.10142 [S]
- https://arxiv.org/html/2505.22642v3 (FastTD3) [V]
- https://arxiv.org/abs/2512.01996 (FastSAC humanoid) [V abstract]
- https://arxiv.org/html/2502.15280v2 (SimbaV2) [S]
- https://arxiv.org/html/2407.04811v2 (PQN) [S]
- https://arxiv.org/abs/2109.11978 (Rudin) [S]
- https://arxiv.org/abs/2107.04034 (RMA) [S]
- https://arxiv.org/abs/2309.14341 (Extreme Parkour) [S]
- https://arxiv.org/html/2306.14874v1 (ANYmal Parkour) [S]
- https://arxiv.org/abs/2303.03381 (Humanoid transformer) [S]
- https://arxiv.org/abs/2012.06644 (CAPS) [S]
- https://arxiv.org/abs/1901.08652 (Hwangbo actuator net) [S]
- https://arxiv.org/abs/1910.07113 (OpenAI ADR) [S]
- https://arxiv.org/abs/2203.11931 (MetaMorph) [S]
- https://arxiv.org/abs/2402.06143 (Ascento stairs) [S]
- https://arxiv.org/abs/2509.02986 (CTBC) [S]
- https://arxiv.org/abs/2506.15132 (Booster Gym), https://arxiv.org/abs/2605.30770 (SSR), https://arxiv.org/abs/2509.12858,
  https://arxiv.org/abs/2508.01247, https://arxiv.org/abs/2404.05695 (Humanoid-Gym) [S]
- https://arxiv.org/abs/2502.08844 (MuJoCo Playground) [S]
