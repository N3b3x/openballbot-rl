# Annotated bibliography: learning-based and simulation work relevant to ballbots

Compiled 2026-11-04 from web search and page fetches only (no code run, nothing downloaded). Each entry is tagged with how well it was verified:

- [V] = abstract/page fetched and read, details below taken from it.
- [S] = seen only in search snippets or secondary summaries; details may be imprecise.
- [M] = cited from memory of the well-known paper, not re-fetched in this session; treat the venue/year as likely but check.

Where a number is not stated, it says "not stated". No quantitative claim here was invented. Several arXiv IDs in the 26xx range surfaced in search results (the current date is Nov 2026); they are listed as found and flagged [S].

Correction to the task brief: the repo this project builds on, `salehiac/OpenBallBot-RL`, is by Achkan Salehi (independent researcher, ex ISIR/CNRS Sorbonne). Nothing found links it to JHU/Hopkins. Also "Johannes et al." for residual RL is actually Johannink et al. (Siemens/Berkeley); "Silver et al." is Tom Silver (MIT).

---

## 1. RL / learning on ballbots and ball-balancing robots

The literature is very thin. Everything found is listed.

### 1.1 Salehi, "Reinforcement Learning for Ballbot Navigation in Uneven Terrain" [V]
- Author: Achkan Salehi. arXiv:2505.18417, v1 May 2025, v2 30 Jan 2026 (fixes Fig. 4, adds experiments). No peer-reviewed venue listed.
- URL: https://arxiv.org/abs/2505.18417 ; code: https://github.com/salehiac/OpenBallBot-RL (Apache-2.0, ~24 stars, 126 commits).
- Method: MuJoCo ballbot, each omniwheel a capsule with anisotropic tangential friction (high along drive direction, negligible along idler-roller direction). Needs a patch to MuJoCo (`mujoco_fix.patch`, build from source at a recommended commit) for the anisotropic friction to behave. Control 500 Hz, depth cameras ~80 Hz. SB3 PPO, gamma 0.99, GAE 0.95, clip 0.015, ent 0.001, vf coef 2.0, batch 256, 5 epochs, 2048 steps x 10 envs, lr 1e-4 manually decayed, 8e6 steps total. Observation: 15-D proprioception (orientation, ang vel, body vel, wheel speeds, prev action) plus two 128x128 depth cameras aimed at the ball-ground contact, pretrained CNN encoder to 20-D each, plus dt since last image; 56-D total into a 5-layer 128-wide MLP. Reward: 0.01 * v.g + 0.02 survival - 1e-4 * |a|^2; fail at 20 deg tilt. Terrain: Perlin noise (snoise2, scale 25, 4 octaves, persistence 0.2, lacunarity 2.0).
- Results: balance learned by ~3e6 steps; median best eval reward ~85.5, about 0.5 m/s toward goal; 5 seeds. Cascaded PID (tuned on flat) keeps balance ~3000 steps on rough terrain but with near-zero goal-directed velocity reward. ~4.4 h of real-time data per run.
- Limitations stated: sim only, no sim-to-real, no data-efficiency work, weak PID baseline, simplified wheel contact, solver/integrator settings not reported.
- Public: code, trained agent in `trained_agents/`, pretrained depth encoder `encoder_frozen/encoder_epoch_53`, scripts `train.py`, `test.py`, `test_pid.py`, `gather_data.py`, `pretrain_encoder.py`.

### 1.2 Wan et al., "asRoBallet: Closing the Sim2Real Gap via Friction-Aware Reinforcement Learning for Underactuated Spherical Dynamics" [V]
- Authors: Fang Wan, Guangyi Huang, Tianyu Wu, Zishang Zhang, Bangchao Huang, Haoran Sun, Mingdong Chen, Chaoyang Song. arXiv:2604.24916 (27 Apr 2026, v2 7 May 2026), accepted RSS 2026 (roboticsproceedings.org/rss22/p105.html). CC BY-NC-ND.
- URLs: https://arxiv.org/abs/2604.24916 , https://arxiv.org/html/2604.24916 ; videos at bionicdl.ancorasir.com. No code repo found.
- Claims the first end-to-end RL policy deployed on humanoid-ballbot hardware, zero-shot sim-to-real. Hardware built from repurposed quadruped parts, iPhone VIO for station keeping, Jetson Orin for inference.
- Sim details (very relevant to us): MuJoCo, ETH-type omniwheels with 12 free-spinning rollers each modelled explicitly; elliptic cone, impratio 10, Newton solver, noslip_iterations 1; ball-floor sliding mu 1, torsional 0.01, rolling 1e-4; wheel-ball soft rubber-like layer solimp "0.85 0.99 0.003"; timestep 0.002 s, frame_skip 5 (100 Hz control); motor Coulomb friction 80-120 mNm (cites Rezero 42 mNm, Kugle 26 mNm); a low-damped virtual ball joint between torso and ball used in training only.
- RL: SB3 PPO, separate actor/critic MLPs 2x64 tanh, 3 torque outputs in [-1,1]. Velocity-tracking obs 16-D (body vel, command, tilt, COM position, ang rate, prev action); station-keeping obs 17-D (adds position error, sin/cos yaw error). Rewards: lin vel 0.5, yaw rate 0.5, ang vel -0.1, torque -0.1, action rate -0.1, termination 1. DR: ball-floor friction U(0.6,1.2), motor Coulomb friction U(0.08,0.12), gyro noise N(0,0.009). Early termination at 20 deg roll/pitch, 1000-step (10 s) episodes. 8 envs, 4.0e6 env steps, ~2.5 h on a Core i9-13900KF.
- Results (sim, Table II): velocity tracking MAE RL vs LQR: 4.0 vs 7.7 cm/s (home pose); random orientation success 100% vs 56%; random arm pose MAE 6.6 vs 23.1 cm/s, success 100% vs 86%; station keeping under random friction 100% vs 18%, random orientation 100% vs 0%, random arm pose 100% vs 6%. Ablation velocity-tracking success: full 96%, no IMU noise 78%, no friction DR 82%. Hardware: 0.05 m/s MAE on carpet, station keeping within 5 cm (tile) and 3 cm (yoga mat), push recovery from 0.3 m displacement in 7 trials.
- Caveat: numbers come from a fetch through a summarizer of the HTML version; check against the PDF before quoting.

### 1.3 Zhou, Lin, Wang, Zhang, "Learning Ball-balancing Robot Through Deep Reinforcement Learning" [V]
- ICCCR 2021 per Salehi's citation; arXiv:2208.10142 (Aug 2022). https://arxiv.org/abs/2208.10142
- Compound controller: conventional feedback controller plus an RL agent (DDPG per the asRoBallet citation) for recovery from large initial tilts. Simulation only; abstract gives no numbers; claims a wider range of recoverable initial tilts than the model-based controller alone. No code found.
- This is the closest prior art to our residual-on-LQR design.

### 1.4 Buzzetti, Zappetti, Iacca, "A Reinforcement Learning Method to Minimize the Damage on a Falling Ballbot" [S]
- ERF 2024 (as cited in asRoBallet). PPO controlling only arms to reduce fall damage, sim only. Not independently fetched.

### 1.5 Farshidian, Neunert, Buchli, "Learning of Closed-Loop Motion Control" [S]
- IROS 2014 (as cited in asRoBallet). RL refinement of an iLQG controller on the ETH Rezero ballbot, initialised from model-based solution. Not independently fetched.

### 1.6 Carius, Ranftl, Farshidian, Hutter, "Constrained stochastic optimal control with learned importance sampling: a path integral approach" [S]
- IJRR 2022 (as cited by Salehi). Uses ballbot on uneven terrain as a benchmark, proprioceptive only. Relevant as a model-based/learned hybrid baseline idea; not fetched.

### 1.7 Not found
- No Disney ballbot RL paper found despite a direct search. No Isaac Sim/Isaac Lab ballbot paper found. No sim-to-real ballbot paper other than asRoBallet. No work found on morphology co-design for ballbots specifically, or on ballbot step climbing with RL.

---

## 2. Wheeled-legged and wheeled-bipedal RL

### 2.1 Chamorro, Klemm, de la Iglesia Valls, Pal, Siegwart, "Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots" [V]
- arXiv:2402.06143 (Feb 2024; ICRA 2024 per search snippet, the arXiv page lists no venue). https://arxiv.org/abs/2402.06143 ; video https://youtu.be/Ec6ar8BVJh4 . No code repo found.
- Single position-based (not velocity-based) RL formulation, asymmetric actor-critic with privileged sim info, no exteroception at deployment, boolean "stair mode" input. Ascento (a balancing two-wheeled legged robot) climbs 15 cm stairs on hardware; quadrupeds and bipeds in sim. Success rates not retrieved.
- Related: Polytechnique Montreal MSc thesis (Chamorro 2024), https://publications.polymtl.ca/57991/1/2024_SimonChamorro.pdf [S].

### 2.2 Klemm et al., "Ascento: A Two-Wheeled Jumping Robot" [S]
- ICRA 2019, arXiv:2005.11435. LQR stabilisation and driving, 3D-printed topology-optimised parts. Useful as the model-based baseline that RL later replaced.

### 2.3 Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter, "Learning robust autonomous navigation and locomotion for wheeled-legged robots" [V via search]
- Science Robotics 9(89), April 2024, DOI 10.1126/scirobotics.adi9641; arXiv:2405.01792. https://arxiv.org/abs/2405.01792
- Model-free RL plus privileged learning for a locomotion controller that blends walking and driving, plus hierarchical RL navigation, km-scale missions in Zurich and Seville. Code not found.

### 2.4 Contact-triggered RL for wheeled bipedal stair climbing (LimX Tron1) [S]
- arXiv:2509.02986. https://arxiv.org/pdf/2509.02986v3 . Feedforward trajectory instruction plus RL for leg lifting across tire sizes. Only seen as a snippet.

### 2.5 Others seen only as snippets [S]
- ATRos, energy-efficient agile wheeled-legged locomotion, PPO with predictive encoder, zero-shot sim-to-real, arXiv:2510.09980 (platform unconfirmed).
- Ultra Mobility Vehicle, constrained RL on a bicycle robot (track stand, jumps), arXiv:2602.22118.
- Baek, Purushottam, Ramos, "Hybrid LMC: Hybrid Learning and Model-based Control for Wheeled Humanoid Robot via Ensemble Deep RL", arXiv:2204.03159 (submitted to IROS 2022, acceptance unconfirmed) [V for abstract]. LQR plus an ensemble of SAC policies on a wheeled humanoid in MuJoCo; LQR stabilises early training; reports better sample efficiency than comparison methods, no numbers on the page, no code. Very close in spirit to our residual-on-LQR.
- evoBOT, two-wheeled compound inverted pendulum with learning-based control (Lamarr/Fraunhofer) [S], https://lamarr-institute.org/publication/evobot-design-and-learning-based-control-of-a-two-wheeled-compound-inverted-pendulum-robot/
- Wheeled inverted pendulum equilibrium-point estimation via real-to-sim adaptation, arXiv:2403.10948 [S].

---

## 3. Cart-pole / Furuta sim-to-real and domain randomization

- Peng, Andrychowicz, Zaremba, Abbeel, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA 2018, arXiv:1710.06537 [S/M]. Randomize dynamics during training, zero-shot to a real arm pushing task. https://xbpeng.github.io/projects/SimToReal/index.html
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017 [M]. Visual randomization.
- OpenAI et al., "Solving Rubik's Cube with a Robot Hand", arXiv:1910.07113 [S]. Automatic Domain Randomization widens ranges as the policy succeeds; 20% success on maximally hard scrambles (26 face rotations).
- Hwangbo et al., "Learning agile and dynamic motor skills for legged robots", Science Robotics 4(26), 2019, arXiv:1901.08652 [S]. Learned actuator network (MLP, 3 hidden layers x 32) mapping position-error/velocity history to torque; large sim-to-real gain over ideal actuator model.
- Muratore et al. line (TU Darmstadt): "Data-efficient Domain Randomization with Bayesian Optimization" (BayRn), arXiv:2003.02471 [S], tested on a Furuta pendulum swing-up, code released; "Robust Value Iteration for Continuous Control Tasks", arXiv:2105.12189 [S]; Menzenbach thesis benchmarking Sim2Real algorithms on Furuta and ball balancing (TU Darmstadt IAS PDF) [S].
- "Distilled Domain Randomization" (DiDoR), arXiv:2112.03149 [S]: zero-shot on Furuta and cart-pole; attributes return variance to overly aggressive randomization, i.e. wide ranges can hurt.
- Shakerimov et al., "Efficient Sim-to-Real Transfer in RL through ..." IEEE Access 2023 [S]: 20 to 50 extra real episodes on top of DR improves cart-pole etc.
- Zero-shot cart-pole swing-up/stabilisation, arXiv:2606.22145 [S], only the reference list was seen.
- Takeaway: for balancing systems the dominant gaps are actuator dynamics (friction, delay), latency and parameter mass/inertia error; narrow, well-chosen DR plus a learned or identified actuator model beats broad DR.

---

## 4. Residual RL on top of a model-based controller

- Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Aparicio Ojea, Solowjow, Levine, "Residual Reinforcement Learning for Robot Control", ICRA 2019, arXiv:1812.03201 [V via search]. Feedback controller handles the easy part, RL learns the residual for contact-rich, unstable-object block assembly on real hardware.
- Silver, Allen, Tenenbaum, Kaelbling, "Residual Policy Learning", arXiv:1812.06298 (Dec 2018) [V via search]. Residual output initialised to zero so the policy starts equal to the base controller, base may be non-differentiable. Project page with code/video exists (URL not retrieved).
- Zhou et al. 2021 (Sec. 1.3) and Hybrid LMC (Sec. 2.5) are the balance-robot instances.
- asRoBallet shows pure RL beats LQR in velocity MAE and in robustness to orientation/arm-pose/friction shifts (Sec. 1.2), which is the argument against residual being necessary, but also the argument for gating: LQR still wins on flat, nominal cases in some metrics (e.g. 56%/100% success flips only under shifts).

## 5. Hybrid / gated mixture-of-experts policies

No paper found that gates between an LQR and an RL policy for a balancing robot. Nearest relatives [S]:
- SAC-MoE, arXiv:2511.12361: SAC actor replaced by a mixture of experts with learned router; up to 6x zero-shot gain on hybrid racing/locomotion.
- Gated expert switching for contact-rich hybrid systems (dergipark, 2024-25): gate learns validity regions of experts.
- Carius et al., MPC-Net with a mixture-of-experts network, arXiv:2103.14331: imitation of MPC with MoE for multi-modal gaits.
- Fault-tolerant MoE legged RL, arXiv:2606.25965; hybrid expert controller for humanoid combat, arXiv:2604.21355 (uncertainty-based arbitration).
Treat our gated LQR/PPO controller as a gap in the literature rather than a replication; but claim novelty cautiously since the search was one pass.

## 6. Teacher-student, privileged terrain, height scans, ToF

- Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning quadrupedal locomotion over challenging terrain", Science Robotics 2020, arXiv:2010.11251 [S]. Teacher with privileged terrain/contact info, proprioceptive student by imitation.
- Kumar et al., RMA, RSS 2021 [M]. Privileged-env-encoder teacher, adaptation module student.
- Miki et al., "Learning robust perceptive locomotion for quadrupedal robots in the wild", Science Robotics 2022 [M]. Elevation map with belief encoder; teacher-student with noisy map.
- Height-scan teacher of ~1.5 m x 1 m at 5 cm resolution relative to the base: Learning Vision-Based Bipedal Locomotion for Challenging Terrain, arXiv:2309.14594 [S]; bipedal student reconstructing privileged info from history then RL fine-tune, arXiv:2504.00614 [S].
- Salehi (Sec. 1.1) conditions on ball-contact depth images through a frozen pretrained CNN encoder; no teacher-student distillation.
- Multizone ToF sensors (ST VL53L5CX: 8x8 zones, 63 deg diagonal FOV, ~4 m, 60 Hz at 4x4 and about 15 Hz at 8x8 per the ST pages [S]). No RL-with-ToF sim paper found. Our ToF sim is therefore a likely gap-filler.

## 7. Morphology, co-design, universal controllers

- Gupta, Fan, Ganguli, Fei-Fei, "MetaMorph: Learning Universal Controllers with Transformers", ICLR 2022, arXiv:2203.11931 [V via search]. Kinematic tree to token sequence, Transformer encoder, per-joint decoder; pretraining over many morphologies gives zero-shot to unseen robots and faster fine-tuning. Code: https://github.com/agrimgupta92/metamorph .
- Schaff, Yunis, Chakrabarti, Walter, "Jointly Learning to Construct and Control Agents using Deep RL", ICRA 2019, arXiv:1801.01432 [S]. Distribution over designs plus one shared policy; trained on link lengths/radii.
- Luck, Ben Amor, "Data-efficient Co-Adaptation of Morphology and Behaviour with Deep RL", CoRL 2019, arXiv:1911.06832 [S]. Latent-conditioned policy, RL value as design objective.
- Chen et al., "C^2: Co-design of Robots via Concurrent Networks Coupling Online and Offline RL", arXiv:2209.06579 (v3 Nov 2023) [V]. Behaviour-cloning term links online RL and an offline population network; validated on a physical detachable four-legged robot. No numbers in abstract, no code.
- Strgar, Kriegman, "Accelerated co-design of robots through morphological pretraining", arXiv:2502.10862 [S]. Pretrain over body plans to cut co-design sample cost.
- RoboMoRe, LLM-driven morphology plus reward co-design, arXiv:2506.00276 [S]. Chen et al. 2024 IEEE/ASME TMech co-optimisation of morphology and gait for small legged robots [S].
- Amorpheus / shared modular policies: search returned nothing; cite from memory only [M]: Wang et al. NerveNet ICLR 2018, Huang et al. Shared Modular Policies ICML 2020, Kurin et al. AMORPHEUS ICLR 2021.
- None of these cover wheeled or ballbot morphologies; our design-space study (scale, ball mass, COM height, torque, rotor inertia) has no direct precedent found.

## 8. GPU-parallel physics and throughput

- MuJoCo Playground, Zakka et al., arXiv:2502.08844 (RSS 2025 demo) [V]. Throughput of Brax PPO on one A100 (env steps/s): CartpoleBalance ~718,626; CheetahRun ~435,162; WalkerWalk ~139,818; HumanoidWalk ~91,563; Go1 flat joystick ~417,451; G1 joystick ~106,093; BerkeleyHumanoid rough ~30,393; PandaPickCube ~140,386; LeapCubeReorient ~76,354. Wall clock: Go1 joystick ~5 min and Berkeley Humanoid <15 min on 2x RTX 4090. Per-env MJX XML tweaks: cartpole-balance iterations=1, ls_iterations=4; walker-stand timestep 0.005, iterations 2; locomotion uses feet-only reduced collision. No direct MJX vs CPU MuJoCo comparison in the portion read. https://playground.mujoco.org/ ; repo: github.com/google-deepmind/mujoco_playground (URL from memory).
- MJX docs [V]: MJX-JAX "can be up to 10x slower than CPU MuJoCo for a single scene"; best with thousands to tens of thousands of parallel scenes; cost scales with possible contacts; list explicit `contact/pair`; lower `iterations`/`ls_iterations`; reference humanoid SPS 650K (M3 Max CPU), 1.8M (AMD 3995WX), 950K (A100 MJX-JAX, batch 8192), 2.7M (TPU v5, batch 16384) as the page's four values (hardware mapping is by listed order, not explicit). https://mujoco.readthedocs.io/en/stable/mjx.html
- MJX-Warp / MuJoCo Warp [V docs]: humanoid ~2.67M to 3.35M SPS depending on JAX FFI graph mode, Aloha pot 1.96M to 2.45M. Supports all geoms and joints and solvers except PGS and **noslip**. Performance degrades for scenes beyond ~60 DoF; MJWarp optimises throughput, not latency. https://mujoco.readthedocs.io/en/stable/mjwarp ; https://github.com/google-deepmind/mujoco_warp . No standalone MuJoCo Warp paper found.
- Implication: asRoBallet and Salehi both use features that are fragile on GPU: asRoBallet uses `noslip_iterations=1` (unsupported in MJX-Warp), Salehi relies on a patched CPU MuJoCo for anisotropic capsule friction (a patch on CPU MuJoCo does not carry to MJX). A small ballbot (nv around 6 plus ball, 3 wheels) is a small scene; GPU throughput comes only from many thousand parallel worlds.
- Isaac Gym, Makoviychuk et al., arXiv:2108.10470 [S]: "2-3 orders of magnitude" over CPU simulators. Isaac Lab, Mittal et al. (NVIDIA, 105 authors), arXiv:2511.04831 [S], BSD-3, integrates Newton/MJWarp. Isaac Lab has a guide for tuning MJWarp (docs.nvidia.com/isaaclab/isaac-lab/how-to-guides/tune-mj-warp.md) [S].
- Brax, Freeman et al., NeurIPS 2021 D&B, arXiv:2106.13281 [S]: JAX simulator with built-in PPO/SAC/ES; spring-based joints and Euler integration need careful tuning.
- Genesis: no formal paper found. Project claims 10-80x over GPU simulators; a survey cites 2.70x to 11.79x over Isaac Gym; Stone Tao critique says original benchmark used physics substeps=1 and was inflated, no public MJX comparison [S]. Treat as unverified.

## 9. Stiff omniwheel and ball contacts in simulation

What was found (mostly community guides, not authoritative):
- asRoBallet recipe (Sec. 1.2): explicit rollers, elliptic cone, impratio 10, Newton, noslip 1, soft wheel-ball layer solimp "0.85 0.99 0.003", dt 0.002 with frame_skip 5.
- Salehi recipe: single capsule with anisotropic friction (needs patched MuJoCo), 500 Hz control.
- General: solref sets time constant/damping ratio; community starting points are 0.004/1 for rigid links, 0.015 to 0.02 /1 for rubber wheels; if the sim explodes, soften solref or reduce timestep [S, robolaweb guides]; verify with the official MuJoCo modelling docs.
- A MuJoCo GitHub issue (#1649) discusses anisotropic slip on soft contacts, unresolved in the snippet [S].
- Gap: no published study of roller-count vs fidelity vs speed trade-off. Our own ablation of this would be new.

---

## Ideas to borrow / where we stand / our edge

### Where we stand
- Prior art on ballbot RL is two real works: Salehi (2025, uneven terrain, sim only, open code) and asRoBallet (RSS 2026, sim-to-real, no code). Everything else is small-scale (DDPG assist, fall-damage arms, iLQG refinement). Our project already overlaps both: Perlin terrain and depth/ToF like Salehi, velocity tracking and robustness DR like asRoBallet, residual-on-LQR like Zhou 2021 / Hybrid LMC.
- Our sim speed (~500-800 steps/s per run on CPU) is in line with a single-process MuJoCo step; asRoBallet reports 4e6 steps in ~2.5 h with 8 envs (about 440 steps/s aggregate, so we are faster per run, plausibly because of config differences; not verified) and Salehi used 8e6 steps with 10 envs. GPU (MJX-JAX) is documented as up to 10x slower than CPU for single scenes and only pays with thousands of envs; for a 3-wheel ballbot that means a rewrite and loss of noslip (MJX-Warp) or custom friction. CPU multiprocessing remains the sensible default.

### Ideas to borrow
1. asRoBallet DR set: friction U(0.6,1.2), motor Coulomb friction 80-120 mNm, gyro noise 0.009 rad/s. Ablations (no IMU noise: 78%, no friction DR: 82% vs 96%) show which randomization matters most; copy those two first. Add actuator friction/delay in the Hwangbo spirit (actuator network) if we ever go to hardware.
2. asRoBallet virtual low-damped ball joint during training only, to ease early learning; test as a curriculum aid for the hybrid controller and for climbing.
3. asRoBallet evaluation protocol: success rate and velocity MAE over fixed perturbations (random orientation, arm pose, friction), with LQR as the baseline. Reusing the same table layout makes our numbers directly comparable (LQR MAE 7.7 cm/s vs RL 4.0 cm/s home pose is a concrete target).
4. Residual policy learning trick (Silver): zero-initialise the final layer of the residual head so training starts at the LQR performance; Johannink: bound the residual magnitude.
5. Teacher-student (Lee 2020, Miki 2022): train the height-scan teacher, distil to the ToF student with noise/dropout on the zone grid; use asymmetric actor-critic like Chamorro so that the critic sees privileged terrain.
6. Chamorro: position-based/mode-flag formulation for stairs. A boolean "climb mode" input may help our 2-3 cm step climbing where velocity-based reward is insufficient.
7. Morphology: Schaff style design distribution with one policy is the cheapest path for our pool; MetaMorph-style tokenisation is overkill because our morphologies share topology and differ in scalar parameters (scale, ball mass, COM height, torque, rotor inertia). Simply feeding the normalised design vector to actor and critic (Luck and Ben Amor latent conditioning) is the proven minimal approach. Consider morphological pretraining (Strgar and Kriegman) to cut cost of the design study.
8. DiDoR caution: very wide DR raises return variance; use ranges per-morphology or ADR (OpenAI) rather than one blanket range.
9. Gating literature (SAC-MoE, MPC-Net MoE): use load-balancing and entropy regularisers on the gate; consider uncertainty-based fallbacks to LQR (arbitration idea in arXiv:2604.21355).
10. Contact fidelity: try asRoBallet's explicit rollers with noslip and impratio 10 as a high-fidelity reference, and evaluate whether the cheaper capsule/anisotropic model reproduces the same climbing limit (Section 9 gap). Make step-climbing limit a function of the friction mu range used in DR (their ball-floor mu spans 0.6-1.2).

### Our edge (claims we can defend, given what the search found)
- Hybrid gated LQR/PPO with morphology-pool training: no precedent found. Closest are Hybrid LMC (residual, single morphology) and MoE-RL for legged robots.
- Parametric design-space study of ballbots with RL-trained controllers (scale, ball mass, COM height, torque, rotor inertia): no precedent found for ballbots; co-design literature is legged or generic.
- ToF multizone sensor simulation plus height-scan teacher on a ballbot: none found. Salehi uses depth images; asRoBallet is blind except for VIO.
- Quantified traction-limited step climbing (2-3 cm) for a ballbot: not found in the literature; Ascento climbs 15 cm but with legs.
- Combination of velocity-integral outer loop with PPO velocity-command tracking is not described in either ballbot paper (asRoBallet tracks velocity directly and uses VIO for station keeping).
- Reproducibility: asRoBallet has no released code; our repo plus Salehi's Apache-2.0 base give open, runnable baselines.

### Honest caveats
- asRoBallet beats LQR strongly under shifted conditions; our residual/gated design must show benefit beyond a well-tuned pure PPO, so include pure PPO and pure LQR as baselines in the same table.
- We have only simulation evidence; asRoBallet's contribution is real hardware. Do not claim sim-to-real.
- Sim-to-real fidelity of our step-climbing result depends on contact model; Section 9 shows no validated recipe exists in the literature beyond asRoBallet's rollers.
- Search tool results may be incomplete: one pass per topic, US-only index, and several 2026 arXiv IDs are seen only through snippets. Verify before citing.

### Unverified / to check before citing
- Venue/year of Zhou et al. (ICCCR 2021 vs arXiv 2022), Buzzetti ERF 2024, Farshidian IROS 2014, Carius IJRR 2022 (all via asRoBallet/Salehi citation lists).
- Whether ICRA 2024 accepted Chamorro et al.; success rates (not retrieved).
- All 26xx arXiv IDs (2606.22145, 2606.25965, 2604.21355, 2602.22118, 2602.22118) and all "M" items.
- MJX CPU/GPU hardware mapping of the 650K/1.8M/950K/2.7M numbers.
- Genesis speed claims.
