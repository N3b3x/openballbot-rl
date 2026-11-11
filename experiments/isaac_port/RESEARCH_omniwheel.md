# Omni-wheel modelling in PhysX / Isaac Sim: research notes (simulation only)

Source: subagent web research, saved by the lead in condensed form. Tags: [V] stated by a source the agent read (web tools returned summaries, so not a full read), [I] inference, [U] unknown. Nothing here was run.

## Findings
- **Anisotropic friction is absent in PhysX 5 [V].** Deprecated in PhysX 3.2 with no replacement (NVIDIA forum "Anisotropic friction in PhysX 3.x"); PxMaterialFlag in 5.6 lists only eDISABLE_FRICTION, eDISABLE_STRONG_FRICTION, eCOMPLIANT_ACCELERATION_SPRING. Exact removal release [U].
- **Contact modification is not a friction hook [V/I].** PxContactModifyCallback can edit contact position, normal, separation, target velocity, max impulse, mass/inertia scale, not friction. It is CPU-side; no Python/GPU-pipeline route found [U]. Treat as unavailable.
- **Isaac Sim does not offer a directional friction knob** (only static/dynamic friction, combine mode, compliant contact) [V].
- **NVIDIA's own mecanum asset (Fraunhofer O3dyn)** approximates each roller with 6 spheres along its axis [V, forum]. Trimesh rollers are unstable; convex hulls help a little; advice: real inertia, joint damping, timestep 8 ms or smaller [V, forum 351181]. The 2023 ICAR paper replaces roller joints by fixed joints plus anisotropic friction for 3x speed [V, abstract], i.e. the same idea as our MuJoCo model.
- **Ballbots:** OpenBallBot-RL (arXiv 2505.18417) uses capsule wheels with anisotropic friction (our lineage). asRoBallet (arXiv 2604.24916) uses 12 hinged rollers per wheel in MuJoCo with soft wheel-ball contact (solimp 0.85/0.99/0.003, impratio 10, noslip 1) and actuator Coulomb friction 80-120 mNm; it has no ablation against anisotropic friction. No maintained open ballbot in Isaac, Bullet, Unity, Drake or Webots [U, absence of evidence].
- **Our MuJoCo wheel is an idealised infinite-roller wheel** (one capsule, pair friction 0.001/1.0, armature 0.005, damping 0.1). A real wheel has roller handover, bearing friction and roller inertia, so neither engine is obviously closer to hardware [I].

## Possible causes of "PhysX holds at 1.1x, MuJoCo falls" (all [I])
1. Non-zero effective free-direction friction in PhysX (bearing friction, strong-friction anchors).
2. Wheel reflected inertia / armature possibly not applied in PhysX (check first, free).
3. Rigid PhysX contact versus soft MuJoCo contact (more solver iterations did not help earlier).
4. Roller ripple and contact flicker (mean contacts 2.6-2.9 vs 3.0).
5. Ball/roller collider type (analytic sphere + capsules/convex, not meshes).

## Ranked ideas
armature and damping match (tiny) > roller bearing damping sweep 0..1e-3 (small) > barrel rollers 24-32 per wheel with overlap (medium) > compliant contact tuned to ~3 mm penetration at 17 N (small) > timestep 1 ms (small) > combine mode / offsets (small) > CCD (unlikely).

## Experiment plan (when the GPU is free; about 12-16 variants, 1-2 GPU hours)
Baseline reproduction -> inertia match -> roller damping sweep -> barrel rollers (24, 32) -> compliant contact -> timestep -> extra designs (zeta41, s1.5) for the winner.
Protocol: eval_design.py (speed ratio at 0.4/0.8/1.1, 4 headings, push falls 0.5/1/1.5x). "Good enough": speed ratio within 0.03 at 0.4 and 0.8; at least 3 of 4 fall at 1.1x like MuJoCo; push-fall rates within 0.25 at all levels; nominal holds in 3 of 4 perturbed roller configs; fall counts at 0.8 agree on 2 of 3 designs. If nothing passes within about a day, stop and use Newton/MuJoCo-Warp. If it only passes by tuning on the failure boundary, use PhysX as a cross-check only.

## Alternatives
- **Isaac Lab 3.0 with Newton (MJWarp solver)** is described as released, targeting Isaac Sim 6.2, Python 3.12, PyTorch 2.11, Warp 1.17, Newton 1.6.1; a Windows install route is documented but was not run [V docs, U in practice]. Risks: whether `<pair>` friction and our contact-frame fix are honoured through Newton [U]; its default cone is pyramidal with impratio 1 (we need elliptic, high impratio).
- **MuJoCo-Warp directly** already tested: see experiments/mjx_port/REPORT.md and experiments/warp_ppo.

## Open questions
1. Is armature set in our PhysX build? 2. Does Newton honour pair friction / allow our contact-frame kernel? 3. Would the asRoBallet ring model in plain MuJoCo be a better "real wheel" reference than the idealised anisotropic one? Test that in MuJoCo before tuning PhysX to match the idealisation.

## Follow-up checks (CPU, subagent, saved by the lead)
- **Armature/damping in PhysX: applied correctly.** Hanging-wheel test (no contacts): first-step acceleration 19.12 (PhysX) vs 19.14 (MuJoCo) rad/s^2 with armature 0.005 + damping 0.1; terminal speed 0.981 vs 1.000 rad/s (about 2 % extra damping from implicit-drive discretisation, negligible here). `damping*pi/180` conversion confirmed. The older REPORT line about `mjc:`/`newton:` attributes describes the official converter output, not our `build_usd.py`, which writes PhysxJointAPI armature and an angular drive. So cause #2 above is ruled out. Files: `experiments/isaac_port/check_armature_*.py`.
- **MuJoCo roller-ring wheel (12 or 24 hinged rollers, isotropic friction, no pair friction) as an alternative reference: not cleaner.** 48 valid configs (roller count, radius, bearing damping, soft contact, phase). At 0.4 it matches the idealised model (ratio 0.986-0.990). At 0.8 only 2 of 48 configs hold in all 4 headings, and which headings fall changes with every parameter. At 1.1 all 48 fall or stall (mean 2.9 of 4 falls), unlike PhysX, which holds. Pushes: ring is more robust than the idealised model (push falls about 0.003/0.29/0.76 vs 0/0.81/1.0). Bearing damping 1e-3 makes it worse. Stage 3 (larger rims) was a geometry error and is invalid. Files: `experiments/ring_wheel/`, `outputs/study/ring_wheel/`.
- **Conclusion:** neither MuJoCo wheel model matches PhysX at the failure edges; the idealised capsule model is the only stable, repeatable reference. Keep PhysX as a cross-check only. The 1.1x disagreement is more likely contact/friction model structure than wheel inertia; not isolated.
