# Research notes

Notes behind the numbers and design decisions. Everything is simulation; there is no physical robot. Start with the study
log; the other documents refer to its entry numbers (S-numbers).

## Current

| Document | What it is |
|---|---|
| [tracking_study_log.md](tracking_study_log.md) | The study log (S0 to S122): every experiment with its protocol, result and decision. The source of truth for every number in this repository |
| [design_guidelines.md](design_guidelines.md) | Guidelines for building the robot, from the S119 design sweep: ball mass, motor torque, COM height, rotor inertia, contact angle |
| [design_rules.md](design_rules.md) | Analytic design constraints from the planar model, with simulation and literature checks |
| [generalist_program.md](generalist_program.md) | Research program for the generalist controller: workstreams and infrastructure status (as of 2026-11-05) |
| [ballbot_v2_roadmap.md](ballbot_v2_roadmap.md) | Roadmap from simulation to a small real robot. Written earlier and cites the CPU-path paper; read it as a plan, not a result |
| [site_crosscheck.md](site_crosscheck.md) | Generated check of the website and paper numbers against the result files (written by `site/build_data.py`) |
| [analysis/](analysis/) | Small analysis outputs used by the notes |

## Literature notes

| Document | What it is |
|---|---|
| [lit_ballbot_classical.md](lit_ballbot_classical.md) | Classical ballbot modelling and control |
| [lit_ballbot_learning.md](lit_ballbot_learning.md) | Learning-based ballbot control |
| [lit_ballbot_hardware.md](lit_ballbot_hardware.md) | Published ballbot hardware designs |
| [lit_rl_sota.md](lit_rl_sota.md) | Reinforcement learning methods relevant to the controller |
| [papers/](papers/) | Reference papers, including the upstream paper (Salehi 2025) |

## Background (written for the upstream CPU environment)

| Document | What it is |
|---|---|
| [mechanics_to_rl.md](mechanics_to_rl.md) | From Lagrangian mechanics to the RL formulation |
| [timeline.md](timeline.md) | Ballbot research history, 2006 to 2025 |
| [terrain_types_for_ballbot.md](terrain_types_for_ballbot.md) | Terrain types and why they matter |
| [code_mapping.md](code_mapping.md) | Legacy: mapping from the referenced papers to the upstream code |

Back to the [documentation index](../README.md).
