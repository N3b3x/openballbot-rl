"""Assemble experiments/isaac_port/REPORT.md from the CSV/JSON results plus the written analysis (numbers are pulled from the files)."""
import json

import numpy as np
import pandas as pd

from common import ROOT

OUT = ROOT / "outputs/study/isaac_port"
cmp_ = pd.read_csv(OUT / "comparison_px_r.csv")
nd = pd.read_csv(OUT / "nominal_detail_recordings.csv")
bm = json.load(open(OUT / "bench_mj.json"))
g1 = json.load(open(OUT / "bench_px_gpu_small.json"))
g2 = json.load(open(OUT / "bench_px_gpu_mid.json"))
g3 = json.load(open(OUT / "bench_px_gpu_4096.json"))
clips = json.load(open(OUT / "videos" / "clips.json"))
cpu_step = bm["contact_frame_stepper(env default)"]


def f(x, nd_=2):
    return "fell" if (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd_}f}" if isinstance(x, (float, np.floating)) else str(x))


def table():
    L = ["| design | ratio 0.4 MJ / PX | ratio 0.8 MJ / PX | ratio 1.1 MJ / PX | falls /2 at 0.4, 0.8, 1.1 (MJ ; PX) | push-fall 0.5x, 1x, 1.5x MJ | push-fall 0.5x, 1x, 1.5x PX | step final ratio MJ / PX | step RMS diff m/s | pulse tilt RMS diff rad |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cmp_.iterrows():
        L.append(f"| {r.design} | {f(r['ratio_0.4_mj'])} / {f(r['ratio_0.4_px'])} | {f(r['ratio_0.8_mj'])} / {f(r['ratio_0.8_px'])} | {f(r['ratio_1.1_mj'])} / {f(r['ratio_1.1_px'])} | "
                 f"{int(r['falls_0.4_mj'])},{int(r['falls_0.8_mj'])},{int(r['falls_1.1_mj'])} ; {int(r['falls_0.4_px'])},{int(r['falls_0.8_px'])},{int(r['falls_1.1_px'])} | "
                 f"{r['pushfall_0.5_mj']:.2f}, {r['pushfall_1.0_mj']:.2f}, {r['pushfall_1.5_mj']:.2f} | {r['pushfall_0.5_px']:.2f}, {r['pushfall_1.0_px']:.2f}, {r['pushfall_1.5_px']:.2f} | "
                 f"{f(r['step_final_ratio_mj'])} / {f(r['step_final_ratio_px'])} | {f(r.get('step_rms_diff_m_s', np.nan), 3)} | {f(r.get('pulse_tilt_rms_rad', np.nan), 3)} |")
    return "\n".join(L)


def ndtable():
    L = ["| scenario | tilt RMS diff (deg) | speed RMS diff (m/s) | final speed MJ / PX | mean wheel-ball normal force MJ / PX (N) | mean friction MJ / PX (N) | fell at step MJ / PX |", "|---|---|---|---|---|---|---|"]
    for _, r in nd.iterrows():
        L.append(f"| {r.design} {r.scenario} | {r.tilt_rms_diff_deg:.2f} | {r.speed_rms_diff_m_s:.3f} | {r.final_speed_mj:.3f} / {r.final_speed_px:.3f} | "
                 f"{r.mean_normal_N_mj:.1f} / {r.mean_normal_N_px:.1f} | {r.mean_friction_N_mj:.2f} / {r.mean_friction_N_px:.2f} | {int(r.fell_at_mj)} / {int(r.fell_at_px)} |")
    return "\n".join(L)


gp = [g1[0], g1[1], g2[0], g2[1], g3[0]]
tp = ["| engine | envs | physics steps/s (sum over envs) | control steps/s (100 Hz) | ms per 2 ms physics step | note |", "|---|---|---|---|---|---|",
      f"| MuJoCo CPU, 1 process, LQR loop, contact-frame stepper | 1 | {cpu_step['physics_steps_per_s']:.0f} | {cpu_step['control_steps_per_s']:.0f} | {1e3 / cpu_step['physics_steps_per_s']:.2f} | busy machine, BelowNormal priority |"]
for g in gp:
    tp.append(f"| PhysX GPU (warp frontend), closed loop | {g['N']} | {g['closed_loop_envsteps_per_s']:.0f} | {g['closed_loop_envsteps_per_s'] / 5:.0f} | {g['sim_only_phys_step_ms']:.1f} | sim-only {g['sim_only_envsteps_per_s']:.0f} steps/s |")
tp = "\n".join(tp)
cl = "\n".join(f"- `{c['file'].split('/')[-1]}` ({c['size_mb']} MB): {c['title']}. {c['outcome']}" for c in clips)

d = cmp_
dr4 = (d["ratio_0.4_px"] - d["ratio_0.4_mj"]).abs()
dr8 = (d["ratio_0.8_px"] - d["ratio_0.8_mj"]).abs().dropna()
agree08 = int((d["falls_0.8_mj"] == d["falls_0.8_px"]).sum())
agree11 = int((d["falls_1.1_mj"] == d["falls_1.1_px"]).sum())
dpush = (d["pushfall_1.0_px"] - d["pushfall_1.0_mj"])
summary = (f"Across the {len(d)} designs that finished: mean |speed ratio difference| at 0.4 is {dr4.mean():.3f} (max {dr4.max():.3f}, design {d.loc[dr4.idxmax(), 'design']}); at 0.8 {dr8.mean():.3f} where both engines survive; "
           f"fall counts at 0.8 agree in {agree08}/{len(d)} designs, at 1.1 in {agree11}/{len(d)}; PhysX falls more than MuJoCo in the 1x push in {int((dpush > 0).sum())} designs and less in {int((dpush < 0).sum())}.")
txt = f"""# Isaac Sim / PhysX port of the ballbot: feasibility and fidelity (simulation only)

Everything below is simulation. No hardware claims. All numbers come from files in `outputs/study/isaac_port/` produced by the scripts in this folder.

## Verdict (short)

**No-go as a replacement for the MuJoCo study engine; go-with-caveats only as a second, independent engine for cross-checks.**

Evidence for fidelity:

- PhysX has no anisotropic friction. `<pair friction="0.001 1.0">` (free sliding along the wheel axle, grip around the wheel) cannot be expressed; the official MJCF to USD converter prints `pairs are not supported` and keeps one isotropic material. A naive import (capsule wheels, mu 1 everywhere) loses balance after 0.39 s in closed loop (clip `naive_track04`).
- With a custom emulation (ring of 16 passive rollers per wheel, 52 links per robot) the nominal design reproduces MuJoCo's well-behaved regimes closely: speed ratio 0.975 vs 0.987 at 0.4 and 1.013 vs 1.030 at 0.8, wheel-ball normal force within about 5 %, open-loop torque pulse tilt RMS difference {nd[nd.scenario == 'pulse'].tilt_rms_diff_deg.iloc[0]:.2f} deg.
- It does **not** reproduce MuJoCo at the edges that matter for the design studies: at 1.1 scaled speed PhysX holds balance (ratio about 1.1) where MuJoCo falls (in 8 of 9 designs every run, in kb4 one of two), and 1x-weight pushes on the nominal design fall in 4 of 4 PhysX trials vs 3 of 4 in MuJoCo (borderline). The zenith ranking (66 deg survives pushes, 41 deg does not) is reproduced in the push test.
- The emulation is fragile: of four roller-ring configurations compared on the same nominal 0.4 run, two held balance and two fell; a single roller per wheel and a ring of 12 thin rollers failed earlier (see section 4).

Evidence for speed: not a large speedup. 4096 cloned robots on the shared RTX 5080 give {g3[0]['closed_loop_envsteps_per_s']:.0f} physics steps/s closed loop (about {g3[0]['closed_loop_envsteps_per_s'] / 5:.0f} control steps/s), about {g3[0]['closed_loop_envsteps_per_s'] / cpu_step['physics_steps_per_s']:.0f}x one MuJoCo CPU process ({cpu_step['physics_steps_per_s']:.0f} physics steps/s on this busy machine) and, extrapolating linearly (not measured), less than the 24 CPU threads would give if free (about 24 x 7.7k). The roller ring multiplies bodies by 17 and the step latency is 35 to 59 ms per physics step whatever the number of envs. (Measured while training and other jobs were running, so absolute numbers are pessimistic; the rented-box test below would settle that.)

Better route if the aim is GPU speed with MuJoCo physics: Isaac Sim 6.2 itself ships `mujoco-warp 3.12` and Newton, and `experiments/mjx_port` already compares MuJoCo-Warp on this model. That keeps the contact model; PhysX does not.

## 1. Feasibility on this machine (verified)

| item | result |
|---|---|
| OS, GPU | Windows 11 Pro, RTX 5080 16 GB (sm_120), driver 610.88, CUDA 13.3. WSL not installed (enabling it needs admin, not attempted) |
| Isaac Sim | `isaacsim[all,extscache]==6.2.0.0` installs with plain pip into a venv (`C:/Users/Owner/isaac_venv`), CPython 3.12 only (5.x needs 3.11, 4.x needs 3.10), Windows wheels exist on PyPI + pypi.nvidia.com. No admin, no installer |
| Size | download 7.5 GB (extscache-kit 5.5 GB, kit-sdk 0.72 GB, physics 0.25 GB, rest small); on disk 16.3 GB in the venv plus 5.4 GB pip cache. **The 15 GB guideline was judged on download size; the installed size is above it.** Delete with `rmdir /s C:\\Users\\Owner\\isaac_venv` and `pip cache purge` |
| Blackwell | Works: Warp 1.17 sees the 5080 (CUDA 12.9 toolkit, driver 13.3), PhysX GPU dynamics ran with the tensor API (warp frontend), headless start 18 to 35 s. The bundled torch is 2.11 CPU only; CUDA torch (about 2.5 GB more) and Isaac Lab were not installed because raw Isaac Sim + `omni.physics.tensors` was enough |
| Linux | Same wheels exist for manylinux_2_35 x86_64/aarch64 |

Requirements for a rented Linux box: Ubuntu 22.04 or 24.04 (glibc 2.35+), a current NVIDIA driver (here 610.88 worked on the RTX 5080; the minimum was not tested), Python 3.12, 25 GB free disk, 16 GB+ GPU, `pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com`, `OMNI_KIT_ACCEPT_EULA=YES`. `bench_px.py` and `sweep.py --engine px` run unchanged.

## 2. What survives the MJCF import (official route)

`mujoco-usd-converter 0.6.1` (shipped with Isaac Sim 6.2) on `dz_nom.xml` (hfield replaced by a plane, `import_check.py`, details in `experiments/isaac_port/import_test/inspect.json`):

- Survives exactly: base mass 3.4512 kg, centre of mass, inertia diagonal and axes; three revolute wheel joints with axes; fixed camera bodies; capsule/sphere/cylinder geometry; one physics material with mu 1.0.
- Lost or not used by PhysX: `<pair>` anisotropic friction ("pairs are not supported"), cameras, lights, sensors. Wheel damping and armature are written as `mjc:damping` / `newton:armature` attributes that only the Newton/MuJoCo-Warp backend reads, not as PhysX joint attributes. Rolling and torsional friction (1e-4, 5e-3) are also Newton attributes.
- The hfield (floor) is not supported; the contact-frame fix (`contact_frame.py`) has no PhysX counterpart.

## 3. How PhysX treats sphere-on-wheel rolling contact (schema facts verified in the installed PhysX schema; behaviour statements marked)

- Friction model: one Coulomb friction type, "patch" (the only allowed value of `physxScene:frictionType`), static and dynamic coefficient per material, combine mode average/min/multiply/max (`physxMaterial:frictionCombineMode`). Contacts are merged into friction anchors within `frictionCorrelationDistance` (0.025). There is no per-direction coefficient, so no anisotropic friction. Torsional friction only through `physxCollision:torsionalPatchRadius`; no rolling friction. Optional compliant contact via `physxMaterial:compliantContactStiffness` (spring-damper instead of rigid).
- Solver: TGS (default) or PGS; per-body and per-articulation `solverPositionIterationCount` / `solverVelocityIterationCount` (we used 32/4); `timeStepsPerSecond` set to 500 gives one scene step per 2 ms and TGS does its own internal substeps (position iterations). There is no extra substepping knob beyond that. Contact `contactOffset` 2 mm, `restOffset` 0.
- Consequence (inference, tested only through the emulation below): the ball-wheel rolling contact is a rigid unilateral contact with Coulomb friction, whereas MuJoCo uses soft contacts (solref default, 2 to 5 mm penetration under the 17 N normal load seen in these runs) with an elliptic cone. Rigid contacts plus a discretised roller ring produce contact flicker (the number of wheels in contact fluctuates, `mean_contacts` 2.6 to 2.9 vs 3.0 in MuJoCo).
- Pitfall found: USD angular drive damping is per degree per second; passing the MuJoCo value 0.1 N m s/rad unscaled made the wheels 57x too damped (first runs showed wheels frozen). Fixed in `build_usd.py`.

## 4. What was built and tried

`export_model.py` dumps masses, inertias, joint anchors/axes and the MuJoCo contact frames; `build_usd.py` writes the USD/PhysX articulation; `backend_px.py` and `backend_mj.py` expose the same interface (`state`, `step_ctrl`, `pose`, `wheel_ball_forces`) so `scenarios.py` and `LQRCore` (a copy of `ballbot_gym/controllers/lqr.py` on plain arrays) run identically on both. The MuJoCo backend reproduces the study's own reference (`dz_lqr.csv`: 0.988/1.029, push 0.875 at 1x; here 0.987/1.030, 0.75 with 4 trials, 0.875 with 8).

Roller emulation: each wheel is a ring of N passive capsule rollers whose long axis is the circumferential tangent (free sliding along the axle comes from rolling, grip around the wheel from friction). Outer ring radius = capsule radius minus the MuJoCo contact penetration, so the settled geometry matches.

Tried (nominal, 0.4 scaled speed, 400 control steps; the first three were judged from short debug traces, the next four are one comparison run `t7.py`): 1 roller sphere per wheel: contact leaves as the wheel turns, fell; ring of 12 capsules r 6 mm: wheel-ball contact lost at about 8 deg lean, fell; sliding-element variant (prismatic along the axle, 12 elements): normal forces doubled, unstable; 10 rollers r 12 mm: fell after about 1.4 s; 20 rollers r 12 mm: held (ratio 0.981); 16 rollers r 20 mm: held (0.979, used for everything below); 12 rollers r 12 mm: fell; 20 rollers r 12 mm with 64/8 solver iterations: fell (more iterations did not help). So the result depends on discretisation, not on a physical parameter that can be derived.

## 5. Per-design comparison

{summary}
 (same LQR gains, same protocol, reduced to 2 headings x 3 speeds and 4 pushes per alpha for PhysX run time)

Protocol as `scripts/study/eval_design.py` (speed grid 0.4/0.8/1.1 times sqrt(scale), 4 s; 50 ms push of 0.5/1/1.5 x total weight at 0.4 speed); full-protocol MuJoCo is in `sweep_mj.csv`. "fell" in a ratio column means every run fell. Step response: velocity command step 0 to 0.4 scaled speed; "step final ratio" is steady speed over command, "step RMS diff" the RMS speed difference to MuJoCo over the 3 s. "pulse tilt RMS" compares the open-loop 0.5 N m pulse (first 0.4 s). CSV: `comparison_px_r.csv`. Only the designs that finished within the time budget are listed.

{table()}

Nominal-design recordings (identical scenarios, from `nominal_detail_recordings.csv`; "friction" for PhysX is the reported contact-report tangential force and is about 15x smaller than MuJoCo's, which I do not trust as a like-for-like quantity: the contact-report friction of the roller joints may not include the part carried through the roller constraint, so only the normal forces are compared):

{ndtable()}

## 6. Throughput (measured, short GPU runs, machine busy)

{tp}

Build/start cost for 4096 envs: {g3[0]['build_s']:.0f} s cloning + {g3[0]['start_s']:.0f} s physics start. Each env is 52 links (base, 3 wheels, 3x16 rollers) plus the ball. CPU PhysX single env: about 20 ms per control step (3 to 6 min per design in the sweep). The CPU MuJoCo number is the raw harness (no gym wrapper), so env-level numbers of 500 to 800 steps/s per process include Python overhead on top.

## 7. Videos and data for the site

Side by side, MuJoCo (left) vs PhysX (right), same LQR gains, contact points (red) and wheel-ball normal (blue) / friction (orange, 4x) force vectors from each engine's own contact data, drawn in one renderer so both panels look alike (poses and forces come from each engine's log; the PhysX run is not re-simulated in MuJoCo). The task suggested mjv scene flags for MuJoCo; custom arrows were used instead so both panels use identical drawing code.

{cl}

`site_data.json` lists them with parameters and outcomes; tables are in the CSVs. Failures are shown on purpose: `naive_track04` (plain import cannot balance) and `zeta41_track08` (MuJoCo falls, PhysX does not in this heading).

## 8. Blockers and effort

Blockers: (1) no anisotropic friction in PhysX (emulation required, fragile, 17x body count); (2) the study's failure boundaries (high speed, push recovery) differ from MuJoCo's, so conclusions about design margins would shift; (3) speed advantage not demonstrated; (4) the RL side (Isaac Lab, CUDA torch, observations/rewards) was not installed or tested; only physics and control were.

Effort: a nominal PhysX environment with the roller emulation took about one working day including install (done here). An Isaac Lab environment with observations, rewards, command sampling, domain randomisation, terrain and perception matching the current env: 2 to 4 weeks, plus an open-ended tuning risk on the roller ring. Not recommended unless a second engine is wanted for robustness checks.

## 9. Cheapest next test

Rent a Linux box with one 24 GB GPU (L4/A10/4090) for about 2 hours: run `python bench_px.py --n 1 1024 4096 8192 --device gpu` and `python sweep.py --engine px --tag _r --dirs 2 --trials 4 --designs nom zeta66 zeta41 s1.5 kb4 ...` with the machine otherwise idle. That decides whether the 35 ms step latency is contention or intrinsic, and gives the missing designs. Then compare with `experiments/mjx_port` (MuJoCo-Warp), which is the like-for-like GPU option.

## Files

- `common.py`, `backend_mj.py`, `backend_px.py`, `build_usd.py`, `export_model.py`, `scenarios.py`: model, backends, scenarios
- `sweep.py`, `compare.py`, `analyze_nom.py`, `bench_mj.py`, `bench_px.py`: experiments
- `record.py`, `render.py`, `render_all.py`, `make_site_data.py`, `make_report.py`: videos, site data, this report
- `probe.py`, `import_check.py`, `t*.py`: probes and debugging runs
- Outputs: `outputs/study/isaac_port/` (`sweep_*.csv`, `comparison_px_r.csv`, `nominal_detail_recordings.csv`, `bench_*.json`, `models/*.json`, `rec/*.npz`, `traces/*.npz`, `videos/`, `site_data.json`)
"""
open(ROOT / "experiments/isaac_port/REPORT.md", "w", encoding="utf-8").write(txt)
print("report written", len(txt))
