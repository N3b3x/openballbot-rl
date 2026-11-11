# MJX / MuJoCo-Warp port feasibility (simulation only)

Source: subagent report, saved by the lead. Numbers come from the scripts and CSVs in this folder and in `outputs/study/mjx_port/`;
the lead spot-checked that the files exist, not every number.

**Verdict: go with caveats for MuJoCo-Warp (GPU). No-go for MJX/JAX on this machine** (no native-Windows CUDA JAX build, WSL not installed; MJX was not benchmarked).

## Feature status
mujoco 3.17.0, mujoco-warp 3.17.0, warp-lang 1.19.0 on the RTX 5080 (sm_120, CUDA 13.4) under Windows: no Blackwell or Windows blocker.
`dz_nom.xml` loads unchanged. Ball/wheel contacts match the CPU (3 contacts, same friction and dim). Elliptic cone, condim 3, Newton solver, implicitfast, capsule-sphere,
sphere-hfield, cylinder all work; one-step qacc agrees with the CPU to about 1e-4 relative. noslip exists in 3.17 (not needed: our xml sets only elliptic cone).
The contact-frame fix has no native support: a 30-line Warp kernel hooked by monkeypatching a private function (`warp_sim.py`); without it one-step qacc reaches about 760. Warp is float32 only.

## Fidelity, nominal design (CPU 3.17 float64 + ContactFrameStepper vs Warp, LQR)
| Metric | Warp vs CPU | CPU 3.13 vs 3.17 | CPU +1e-6 rad vs CPU |
|---|---|---|---|
| Speed ratio 0.4 (CPU / other) | 0.990 / 0.982 | 0.990 / 0.990 | 0.990 / 0.990 |
| Speed ratio 0.8 | 1.029 / 1.036 | 1.029 / 1.028 | 1.029 / 1.029 |
| Falls at 1.1 (of 4) | 4 / 4 | 4 / 4 | 4 / 4 |
| Velocity RMS error 0.4 / 0.8 (m/s) | 0.023 / 0.022 | 0.0003 / 0.0037 | 0.0003 / 0.0013 |
| Tilt RMS error 0.4 / 0.8 (deg) | 0.90 / 0.86 | 0.007 / 0.13 | 0.007 / 0.04 |
| Push falls of 16 at 0.5x / 1x / 1.5x | 0 / 14 / 16 vs 0 / 12 to 13 / 16 | identical | 0 / 14 / 16 vs 0 / 13 / 16 |
| Torque-step forced ball-speed RMS error | 5.5 % of peak | 0.35 % | 0.20 % |
| Open-loop tilt growth rate (1/s) | 6.27 / 5.77 | 6.27 / 6.21 | 6.27 / 6.29 |

Warp reproduces the headline behaviour, but its per-trajectory error is 25 to 100 times the CPU version-to-version spread. Likely a float32 plus solver effect amplified by the unstable plant
(open loop e-folds in about 0.16 s); not isolated. Open-loop trajectories after about 0.6 s are not comparable between any two engines. GPU runs are not bit-reproducible (12 or 13 push falls on repeats).

## Design sweep, 17 designs, both engines
Fall/no-fall parity 193 of 204 (94.6 %) for speed tests, 784 of 816 (96.1 %) for pushes; speed ratios of non-falling runs agree within 0.01 except where noted.
Design ranking is preserved (zhigh and s1.5 robust; zlow, zeta41, bal, kb, J fragile; s0.5 fails on both). Marginal designs (bal4, bal025, kb4, J0005) flip individual runs.
Largest disagreement: nom+mu0.25 (CPU 4/4 falls at 0.4, Warp 1/4), cause not isolated. mu0.5 and mu0.25 are agent-made variants. Full table: `outputs/study/mjx_port/sweep_cpu_vs_warp_wide.csv`.

## Throughput (GPU shared with training: lower bounds)
CPU 1 process (Python ContactFrameStepper) 1364 control steps/s; Warp 1 world 281; 256 worlds 59 k; 4096 worlds 355 k (about 260 times one CPU process);
with a host-side numpy LQR in the loop 47.5 k (256) and 283 k (4096). CUDA graphs are mandatory (about 8 times slower without).

## Blockers and risks
1. Contact-frame fix is a monkeypatch of a private function: pin the version.
2. float32 gap measured with LQR only, not with trained policies; edge cases flip, so GPU-trained policies need CPU validation.
3. Not bit-reproducible. 4. NaN worlds (1 of 4096 over 3 s): needs detection and auto-reset.
5. hfield overflow warning in toppled states; per-world terrain untested. 6. Not tested: ToF/ray sensors, cameras, per-world DR, motor model and latency, PPO at 4096 envs. 7. MJX/JAX needs WSL or Linux.

## Porting effort (estimate, not measured)
Done: model load, contact-frame kernel, batched step, LQR port, parity harness. Remaining for a proprioceptive PPO env about 3 to 4 weeks (batched obs/reward/reset 3 to 4 days, per-world randomisation 2 to 3, latency and motor 1,
GPU PPO loop 3 to 5, CPU validation 2); terrain, height-scan and ToF on GPU 1 to 2 more weeks (highest uncertainty).

Videos: `outputs/study/mjx_port/videos/` (7 clips, CPU left and Warp right, contact points, normal force orange, friction force cyan, HUD); site data: `outputs/study/mjx_port/site_data.json`.
