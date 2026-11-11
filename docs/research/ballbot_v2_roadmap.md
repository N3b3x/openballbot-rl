# Ballbot v2 roadmap: from simulation to a small real robot

Goal: a small, self-controlling ballbot that can drive over uneven ground and climb a low obstacle (a book on the floor), with onboard sensing and onboard compute. This document records what has been measured, what is known from public sources, and the experiment path. Items marked **TODO** are not yet done; nothing below is claimed as a result unless stated.

## 1. Where the simulation stands (measured, see `paper/main.pdf`)

- The simulated robot is already desk-scale: 18 cm ball, 28 cm tower, 4.0 kg total. A 2 to 3 cm book is about one third of the ball radius.
- After the fixes in `scripts/study/README.md`, PPO policies balance and drive at 0.1 to 0.2 m/s on terrain up to about 9 degrees mean slope.
- Frozen depth features from the two ball-facing cameras were ignored by the policy (blank depth changed nothing). The cameras mostly see the ball itself (`paper/figures/fig_episode_strip.png`), so they carry little forward terrain information.
- Speed is the weak point (about 0.15 m/s). The reward is speed-capped at 0.6 m/s; the policies have not yet learned to approach it.

## 2. Sensing plan

| Sensor | Role | Status |
|---|---|---|
| IMU (tilt, rates) | balance | in sim (proprioception) |
| Wheel encoders | wheel speeds | in sim |
| 2 side multizone ToF, aimed forward and down | see steps and slopes just ahead of the ball | **implemented** (`ballbot_gym/sensors/tof.py`), under test |
| Head ToF or small camera, aimed forward | longer-range look-ahead | **TODO** (config supports a third sensor; not yet trained) |
| Privileged height scan (simulation only) | teacher / upper bound | **implemented** (`height_scan`) |

Public specs for the ST VL53L9CX (secondary sources disagree slightly, check the ST datasheet before designing a PCB): up to 54 x 42 zones (about 2.3k), roughly 54 x 42 degree field of view, range about 5 cm to 9 to 10 m, 60 to 100 Hz depending on profile, 12.8 x 6.1 x 4.6 mm module, 940 nm Class 1 laser.
Sources: [CNX Software, VL53L9CX](https://www.cnx-software.com/2026/06/22/st-vl53l9cx-direct-time-of-flight-3d-lidar-supports-5cm-to-9m-range-2-3k-zones-resolution/), [Hackster](https://hackster.io/news/stmicro-unveils-the-high-resolution-all-in-one-vl53l9-direct-time-of-flight-lidar-sensor-2e9616eac192).

The simulated sensor uses an 8 x 8 binned grid per sensor (64 zones), 2 m range, Gaussian noise (5 mm plus 1% of range) and 2% dropout. A real part would be read at lower resolution to keep bus bandwidth and the policy input small. Real ToF has failure modes the sim does not yet model (multipath, dark or glossy surfaces such as a book cover, ambient IR); these are on the **TODO** list for domain randomisation.

## 3. Learning architecture path (what the literature suggests)

1. **Teacher with privileged information.** Train a PPO teacher that sees the height scan. It gives an upper bound on what the task allows.
2. **Student with realistic sensors.** Distil the teacher into a student that sees only IMU, encoders and ToF (plus short history). This teacher-student recipe is how perceptive legged locomotion was made robust to noisy and missing perception ([Lee et al. 2020](https://www.science.org/doi/10.1126/scirobotics.abc5986), [Miki et al. 2022](https://www.science.org/doi/10.1126/scirobotics.abk2822)).
3. **History or recurrence.** A short stack of past observations or a small GRU, so the policy can infer slip and contact. **TODO**.
4. **Domain randomisation.** Mass, friction, motor lag and gain, sensor noise and latency, terrain parameters, pushes. **TODO**; required before any hardware.
5. **Larger parallel simulation.** The current bottleneck is CPU MuJoCo (about 1,500 to 2,000 decisions/s on this machine), not the GPU. The omniwheel friction fix is Python-level and would have to be ported to a GPU engine (MuJoCo-Warp or MJX) to use thousands of environments. **TODO**, expected to be a few days of work and to need Linux or WSL.

## 4. Onboard compute (public numbers)

- NXP i.MX 8M Plus: NPU rated 2.3 TOPS ([NXP](https://nxp.com/products/IMX8MPLUS), [CNX Software](https://www.cnx-software.com/?p=74230)).
- NVIDIA Jetson Orin Nano class: tens of TOPS (40 TOPS for the original module; a newer module is reported at 78 TOPS, a vendor claim) ([CNX Software](https://www.cnx-software.com/2023/03/22/499-nvidia-jetson-orin-nano-developer-kit-80x-jetson-nano-devkit-performance/)).
- The policy itself is tiny: a 2 x 256 MLP with about 0.14M parameters, a few hundred thousand multiply-adds per decision at 100 Hz. It does not need an NPU; a Cortex-A53 CPU or a microcontroller-class core can run it. The balance loop should run on a real-time core (a microcontroller, or the i.MX 8M Plus Cortex-M7) with the compute module handling perception and high-level commands. An NPU only matters for a camera-based encoder. **TODO**: export to ONNX and measure latency on the target.

## 5. Hardware sizing (needs measurement, not guesses)

The simulator gives peak motor torque and speed demands directly. **TODO**: log wheel torque, speed and power during book climbing and on rough terrain, add the margin for friction and motor lag, and pick brushless motors (field-oriented control, encoder feedback, current limiting) and a battery from those numbers. The simulated actuators are idealised torque motors; a real motor adds delay and torque limits that must be randomised in training.

## 6. Experiment path

| Step | Question | Status |
|---|---|---|
| A | Can the policy climb a 0 to 4 cm book? How does success fall with height? | running (`scripts/study/run_queue_book.sh`) |
| B | Do side ToF sensors help over proprioception only? How close to the privileged scan? | running (same queue) |
| C | End-to-end depth CNN on the ball cameras | running (`run_queue_long.sh`) |
| D | Teacher-student distillation, history, randomisation | TODO |
| E | Longer episodes and speed (more than 0.5 m/s) | TODO |
| F | ONNX export and latency on Jetson / i.MX 8M Plus | TODO |
| G | Physical prototype: motors, IMU, ToF, sim-to-real gap | TODO |

## 7. Honest risks

- A ballbot climbing an obstacle is a hard contact problem: the ball must roll up an edge while the body leans. There is no guarantee a small ball with this drive can climb 3 to 4 cm; experiment A measures it.
- Sim-to-real for a ballbot is difficult mainly because of ball-floor and ball-wheel contact, which are the least accurate parts of the simulator.
- Everything above is simulation until a prototype exists.
