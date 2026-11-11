# Live policy in the browser: feasibility and evidence

Verdict: **feasible, and faithful for the cases tested.** The official MuJoCo WebAssembly build, the same MuJoCo version as the
Python training environment, loads the unmodified `ballbot.xml`, and the Python `ContactFrameStepper` can be reproduced in
JavaScript. Closed-loop rollouts of the trained policy in the browser track the Python reference to within 0.2 mm to 35 mm of base
position over the 8 s training-episode horizon (details below). Simulation only; no hardware.

## 1. Which WebAssembly build

| Candidate | Status |
|---|---|
| `@mujoco/mujoco` (Google DeepMind, official) | versions 3.6.0 ... 3.17.0 on npm; **3.13.0 exists** and is served by jsDelivr (`mujoco.js` 293 KB, `mujoco.wasm` 10,175,239 bytes). Python training used `mujoco` 3.13.0, so the engine version matches exactly. Pinned URL: `https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/mujoco.js`. |
| npm `mujoco` | only 3.1.6 and 3.1.16 (older, different bindings). Not used. |
| `mujoco_wasm` (community port) | unofficial, versions 3.2.x to 3.3.x. Not used. |

Single-threaded build; needs no COOP/COEP headers, so it works on GitHub Pages.

## 2. Does it load the model?

Yes (`site/live/model/` holds a snapshot of `ballbot_gym/models/ballbot.xml` + `meshes/cone.stl` + a down-scaled
`textures/basketball_texture.png`, written into the WASM virtual file system; the texture is visual only). Result: `nq=17, nv=15, nu=3,
ngeom=11`, the 293 x 293 heightfield (flat, all zeros, as in the flat-ground training config), the three anisotropic `<pair friction="0.001 1.0">`
contacts, RK4 in the XML. The environment settings are re-applied in JS exactly as `_load_model` does: `integrator = implicitfast`,
`actuator_ctrlrange = +-30`, wheel joint damping 0.1 (armature 0.005 from the XML). Terrain generators other than flat are not ported
(the hfield data could be written from JS, but this was not done or tested).

## 3. Can the contact-frame fix be reproduced?

`ContactFrameStepper.forward` calls, in order: `mj_kinematics, mj_comPos, mj_camlight, mj_crb, mj_factorM, mj_collision`, **rewrites
`contact.frame` of every ball-wheel contact**, then `mj_makeConstraint, mj_island, mj_transmission, mj_fwdVelocity, mj_fwdActuation,
mj_fwdAcceleration, mj_fwdConstraint`, and then `mj_implicit` (implicitfast).

* All of these stage functions, plus `mj_implicit`, are exported by the 3.13.0 JS bindings. So splitting the step works.
* **`data.contact` is returned by copy** (documented in the package README). Verified: writing `data.contact.get(0).frame[3]` and reading
  `data.contact` again shows the old value. So the frame cannot be overwritten through the contact API. This is the blocker the task asked about.
* Workaround that works: `data.arena` is a live typed-array view of the arena inside `mjData`, and the contact array is its first
  object. The layout in 3.13.0 is: `dist` at double index 0, `pos` 1..3, `frame` 4..12, stride 73 doubles (584 bytes), `geom1/geom2` as
  int32 pair found at an offset that `calibrate()` locates by matching two contacts. `site/live/sim.js` calibrates this at start-up
  and `verify()` compares the heap with `data.contact` exactly (geom pair and all nine frame values for every contact).
  Result: layout verified at **every control step** of all five reference runs in Node (2,726 steps) and in the browser at page load.
* This is a hack on an undocumented struct layout: it is valid only for this pinned build. If `verify()` fails, the page says so
  (badge turns red, banner explains) and does not claim faithful physics.

## 4. Evidence of parity (script: `site/live/parity_node.mjs`, numbers in `site/live/parity.json`)

Reference: `scripts/site/ref_rollout.py` records deterministic rollouts of run `robs_c_s0` in the real Python env (`configs/env/robs_c.yaml`
with `dr: {}` and `init_yaw_random: false`, flat ground, fixed command, MuJoCo 3.13.0). JS: `policy.js` + `sim.js` on `@mujoco/mujoco` 3.13.0 (Node 24).

* **Policy export.** `scripts/site/export_policy.py` writes the pi-network (22 -> 256 -> 256 -> 3, tanh), the VecNormalize mean/var for the five
  normalised keys (clip 10, eps 1e-8), key order and action scaling. JS vs SB3 `predict(deterministic=True)` on 500 random observations:
  max difference 6.0e-7. On all recorded Python observations: 3.3e-6 (limited by the 7-digit rounding of the stored observations).
* **Initial state and first observation:** identical (error 0).
* **Single control step from the same state and action** (300 steps, Python states fed in each step): median base-position difference 4.5e-7 m
  (diag), 1.4e-8 m (yaw), 1.5e-8 m; median velocity difference 1e-4, 2.4e-6, 2.5e-6 m/s. Occasional spikes (max 0.1 m/s) at contact
  transitions, probably from the solver warm-start differing.
* **Closed loop, base-xy difference JS+WASM vs Python** (mm):

  | run (command vx, vy, wz) | 1 s | 4 s | 6 s | 8 s | max |
  |---|---|---|---|---|---|
  | diagonal (0.4, 0.4, 0) | 0.05 | 0.09 | 0.09 | 0.20 | 0.23 |
  | forward + yaw (0, 0.8, 0.5) | 0.04 | 0.40 | 3.1 | 4.8 | 5.6 |
  | reverse + yaw (-0.6, 0.3, -0.8) | 0.63 | 1.8 | 1.8 | **35** | 35 |
  | hold (0, 0, 0), 3 s | 0.03 | 0.14 | 0.14 | - | 0.19 |

  The reverse + yaw run shows the divergence growing late (1.8 mm at 6 s, 35 mm at 8 s): chaotic contact dynamics amplify differences
  of floating-point order. We have **not** measured beyond 8 s or for other commands, so no claim is made there.
* **Contact fix off:** Python (`fix_wheel_contacts: false`) falls at control step 25; the WASM sim with `fixFrames = false` falls at step 25,
  with a 0.45 microns position difference up to then. So the fix is necessary (not cosmetic) and is reproduced.
* **In a browser** (the built-in Chromium-based pane, same code path as the page): max base difference 15 mm over the four runs;
  per run: diagonal 0.40 mm, yaw 6.2 mm, reverse + yaw 15 mm at 8 s. Browser and Node differ at the millimetre level in the late part of
  chaotic runs, which is the expected effect of different `Math.tanh/atan2` implementations; only Chromium and Node were tested (not Firefox/Safari).

## 5. Things that are NOT reproduced (and the page says so)

* Domain randomisation used in training (mass +-15 %, friction +-20 %, torque +-15 %, wheel damping, one-step latency with p = 0.2,
  observation noise, random pushes 5 to 20 N). The page runs the nominal model with no noise; the push buttons add a deliberate 15 N, 50 ms shove.
* Rough terrain, the book-climbing task, ToF/depth/height-scan observations (policy `robs_c_s0` is proprioception only).
* Cameras and any rendering inside MuJoCo (three.js draws the geoms from `geom_xpos/geom_xmat`).
* Speed on slow devices; the 10 MB WASM download; offline use (the page degrades to the static numbers and explains why).

## 6. An environment quirk the browser copy had to reproduce

`BallbotEnv._get_obs` builds `motor_state` from `qvel[model.joint("wheel_joint_k").id]`, i.e. the **joint index** (1, 2, 3), not the DoF address (6, 7, 8).
So the policy's "motor_state" is `[base vy, base vz, base roll rate] / 30`, not the wheel speeds. The policy was trained that way, so `sim.js`
reproduces it (before that fix the closed loop fell within 20 steps). The motor model (`_wheel_dofs`) uses the right addresses; only the observation is affected.
It is worth fixing in a future version of the environment, but changing it would require retraining.
