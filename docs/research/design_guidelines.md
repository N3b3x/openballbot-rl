# Ballbot design guidelines (from simulation)

What the simulation study says about building a small ballbot that this controller can balance, drive and hold goals with. There is no physical robot yet: every number here comes from MuJoCo-Warp simulation (float32, GPU) with randomised physics (mass, friction, damping, torque, latency) and sensor noise, using the controller recommended at the time of the sweep (LQR plus a learned residual, checkpoint g34_mref_rsad_db_s0). The recommendation has since moved to g39_goalcentre_s0, the same recipe with goal logic on the true ball centre (S118, S121); its balance and battery robustness are within the g34 range, so these design guidelines are expected to carry over, but the sweep has not been rerun with it (set SPEC_RUN=g39_goalcentre_s0 to do so). Sources are the study log entries in brackets (docs/research/tracking_study_log.md).

Reference robot ("nominal"): 9 cm radius ball of 0.17 kg, a 3.85 kg body whose centre of mass sits 13 cm above the ball centre, 4.0 kg in total, three omni-wheels touching the ball at 51 degrees from its top, direct-drive wheel torque limit 30 N m, rotor inertia 0.005 kg m^2.

## The rules, in order of importance

| # | Property | Guideline | Why (evidence) |
|---|---|---|---|
| 1 | Ball mass | At least 4 percent of the body mass; 8 to 15 percent is better | 1 percent: 41 percent of robots fall while standing still; 2 percent: 5 percent; 4.4 percent (nominal): 0.7 percent; 9 to 17 percent: none (S119). The simple planar physics model does not see this; it shows up only in full contact simulation. |
| 2 | Motor torque at the wheel | Peak at least 15 N m per wheel for a 4 kg robot (half the simulated limit); size the continuous rating for about 3 N m | Peak 15: as good as 30; 10: 6 times more still-falls; 6: 13 percent fall; 3: cannot stand (S119). Torque actually used: median 0.7 N m, 90th percentile 2.6, 99th 8, short peaks up to the limit during push recoveries (S119). |
| 3 | Rotor inertia (motor + wheel, as seen at the wheel) | Do not go below the nominal 0.005 kg m^2; more (up to 0.02) helped | 0.02: the best design of the sweep (terrain falls 0.109 against 0.138); 0.001: 43 percent fall, and the plain LQR falls just as often (49 percent), so this is a physical limit of the build, not a training gap (S119). |
| 4 | Body centre-of-mass height | At least the nominal 1.4 ball radii above the ball centre, and higher is better up to 2.4 (the highest tested); do not build it low | Higher is easier over the whole tested range: still-falls 3.7 percent at 0.9 radii, 0.7 at 1.45, 0.3 at 2.4; terrain goal falls 20 to 12.5 percent (S119). A tall body tips more slowly and needs less lean per acceleration. The analytic grip check warns above about 2 radii for recovering a 15 degree tilt; the controller never lets the robot tilt that far. |
| 5 | Body mass | 3 to 5 kg for these motors; not below 3 kg if the robot must resist steady pushing or pulling | Balance and terrain hardly change with body mass, but steady-pull resistance does: half-strength pulls topple 24 percent at 2 kg, 3 percent at 3 kg, about 2 percent from 3.85 kg up (S119). Above 5 kg still-falls start to rise with the same motors (2.1 percent at 6.5 kg). |
| 6 | Wheel contact angle | About 50 degrees from the top of the ball | 41 and 61 degrees are both somewhat worse than 51 (S119). |
| 7 | Ball size | 9 cm or larger; avoid smaller. A larger ball needs the controller retrained (or at least the learned part) for that size | 7 cm is genuinely harder: the plain LQR falls 28 percent, the full controller 11 percent (grip becomes the limit and the robot tips faster). 12 cm is physically easy, the plain LQR falls only 2 percent, but the current learned controller falls 69 percent because it never trained on another ball size (S119). |

## Limits to design around

- **Steady pushes and pulls.** Short shoves are absorbed completely. A steady sideways force is resisted only up to roughly m g sin(5 degrees) (about 3.3 N for the nominal 3.85 kg body); at 3 to 12 N about a third of robots fall on flat ground, and neither longer training, stronger training pushes nor a longer controller memory changes that (S116, S117). More body mass is the only lever found.
- **Steps.** A 1.5 cm step is crossed most of the time (2 of 3 random trials); about 4 in 10 robots fall on the step-ups of the standard terrain set, and a 3 cm step at 0.5 m/s is beyond the robot (S113 note, S116).
- **Hard designs.** A very light ball, a very light body and weak motors are the designs the controller struggles with most, and for heavy-body and weak-motor builds it is already close to what the plain LQR controller achieves, i.e. near the build's physical limit (S114, S116).

## How to check a candidate design

Write the design as a spec file in physical units and run the spec tool; it builds the simulated robot, reports the physics margins and, with `--eval`, runs the controller on it:

```
python scripts/study/robot_spec.py build specs/nominal.yaml --eval
```

Copy `specs/nominal.yaml`, change the numbers, and compare the five fall rates with the nominal row of S119 (still 0.007, pushes 0.007, half pulls 0.017, drive 0.001, terrain goals 0.138). For a family of variants use the sweep mode with a file like `specs/sweep_one_at_a_time.yaml`.

## Caveats

- Simulation only. Contact, friction, motor and sensor models are idealised; real numbers will differ.
- One property was varied at a time around the nominal robot; combinations were not tested.
- The controller was trained on 11 designs around the nominal robot. Designs it never saw can fail because the controller is unfamiliar with them, not because the build is bad. An LQR-only check separates the two: the 12 cm ball is such a case (plain LQR 2 percent falls, learned controller 69 percent), while the 7 cm ball, very low rotor inertia and very weak motors fail with the plain LQR too, i.e. they are genuine physical limits (S119).
- The motor torque numbers assume direct drive at the wheel. With a gearbox, divide the wheel torque by the gear ratio for the motor torque and check the motor's speed range against the wheel speeds needed (about 1 m/s of ball speed in the study).
- The guidelines are per controller. The same sweep with the later checkpoint g44 (S131, S132) keeps the directions but loses margin for heavy bodies (still-falls 0.21 at 6.5 kg against 0.02), low body COM and wide contact angles, because the recipe did not train on them; it gains on 7 cm balls and 6 N m motors, which it did. Treat a rule as holding only for designs inside the controller's training range.
- Every guideline rests on the S119 sweep: 256 simulated robots per case and one controller checkpoint. Differences of 1 to 2 percentage points between designs are within noise.
