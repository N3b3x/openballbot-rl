# Ballbot design rules: constraints, guidelines and what is still unverified

Generated with `scripts/study/design_constraints.py` (results in `outputs/study/results/design_constraints.csv`).
Everything marked MODEL comes from the derived planar model; SIM from MuJoCo experiments in this repo; LIT from the
literature reports in this folder (`lit_ballbot_*.md`), which tag their own confidence.

## Hard rules (a design that violates these cannot work, regardless of controller)

1. **Structural (MODEL).** The tau -> ball-position transfer function has one unstable pole p = sqrt(g/l_eff) (4.8 1/s
   here) and one non-minimum-phase zero z (5.8 1/s). Any linear controller has peak sensitivity
   Ms >= (z+p)/|z-p| * exp(p T) (Skogestad bound, T = total loop delay). Nominal floor: 10.7 (about 20 dB). The usual
   crossover window p < wc < z/2 is EMPTY because z/p < 1.3 for every design we tried, so the robot is intrinsically a
   marginal-robustness plant. Moving the COM down helps (floor 9.0 at l = 8.7 cm), up hurts (26 at l = 38 cm).
2. **Traction (MODEL + LIT).** Each wheel can transmit at most mu N r_w of torque, N = m_a g / (3 cos zeta). Nominal:
   N = 20 N, r_w = 15 mm, so 0.31 N m per wheel, about 1.7 N m planar torque on the ball, against m_a g l = 4.9 N m.
   Slip-free tilt recovery therefore tops out near 21 degrees (static). The wheel torque beyond 0.3 N m is useless in
   our sim (motor limit 30 N m has a 133x margin): real motors need to be sized by traction, not by peak torque.
   Rule of thumb: l <= mu r_w fac / (3 cos(zeta) sin(theta_rec)), nominal l_max = 0.18 m for 15 degree recovery.
3. **Torque authority (MODEL).** tau_plane,max / (m_a g l) >= ~7 for LQR to recover a 20 degree tilt within the actuator
   limit. Only binding for weak motors (< 6 N m at s = 1); otherwise traction binds first.
4. **Lean (MODEL).** Steady acceleration a needs lean kappa a with kappa about 7 to 15 deg per m/s^2 over the sweep.
   With a 20 degree lean budget the acceleration ceiling is about 2 m/s^2 for all designs.
5. **Scaling (MODEL, exact).** Geometric scaling s: p ~ s^(-1/2), torques ~ s^4, traction-limited tilt is
   scale-invariant (about 21 degrees), loop-delay budget T_max = ln2/p grows as s^(1/2).
6. **Delay (MODEL).** For a factor-two inflation of the sensitivity floor: T <= ln2/p = 145 ms (nominal). A 100 Hz
   loop (5 ms hold + compute + actuator) uses about 10 percent of the budget. SIM: LQR at 100, 250 and 500 Hz behaves
   identically (same falls at 1 m/s, same failure on the half-scale robot), so loop rate is NOT the limit.

## Findings that contradict the first-order model (SIM)

- **Wheel contact angle matters.** Model: traction-limited planar torque is independent of zeta (the extra normal force
  N ~ 1/cos zeta is cancelled by the shorter lever arm, |n|^2/max|n|). SIM (LQR, 6 trials per cell): zeta = 66 degrees
  survives pushes of 1 x body weight in 6/6 trials, nominal 51 degrees only 1/8, zeta = 41 degrees 0/6. The mechanism
  is something the planar min-norm model misses (3D force distribution, lower reflected rotor inertia
  J_eff = 0.043 vs 0.103 kg m^2, or non-planar traction). Treat steeper contact (60 to 70 degrees, as in LIT) as
  beneficial, upper bound about 70 degrees by geometry.
- **The LQR 1 m/s limit and the half-scale failure are not loop rate, not wheel friction.** Friction x2, x4, roll
  friction x10 and rates up to 500 Hz change nothing; PPO policies do reach 1 m/s (ratio 0.91 to 0.95). So these are
  limits of the linear controller / planar model, which is the argument for the learned gate.
- **Mass ratio.** Our body : ball ratio is 23, the literature (BallIP, Kugle, CMU) uses roughly 2 to 12. The planar
  model is least faithful for light balls (the 0.25x ball fails under LQR). A realistic character-scale build should use
  a ball of at least a few hundred grams, which also eases the contact stiffness in simulation.

## Guidelines that follow

| Choose | Because |
|---|---|
| COM as low as the mechanics allow (l <= 0.18 m at r = 0.09) | lowers the sensitivity floor and keeps the traction constraint satisfied |
| wheel-ball contact zenith 60 to 70 degrees | pushes survived (SIM), lower reflected inertia; upper bound by geometry |
| high-friction ball skin (urethane / rubber), mu >= 1 | traction is the binding limit (also for step climbing: F ~ W tan(alpha)) |
| motors sized by traction: wheel torque ~ mu N r_w with 2x margin, back-drivable, low Coulomb friction | peak torque beyond that is wasted; LIT: low friction beats more torque |
| sensors: IMU >= 1 kHz, encoders on all wheels | observability of tilt rate and ball velocity |
| loop: inner stabiliser 500 Hz to 2 kHz, policy 100 to 200 Hz | model says rate is not binding; faster inner loop is cheap insurance |
| ball mass ratio nearer 5 to 10 than 23 | model fidelity, contact stiffness |

## Compute architecture (Jetson-class hardware) - suggestion, not verified on hardware

Our hybrid controller already has the right structure: a fast model-based inner layer (LQR + governor) and a slower
learned layer (gate + policy). Map them to a real-time MCU (STM32H7 class, 1 to 2 kHz: IMU read, LQR, current loops,
wheel encoders) and a Jetson Orin Nano class module (100 to 200 Hz policy producing gate/residual/velocity setpoints
over CAN/UART/SPI). The delay budget (above) is about 145 ms, so compute latency of a small MLP is not the concern;
timing jitter, bus latency and sensor bandwidth are. Do not run the stabilising loop on a non-real-time Linux
process. We have not verified Jetson numbers here; check TensorRT latency of the exported policy before committing.
