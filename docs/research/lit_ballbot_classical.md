# Classical / model-based ballbot literature: annotated bibliography

Method note. Compiled from web search results and a handful of page fetches. Several PDFs (Fankhauser thesis, Pham 2024, Watson 2018, CMU ICRA 2006) could not be parsed, so most numbers below come from abstracts, publisher pages, press or secondary sources. Every number carries its source; "not verified" means I could not read the primary text. Nothing here was downloaded or run. Coverage is NOT exhaustive: several requested topics returned no ballbot-specific hits (see "Gaps").

## 1. Carnegie Mellon (Hollis group)

**1.1 Lauwers, Kantor, Hollis, "One is Enough!", ISRR 2005.**
URL: https://www.ri.cmu.edu/publications/one-is-enough
First ballbot paper: overall design, inverse mouse-ball actuator (rollers drive the ball), first balancing and point-to-point results. Only the abstract-level description was verified. Code/CAD: none public found (a CMU patent exists: US 7,847,504 "Dynamic balancing mobile robot", per Wikipedia reference list; grant for a later "Ball-balancing robot and drive assembly" also exists).

**1.2 Lauwers, Kantor, Hollis, "A Dynamically Stable Single-Wheeled Mobile Robot with Inverse Mouse-Ball Drive", ICRA 2006, pp. 2884-2889.**
URL: https://ri.cmu.edu/?p=9475 (text copy: scispace.com/pdf/a-dynamically-stable-single-wheeled-mobile-robot-with-2smbf1eczo.pdf, not readable via my fetcher).
Person-sized robot balancing on one spherical wheel, moving in any direction without turning first (unlike Segway-type). Reported (secondary sources quoting the paper): body a cylinder 1.5 m tall, 400 mm diameter, 45 kg; ball 200 mm hydroformed steel shell with 3.2 mm urethane layer; four rollers each with its own motor; yaw via bearing + slip ring + separate motor. Controller type, control rate, top speed: NOT verified. Code/CAD: no.

**1.3 Nagarajan, Kantor, Hollis, "The ballbot: An omnidirectional balancing mobile robot", IJRR 33(6), 917-930, 2014. DOI 10.1177/0278364913509126.**
URL: https://ri.cmu.edu/publications/the-ballbot-an-omnidirectional-balancing-mobile-robot
Evolved hardware: four-wheel inverse mouse-ball drive, yaw mechanism with unlimited rotation, triad of retractable legs for static stability when powered down; experiments on balancing and locomotion. Mass, ball size, speeds and tilt limits for this version: NOT verified (abstract only). Code/CAD: no.

**1.4 Nagarajan, Mampetta, Kantor, Hollis, "State transition, balancing, station keeping, and yaw control for a dynamically stable single spherical wheel mobile robot", ICRA 2009** (title from Wikipedia reference list; I did not open it). Covers getting up from the legs to balance, station keeping, yaw. Numbers not verified.

**1.5 Nagarajan, Kantor, Hollis, "Trajectory planning and control of a dynamically stable single spherical wheeled mobile robot", ICRA 2009.**
URL: https://www.ri.cmu.edu/?p=10213 (listing only). Trajectory planning with planar (decoupled) models; numbers not verified.

**1.6 Nagarajan, Kantor, Hollis, "Human-robot physical interaction with dynamically stable mobile robots", HRI/ICRA 2009.**
Relevant to pushable ballbots: the ballbot is compliant to pushes (and the CMU group describes tolerance to kicks and shoves and collisions with furniture and walls, secondary source: https://thefutureofthings.com/4194-the-ballbot-robot and similar). Quantitative push limits: not verified.

**1.7 Nagarajan, "Dynamic constraint-based optimal shape trajectory planner for shape-accelerated underactuated balancing systems", RSS 2010.**
URL: https://m.roboticsproceedings.org/rss06/p31.html
Key concept: "shape-accelerated" systems, where internal shape (body lean, arm angles) maps to accelerations in position space. Planner uses the dynamic constraint to compute shape trajectories that optimally track desired position trajectories; examples on 3D ballbot. Numbers not verified.

**1.8 Nagarajan, Kantor, Hollis, "Hybrid control for navigation of shape-accelerated underactuated balancing systems", CDC 2010.**
URL: https://ri.cmu.edu/publications/hybrid-control-for-navigation-of-shape-accelerated-underactuated-balancing-systems-2
Sequential composition of controllers for obstacle navigation; simulation of 3D ballbot.

**1.9 Nagarajan, Kantor, Hollis, "Integrated motion planning and control for graceful balancing mobile robots", IJRR 32(9-10), 1005-1029, 2013. DOI 10.1177/0278364913488011.**
URL: https://ri.cmu.edu/?p=122036
Palette of "motion policies" (fast, graceful motions inside collision-free domains), a directed "gracefully prepares" graph, Dijkstra for a goal-rooted policy tree with fast replanning for dynamic obstacles, hybrid supervisory controller. Validated on the ballbot. Numbers not verified. Earlier version: ICRA/CDC 2012 (https://www.ri.cmu.edu/?p=7475).

**1.10 Nagarajan, Kim, Hollis, "Planning in high-dimensional shape space for a single-wheeled balancing mobile robot with arms", ICRA 2012.**
URL: https://ri.cmu.edu/?p=7474. Ballbot with 2-DOF series-elastic arms (arm detail from Wikipedia); weight-matrix chooses among redundant shape trajectories.

**1.11 Nagarajan, "Shape space planner for shape-accelerated balancing mobile robots", IJRR 32(11), 1323-1341, 2013.**
URL: https://www.ri.cmu.edu/publications/shape-space-planner-for-shape-accelerated-balancing-mobile-robots
Journal version of 1.7 and 1.10; experiments move the ballbot using body lean, arm angles or both, including arm-constrained cases.

**1.12 Nagarajan PhD thesis, "Fast and Graceful Balancing Mobile Robots", CMU.** https://www.ri.cmu.edu/event/fast-and-graceful-balancing-mobile-robots (event page). Combines shape planning with sequential composition; reports navigation with disturbances and dynamic obstacles.

Note: the requested "Inkwell/Hollis 'Insect'" item could not be matched to any real publication; I did not find it and do not cite it. Also the claimed CMU "0.6 m/s" figure I had in the query is unsupported by sources I found.

## 2. Japan: BallIP (Kumagai, Tohoku Gakuin University)

**2.1 Kumagai, Ochiai, "Development of a robot balancing on a ball", ICCAS 2008, pp. 433-438** (it is ICCAS, not ICRA). 
**2.2 Kumagai, Ochiai, "Development of a robot balanced on a ball: Application of passive motion to transport", ICRA 2009.** DOI 10.1109/ROBOT.2009.5152324.
**2.3 Kumagai, Ochiai, "Development of a robot balanced on a ball: First report, implementation of the robot and basic control", J. Robotics and Mechatronics 22(3), 348-355, 2010.**
URL: https://www.fujipress.jp/jrm/rb/robot002200030348
Three omniwheels driven by stepping motors; two sets of rate gyros + accelerometers; can station-keep, translate in any direction, yaw. Key idea: control on two decoupled axes using BALL ACCELERATION as the input (not torque), which is claimed (Nyquist plots) to make the loop robust to inertia-parameter changes. Passive mode: stable but easily pushed (a compliance mode; ICRA 2009 uses it to transport). Hardware (IEEE Spectrum, https://spectrum.ieee.org/042910-a-robot-that-balances-on-a-ball): 7.5 kg robot, about 0.5 m tall, rubber-coated 3.6 kg bowling ball 20 cm diameter, RIKEN-developed omniwheels, NIDEC stepper motors with micro-step drivers (0.225 deg step), 16-bit MCU. Speed, control rate: not verified.
**2.4 Kumagai, "Development of a ball drive unit using partially sliding rollers", IROS 2010** (title from Wikipedia list, not opened): addresses omniwheel/roller mechanism.
Other Japan: Endo & Nakamura, "An omnidirectional vehicle on a basketball", ICAR 2005 (early ball-riding vehicle, per Wikipedia list; not opened). The query item "Sugiyama/Tohoku" is probably Kumagai's Tohoku Gakuin group; I found no separate Sugiyama paper.
Code/CAD: none found.

## 3. ETH Zurich Rezero

**3.1 Fankhauser, Gwerder, "Modeling and Control of a Ballbot", Bachelor thesis, ETH ASL, spring 2010** (supervisors Leutenegger, Colas; Siegwart lab). Found via a Scribd copy / ETH Research Collection; I could not read the text.
Claims (abstract): planar decomposition is the state of the art; their new 3D modelling and control gives "considerably higher performance" (acceleration, speed, precision, robustness). Three driven omniwheels. Reported secondary numbers: tilt resistance about 17 deg (IEEE Spectrum, https://spectrum.ieee.org/automaton/robotics/diy/project-rezero-ball-balancing-elegance), speed 3.5 m/s per IEEE Spectrum but "about 2 m/s, 20 deg" per Pham 2024 (CONFLICTING, treat as unverified), balance loop 160 Hz (TED demo script). Mass, ball size, motors: not verified.
**3.2 Doessegger et al., "Rezero, Focus Project Report", ETH 2010** (from Wikipedia list, not opened). Design project, 13 undergraduates (team size varies between sources).
Code/CAD: none found. This 3D (non-planar) model is the most relevant comparison point to our planar-Lagrangian model.

## 4. Aalborg Kugle

**4.1 Jespersen, "Kugle: Modelling and Control of a Ball-balancing Robot", MSc thesis, Aalborg Univ., 2019, 281 pp.** URL https://projekter.aau.dk/kugle-modelling-and-control-of-a-ball-balancing-robot-4450f15c.html ; code https://github.com/mindThomas/Kugle-MATLAB (MIT license; also Kugle-Embedded, -ROS, -Gazebo).
Quaternion-based nonlinear model (thesis concludes it is workable but unnecessary in the operating range). Cascade: sliding-mode balance controller (two sliding surfaces compared) + velocity LQR + ACADO path-following MPC giving attitude references; quaternion EKF + velocity EKF. Hardware: 16 kg Kugle V1, 3 omniwheels, STM32H7 + FreeRTOS, ROS on an Intel NUC, two SICK TiM571 LiDARs. Rates and tests (GitHub README): balance/estimators/velocity controller at 200 Hz; tilt tracking error under 1 deg for references up to 1 Hz; velocity references up to 1 m/s and 1 rad/s tested. Public: MATLAB/Simulink, firmware, ROS, Gazebo; mechanical drawings but native CAD unclear.

## 5. LQR / MPC / sliding mode / H-infinity papers

**5.1 Pham, Kim, Kim, Lee, "Balancing and transferring control of a ball segway using a double-loop approach", IEEE Control Systems Magazine 38(2), 15-37, 2018.** Outer-loop LQR + inner-loop PI; hardware experiments of balancing, trajectory tracking and simultaneous yaw rotation. Copy at https://njcpe.github.io/pdfs/ele503paper.pdf (not parsed). Numbers not verified.
**5.2 Do, Lee, "Adaptive backstepping + hierarchical sliding mode for a ridable ballbot", IEEE Trans. SMC, 2021** (as reported by a search snippet; also "Aggregated hierarchical sliding mode control for a spatial ridable ballbot" on bohrium.com). Handles input constraints and mismatched uncertainty; reported numerical studies (aggregator source, not verified).
**5.3 Lal, Codrean, Busoniu, IFAC 2020, https://ifatwww.et.uni-magdeburg.de/ifac2020/media/pdfs/1224.pdf.** Sliding mode on linearized plant for stabilization and tracking, compared with LQR; abstract-level only.
**5.4 Studt, Zhavzharov, Abbas, "Parameter identification and LQR/MPC balancing control of a ballbot", ECC 2022, pp. 1315-1321, DOI 10.23919/ECC55457.2022.9837996.** No abstract retrieved.
**5.5 Fischer, Karachalios, Zhavzharov, Abbas, "Closed-loop identification and tracking control of a ballbot", arXiv:2404.14845.** Double loop; LQR inner, MPC outer. Hardware (cheap): bowling ball, 3 NEMA 17 steppers, 58 mm omniwheels, ICM IMU, mouse-trackball sensor, XMC4700 + Raspberry Pi 4. LQR Q=diag(20,100,10,50), R=200 at 5 ms; MPC 0.1 s, horizon 4 s, constraints |theta| <= 3 deg, |ydot| <= 15 cm/s. Model fit only 68% (theta), 46% (ydot). Settling about 10 s on a 20 cm step. Code not stated as released.
**5.6 Karachalios, Abbas, "Parameter refinement of a ballbot and predictive control for reference tracking with LPV embedding", arXiv:2402.12092.** LPV-MPC (tilt and rate as scheduling), RK4 ZOH, N=20, ts=0.05 s, input bound 1.5 Nm, tilt bound pi/3, QP solve about 6-9 ms. Simulation (plus parameters from 5.4).
**5.7 Pham, Thai, Dang, Vo, "Analysis of the parametric configuration impact on BallBot control performance", Int. J. Mech. Syst. Dyn. 4(4), 446-460, 2024. DOI 10.1002/msd2.12133.** https://journal.hep.com.cn/ijmsd/EN/10.1002/msd2.12133
LQR-based simulation sweep of body mass, chassis size, ball diameter. Headline: robot mass noticeably affects response, ball diameter has relatively small effect. Also four-wheel inverse mouse-ball drive, yaw drive, tripod, trajectory planning from rest. (Author is Pham, not "Pratt"; I found no Pratt ballbot paper.) Numbers not retrieved.
**5.8 Pham et al., IEEE Access 2025, control-barrier-function MPC for ballbot obstacle avoidance with tilt limits** (search snippet only; not verified).
**5.9 H-infinity full-state feedback for a ball-balancing robot, LMI-based, one plane, Vietnamese journal** https://www.vjol.info.vn/index.php/dhcnhn/article/view/78699. Simulation only.
**5.10 Puychaison et al., "Mouse type ballbot identification and control using convex-concave optimization", CACS 2019** (cited via 5.6, not opened).
**5.11 Sanchez, Arribas, Gomez, Polo, "A monoball robot based on LEGO Mindstorms", IEEE Control Systems Magazine, 2012** (via Wikipedia list). Low-cost educational.

## 6. Adjacent: mechanisms, yaw, slip, alternative drives

- Watson, Gladwin, Prescott, IECON 2018 (https://eprints.whiterose.ac.uk/132164/1/Watson%202018%20Design%20OmniDirectional.pdf): NOT a ballbot; Collinear Mecanum Drive, balancing in a single axis, motivated as an alternative to ballbots. Follow-up: "Collinear mecanum drive: modelling, analysis, partial feedback linearisation, and nonlinear control", https://eprints.whiterose.ac.uk/157762/1/journal_backstepping.pdf. Note from search snippet: ballbot yaw torque is capped by ball-floor friction.
- Sihite, UCSD 2019 dissertation (catalog record only): reduced nonlinear model for a micro ball-balancing robot capturing high yaw rate, with EKF; closest to yaw-coupling modelling. Not verified in full.
- Holland, Hayes, Langlois, kinematic slip ratio for omniwheels on a spherical platform (https://carleton.ca/johnhayes/wp-content/uploads/R05-CSME-44-HHL-Rev-F-W-Abs.pdf); Stonier et al., "Nonlinear slip dynamics for an omniwheel mobile robot platform" (non-ballbot).
- Drive-mechanism summary from Wikipedia list: CMU inverse mouse-ball (4 rollers + separate yaw motor), BallIP/Rezero/Kugle 3 omniwheels with yaw from the same motors, Festo BionicMobileAssistant and Bossa Nova mObi (industrial), UIUC ballbot wheelchair 2023.

## 7. Gaps (searched, nothing verified)

- Disney Research ballbot / "rolling robot" character: no ballbot paper found. Disney work found is a biped on a rolling ball/cylinder (https://la.disneyresearch.com/publication/optimization-and-control-of-cyclic-biped-locomotion-on-a-rolling-cylinder/). Rezero (ETH) is the character-oriented ballbot.
- Differential flatness, energy-based / passivity control, disturbance observer specifically for ballbots: no direct hits. CMU "shape-accelerated" planning is the closest planning analogue.
- TU Delft, Univ. Adelaide (Wikipedia mentions a 2009 Adelaide LEGO and full-scale bowling-ball version), "Bhattacharya/Han ballbot with arms", Liao/Zhou, "BallIP" beyond Kumagai: no primary sources found.
- Primary numbers (speeds, accelerations, tilt limits) for CMU 2014 and Rezero are unconfirmed.

## 8. Ideas to borrow / how our approach compares

Our setup: 3-omniwheel ballbot in MuJoCo, ball r=0.09 m (0.18 m dia, 0.17 kg), body 3.9 kg, planar Lagrangian model (unstable pole 4.8 1/s, NMP zero 5.8 1/s), discrete LQR + velocity reference governor + outer integral, PPO reaching about 0.75 m/s robust tracking, push recovery to 50 N.

Scale comparison. Our body (3.9 kg, 0.18 m ball) is in the BallIP/Rezero/Kugle class (7.5 kg with 0.20 m ball; Kugle 16 kg), far lighter than CMU's 45 kg person-sized robot with 0.20 m ball. A 0.18 m ball under 3.9 kg is a high ball-to-body size ratio, so our unstable pole of 4.8 1/s is faster than a tall heavy CMU-type robot would have (inverted-pendulum time scales shrink with height); expect that to make our loop demanding on control rate (Kugle 200 Hz, Rezero about 160 Hz are reference rates; Fischer used 5 ms LQR).

Ideas to borrow:
1. Ball acceleration as control input (Kumagai): abstracting torque to ball acceleration made the loop robust to inertia error and decouples the axes. Our discrete LQR could be fed a ball-acceleration command with a low-level torque tracker; this also makes PPO's action space more physical.
2. Passive/compliant mode (BallIP, CMU HRI): a low-gain mode to accept pushes. We could add this as an explicit mode or as an RL reward term, to compare against our 50 N hard recovery.
3. Cascade structure (Kugle, Pham, Fischer): fast balance loop + slower velocity LQR + MPC with tilt and velocity constraints. Our reference governor is the same idea; consider constraint-aware MPC (CBF-MPC or Fischer-style tilt <= 3 deg constraint) as a stronger classical baseline against PPO, with Kugle's 200 Hz and 1 m/s test as a benchmark target.
4. Shape-accelerated planning (Nagarajan): lean deliberately to generate acceleration; for non-minimum-phase behaviour (our zero 5.8 1/s) the planner should precompute an initial counter-lean. Use for feed-forward trajectories, and as a teacher/oracle for PPO.
5. Sliding mode and adaptive backstepping (Lal 2020, Do and Lee 2021, Jespersen 2019): robustness baselines under model mismatch; useful for comparing PPO's robustness claims.
6. Parameter study (Pham 2024): body mass matters more than ball diameter for LQR response; supports running a mass/inertia domain-randomization and sensitivity table for ours (ball mass 0.17 kg vs body 3.9 kg).
7. 3D vs planar model (Fankhauser and Gwerder): they argue planar decomposition limits speed and accuracy and claim large gains with a 3D model; Jespersen found a quaternion model unnecessary in the normal range. Our planar Lagrangian is likely fine below roughly 1 m/s, but a test at higher speed/yaw rate against the 3D MuJoCo dynamics would quantify where the decoupling fails. Yaw-ball friction coupling (Watson, Sihite) is the likely failure mode.
8. Slip modelling: use omniwheel slip-ratio and friction-limited yaw torque as extra randomization.

How we compare: our 0.75 m/s tracking is within the range tested by Kugle (1 m/s) and below Rezero's claimed 2 to 3.5 m/s (unverified). Our push recovery of 50 N has no direct published counterpart I could verify (CMU/BallIP report qualitative push tolerance only), so it would be a novel quantitative metric to report; Fischer's hardware shows only 3 deg and 15 cm/s tracking limits, so classical baselines are typically much more conservative than a learned policy. Caveat: all comparisons above rely on secondary numbers.
