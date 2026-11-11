# Ballbot hardware, designs and open-source codebases: literature and web survey

Scope: web-only survey (search plus page reads, nothing downloaded or run). Date of survey: 2026-11-04.
Honesty convention: every number below is tagged by what I actually saw. "Not found" means the page I could read did not give it, not that it does not exist. Several key papers (Nagarajan 2014 IJRR, Xiao 2023 drivetrain paper, Kugle thesis PDF, Illinois dissertation, MBBR 2015 paper, Watson 2018) were only reachable as abstracts, snippets or unreadable PDFs, so their detailed tables are NOT reproduced here. Nothing in this file was invented; gaps are listed in section 6.

## 1. Catalogue of projects

### 1.1 Research / university platforms

| Project | URL | License | What I verified |
|---|---|---|---|
| CMU Ballbot (Lauwers, Kantor, Hollis 2006; Nagarajan, Kantor, Hollis 2014 IJRR) | https://ri.cmu.edu/publications/the-ballbot-an-omnidirectional-balancing-mobile-robot ; https://ri.cmu.edu/?p=9475 | n/a (paper) | 2006 ball: 200 mm diameter hydroformed steel shell with 3.2 mm urethane layer, "inverse mouse-ball" drive (2006: 2 roller pairs; 2014 IJRR evolved to 4-wheel inverse mouse-ball drive), yaw drive, triad of landing legs. Press release: 95 lb (about 43 kg), roughly human height and width. Ball mass, COM height, motors: not found. |
| ETH Zurich Rezero (Project Rezero) | https://spectrum.ieee.org/project-rezero-ball-balancing-elegance ; https://robohub.org/designed-to-mingle-rezero-ballbots-smooth-moves-can-handle-a-crowd ; https://ien.eu/article/stability-and-dynamics-on-a-ball | Closed (not open source as far as I found) | Three ETH-type omniwheels at 120 deg, an arrestor opposite each wheel to hold the ball. Ball: aluminium hollow sphere with about 4 mm rubberised low-wear coating (IEN article, via search snippet). Motors: 4-pole BLDC with 42 mm planetary gearhead and encoder, maxon EPOS 70/10 (via search snippet, not confirmed on the page I fetched). Control loop 160 Hz (Robohub). Lean up to 17 deg (IEEE Spectrum, Robohub); speed 3.5 m/s (IEEE Spectrum) vs 3.6 m/s and about 3 m/s^2 acceleration, 20 deg lean (press snippets, less reliable). Entertainment / character intent, Disney supported (IEEE Spectrum). Mass, height, ball diameter: not found. Rezero wheel-motor Coulomb friction about 42 mNm (quoted in asRoBallet paper). |
| Kugle (Aalborg, T. K. Jespersen, MSc 2019) | https://projekter.aau.dk/kugle-modelling-and-control-of-a-ball-balancing-robot-4450f15c.html ; https://github.com/mindThomas/Kugle-MATLAB (also Kugle-ROS, Kugle-Embedded, Kugle-Misc) | MIT (README states MIT across repos) | 16 kg Kugle V1, three omniwheels, STM32H7 firmware (NUCLEO-H743ZI), two EKFs, quaternion model, sliding-mode attitude controller, velocity LQR plus shape-accelerated MPC. Tracking error under 1 deg for commands up to 1 Hz; references up to 1 m/s and 1 rad/s tested. Vicon validation. Ball diameter, motors: not found on pages read (should be in Parameters folder / thesis). Kugle wheel-motor Coulomb friction about 26 mNm (quoted in asRoBallet paper). Thesis notes the quaternion model is unnecessary in normal operating range. |
| MiaPURE / PURE (UIUC, C. Xiao, Hsiao-Wecksler; arXiv 2304.02887, IROS 2023) | https://arxiv.org/abs/2304.02887 | n/a (paper) | Off-the-shelf-parts drivetrain, 60 kg payload, max speed 2.3 m/s, stops from 1.4 m/s in 2 s in a chosen direction, LQR-PI controller validated first on a planar inverted pendulum testbed (PIPTB). Noted high stiction from elastomeric components in torque transmission. Ball size, motors: not found (PDF unreadable via my tool). |
| Xiao PhD dissertation, UIUC 2022, "Personal unique rolling experience: design, modeling, and control of a riding ballbot" | https://www.ideals.illinois.edu/items/126838 (403 for my fetch; known from search snippet only) | unknown | Contains a design optimisation of 3- and 4-omniwheel drivetrain configurations for torque generation and contact stability. Motivation: the first prototype lost stability from omniwheel-ball slip. Results not read. This is the closest thing found to a published design-optimisation study; worth reading in full. |
| Micro Ball Balancing Robot, MBBR (Yang, Sihite, Friesen, Bewley, IROS 2015; Sihite UCSD thesis) | https://escholarship.org/uc/item/2cd979dq ; DOI 10.1109/IROS.2015.7353956 | unknown | Key idea: omniwheels placed at "midlatitude" (about 20 to 30 deg N latitude i.e. contact roughly 60 to 70 deg from the north pole) and orthogonal to each other, to raise normal force because miniaturisation cuts weight and thus normal force and increases stiction/slip problems. Ball size and mass: not found. |
| Kumagai and Ochiai, Tohoku Gakuin Univ, "BallIP", ICCAS 2008, JRM 2010 | https://spectrum.ieee.org/042910-a-robot-that-balances-on-a-ball ; https://robotsguide.com/robots/ballip | unknown | Three omniwheels, 16-bit MCU, gyros plus accelerometer. IEEE Spectrum: 20 cm bowling ball (3.6 kg) coated with rubber spray, robot about half a metre tall, 7.5 kg. Robots Guide: 12.5 kg, 65 cm, payload up to 15 kg. Sources disagree on mass (probably robot-only vs total); I did not resolve it. |
| Watson et al., Univ. of Sheffield, IEEE ICIT 2018, omnidirectional balancing platform for confined-space inspection | https://eprints.whiterose.ac.uk/132164/1/Watson%202018%20Design%20OmniDirectional.pdf | unknown | PDF unreadable via my tool; details not extracted. |
| asRoBallet (humanoid ballbot, arXiv 2604.24916, accepted RSS 2026) | https://arxiv.org/html/2604.24916v1 | Code "to be released" at https://doc.ancoraspring.com/ (not verified live) | See section 2. Most useful sim-to-real friction source found. |
| Salehi, OpenBallBot-RL (arXiv 2505.18417) | https://github.com/salehiac/OpenBallBot-RL ; mirror https://github.com/standardgalactic/OpenBallBot-RL | Apache-2.0 | MuJoCo ballbot (needs patched MuJoCo build, `mujoco_fix.patch`), omniwheels as capsules with anisotropic friction, PPO, depth camera, 500 Hz sim step, failure at 20 deg tilt. This is the repo our study builds on. Physical masses/radii not in the paper text I read. |
| Studt, Zhavzharov, Abbas, ECC 2022; Zhou et al. ICCCR 2021 (deep RL for ball-balancing robots); Shu and Hollis IROS 2021; Jespersen et al. ICRA 2020 | cited in asRoBallet reference list | unknown | Only seen as citations; not read. |

### 1.2 Hobby / course / portfolio projects

| Project | URL | License | Verified details |
|---|---|---|---|
| "Paul" by Jochen Alt | https://github.com/jochenalt/Paul ; https://hackaday.io/project/20045-paul ; https://pololu.com/blog/653/ball-balancing-robot | Not stated (GitHub page showed none) | Three omniwheels, Pololu 37D metal gearmotors with encoders, gear ratio 1:19 (author comment), 3x VNH2SP30 drivers, MPU-6050 with Kalman filter, two ATmega644, about 100 Hz balance loop, modified manually-tuned PID, fixed-point math. Lessons: loose IMU wire caused a violent fall; manual PID beat the academic tuning he tried; "a high body mass is fine if the ball and wheels are light and the centre of gravity is high"; would use an ARM board (Teensy 3.6) next time. Ball size, mass: not found. Repo has Mechanics, Schematics, datasheets, source folders (44 stars, 13 forks at fetch). |
| algarv Ball Balancing Robot | https://algarv.github.io/Portfolio/BallBot ; https://github.com/algarv/Ball_Balancing_Robot | not checked | Three steppers, three omniwheels, RP2040, complementary filter, PD controller, IMU read at about 10 Hz (too slow) and 500 Hz step timer. Balanced only about 2 s. Failure modes: motor-to-motor power interference (fixed with higher-voltage drivers), accelerometer-only spikes up to +/-40 deg, timer contention. Author chose a larger ball to reduce lean angle and COM lever arm. |
| RMI-NITT/Ballbot | https://github.com/RMI-NITT/Ballbot | not checked | ROS Kinetic and Gazebo; single-point stability and go-to-goal. Only seen via search summary. |
| Prasheel24/ballbot | https://gh.r-mj.com/Prasheel24/ballbot (mirror) | not checked | Simulation-only, MATLAB/Simulink, balancing and station keeping. Search summary only. |
| I-M-Robotics-Lab Ball-Balancing-Robot | https://www.github.com/I-M-Robotics-Lab/Ball-Balancing-Robot | not checked | Listed in search results only; contents unverified. Note: several "ball balancing" repos are ball-on-plate platforms, a different system. |
| Other thesis hits (not opened) | Padova MSc (Singh Bhawandeep) https://thesis.unipd.it/retrieve/c745da0f-ce3b-483e-bb9b-eb2ac529748f/Singh_Bhawandeep.pdf ; UW capstone PIICCAR https://www.engr.washington.edu/files/industry/docs/capstone-projects/2020-2021/PIICCAR_Billy-Lin_Wilson-Martinez_Ariel-Chang_Junyi-Wu-2.pdf ; Univ. Scranton closeout https://ece454designprojects.scranton.edu/bradleyhodousek/wp-content/uploads/sites/28/2017/05/project-closeout.pdf ; ZCU bachelor thesis https://otik.zcu.cz/bitstream/11025/44983/1/Bakalarska%20prace%20Lukas%20Kolbl.pdf | n/a | Appeared in search for "Kugle ball balancing robot"; I did not read them, so I cannot say they are ballbots. Candidates for follow-up. |
| Bilkent CS548 course project | https://www.ee.bilkent.edu.tr/~inal/cs548project.htm | n/a | Simplified model values listed: ball radius 0.106 m, ball mass 2.44 kg, COM 0.69 m above ball centre. These are model parameters, not necessarily hardware. |

### 1.3 Commercial / entertainment / patents

| Item | URL | Notes |
|---|---|---|
| US Patent 10,189,342 "Ball-balancing robot and drive assembly therefor" | https://patents.justia.com/patent/10189342 (403 on fetch; content from search snippets) | Wheel normal force N = m g / (3 cos theta), theta measured from north pole. Lowering wheels to about 70 deg from the pole raises normal force. Scaling a design down by length l cuts mass roughly l^3 and so cuts contact normal force, causing slip. Adding body mass just to raise normal force is called undesirable. Standard layout mentioned: 120 deg wheel separation, about 45 deg from the pole. Assignee not confirmed. |
| Disney Research "ball walker" | https://la.disneyresearch.com/?p=9116 | A biped walking on a rolling ball (wheeled LIP model). Not a ballbot. No Disney ballbot hardware found. |
| Disney BB-8 | https://hackaday.com/2015/04/19/bb-8-is-real-but-how-did-they-do-it | Different mechanism (internal drive in a rolling shell). Not relevant to ballbot design except for character styling. |
| "Segway-style ball drive", "Ballerina", Sphero-like, commercial hobby kits | not found | I did not find a verified commercial ballbot product or hobby kit. Do not assume one exists from this survey. |
| Ubergizmo "Ballbot shrunk to desktop size" (2010) | https://ubergizmo.com/2010/03/ballbot-shrunk-to-desktop-size/amp | Press item, no dimensions. Likely a Japanese or CMU-descendant demo; unverified. |
| Omni-ball hobby builds (James Bruton) | https://hackaday.com/2021/05/20/3d-printing-omni-balls-for-robot-locomotion/ | Not ballbots; TPU pads for grip is the transferable tip. |

### 1.4 Public simulation models of ballbots

| Model | Simulator | License | Note |
|---|---|---|---|
| OpenBallBot-RL (`ballbotgym`) | MuJoCo (patched) | Apache-2.0 | Capsule omniwheels with anisotropic friction. Our base. |
| asRoBallet sim | MuJoCo, elliptic cone, impratio 10, Newton solver, noslip_iterations 1, discrete roller mechanics of ETH-type omniwheels | code release pending | Most faithful wheel model described. |
| Kugle-MATLAB / Kugle-ROS | MATLAB/Simulink, ACADO MPC, ROS (Gazebo keyword) | MIT | Analytical models and ROS integration. |
| RMI-NITT/Ballbot | Gazebo (ROS Kinetic) | unchecked | Old. |
| Prasheel24/ballbot | MATLAB/Simulink | unchecked | |
| Isaac Sim / Isaac Lab / Unity ballbot | none found | | One search only; absence is not proven. |

## 2. Quantitative numbers collected (with sources)

| Quantity | Value | Source / confidence |
|---|---|---|
| asRoBallet ball | 220 mm diameter, SBR-coated hollow steel sphere | arXiv 2604.24916, read directly |
| asRoBallet wheels | 3 ETH-type omniwheels, 12 rollers each, 120 deg apart | same |
| asRoBallet wheel motors | RobStride 05, 5.5 N m | same |
| Wheel-motor Coulomb friction | asRoBallet about 130 mNm, Rezero about 42 mNm, Kugle about 26 mNm | same (quoted comparison) |
| Friction domain randomisation used | floor tangential/torsional 0.6 to 1.2; motor Coulomb 0.08 to 0.12 N m; LQR assumption of high torsional friction broke at 0.01 to 0.5 | same |
| asRoBallet control period | 10 ms policy, frame_skip 5 (sim step 2 ms), IMU 2 kHz | same |
| asRoBallet hardware results | station-keeping within 5 cm (tile) and 3 cm (yoga mat); pushes up to 0.3 m recovered in 7 trials; joystick tracking MAE 0.05 m/s | same |
| asRoBallet sim | RL 100% success vs LQR 86% (random arm poses), LQR 56% with random orientation | same |
| Training pitfall | training failed when wheel motors were replaced with velocity-type actuators; Coulomb dead zone causes limit-cycle jitter in station keeping | same |
| Rezero | 160 Hz loop, 17 deg lean, 3.5 m/s | IEEE Spectrum, Robohub |
| MiaPURE | 2.3 m/s, 60 kg payload, stop from 1.4 m/s in 2 s | arXiv abstract |
| Kugle V1 | 16 kg, tracking under 1 deg, 1 m/s refs | AAU abstract |
| BallIP (Kumagai) | 20 cm ball, 3.6 kg ball, robot 7.5 kg (IEEE) or 12.5 kg (Robots Guide) | conflicting |
| CMU 2006 ball | 200 mm steel shell + 3.2 mm urethane | Lauwers 2006 via snippet |
| CMU total mass | about 43 kg | press release via snippet |
| Paul motors | Pololu 37D, 1:19 | Hackaday comment |
| OpenBallBot-RL | 500 Hz sim, depth cam about 80 Hz, fail at 20 deg tilt, PPO learns balance by about 3e6 steps, about 0.5 m/s toward goal | arXiv 2505.18417v2 |
| Wheel contact latitude | standard about 45 deg from pole (patent); MBBR 60 to 70 deg from pole; patent suggests 70 deg to cut slip | patent / IROS 2015 snippets |

Mass ratios actually found in practice: BallIP ball 3.6 kg vs robot 7.5 kg gives body:ball of about 2 (if 7.5 kg is robot-only), or about 2.5 if total is 12.5 kg and ball is 3.6 kg (body about 8.9 kg). I found no source with a body:ball ratio anywhere near 23. Kugle and CMU ball masses were not found. So the ratio claim is "BallIP is about 2 to 3; others unknown", not a survey of many.

## 3. Lessons learned and failure modes reported

1. Wheel-ball slip is THE recurring failure and gets worse as the robot shrinks (patent, MBBR, Xiao dissertation motivation). Normal force from weight alone scales with m while ball/wheel size scale with l, so small robots are traction-starved.
2. Contact latitude is the main lever: lower the wheels (larger theta from the pole) to raise N = mg/(3 cos theta).
3. Stiction and Coulomb friction in the drive (elastomer transmission, gearheads) cause limit cycles and dead zones; Rezero 42 mNm, Kugle 26 mNm, asRoBallet 130 mNm. Low-friction, back-drivable actuators matter more than peak torque for a good station-keeping feel.
4. Actuator interface matters for learned policies: torque-mode worked, velocity-mode broke training (asRoBallet).
5. Hobby builds fail on boring things: IMU wiring, filter choice (accelerometer-only gives +/-40 deg spikes), loop rate (10 Hz IMU too slow, 100 to 500 Hz needed), power-rail noise between motor drivers.
6. Larger ball reduces required lean angle and lever arm (algarv). Heavy body with light ball and light wheels with high COM is acceptable (Paul author).
7. RL beats LQR when friction deviates from the linear model assumptions and when body configuration changes (asRoBallet), but needs friction and motor-friction randomisation.

## 4. Gaps I could not fill (explicitly unverified)

- Rezero mass, height, ball diameter, motor torque; CMU ball mass and COM; Kugle ball diameter, motor model; MiaPURE ball diameter and motor torque; Paul ball diameter and mass; MBBR ball diameter and mass; Sihite thesis contents; Xiao dissertation optimisation outcomes.
- Torque-to-weight ratios as a design rule: no source found that states one. Any such ratio below is derived by me.
- Ball surface material guidance beyond: urethane 3.2 mm (CMU), rubberised 4 mm on aluminium (Rezero), SBR on steel (asRoBallet), rubber spray on bowling ball (BallIP).
- Preload: Rezero uses arrestors opposite each wheel; a search snippet mentioned spring-loaded idlers for slip prevention. No numbers.
- Scaling studies: only the patent's l^3 argument and the Xiao dissertation abstract. No quantitative scaling paper found.
- YouTube build logs, Printables/Thingiverse ballbots, Instructables, GitLab, Hackaday beyond Paul: searched only once, nothing concrete found. Other universities requested (TU Delft, Michigan, Georgia Tech, JHU, Stanford, Tokyo, KAIST): no specific ballbot project located; not proven absent.

## 5. What this means for our design study and for a small character-scale build

Our numbers: ball r = 0.09 m, body 3.9 kg, ball 0.17 kg (body:ball about 23), COM 0.13 m above ball centre, wheel zenith about 51 deg, 3 omniwheels, 30 N m wheel torque limit in sim, climbing 3 cm steps limited by traction.

Back-of-envelope (my own arithmetic, not from a source; assumes static load only and the patent's N formula):
- Static normal force per wheel N = m g / (3 cos 51 deg) = 3.9 * 9.81 / (3 * 0.629) about 20 N (counting only the 3.9 kg body weight; including the ball is a small correction).
- At wheel-ball friction coefficient mu of 0.6 to 1.0 (assumed, rubber on rubberised ball), tangential force per wheel before slip is about 12 to 20 N, so the three wheels can transmit very roughly 1 to 2 N m of torque to the ball about a horizontal axis (after projecting wheel directions; order-of-magnitude only). Gravity torque at a lean angle is m g h sin(phi) = 3.9 * 9.81 * 0.13 * sin(phi) about 5.0 sin(phi) N m, so holding 10 deg of lean needs about 0.9 N m, and any acceleration or a step climb adds to that.
- Consequence: the traction envelope is about 1 to 2 N m at the ball, tiny next to a 30 N m per wheel torque limit. If the 30 N m is applied at the wheel shaft with a small wheel radius, the equivalent tangential force is far above the slip limit. The torque limit is therefore not the binding constraint; friction is. This matches the literature (slip, not torque, as the dominant failure) and supports our observation that step climbing is traction-limited. The sim should be checked to see whether wheel-ball friction is actually the limiting factor (impratio, cone type, noslip iterations) and whether the capsule-with-anisotropic-friction wheel model is generous or stingy relative to real rollers.

Design implications:
1. Zenith: our 51 deg is slightly lower than the patent's 70 deg suggestion and the MBBR's 60 to 70 deg. Going from 51 to 65 deg raises N by cos51/cos65 = 0.629/0.423 = about 1.49x for the same weight. This is a cheap, literature-backed way to raise traction and should be a swept variable in the design study (try 45, 55, 65, 70). Costs: more horizontal force component pushing the ball sideways, packaging near the equator, less wheel clearance, and more coupling between wheels, so check actuation-matrix conditioning.
2. Mass ratio 23 is far outside anything I could document (BallIP about 2 to 3). Real builds either use a heavy ball (3.6 kg bowling ball, steel shells) or keep the body light. A 170 g ball is closer to a hollow plastic or thin shell. Low ball inertia helps agility, but low ball mass also means little rolling-inertia buffering and the ball is easily dragged by wheel slip. The study should sweep ball mass (0.17 to about 1 kg) and report whether traction margin or control authority improves; expect this to be an open question rather than a settled rule.
3. Ball radius: larger ball means less lean for the same acceleration and a bigger lever to cross steps (algarv, qualitative), but for a step of 3 cm the geometry favours a bigger r; r = 0.09 m is small compared to 0.10 to 0.11 m used in BallIP, CMU, asRoBallet and likely Rezero. A character build of about 0.2 m ball diameter is within the range of real ballbots, which makes our 0.18 m diameter plausible but at the small end.
4. COM height 0.13 m is low for a 0.18 m ball; real humanoid-scale ballbots have COM 0.5 to 0.7 m (Bilkent model 0.69 m, not hardware-verified). A low COM gives faster pendulum dynamics (higher natural frequency sqrt(g/h)) and less time to react but also smaller gravitational toppling torque; it should be considered when comparing control rates. Real hobby ones run 100 to 500 Hz; keep our sim at 500 Hz with a realistic 100 to 200 Hz policy and add latency randomisation.
5. Sim fidelity items to add, all backed by asRoBallet: motor Coulomb friction randomisation 0.08 to 0.12 N m (hardware 26 to 130 mNm for large robots; a small build with small gearmotors may have a comparable or larger fraction relative to its 1 to 2 N m traction envelope, so friction is more significant for us); floor friction 0.6 to 1.2 and torsional friction down to 0.01; elliptic cone, impratio 10, noslip iterations 1; torque-mode actuation.
6. Actuator choice for a small character build: pick low-ratio, back-drivable, low-cogging motors (quasi-direct drive like the RobStride class at 5.5 N m is more than enough; the traction limit is 1 to 2 N m at the ball). Peak torque beyond the slip limit is wasted; spend the budget on low friction and a good IMU at kHz rates. Hobby gearmotors (Paul: 1:19 Pololu 37D) work but with PID at about 100 Hz and limited agility.
7. Preload: Rezero uses arrestors opposite each drive wheel to hold the ball; with a character shell this also provides a way to add preload without adding body mass, which the patent calls the undesirable way to get traction. We should model a preload force (spring or constant normal force) as a design variable. I have no published numeric preload.
8. Ball surface: use compliant rubber or urethane over a stiff core (CMU 3.2 mm urethane; Rezero about 4 mm rubberised; asRoBallet SBR). TPU grip pads (Bruton) are the cheap 3D-print analogue. Roller material on the omniwheels is an equally important knob and I found no quantitative data.
9. Character aesthetics: Rezero's "dynamic hull" (breathing, tilt up to 17 deg while turning) shows the lean and yaw envelope that looks expressive; large lean angles require traction margin in accordance with point 1 and a ball larger than the body footprint requirement. A small shell around a low COM makes expressive leaning hard; consider allowing a movable internal mass or head tilt (as asRoBallet's arms/head) rather than the whole body leaning.
10. Study design: report (a) traction margin = (mu N_wheel) / required tangential force across tasks; (b) sweep zenith, ball mass ratio, ball radius, COM height, friction; (c) report slip events as a failure mode distinct from tilt failure. Follow the Xiao dissertation's framing (torque generation capability and contact stability vs configuration) and cite it once the full text has been read.

## 6. Suggested follow-up reading (not done)

- Nagarajan, Kantor, Hollis, "The Ballbot", IJRR 2014, DOI 10.1177/0278364913509126 (for CMU masses and motor sizing).
- Xiao et al., arXiv 2304.02887 full text and Xiao UIUC 2022 dissertation (design optimisation, contact stability).
- Jespersen Kugle thesis PDF and the Parameters folder in Kugle-MATLAB (ball diameter, masses, motors).
- Yang, Sihite, Friesen, Bewley IROS 2015 and Sihite thesis (micro-scale traction).
- Skrabel 2013 ETH bachelor thesis (Rezero mechanical design), cited in asRoBallet.
- Watson et al. 2018 (Sheffield).
- Targeted searches for Isaac/Unity ballbot models, Printables/Thingiverse files, and TU Delft, Michigan, Georgia Tech, JHU, Stanford, KAIST theses.
