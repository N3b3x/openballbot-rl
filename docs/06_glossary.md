# 📖 Glossary: Key Terms and Concepts

*A comprehensive glossary of technical terms used in OpenBallBot-RL documentation*

---

## 📋 Table of Contents

1. [Physics & Dynamics](#physics--dynamics)
2. [Reinforcement Learning](#reinforcement-learning)
3. [Robotics](#robotics)
4. [Simulation](#simulation)
5. [Control Theory](#control-theory)
6. [Mathematical Notation](#mathematical-notation)

---

## ⚙️ Physics & Dynamics

### Anisotropic Friction
**Definition:** Friction that varies with direction. In omniwheels, low friction along the rolling direction and high friction perpendicular to it.

**Context:** The MuJoCo patch enables anisotropic friction (`friction="0.001 1.0"`) to model omniwheel behavior without simulating individual rollers.

**Related:** Isotropic friction, omniwheel, contact forces

---

### Ballbot
**Definition:** A dynamically balanced mobile robot that balances on a single spherical ball using three omniwheels.

**Context:** The robot must actively control the wheels to maintain balance while moving, making it an underactuated system.

**Related:** Underactuated system, dynamic balance, omniwheel

---

### Center of Mass (CoM)
**Definition:** The point where the mass of a body is concentrated. For stability, the CoM must stay above the contact point.

**Context:** The ballbot maintains balance by keeping its CoM above the ball contact point through active control.

**Related:** Stability, balance, contact point

---

### Contact Forces
**Definition:** Forces that arise when two bodies come into contact. Includes normal forces (perpendicular) and friction forces (tangential).

**Context:** MuJoCo computes contact forces between the ball and wheels, and between the ball and terrain.

**Related:** Anisotropic friction, normal force, friction coefficient

---

### Degrees of Freedom (DoF)
**Definition:** The number of independent parameters needed to specify the configuration of a system.

**Context:** The ballbot has 6 DoF (3 orientation angles + 3 position coordinates) but only 3 control inputs (wheel torques), making it underactuated.

**Related:** Underactuated system, configuration space

---

### Dynamic Balance
**Definition:** Maintaining stability through active control rather than passive stability (like a static base).

**Context:** The ballbot cannot be statically stable (support polygon is a point), so it must use dynamic balance through wheel control.

**Related:** Static stability, support polygon, active control

---

### Euler Angles
**Definition:** A set of three angles (roll φ, pitch θ, yaw ψ) that describe orientation in 3D space.

**Context:** The observation space includes Euler angles \((\phi, \theta, \psi)\) to represent robot orientation.

**Related:** Quaternion, rotation matrix, orientation

---

### Inertia Matrix
**Definition:** A matrix \(\mathbf{M}(\mathbf{q})\) that relates angular accelerations to torques. Depends on the robot's configuration.

**Context:** The Lagrangian dynamics include the inertia matrix \(\mathbf{M}(\mathbf{q})\) which MuJoCo computes automatically.

**Related:** Lagrangian dynamics, configuration-dependent, mass matrix

---

### Isotropic Friction
**Definition:** Friction that is the same in all directions (standard MuJoCo behavior).

**Context:** Standard MuJoCo only supports isotropic friction, which doesn't work for omniwheels. The patch enables anisotropic friction.

**Related:** Anisotropic friction, friction coefficient

---

### Lagrangian Dynamics
**Definition:** A formulation of classical mechanics using the Lagrangian \(L = T - V\) (kinetic minus potential energy).

**Context:** Nagarajan (2014) derived the ballbot dynamics using Lagrangian mechanics, yielding:
\[
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]

**Related:** Euler-Lagrange equations, kinetic energy, potential energy

---

### Omniwheel
**Definition:** A wheel with small rollers around its circumference, allowing it to roll smoothly in one direction while providing high friction perpendicular to it.

**Context:** The ballbot uses three omniwheels arranged at 120° angles to control the ball's motion.

**Related:** Anisotropic friction, inverse mouse-ball drive, directional friction

---

### Quaternion
**Definition:** A four-component representation of 3D rotation (more numerically stable than Euler angles).

**Context:** MuJoCo stores orientation as quaternions `[w, x, y, z]`, which are converted to Euler angles for observations.

**Related:** Euler angles, rotation matrix, orientation

---

### Support Polygon
**Definition:** The convex hull of contact points between a robot and the ground. Determines static stability.

**Context:** For a ballbot, the support polygon collapses to a point (the ball contact), making static stability impossible.

**Related:** Static stability, dynamic balance, contact point

---

### Underactuated System
**Definition:** A system with fewer control inputs than degrees of freedom.

**Context:** The ballbot has 6 DoF but only 3 control inputs (wheel torques), making it underactuated. This means not all states can be directly controlled.

**Related:** Degrees of freedom, control inputs, actuation

---

## 🤖 Reinforcement Learning

### Actor-Critic
**Definition:** An RL algorithm that learns both a policy (actor) and a value function (critic).

**Context:** PPO is an actor-critic algorithm that learns both \(\pi_\theta(a|s)\) (policy) and \(V_\theta(s)\) (value function).

**Related:** Policy gradient, value function, PPO

---

### Advantage Function
**Definition:** \(A(s, a) = Q(s, a) - V(s)\), measuring how much better an action is than average.

**Context:** PPO uses Generalized Advantage Estimation (GAE) to compute advantages from rollouts.

**Related:** Q-function, value function, GAE

---

### Episode
**Definition:** A single trial in an RL environment, from initial state to termination.

**Context:** Each episode starts with `reset()` and ends when the robot falls (tilt > 20°) or reaches max steps (4000).

**Related:** Reset, termination, rollout

---

### Exploration vs Exploitation
**Definition:** The trade-off between trying new actions (exploration) and using known good actions (exploitation).

**Context:** PPO balances exploration (via entropy bonus) and exploitation (via policy updates).

**Related:** Entropy bonus, policy gradient, exploration

---

### Feature Extractor
**Definition:** A neural network that processes raw observations into feature vectors for the policy/value networks.

**Context:** The `Extractor` class processes multi-modal observations (proprioceptive + RGB-D) into a single feature vector.

**Related:** Observation space, neural network, multi-modal

---

### Generalized Advantage Estimation (GAE)
**Definition:** A method for estimating advantages using a weighted combination of TD errors.

**Context:** Stable-Baselines3 uses GAE to compute advantages from rollouts for PPO updates.

**Related:** Advantage function, TD error, PPO

---

### Policy
**Definition:** A function \(\pi(a|s)\) that maps states to action probabilities (or distributions).

**Context:** The PPO policy \(\pi_\theta(a|s)\) is a neural network that outputs a Gaussian distribution over actions.

**Related:** Policy gradient, neural network, action distribution

---

### Proximal Policy Optimization (PPO)
**Definition:** A policy gradient algorithm that prevents large policy updates by clipping the importance ratio.

**Context:** OpenBallBot-RL uses PPO from Stable-Baselines3 to train the ballbot navigation policy.

**Related:** Policy gradient, clipping, importance ratio, Stable-Baselines3

---

### Reward Function
**Definition:** A function \(r(s, a, s')\) that provides feedback to the RL agent.

**Context:** The ballbot reward combines directional progress, action regularization, and survival bonus:
\[
r = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100} - 0.0001\|\mathbf{a}\|^2 + 0.02 \cdot \mathbb{1}[\text{upright}]
\]

**Related:** Reward shaping, reward design, signal

---

### Reward Shaping
**Definition:** Modifying the reward function to guide learning without changing the optimal policy.

**Context:** The reward function is carefully designed to encourage balance (survival bonus) and navigation (directional reward) while penalizing unsafe actions.

**Related:** Reward function, reward design, optimal policy

---

### Rollout
**Definition:** A sequence of state-action-reward tuples collected by running the policy in the environment.

**Context:** PPO collects rollouts of length `n_steps=2048` before updating the policy.

**Related:** Episode, trajectory, experience buffer

---

### State Space
**Definition:** The set of all possible states the agent can be in.

**Context:** The ballbot state space includes orientation, velocities, motor states, and camera observations.

**Related:** Observation space, state representation

---

### Value Function
**Definition:** \(V(s) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]\), the expected cumulative reward from state \(s\).

**Context:** PPO learns a value function \(V_\theta(s)\) to estimate expected returns and compute advantages.

**Related:** Q-function, advantage function, critic

---

## 🦾 Robotics

### Actuator
**Definition:** A device that converts control signals into physical motion (e.g., motors).

**Context:** The ballbot has three actuators (wheel motors) that apply torques to control the ball.

**Related:** Motor, torque, control input

---

### Inverse Mouse-Ball Drive
**Definition:** A drive mechanism where wheels contact a ball from below, spinning the ball to move the robot (opposite of a computer mouse).

**Context:** Lauwers (2006) invented this mechanism for the first ballbot prototype.

**Related:** Omniwheel, ballbot, drive mechanism

---

### Proprioceptive Sensors
**Definition:** Sensors that measure the robot's own state (orientation, velocity, etc.), as opposed to exteroceptive sensors (cameras, lidar).

**Context:** The ballbot uses proprioceptive sensors (IMU, encoders) and exteroceptive sensors (RGB-D cameras).

**Related:** Exteroceptive sensors, IMU, encoders

---

### RGB-D Camera
**Definition:** A camera that captures both color (RGB) and depth (D) information.

**Context:** The ballbot uses two RGB-D cameras to perceive terrain and obstacles.

**Related:** Depth camera, visual perception, exteroceptive sensors

---

### Sensor Fusion
**Definition:** Combining information from multiple sensors to improve state estimation.

**Context:** The observation space combines proprioceptive sensors (IMU) and exteroceptive sensors (RGB-D cameras).

**Related:** Multi-modal, sensor integration, state estimation

---

## 🖥️ Simulation

### Gymnasium
**Definition:** The standard Python API for reinforcement learning environments (successor to OpenAI Gym).

**Context:** `BBotSimulation` implements the Gymnasium interface (`reset()`, `step()`, `action_space`, `observation_space`).

**Related:** OpenAI Gym, environment API, RL interface

---

### MuJoCo
**Definition:** A physics engine for robotics simulation with fast dynamics computation and contact handling.

**Context:** OpenBallBot-RL uses MuJoCo (patched for anisotropic friction) to simulate the ballbot physics.

**Related:** Physics engine, simulation, contact forces

---

### Perlin Noise
**Definition:** A procedural noise function that generates smooth, natural-looking random patterns.

**Context:** The terrain generator uses Perlin noise to create randomized uneven terrain for training.

**Related:** Procedural generation, terrain generation, randomization

---

### Sim-to-Real Transfer
**Definition:** The challenge of transferring policies trained in simulation to real robots.

**Context:** While not explicitly addressed in this repo, realistic physics (anisotropic friction) helps with sim-to-real transfer.

**Related:** Domain adaptation, reality gap, transfer learning

---

## 🎛️ Control Theory

### Constraint-Aware Control
**Definition:** Control methods that explicitly account for safety constraints (e.g., tilt limits).

**Context:** Carius (2022) showed how to encode constraints in the control objective, which translates to reward penalties in RL.

**Related:** Constrained optimization, safety constraints, reward penalties

---

### Feedback Control
**Definition:** Control that uses sensor measurements to adjust actions (closed-loop control).

**Context:** The RL policy uses observations (orientation, velocities) to compute actions (closed-loop feedback).

**Related:** Open-loop control, closed-loop, sensor feedback

---

### PID Controller
**Definition:** A classical controller using proportional, integral, and derivative terms.

**Context:** The repository includes a PID baseline (`policies.PID`) for comparison with RL.

**Related:** Classical control, feedback control, baseline

---

### Trajectory Tracking
**Definition:** Following a predefined path or trajectory.

**Context:** While not the focus of this repo, classical ballbot control often uses trajectory tracking for navigation.

**Related:** Path planning, waypoint following, navigation

---

## 🔢 Mathematical Notation

### \(\mathbf{q}\)
**Definition:** Generalized coordinates (configuration variables).

**Context:** For the ballbot, \(\mathbf{q} = [\phi, \theta, \psi]^T\) (Euler angles).

**Related:** Configuration space, degrees of freedom

---

### \(\dot{\mathbf{q}}\)
**Definition:** Time derivative of \(\mathbf{q}\) (velocities).

**Context:** \(\dot{\mathbf{q}} = [\dot{\phi}, \dot{\theta}, \dot{\psi}]^T\) (angular velocities).

**Related:** Velocity, time derivative

---

### \(\ddot{\mathbf{q}}\)
**Definition:** Second time derivative of \(\mathbf{q}\) (accelerations).

**Context:** Appears in the dynamics equation: \(\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + ... = \boldsymbol{\tau}\).

**Related:** Acceleration, dynamics

---

### \(\boldsymbol{\tau}\)
**Definition:** Control torques (control inputs).

**Context:** \(\boldsymbol{\tau} = [\tau_0, \tau_1, \tau_2]^T\) (three wheel torques).

**Related:** Torque, control input, actuator

---

### \(\mathbf{M}(\mathbf{q})\)
**Definition:** Inertia matrix (configuration-dependent).

**Context:** Appears in Lagrangian dynamics: \(\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + ... = \boldsymbol{\tau}\).

**Related:** Inertia, mass matrix, configuration-dependent

---

### \(\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\)
**Definition:** Coriolis and centrifugal force matrix.

**Context:** Captures velocity-dependent forces in the dynamics equation.

**Related:** Coriolis force, centrifugal force, dynamics

---

### \(\mathbf{G}(\mathbf{q})\)
**Definition:** Gravitational force vector.

**Context:** \(\mathbf{G}(\mathbf{q}) = mg\frac{\partial h}{\partial \mathbf{q}}\), where \(h\) is center of mass height.

**Related:** Gravity, potential energy, dynamics

---

### \(\pi_\theta(a|s)\)
**Definition:** Policy parameterized by \(\theta\), mapping state \(s\) to action distribution.

**Context:** The PPO policy \(\pi_\theta(a|s)\) is a neural network with parameters \(\theta\).

**Related:** Policy, neural network, parameters

---

### \(V_\theta(s)\)
**Definition:** Value function parameterized by \(\theta\), estimating expected return from state \(s\).

**Context:** PPO learns both \(\pi_\theta(a|s)\) and \(V_\theta(s)\) (actor-critic).

**Related:** Value function, critic, expected return

---

### \(\mathbb{E}[\cdot]\)
**Definition:** Expectation operator (average over randomness).

**Context:** Used in RL objectives: \(\mathbb{E}_\pi[\sum_t r_t]\) (expected cumulative reward).

**Related:** Expectation, average, probability

---

### \(\mathbb{1}[\cdot]\)
**Definition:** Indicator function (1 if condition is true, 0 otherwise).

**Context:** Used in survival bonus: \(\mathbb{1}[\text{upright}]\) (1 if robot is upright, 0 otherwise).

**Related:** Indicator, boolean, conditional

---

**Happy Learning! 📚**

*Last Updated: 2025*

