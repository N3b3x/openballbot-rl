# 🌍 Sim-to-Real Transfer for Robotics Reinforcement Learning

*A comprehensive guide to transferring policies from simulation to real robots*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [The Reality Gap](#the-reality-gap)
3. [Domain Randomization](#domain-randomization)
4. [Sensor Noise Modeling](#sensor-noise-modeling)
5. [Actuator Dynamics](#actuator-dynamics)
6. [Safety Considerations](#safety-considerations)
7. [Deployment Checklist](#deployment-checklist)
8. [Real-World Example: Ballbot Transfer](#real-world-example-ballbot-transfer)
9. [Advanced Techniques](#advanced-techniques)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

Sim-to-real transfer is the process of deploying policies trained in simulation to real robots. This is one of the most critical and challenging aspects of robotics RL.

> "The reality gap is the bane of simulation-based RL. But with careful design, we can bridge it."  
> — *Sergey Levine, UC Berkeley*

**Key Challenges:**
- **Reality gap**: Simulation never perfectly matches reality
- **Sensor noise**: Real sensors are noisy
- **Actuator limitations**: Real actuators have delays and limits
- **Safety**: Real robots can break or cause harm
- **Generalization**: Policies must work in unseen conditions

**Key Questions This Tutorial Answers:**
- What is the reality gap and how do we minimize it?
- How do we add realistic noise to sensors?
- How do we model actuator dynamics?
- What safety measures are needed?
- How do we deploy policies to real robots?

---

## 🌉 The Reality Gap

### What is the Reality Gap?

The **reality gap** is the difference between simulation and reality:

```
Reality Gap = |Simulation Behavior - Real Robot Behavior|
```

**Sources of Reality Gap:**

1. **Physics Modeling Errors**
   - Friction models imperfect
   - Contact dynamics simplified
   - Material properties approximate

2. **Sensor Differences**
   - Simulation: Perfect measurements
   - Reality: Noisy, delayed, missing data

3. **Actuator Differences**
   - Simulation: Ideal actuators
   - Reality: Delays, limits, backlash

4. **Environmental Differences**
   - Simulation: Controlled conditions
   - Reality: Variable lighting, temperature, etc.

### Measuring the Reality Gap

**Quantitative Metrics:**
```python
def measure_reality_gap(sim_trajectory, real_trajectory):
    """
    Measure difference between simulation and real trajectories.
    """
    # Position error
    pos_error = np.mean(np.linalg.norm(
        sim_trajectory['positions'] - real_trajectory['positions'], 
        axis=1
    ))
    
    # Velocity error
    vel_error = np.mean(np.linalg.norm(
        sim_trajectory['velocities'] - real_trajectory['velocities'],
        axis=1
    ))
    
    # Action error
    action_error = np.mean(np.linalg.norm(
        sim_trajectory['actions'] - real_trajectory['actions'],
        axis=1
    ))
    
    return {
        'position_error': pos_error,
        'velocity_error': vel_error,
        'action_error': action_error
    }
```

**Qualitative Assessment:**
- Does policy work on real robot?
- Does it generalize to new conditions?
- Is it safe and stable?

### Strategies to Minimize Reality Gap

1. **Domain Randomization** (most effective)
2. **Realistic Sensor Modeling**
3. **Actuator Dynamics**
4. **System Identification**
5. **Fine-tuning on Real Data**

---

## 🎲 Domain Randomization

### What is Domain Randomization?

**Domain randomization** is the practice of randomizing simulation parameters during training to make policies robust to reality gap.

**Key Idea:**
> "If you train on a distribution of simulations, the policy learns to be robust to simulation errors."

### Physics Randomization

**Randomize Physical Parameters:**

```python
class RandomizedBallbotEnv(gym.Env):
    """
    Ballbot environment with physics randomization.
    """
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Randomize physics parameters
        self.model.body_mass[body_id] *= np.random.uniform(0.8, 1.2)  # Mass
        self.model.body_inertia[body_id] *= np.random.uniform(0.8, 1.2)  # Inertia
        
        # Randomize friction
        friction_tangential = np.random.uniform(0.0005, 0.0015)
        friction_normal = np.random.uniform(0.8, 1.2)
        self.model.pair_friction[contact_id, 0] = friction_tangential
        self.model.pair_friction[contact_id, 1] = friction_normal
        
        # Randomize gravity (for robustness)
        self.model.opt.gravity[2] *= np.random.uniform(0.95, 1.05)
        
        return self._get_obs(), {}
```

**Parameters to Randomize:**
- Mass and inertia
- Friction coefficients
- Damping coefficients
- Joint limits
- Gravity (slight variations)

### Visual Randomization

**Randomize Visual Appearance:**

```python
def randomize_visuals(self):
    """
    Randomize visual properties for sim-to-real transfer.
    """
    # Randomize lighting
    self.renderer.scene.option.ambient = np.random.uniform(0.1, 0.5)
    
    # Randomize camera noise
    self.camera_noise_std = np.random.uniform(0.0, 0.02)
    
    # Randomize texture properties
    self.texture_scale = np.random.uniform(0.8, 1.2)
    
    # Randomize background
    self.background_color = np.random.uniform(0.0, 1.0, size=3)
```

**Benefits:**
- Robust to lighting changes
- Handles camera noise
- Generalizes to different environments

### Terrain Randomization

**Randomize Terrain Properties:**

```python
def randomize_terrain(self):
    """
    Randomize terrain for robust navigation.
    """
    # Terrain difficulty
    terrain_scale = np.random.uniform(20.0, 30.0)
    terrain_octaves = np.random.randint(3, 6)
    terrain_persistence = np.random.uniform(0.15, 0.25)
    
    # Generate randomized terrain
    terrain = generate_perlin_terrain(
        n=self.terrain_resolution,
        scale=terrain_scale,
        octaves=terrain_octaves,
        persistence=terrain_persistence,
        seed=np.random.randint(0, 10000)
    )
    
    return terrain
```

**Benefits:**
- Robust to terrain variations
- Handles uneven surfaces
- Generalizes to new environments

### Systematic Domain Randomization

**Progressive Randomization:**

```python
class ProgressiveDomainRandomization:
    """
    Gradually increase randomization during training.
    """
    def __init__(self):
        self.training_step = 0
        self.max_steps = 10_000_000
    
    def get_randomization_level(self):
        """
        Increase randomization as training progresses.
        """
        progress = self.training_step / self.max_steps
        
        # Start with low randomization, increase over time
        if progress < 0.3:
            return 0.3  # 30% randomization
        elif progress < 0.6:
            return 0.6  # 60% randomization
        else:
            return 1.0  # 100% randomization
    
    def randomize(self):
        level = self.get_randomization_level()
        
        # Scale randomization by level
        mass_range = (1.0 - 0.2 * level, 1.0 + 0.2 * level)
        friction_range = (0.001 * (1 - 0.5 * level), 0.001 * (1 + 0.5 * level))
        
        # Apply randomization
        # ...
```

**Benefits:**
- Start with easier conditions
- Gradually increase difficulty
- Better learning curve

---

## 📡 Sensor Noise Modeling

### Why Model Sensor Noise?

**Real sensors are noisy:**
- IMU: Gyroscope drift, accelerometer noise
- Cameras: Image noise, motion blur
- Encoders: Quantization, dead zones

**Simulation sensors are perfect:**
- No noise
- No delays
- Perfect measurements

**Solution:** Add realistic noise to simulation sensors.

### IMU Noise Modeling

**Gyroscope Noise:**

```python
class NoisyIMU:
    """
    Realistic IMU noise model.
    """
    def __init__(self):
        # Gyroscope parameters
        self.gyro_noise_std = 0.01  # rad/s (typical for MEMS gyro)
        self.gyro_bias = np.random.normal(0, 0.001, 3)  # Bias drift
        self.gyro_drift_rate = 0.0001  # rad/s²
        
        # Accelerometer parameters
        self.accel_noise_std = 0.1  # m/s²
        self.accel_bias = np.random.normal(0, 0.01, 3)
    
    def add_gyro_noise(self, true_angular_vel):
        """
        Add realistic gyroscope noise.
        """
        # White noise
        noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        # Bias drift (slowly changing)
        self.gyro_bias += np.random.normal(0, self.gyro_drift_rate, 3)
        
        # Total measurement
        measured = true_angular_vel + noise + self.gyro_bias
        
        return measured
    
    def add_accel_noise(self, true_accel):
        """
        Add realistic accelerometer noise.
        """
        # White noise
        noise = np.random.normal(0, self.accel_noise_std, 3)
        
        # Bias
        measured = true_accel + noise + self.accel_bias
        
        return measured
```

**Usage in Environment:**

```python
def _get_obs(self):
    # Get true angular velocity from MuJoCo
    true_angular_vel = self.data.cvel[body_id, 3:6]
    
    # Add IMU noise
    noisy_angular_vel = self.imu.add_gyro_noise(true_angular_vel)
    
    # Use noisy measurement in observation
    obs = {
        "angular_vel": noisy_angular_vel,
        # ...
    }
    return obs
```

### Camera Noise Modeling

**Image Noise:**

```python
class NoisyCamera:
    """
    Realistic camera noise model.
    """
    def __init__(self):
        self.image_noise_std = 0.02  # For normalized images [0, 1]
        self.motion_blur_prob = 0.1  # Probability of motion blur
    
    def add_image_noise(self, image):
        """
        Add realistic camera noise.
        """
        # Gaussian noise
        noise = np.random.normal(0, self.image_noise_std, image.shape)
        noisy_image = image + noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        # Motion blur (occasionally)
        if np.random.rand() < self.motion_blur_prob:
            noisy_image = self.apply_motion_blur(noisy_image)
        
        return noisy_image
    
    def apply_motion_blur(self, image):
        """
        Apply motion blur effect.
        """
        from scipy.ndimage import gaussian_filter
        # Blur in horizontal direction (simulating motion)
        blurred = gaussian_filter(image, sigma=(0, 1, 0))
        return blurred
```

**Depth Sensor Noise:**

```python
def add_depth_noise(self, depth_image):
    """
    Add realistic depth sensor noise.
    """
    # Depth-dependent noise (farther = noisier)
    noise_std = 0.01 * depth_image  # 1% of depth
    
    noise = np.random.normal(0, noise_std, depth_image.shape)
    noisy_depth = depth_image + noise
    
    # Clip to valid range
    noisy_depth = np.clip(noisy_depth, 0.0, 1.0)
    
    return noisy_depth
```

### Encoder Noise Modeling

**Motor Encoder Noise:**

```python
class NoisyEncoder:
    """
    Realistic encoder noise model.
    """
    def __init__(self):
        self.quantization_levels = 1024  # 10-bit encoder
        self.dead_zone = 0.001  # rad/s (below this, no reading)
    
    def add_encoder_noise(self, true_velocity):
        """
        Add realistic encoder noise.
        """
        # Quantization
        quantized = np.round(true_velocity * self.quantization_levels) / self.quantization_levels
        
        # Dead zone
        quantized[np.abs(quantized) < self.dead_zone] = 0.0
        
        return quantized
```

---

## ⚙️ Actuator Dynamics

### Why Model Actuator Dynamics?

**Real actuators have:**
- **Delays**: Command to execution delay
- **Limits**: Maximum torque/velocity
- **Backlash**: Mechanical play
- **Saturation**: Cannot exceed limits

**Simulation actuators are ideal:**
- Instant response
- No limits (or hard limits)
- Perfect tracking

**Solution:** Model realistic actuator dynamics.

### Actuator Delay

**Command Delay:**

```python
class DelayedActuator:
    """
    Model actuator command delay.
    """
    def __init__(self, delay_steps=2):
        self.delay_steps = delay_steps
        self.command_buffer = []
    
    def apply_action(self, action):
        """
        Apply action with delay.
        """
        # Add to buffer
        self.command_buffer.append(action)
        
        # Return delayed command
        if len(self.command_buffer) > self.delay_steps:
            delayed_action = self.command_buffer.pop(0)
        else:
            delayed_action = np.zeros_like(action)  # No command yet
        
        return delayed_action
```

**Usage:**

```python
def step(self, action):
    # Apply delay
    delayed_action = self.actuator_delay.apply_action(action)
    
    # Use delayed action
    self.data.ctrl[:] = delayed_action * 10
    
    # Step simulation
    mujoco.mj_step(self.model, self.data)
    
    # ...
```

### Actuator Limits

**Torque Saturation:**

```python
class LimitedActuator:
    """
    Model actuator torque limits.
    """
    def __init__(self, max_torque=10.0, max_velocity=10.0):
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        self.current_velocity = np.zeros(3)
    
    def apply_limits(self, desired_torque, current_velocity, dt):
        """
        Apply realistic actuator limits.
        """
        # Velocity limit
        if np.any(np.abs(current_velocity) > self.max_velocity):
            # Reduce torque to stay within velocity limit
            scale = self.max_velocity / np.max(np.abs(current_velocity))
            desired_torque *= scale
        
        # Torque limit
        limited_torque = np.clip(desired_torque, -self.max_torque, self.max_torque)
        
        # Rate limiting (smooth changes)
        max_torque_change = 5.0 * dt  # N⋅m/s
        torque_change = limited_torque - self.last_torque
        torque_change = np.clip(torque_change, -max_torque_change, max_torque_change)
        
        final_torque = self.last_torque + torque_change
        self.last_torque = final_torque
        
        return final_torque
```

### Actuator Dynamics Model

**First-Order Dynamics:**

```python
class ActuatorDynamics:
    """
    Model actuator as first-order system.
    """
    def __init__(self, time_constant=0.02):
        self.time_constant = time_constant  # 20ms time constant
        self.current_torque = np.zeros(3)
    
    def update(self, desired_torque, dt):
        """
        Update actuator state (first-order response).
        """
        # First-order dynamics: τ_dot = (τ_desired - τ_current) / τ_time
        torque_rate = (desired_torque - self.current_torque) / self.time_constant
        
        # Integrate
        self.current_torque += torque_rate * dt
        
        return self.current_torque
```

---

## 🛡️ Safety Considerations

### Why Safety Matters

**Real robots can:**
- Break hardware (motors, joints)
- Cause injury (collisions)
- Damage environment
- Fail catastrophically

**Simulation has no consequences:**
- Can try dangerous actions
- No hardware damage
- No safety risks

**Solution:** Implement safety measures.

### Safety Limits

**Joint Limits:**

```python
def enforce_joint_limits(self):
    """
    Enforce joint limits to prevent damage.
    """
    # Check joint positions
    joint_positions = self.data.qpos[joint_ids]
    joint_limits = self.model.jnt_range[joint_ids]
    
    # Clamp to limits
    joint_positions = np.clip(
        joint_positions,
        joint_limits[:, 0],
        joint_limits[:, 1]
    )
    
    # Apply limits
    self.data.qpos[joint_ids] = joint_positions
```

**Torque Limits:**

```python
def enforce_torque_limits(self, torques):
    """
    Enforce torque limits to prevent motor damage.
    """
    max_torque = 10.0  # N⋅m
    limited_torques = np.clip(torques, -max_torque, max_torque)
    return limited_torques
```

**Velocity Limits:**

```python
def enforce_velocity_limits(self):
    """
    Enforce velocity limits to prevent damage.
    """
    max_velocity = 10.0  # rad/s
    joint_velocities = self.data.qvel[joint_ids]
    
    # Scale down if exceeding limits
    if np.any(np.abs(joint_velocities) > max_velocity):
        scale = max_velocity / np.max(np.abs(joint_velocities))
        self.data.qvel[joint_ids] *= scale
```

### Emergency Stop

**Fail-Safe Mechanisms:**

```python
class EmergencyStop:
    """
    Emergency stop for safety.
    """
    def __init__(self):
        self.enabled = True
        self.max_tilt = np.deg2rad(30)  # 30° emergency limit
        self.max_velocity = 15.0  # rad/s
    
    def check_emergency(self, obs, info):
        """
        Check if emergency stop needed.
        """
        if not self.enabled:
            return False
        
        # Check tilt
        tilt_angle = self.compute_tilt_angle(obs)
        if tilt_angle > self.max_tilt:
            return True  # Emergency stop!
        
        # Check velocity
        if np.any(np.abs(obs['angular_vel']) > self.max_velocity):
            return True  # Emergency stop!
        
        return False
    
    def emergency_stop(self):
        """
        Execute emergency stop.
        """
        # Set all torques to zero
        self.data.ctrl[:] = 0.0
        
        # Log emergency
        print("EMERGENCY STOP ACTIVATED!")
```

### Gradual Deployment

**Start Safe:**

```python
class GradualDeployment:
    """
    Gradually increase policy authority during deployment.
    """
    def __init__(self):
        self.deployment_step = 0
        self.max_steps = 1000
    
    def get_action_scale(self):
        """
        Gradually increase action scale.
        """
        progress = min(self.deployment_step / self.max_steps, 1.0)
        
        # Start at 10% authority, increase to 100%
        scale = 0.1 + 0.9 * progress
        
        return scale
    
    def apply_action(self, action):
        """
        Apply scaled action.
        """
        scale = self.get_action_scale()
        scaled_action = action * scale
        
        # Apply with safety limits
        safe_action = self.enforce_limits(scaled_action)
        
        self.deployment_step += 1
        return safe_action
```

---

## ✅ Deployment Checklist

### Pre-Deployment

- [ ] **Policy tested in simulation** with domain randomization
- [ ] **Safety limits implemented** and tested
- [ ] **Emergency stop** functional
- [ ] **Actuator limits** enforced
- [ ] **Sensor calibration** verified
- [ ] **Hardware inspection** completed

### Deployment Steps

1. **Initial Testing**
   - Deploy with limited authority (10-20%)
   - Test in controlled environment
   - Monitor safety metrics

2. **Gradual Increase**
   - Increase authority gradually
   - Test at each level
   - Monitor for issues

3. **Full Deployment**
   - Deploy at full authority
   - Continuous monitoring
   - Log all data

### Monitoring

**Key Metrics:**
- Tilt angles (should stay within limits)
- Joint velocities (should not exceed limits)
- Torque usage (should not saturate)
- Policy confidence (uncertainty estimates)

**Logging:**
```python
def log_deployment_data(self, obs, action, reward, info):
    """
    Log data during deployment.
    """
    log_data = {
        'timestamp': time.time(),
        'tilt_angle': self.compute_tilt_angle(obs),
        'velocities': obs['angular_vel'],
        'actions': action,
        'reward': reward,
        'safety_metrics': self.compute_safety_metrics(obs)
    }
    
    # Save to file
    self.deployment_log.append(log_data)
```

---

## 🤖 Real-World Example: Ballbot Transfer

### Ballbot-Specific Considerations

**1. Anisotropic Friction:**
- Real omniwheels have directional friction
- MuJoCo patch enables this
- Must match real friction coefficients

**2. IMU Calibration:**
- Real IMU has bias and drift
- Must calibrate before deployment
- Add noise during training

**3. Camera Calibration:**
- Real cameras need calibration
- Add noise and distortion
- Model realistic frame rates

**4. Terrain Adaptation:**
- Real terrain varies
- Train on diverse terrain
- Test on real surfaces

### Implementation Example

```python
class RealisticBallbotEnv(BBotSimulation):
    """
    Ballbot environment with realistic noise for sim-to-real.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add noise models
        self.imu_noise = NoisyIMU()
        self.camera_noise = NoisyCamera()
        self.actuator_delay = DelayedActuator(delay_steps=2)
        self.actuator_limits = LimitedActuator()
        
        # Domain randomization
        self.randomize_physics = True
    
    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        
        # Randomize physics
        if self.randomize_physics:
            self.randomize_physics_params()
        
        return obs, info
    
    def _get_obs(self):
        # Get true observations
        true_obs = super()._get_obs()
        
        # Add sensor noise
        true_obs['angular_vel'] = self.imu_noise.add_gyro_noise(
            true_obs['angular_vel']
        )
        
        if 'rgbd_0' in true_obs:
            true_obs['rgbd_0'] = self.camera_noise.add_image_noise(
                true_obs['rgbd_0']
            )
            true_obs['rgbd_1'] = self.camera_noise.add_image_noise(
                true_obs['rgbd_1']
            )
        
        return true_obs
    
    def step(self, action):
        # Apply actuator delay and limits
        delayed_action = self.actuator_delay.apply_action(action)
        limited_action = self.actuator_limits.apply_limits(
            delayed_action, 
            self.data.qvel[motor_ids],
            self.model.opt.timestep
        )
        
        # Step with limited action
        return super().step(limited_action)
```

---

## 🔬 Advanced Techniques

### System Identification

**Identify Real Robot Parameters:**

```python
def identify_robot_parameters(real_robot_data):
    """
    Identify real robot parameters from data.
    """
    # Collect data from real robot
    # - Apply known torques
    # - Measure responses
    # - Fit dynamics model
    
    # Estimate mass, inertia, friction
    estimated_mass = fit_mass(real_robot_data)
    estimated_inertia = fit_inertia(real_robot_data)
    estimated_friction = fit_friction(real_robot_data)
    
    return {
        'mass': estimated_mass,
        'inertia': estimated_inertia,
        'friction': estimated_friction
    }
```

### Fine-Tuning on Real Data

**Adapt Policy to Real Robot:**

```python
def fine_tune_on_real_data(policy, real_robot_data):
    """
    Fine-tune policy using real robot data.
    """
    # Use real data as demonstrations
    # Fine-tune with small learning rate
    # Preserve sim-learned behavior
    
    fine_tuned_policy = copy.deepcopy(policy)
    
    for episode in real_robot_data:
        # Update policy on real data
        fine_tuned_policy.update(episode)
    
    return fine_tuned_policy
```

### Adversarial Domain Adaptation

**Learn Domain-Invariant Features:**

```python
class DomainAdversarialTraining:
    """
    Train policy to be domain-invariant.
    """
    def __init__(self):
        self.domain_classifier = DomainClassifier()
        self.feature_extractor = FeatureExtractor()
    
    def train(self, sim_data, real_data):
        """
        Train with domain adversarial loss.
        """
        # Extract features
        sim_features = self.feature_extractor(sim_data)
        real_features = self.feature_extractor(real_data)
        
        # Domain classification loss (adversarial)
        domain_loss = self.domain_classifier.loss(sim_features, real_features)
        
        # Policy loss (domain-invariant)
        policy_loss = self.policy.loss(sim_data)
        
        # Total loss
        total_loss = policy_loss - domain_loss  # Adversarial
        
        return total_loss
```

---

## ✅ Best Practices

### 1. Start with Domain Randomization

> "Domain randomization is the most effective sim-to-real technique."  
> — *Common wisdom in robotics RL*

- Randomize physics, visuals, terrain
- Train on diverse conditions
- Test robustness

### 2. Model Realistic Sensors

- Add noise to all sensors
- Model delays and quantization
- Match real sensor characteristics

### 3. Enforce Safety Limits

- Joint limits
- Torque limits
- Velocity limits
- Emergency stop

### 4. Gradual Deployment

- Start with limited authority
- Increase gradually
- Monitor continuously

### 5. Collect Real Data

- Log all deployment data
- Use for fine-tuning
- Improve future policies

---

## 📊 Summary

### Key Takeaways

1. **Reality gap is inevitable** - But can be minimized
2. **Domain randomization is key** - Most effective technique
3. **Model realistic sensors** - Add noise and delays
4. **Enforce safety limits** - Protect hardware
5. **Deploy gradually** - Start safe, increase authority

### Sim-to-Real Checklist

- [ ] Domain randomization implemented
- [ ] Sensor noise modeled
- [ ] Actuator dynamics included
- [ ] Safety limits enforced
- [ ] Emergency stop functional
- [ ] Gradual deployment plan
- [ ] Monitoring system ready
- [ ] Real robot tested

---

## 📚 Further Reading

### Papers

- **Tobin et al. (2017)** - "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **Peng et al. (2018)** - "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- **James et al. (2019)** - "Sim-to-Real via Sim-to-Sim: Data-Efficient Robotic Grasping with Deep Reinforcement Learning"
- **Ramos et al. (2019)** - "From Pixels to Torques: Policy Learning with Deep Dynamical Models"

### Tutorials

- [Domain Randomization Techniques](#domain-randomization)
- [Sensor Noise Modeling](#sensor-noise-modeling)
- [Actuator Dynamics](#actuator-dynamics)

---

*Last Updated: 2025*

**Safe Deployments! 🛡️**

