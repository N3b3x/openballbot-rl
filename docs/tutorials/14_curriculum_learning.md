# 📚 Curriculum Learning for Reinforcement Learning

*A comprehensive guide to designing and implementing curriculum learning for robotics RL*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is Curriculum Learning?](#what-is-curriculum-learning)
3. [Why Use Curriculum Learning?](#why-use-curriculum-learning)
4. [Curriculum Design Principles](#curriculum-design-principles)
5. [Difficulty Metrics](#difficulty-metrics)
6. [Automatic Curriculum Generation](#automatic-curriculum-generation)
7. [Implementation Examples](#implementation-examples)
8. [Real-World Example: Ballbot Curriculum](#real-world-example-ballbot-curriculum)
9. [Advanced Techniques](#advanced-techniques)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

Curriculum learning is a training strategy where tasks are presented in order of increasing difficulty. Instead of training on the full task distribution from the start, the agent first learns easier tasks, then gradually progresses to harder ones.

> "Start simple, then add complexity. This is how humans learn, and it works for machines too."  
> — *Bengio et al. (2009), "Curriculum Learning"*

**Key Concepts:**
- **Easier tasks first**: Build foundational skills
- **Gradual progression**: Increase difficulty over time
- **Automatic adjustment**: Adapt based on performance
- **Sample efficiency**: Learn faster with curriculum

**Key Questions This Tutorial Answers:**
- What is curriculum learning and why does it work?
- How do we design curricula for robotics?
- How do we measure task difficulty?
- How do we automatically adjust difficulty?
- How do we implement curriculum learning?

---

## 📖 What is Curriculum Learning?

### The Core Idea

**Traditional RL:**
```
Train on full task distribution from start
  ↓
Agent struggles with hard tasks
  ↓
Slow learning, poor sample efficiency
```

**Curriculum Learning:**
```
Train on easy tasks first
  ↓
Agent learns basic skills
  ↓
Gradually increase difficulty
  ↓
Faster learning, better performance
```

### Mathematical Formulation

**Curriculum:** A sequence of task distributions \(D_1, D_2, ..., D_T\) where:
- \(D_1\): Easiest tasks
- \(D_T\): Full task distribution
- Difficulty increases: \(D_1 \subset D_2 \subset ... \subset D_T\)

**Training:**
- Train on \(D_1\) until performance threshold
- Switch to \(D_2\), train until threshold
- Continue until \(D_T\)

### Types of Curricula

**1. Fixed Curriculum:**
- Predetermined difficulty progression
- Simple to implement
- May not adapt to agent

**2. Adaptive Curriculum:**
- Adjusts based on agent performance
- More complex
- Better sample efficiency

**3. Automatic Curriculum:**
- Learns curriculum automatically
- State-of-the-art
- Requires additional learning

---

## 🎓 Why Use Curriculum Learning?

### Benefits

**1. Faster Learning:**
- Agent learns basic skills first
- Builds on previous knowledge
- Reduces exploration needed

**2. Better Final Performance:**
- More stable learning
- Avoids local minima
- Better generalization

**3. Sample Efficiency:**
- Fewer samples needed
- Important for real robots
- Reduces training time

**4. Stability:**
- Less variance in learning
- More predictable training
- Easier to debug

### When to Use

**Good For:**
- Complex tasks with easy/hard variants
- Tasks with natural difficulty progression
- Limited sample budget
- Unstable training without curriculum

**Not Needed For:**
- Simple tasks
- Tasks without difficulty variation
- When random sampling works well

---

## 🎨 Curriculum Design Principles

### Principle 1: Start Simple

**Easy Tasks Should:**
- Require basic skills only
- Have clear success criteria
- Provide dense rewards
- Be solvable quickly

**Example for Ballbot:**
- Start with **flat terrain**
- Simple balancing task
- Clear reward signal
- Easy to succeed

### Principle 2: Gradual Progression

**Difficulty Should Increase:**
- Smoothly (not abruptly)
- Based on agent capability
- With clear milestones
- At appropriate pace

**Example Progression:**
1. Flat terrain, short episodes
2. Flat terrain, longer episodes
3. Easy terrain (smooth Perlin)
4. Medium terrain (moderate Perlin)
5. Hard terrain (rough Perlin)

### Principle 3: Measurable Difficulty

**Difficulty Metrics Should:**
- Be objective and measurable
- Correlate with task hardness
- Be computable efficiently
- Guide curriculum progression

**Examples:**
- Terrain roughness
- Episode length
- Initial state difficulty
- Goal distance

### Principle 4: Adaptive

**Curriculum Should:**
- Adjust to agent performance
- Speed up if agent learns fast
- Slow down if agent struggles
- Prevent getting stuck

---

## 📏 Difficulty Metrics

### Terrain Difficulty

**For Ballbot Terrain:**

```python
def compute_terrain_difficulty(terrain_data):
    """
    Compute terrain difficulty metric.
    """
    # Height variation (rougher = harder)
    height_variance = np.var(terrain_data)
    
    # Gradient magnitude (steeper = harder)
    gradients = np.gradient(terrain_data)
    gradient_magnitude = np.mean(np.linalg.norm(gradients, axis=0))
    
    # Combined difficulty
    difficulty = 0.5 * height_variance + 0.5 * gradient_magnitude
    
    return difficulty
```

**Difficulty Levels:**
- **Easy**: Flat terrain (difficulty ≈ 0)
- **Medium**: Smooth Perlin (difficulty ≈ 0.1-0.3)
- **Hard**: Rough Perlin (difficulty ≈ 0.3-0.5)

### Episode Difficulty

**Initial State Difficulty:**

```python
def compute_initial_difficulty(initial_state):
    """
    Compute difficulty based on initial state.
    """
    # Tilt angle (more tilted = harder)
    tilt_angle = compute_tilt_angle(initial_state)
    
    # Initial velocity (moving = harder)
    initial_vel = np.linalg.norm(initial_state['vel'][:2])
    
    # Combined difficulty
    difficulty = tilt_angle / np.pi + 0.1 * initial_vel
    
    return difficulty
```

**Goal Distance:**

```python
def compute_goal_difficulty(initial_pos, goal_pos):
    """
    Compute difficulty based on goal distance.
    """
    distance = np.linalg.norm(goal_pos - initial_pos)
    
    # Farther = harder
    difficulty = distance / max_distance
    
    return difficulty
```

### Performance-Based Difficulty

**Use Agent Performance:**

```python
def compute_performance_difficulty(episode_rewards, episode_lengths):
    """
    Compute difficulty based on agent performance.
    """
    # Low reward = task too hard
    # High reward = task too easy
    
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # If agent succeeds easily, increase difficulty
    if mean_reward > threshold_high:
        difficulty_multiplier = 1.1  # Increase difficulty
    
    # If agent fails, decrease difficulty
    elif mean_reward < threshold_low:
        difficulty_multiplier = 0.9  # Decrease difficulty
    
    else:
        difficulty_multiplier = 1.0  # Keep same
    
    return difficulty_multiplier
```

---

## 🤖 Automatic Curriculum Generation

### Self-Paced Learning

**Concept:** Agent chooses its own difficulty.

**Implementation:**
```python
class SelfPacedCurriculum:
    """
    Agent chooses difficulty based on performance.
    """
    def __init__(self):
        self.current_difficulty = 0.0  # Start easy
        self.performance_history = []
    
    def select_difficulty(self):
        """
        Select difficulty for next episode.
        """
        # If performing well, increase difficulty
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            if recent_performance > 0.8:  # 80% success
                self.current_difficulty = min(
                    self.current_difficulty + 0.1, 
                    1.0
                )
            elif recent_performance < 0.3:  # 30% success
                self.current_difficulty = max(
                    self.current_difficulty - 0.1,
                    0.0
                )
        
        return self.current_difficulty
    
    def update_performance(self, success):
        """
        Update performance history.
        """
        self.performance_history.append(1.0 if success else 0.0)
```

### Prioritized Experience Replay for Curricula

**Concept:** Sample harder tasks more often as agent improves.

```python
class PrioritizedCurriculum:
    """
    Prioritize harder tasks as agent improves.
    """
    def __init__(self):
        self.task_pool = []  # Pool of tasks with difficulties
        self.agent_performance = 0.5  # Current performance
    
    def sample_task(self):
        """
        Sample task based on current performance.
        """
        # Filter tasks: difficulty should match agent capability
        suitable_tasks = [
            task for task in self.task_pool
            if abs(task.difficulty - self.agent_performance) < 0.2
        ]
        
        if suitable_tasks:
            # Sample from suitable tasks
            return np.random.choice(suitable_tasks)
        else:
            # Fallback to random
            return np.random.choice(self.task_pool)
    
    def update_performance(self, performance):
        """
        Update agent performance estimate.
        """
        self.agent_performance = 0.9 * self.agent_performance + 0.1 * performance
```

### GoalGAN-Style Curriculum

**Concept:** Generate tasks at "boundary of competence" (not too easy, not too hard).

```python
class GoalGANCurriculum:
    """
    Generate tasks at boundary of agent competence.
    """
    def __init__(self):
        self.goal_generator = GoalGenerator()  # GAN for goal generation
        self.discriminator = Discriminator()  # Classifies easy/hard
    
    def generate_task(self, agent_policy):
        """
        Generate task at boundary of competence.
        """
        # Generate candidate goals
        candidate_goals = self.goal_generator.sample(100)
        
        # Evaluate difficulty with current policy
        difficulties = []
        for goal in candidate_goals:
            # Run policy on goal
            success_rate = evaluate_policy_on_goal(agent_policy, goal)
            difficulty = 1.0 - success_rate  # Low success = high difficulty
        
        # Select goal at boundary (50% success rate)
        target_difficulty = 0.5
        best_goal = min(
            candidate_goals,
            key=lambda g: abs(self.get_difficulty(g) - target_difficulty)
        )
        
        return best_goal
```

---

## 💻 Implementation Examples

### Fixed Curriculum

**Simple Progression:**

```python
class FixedCurriculum:
    """
    Fixed difficulty progression.
    """
    def __init__(self):
        self.stages = [
            {'difficulty': 0.0, 'min_performance': 0.0},   # Flat terrain
            {'difficulty': 0.2, 'min_performance': 0.5},  # Easy terrain
            {'difficulty': 0.4, 'min_performance': 0.7},  # Medium terrain
            {'difficulty': 0.6, 'min_performance': 0.8},  # Hard terrain
            {'difficulty': 1.0, 'min_performance': 0.9},  # Full difficulty
        ]
        self.current_stage = 0
    
    def get_difficulty(self):
        """
        Get current difficulty level.
        """
        return self.stages[self.current_stage]['difficulty']
    
    def update(self, performance):
        """
        Progress to next stage if performance threshold met.
        """
        if (self.current_stage < len(self.stages) - 1 and
            performance >= self.stages[self.current_stage]['min_performance']):
            self.current_stage += 1
            print(f"Progressed to stage {self.current_stage}")
```

### Adaptive Curriculum

**Performance-Based Adjustment:**

```python
class AdaptiveCurriculum:
    """
    Adaptively adjust difficulty based on performance.
    """
    def __init__(self, initial_difficulty=0.0):
        self.difficulty = initial_difficulty
        self.performance_window = []
        self.window_size = 50
    
    def update(self, episode_reward, episode_length):
        """
        Update curriculum based on performance.
        """
        # Compute success (heuristic)
        success = 1.0 if episode_reward > 50 and episode_length > 1000 else 0.0
        self.performance_window.append(success)
        
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
        
        # Adjust difficulty
        if len(self.performance_window) == self.window_size:
            success_rate = np.mean(self.performance_window)
            
            if success_rate > 0.8:  # Doing well
                self.difficulty = min(self.difficulty + 0.05, 1.0)
            elif success_rate < 0.3:  # Struggling
                self.difficulty = max(self.difficulty - 0.05, 0.0)
    
    def get_difficulty(self):
        return self.difficulty
```

### Integration with Training

**In Training Loop:**

```python
# Initialize curriculum
curriculum = AdaptiveCurriculum(initial_difficulty=0.0)

# Training loop
for iteration in range(num_iterations):
    # Get current difficulty
    difficulty = curriculum.get_difficulty()
    
    # Generate terrain with difficulty
    terrain = generate_terrain_with_difficulty(difficulty)
    
    # Train on this difficulty
    for episode in range(episodes_per_iteration):
        obs, _ = env.reset(terrain=terrain)
        # ... training ...
        
        # Update curriculum
        curriculum.update(episode_reward, episode_length)
```

---

## 🤖 Real-World Example: Ballbot Curriculum

### Ballbot-Specific Curriculum

**Difficulty Progression:**

```python
class BallbotCurriculum:
    """
    Curriculum learning for Ballbot navigation.
    """
    def __init__(self):
        self.stage = 0
        self.stages = [
            {
                'name': 'Flat Terrain',
                'terrain_type': 'flat',
                'max_ep_steps': 2000,
                'min_performance': 0.0
            },
            {
                'name': 'Easy Terrain',
                'terrain_type': 'perlin',
                'terrain_scale': 30.0,  # Smooth
                'max_ep_steps': 3000,
                'min_performance': 0.6
            },
            {
                'name': 'Medium Terrain',
                'terrain_type': 'perlin',
                'terrain_scale': 25.0,  # Moderate
                'max_ep_steps': 4000,
                'min_performance': 0.7
            },
            {
                'name': 'Hard Terrain',
                'terrain_type': 'perlin',
                'terrain_scale': 20.0,  # Rough
                'max_ep_steps': 4000,
                'min_performance': 0.8
            }
        ]
    
    def get_config(self):
        """
        Get current stage configuration.
        """
        return self.stages[self.stage]
    
    def should_progress(self, performance):
        """
        Check if should progress to next stage.
        """
        if (self.stage < len(self.stages) - 1 and
            performance >= self.stages[self.stage]['min_performance']):
            return True
        return False
    
    def progress(self):
        """
        Progress to next stage.
        """
        if self.stage < len(self.stages) - 1:
            self.stage += 1
            print(f"Curriculum: Progressed to {self.stages[self.stage]['name']}")
```

### Implementation in Environment

**Modify Environment:**

```python
class CurriculumBallbotEnv(BBotSimulation):
    """
    Ballbot environment with curriculum learning.
    """
    def __init__(self, curriculum=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum or BallbotCurriculum()
    
    def reset(self, seed=None):
        # Get current curriculum config
        config = self.curriculum.get_config()
        
        # Apply curriculum settings
        self.terrain_type = config['terrain_type']
        if 'terrain_scale' in config:
            self.terrain_scale = config['terrain_scale']
        self.max_ep_steps = config['max_ep_steps']
        
        # Reset environment
        return super().reset(seed=seed)
    
    def update_curriculum(self, performance):
        """
        Update curriculum based on performance.
        """
        if self.curriculum.should_progress(performance):
            self.curriculum.progress()
```

### Integration with Training

**In Training Script:**

```python
# Initialize curriculum
curriculum = BallbotCurriculum()

# Create environment with curriculum
env = CurriculumBallbotEnv(curriculum=curriculum)

# Training loop
for iteration in range(num_iterations):
    # Collect rollouts
    rollouts = collect_rollouts(env, policy, n_rollouts=10)
    
    # Compute performance
    mean_reward = np.mean([r.total_reward for r in rollouts])
    performance = mean_reward / 100.0  # Normalize
    
    # Update curriculum
    env.update_curriculum(performance)
    
    # Update policy
    policy.update(rollouts)
```

---

## 🔬 Advanced Techniques

### Automatic Difficulty Tuning

**Learn Optimal Difficulty:**

```python
class LearnedCurriculum:
    """
    Learn optimal curriculum automatically.
    """
    def __init__(self):
        self.difficulty_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output difficulty in [0, 1]
        )
    
    def predict_difficulty(self, state):
        """
        Predict optimal difficulty for current state.
        """
        return self.difficulty_network(state)
    
    def update(self, states, performances):
        """
        Update difficulty network.
        """
        # Learn to predict difficulty that gives good performance
        # (simplified - actual implementation more complex)
        difficulties = self.predict_difficulty(states)
        loss = self.compute_loss(difficulties, performances)
        loss.backward()
```

### Multi-Objective Curriculum

**Balance Multiple Skills:**

```python
class MultiObjectiveCurriculum:
    """
    Curriculum for multiple objectives.
    """
    def __init__(self):
        self.objectives = ['balance', 'navigation', 'efficiency']
        self.difficulties = {obj: 0.0 for obj in self.objectives}
    
    def get_task(self):
        """
        Generate task balancing all objectives.
        """
        # Sample objective to focus on
        focus_objective = np.random.choice(self.objectives)
        
        # Get difficulty for this objective
        difficulty = self.difficulties[focus_objective]
        
        # Generate task
        task = self.generate_task(focus_objective, difficulty)
        
        return task
    
    def update(self, objective, performance):
        """
        Update difficulty for specific objective.
        """
        if performance > 0.8:
            self.difficulties[objective] = min(
                self.difficulties[objective] + 0.1, 1.0
            )
```

---

## ✅ Best Practices

### 1. Start Simple

> "The best curriculum starts as simple as possible."  
> — *Common wisdom in curriculum learning*

- Begin with easiest possible tasks
- Ensure agent can succeed
- Build confidence

### 2. Measure Progress

- Track performance at each difficulty
- Use objective metrics
- Monitor curriculum progression

### 3. Adapt Gradually

- Don't increase difficulty too fast
- Wait for stable performance
- Allow time to consolidate learning

### 4. Balance Exploration

- Don't make curriculum too restrictive
- Allow some exploration of harder tasks
- Mix easy and hard tasks

### 5. Validate Effectiveness

- Compare with/without curriculum
- Measure sample efficiency
- Check final performance

---

## 📊 Summary

### Key Takeaways

1. **Curriculum learning improves sample efficiency** - Learn faster with easier tasks first
2. **Start simple, progress gradually** - Build skills incrementally
3. **Measure difficulty objectively** - Use metrics, not intuition
4. **Adapt to agent performance** - Adjust based on success
5. **Validate effectiveness** - Compare with baseline

### Curriculum Design Checklist

- [ ] Difficulty metrics defined
- [ ] Initial difficulty set (easy)
- [ ] Progression strategy defined
- [ ] Performance thresholds set
- [ ] Adaptation mechanism implemented
- [ ] Validation plan created

---

## 📚 Further Reading

### Papers

- **Bengio et al. (2009)** - "Curriculum Learning" - Original curriculum learning paper
- **Graves et al. (2017)** - "Automated Curriculum Learning for Neural Networks"
- **Florensa et al. (2018)** - "Automatic Goal Generation for Reinforcement Learning Agents"
- **Portelas et al. (2020)** - "Automatic Curriculum Learning for Deep RL: A Short Survey"

### Tutorials

- [Reward Design for Robotics](04_reward_design_for_robotics.md) - Reward shaping
- [Terrain Generation](15_terrain_generation.md) - Terrain difficulty
- [Complete Training Guide](13_complete_training_guide.md) - Training workflow

---

*Last Updated: 2025*

**Happy Learning! 📚**

