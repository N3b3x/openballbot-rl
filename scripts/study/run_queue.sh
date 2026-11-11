#!/bin/bash
# Sequential training queue (20 envs each is ~2x more efficient than running several jobs concurrently).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
T=1.0e6
run() { # name variant env-config seed total n_envs n_steps
  echo "[$(date +%H:%M:%S)] start $1" >> outputs/study/queue.log
  python scripts/study/train_ppo.py --name "$1" --variant "$2" --env-config "$3" --seed "$4" \
      --total-timesteps "$5" --gamma 0.999 --n-envs "$6" --n-steps "$7" > "outputs/study/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1" >> outputs/study/queue.log
}
E=configs/env/perlin_directional_train.yaml
run proprio_s0       proprio      $E 0 $T 20 256
run depth_frozen_s0  depth_frozen $E 0 $T 20 256
run proprio_s1       proprio      $E 1 $T 20 256
run depth_frozen_s1  depth_frozen $E 1 $T 20 256
# physics-fix ablation on flat ground (stock contact frames)
run flat_nofix       proprio configs/env/flat_directional_nofix.yaml 0 3e5 20 256
run flat_fix         proprio configs/env/flat_directional.yaml 0 3e5 20 256
# H3: waypoint task, closed-loop goal vs open-loop direction (proprio only)
run goal_relative    proprio configs/env/perlin_goal_relative.yaml 0 $T 20 256
run goal_direction   proprio configs/env/perlin_goal_direction.yaml 0 $T 20 256
echo "[$(date +%H:%M:%S)] QUEUE_DONE" >> outputs/study/queue.log
