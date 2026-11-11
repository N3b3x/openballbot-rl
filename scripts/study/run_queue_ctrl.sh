#!/bin/bash
# Flat-ground velocity-command tracking (any direction + yaw rate) at three limit levels.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  echo "[$(date +%H:%M:%S)] start $1" >> outputs/study/queue_ctrl.log
  python scripts/study/train_ppo.py --name "$1" --variant proprio --env-config "configs/env/$2.yaml" --seed 0 \
     --total-timesteps 3e6 --gamma 0.99 --n-envs 12 --n-steps 426 > "outputs/study/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1" >> outputs/study/queue_ctrl.log
}
run ctrl_base_s0 ctrl_base
run ctrl_wide_s0 ctrl_wide
run ctrl_free_s0 ctrl_free
echo "[$(date +%H:%M:%S)] QUEUE_DONE" >> outputs/study/queue_ctrl.log
