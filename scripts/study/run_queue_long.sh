#!/bin/bash
# Round 2: longer budget (3M decisions), harder terrain mix (A up to 0.20), end-to-end CNN variant.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
E=configs/env/perlin_directional_train_hard.yaml
run() {
  echo "[$(date +%H:%M:%S)] start $1" >> outputs/study/queue_long.log
  python scripts/study/train_ppo.py --name "$1" --variant "$2" --env-config "$E" --seed "$3" \
     --total-timesteps 3e6 --gamma 0.999 --n-envs 20 --n-steps 256 > "outputs/study/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1" >> outputs/study/queue_long.log
}
run long_proprio_s0   proprio      0
run long_depth_e2e_s0 depth_e2e    0
run long_depth_frozen_s0 depth_frozen 0
echo "[$(date +%H:%M:%S)] QUEUE_DONE" >> outputs/study/queue_long.log
