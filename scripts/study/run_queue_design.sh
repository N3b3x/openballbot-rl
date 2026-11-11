#!/usr/bin/env bash
# Morphology study: from-scratch PPO (robust recipe: DR + pushes, Froude-scaled command range) per design.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1
C="--variant proprio --total-timesteps ${STEPS:-5e6} --gamma 0.995 --n-envs 4 --n-steps 896 --batch-size 1792 --seed 0 --log-std-init -1.5"
run() { python scripts/study/train_ppo.py --name dz_$1_s0 --env-config configs/env/dz_$1.yaml $C > outputs/study/dz_$1.log 2>&1; }
for batch in "$@"; do
  for t in $batch; do run $t & done
  wait
  echo "batch done: $batch"
done
