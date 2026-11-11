#!/bin/bash
# Stage B: random planar commands, initialised from the fixed-command policy (stage A).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
while [ ! -f outputs/study/runs/ctrl_fixed_s0/summary.json ]; do sleep 20; done
python scripts/study/train_ppo.py --name ctrl_trans2_s0 --variant proprio --env-config configs/env/ctrl_trans.yaml --seed 0 \
   --total-timesteps 2e6 --gamma 0.999 --n-envs 12 --n-steps 426 --init-from outputs/study/runs/ctrl_fixed_s0/final_model.zip \
   > outputs/study/ctrl_trans2_s0.log 2>&1
