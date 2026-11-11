#!/usr/bin/env bash
# Robustness round: domain randomisation + pushes, fine-tuned from the best tracking policies.
# Waits for the spin round so the CPU is not oversubscribed.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
R=outputs/study/runs
until [ -f $R/spin_a_s0/summary.json ] && [ -f $R/spin_b_s0/summary.json ] && [ -f $R/spin_c_s1/summary.json ]; do sleep 120; done
C="--variant proprio --total-timesteps 10e6 --gamma 0.995 --n-envs 7 --n-steps 512 --batch-size 1792 --seed 0"
python scripts/study/train_ppo.py --name rob_a_s0 --env-config configs/env/rob_a.yaml $C --init-from $R/trk_g_s0/final_model.zip > outputs/study/rob_a.log 2>&1 &
python scripts/study/train_ppo.py --name rob_b_s0 --env-config configs/env/rob_b.yaml $C --init-from $R/trk_g_s0/final_model.zip > outputs/study/rob_b.log 2>&1 &
python scripts/study/train_ppo.py --name rob_a_h_s0 --env-config configs/env/rob_a.yaml $C --init-from $R/trk_h_s0/final_model.zip > outputs/study/rob_ah.log 2>&1 &
wait
echo robust round done
