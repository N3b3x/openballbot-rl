#!/usr/bin/env bash
# Architecture tournament over a pool of 12 ballbot designs (one policy for the whole family):
#   hyb = learned LQR/PPO gate, res = residual-on-LQR, ppo = pure PPO with the morphology context.
# Then the remaining single-design runs of the morphology study.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1
R=outputs/study/runs
until [ -f $R/dz_nom_s0/summary.json ] && [ -f $R/dz_s0.5_s0/summary.json ] && [ -f $R/dz_s1.5_s0/summary.json ] && [ -f $R/dz_zhigh_s0/summary.json ] && [ -f $R/dz_kb4_s0/summary.json ] && [ -f $R/dz_tau3_s0/summary.json ]; do sleep 60; done
C="--variant proprio --total-timesteps ${STEPS:-10e6} --gamma 0.995 --n-envs 8 --n-steps 448 --batch-size 1792 --seed 0 --log-std-init -1.5"
python scripts/study/train_ppo.py --name pool_hyb_s0 --env-config configs/env/pool_hyb.yaml $C --gate-bias -0.8 > outputs/study/pool_hyb.log 2>&1 &
python scripts/study/train_ppo.py --name pool_res_s0 --env-config configs/env/pool_res.yaml $C --zero-init-actor > outputs/study/pool_res.log 2>&1 &
python scripts/study/train_ppo.py --name pool_ppo_s0 --env-config configs/env/pool_ppo.yaml $C > outputs/study/pool_ppo.log 2>&1 &
wait
echo pool done
C2="--variant proprio --total-timesteps 5e6 --gamma 0.995 --n-envs 4 --n-steps 896 --batch-size 1792 --seed 0 --log-std-init -1.5"
for t in zlow kb025 bal025 bal4 tau6 J02 J0005; do
  python scripts/study/train_ppo.py --name dz_${t}_s0 --env-config configs/env/dz_$t.yaml $C2 > outputs/study/dz_$t.log 2>&1 &
done
wait
echo batch2 done
