#!/bin/bash
# Book v2: 30 deg tilt limit, 1 m run-up, chamfered 2 cm lip, 0-3 cm books, 3M decisions. Teacher (scan) first.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  echo "[$(date +%H:%M:%S)] start $1" >> outputs/study/queue_book2.log
  python scripts/study/train_ppo.py --name "$1" --variant proprio --env-config "configs/env/$2.yaml" --seed 0 \
     --total-timesteps 3e6 --gamma 0.999 --n-envs 12 --n-steps 426 > "outputs/study/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1" >> outputs/study/queue_book2.log
}
run book2_scan_s0    book2_scan
run book2_tof_s0     book2_tof
run book2_proprio_s0 book2_proprio
echo "[$(date +%H:%M:%S)] QUEUE_DONE" >> outputs/study/queue_book2.log
