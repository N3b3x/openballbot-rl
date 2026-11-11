#!/bin/bash
# Book-climbing study: proprio vs side ToF vs privileged height scan (same PPO budget).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  echo "[$(date +%H:%M:%S)] start $1" >> outputs/study/queue_book.log
  python scripts/study/train_ppo.py --name "$1" --variant proprio --env-config "configs/env/$2.yaml" --seed 0 \
     --total-timesteps 2e6 --gamma 0.999 --n-envs 12 --n-steps 426 > "outputs/study/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1" >> outputs/study/queue_book.log
}
run book_proprio_s0 book_proprio
run book_tof_s0     book_tof
run book_scan_s0    book_scan
echo "[$(date +%H:%M:%S)] QUEUE_DONE" >> outputs/study/queue_book.log
