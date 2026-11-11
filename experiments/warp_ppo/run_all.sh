#!/bin/bash
# Train + evaluate on both engines. Usage: bash run_all.sh   (from repo root)
PY=experiments/mjx_port/.venv/Scripts/python.exe
OUT=outputs/study/warp_ppo
for job in "nom 0" "nom 1" "nom 2" "tau3 0" "tau6 0" "s1.5 0" "s0.5 0" "zhigh 0" "zlow 0"; do
  set -- $job; d=$1; s=$2; name=wpdr_${d}_s${s}
  $PY experiments/warp_ppo/train.py --name $name --design $d --seed $s --nworld 4096 --steps 60e6 > $OUT/$name.out 2>&1
  $PY experiments/warp_ppo/eval_policy.py --engine warp --run $name --design $d > $OUT/$name.warp.out 2>&1
  python experiments/warp_ppo/eval_policy.py --engine cpu --run $name --design $d > $OUT/$name.cpu.out 2>&1 &
done
wait
echo ALLDONE > $OUT/ALLDONE
