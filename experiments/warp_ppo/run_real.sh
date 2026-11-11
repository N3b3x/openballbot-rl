#!/bin/bash
PY=experiments/mjx_port/.venv/Scripts/python.exe; OUT=outputs/study/warp_ppo
for s in 0 1; do name=wpreal_nom_s$s
  $PY experiments/warp_ppo/train.py --name $name --design nom --cfg nom_real --seed $s --nworld 4096 --steps 60e6 > $OUT/$name.out 2>&1
  $PY experiments/warp_ppo/eval_policy.py --engine warp --run $name --design nom --cfg nom_real > $OUT/$name.warp.out 2>&1
  python experiments/warp_ppo/eval_policy.py --engine cpu --run $name --design nom --cfg nom_real > $OUT/$name.cpu.out 2>&1 &
done
wait; echo done > $OUT/REALDONE
