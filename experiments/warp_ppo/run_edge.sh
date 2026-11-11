#!/bin/bash
PY=../mjx_port/.venv/Scripts/python.exe
O=../../outputs/study/warp_ppo
for s in none scan depth; do
  $PY train.py --name wpedge_${s}_s0 --nworld 4096 --steps 60e6 --terrain --sensor $s --edge-focus 0.7 --seed 0 > $O/wpedge_${s}_s0.out 2>&1
done
touch $O/EDGEDONE
