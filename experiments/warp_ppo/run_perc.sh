#!/bin/bash
# blind vs privileged height-scan vs VL53L9-style depth, same ridge, 60M steps each
PY=../mjx_port/.venv/Scripts/python.exe
O=../../outputs/study/warp_ppo
for s in none scan depth; do
  $PY train.py --name wpperc_${s}_s0 --nworld 4096 --steps 60e6 --terrain --sensor $s --seed 0 > $O/wpperc_${s}_s0.out 2>&1
done
touch $O/PERCDONE
