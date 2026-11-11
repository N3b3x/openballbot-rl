#!/bin/bash
# Ablation 1: asymmetric critic and history stack on tiles + surfaces + DR (100M steps, seed 0)
PY=../mjx_port/.venv/Scripts/python.exe
O=../../outputs/study/warp_ppo
run() { n=$1; shift; $PY train.py --name $n --nworld 4096 --steps 100e6 --tiles --surfaces --seed 0 "$@" > $O/$n.out 2>&1; }
run wpab_base_s0
run wpab_asym_s0 --asym
run wpab_hist8_s0 --hist 8
run wpab_both_s0 --hist 8 --asym
touch $O/ABL1DONE
