#!/bin/bash
PY=../mjx_port/.venv/Scripts/python.exe
O=../../outputs/study/warp_ppo
$PY train.py --name wpgen_none_s0 --nworld 4096 --steps 100e6 --tiles --sensor none --seed 0 > $O/wpgen_none_s0.out 2>&1
touch $O/GENDONE
