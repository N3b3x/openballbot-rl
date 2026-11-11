cd /c/Users/Owner/Documents/GitHub/openballbot-rl/experiments/warp_ppo
PY=../mjx_port/.venv/Scripts/python.exe
R=g26_ret_rs025_mu01_s1
for k in 0 3 7 12; do TRAIN_MOD=train_dev PUSH_SCALE=0 $PY eval_nav.py --run $R --source bump --v-max 0.15 --steps 4000 --n 2 --record $k --out rec_bump_$k 2>&1 | grep -E "^rec_" | cut -c1-200; done
echo RECDONE
