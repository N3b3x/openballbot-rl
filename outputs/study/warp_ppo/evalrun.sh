# usage: bash evalrun.sh RUN  -- waits for the run to finish, then motion tests (with the layer), 20 s standing/driving, battery
cd /c/Users/Owner/Documents/GitHub/openballbot-rl/experiments/warp_ppo
O=../../outputs/study/warp_ppo; PY=../mjx_port/.venv/Scripts/python.exe; r=$1
until grep -q "^done " $O/$r.out; do sleep 60; done
export TRAIN_MOD=train_dev
for sc in goal curve zigzag; do $PY eval_motion.py --run $r --scenario $sc --mode ref --out motion_${sc}_ref 2>&1 | grep "^{" | cut -c1-600; done
for sp in 0.0 0.3; do TILES_FLAT=1 EVAL_NOEXIT=1 PUSH_SCALE=0 $PY eval_tiles2.py --run $r --task vel --speed $sp --n 32 --dr --surfaces --steps 2000 2>&1 | grep "^{" | head -3; done
(cd ../..; experiments/mjx_port/.venv/Scripts/python.exe experiments/warp_ppo/eval_battery.py --run $r --dr-scales 1.0 1.75 2.5)
echo "$r EVAL_DONE"
