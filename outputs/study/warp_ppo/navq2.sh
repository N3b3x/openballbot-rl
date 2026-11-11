cd /c/Users/Owner/Documents/GitHub/openballbot-rl/experiments/warp_ppo
O=../../outputs/study/warp_ppo
PY=../mjx_port/.venv/Scripts/python.exe
until [ -f $O/g23_nav_rs025_s1/battery.json ]; do sleep 60; done
for r in g18_ret_rs025_s1 g23_nav_rs025_s0 g23_nav_rs025_s1; do for ps in 0 1; do pl=""; sfx=""; [ $ps = 1 ] && pl="--pull" && sfx="_pp"; TRAIN_MOD=train_dev PUSH_SCALE=$ps $PY eval_nav.py --run $r --source sensor --res 18x14 --fov 55,42 --planner smooth --carrot 0.3 --w-max 2.0 --aim target $pl --out navS_tof18aim$sfx 2>&1 | grep -E "^navS|Traceback"; done; done
