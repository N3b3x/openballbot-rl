cd /c/Users/Owner/Documents/GitHub/openballbot-rl/experiments/warp_ppo
O=../../outputs/study/warp_ppo
PY=../mjx_port/.venv/Scripts/python.exe
navs(){ r=$1; for ps in 0 1; do pl=""; sfx=""; [ $ps = 1 ] && pl="--pull" && sfx="_pp"; TRAIN_MOD=train_dev PUSH_SCALE=$ps $PY eval_nav.py --run $r --source oracle --planner smooth --carrot 0.3 --w-max 2.0 $pl --out navS_oracle$sfx 2>&1 | grep -E "^navS"; TRAIN_MOD=train_dev PUSH_SCALE=$ps $PY eval_nav.py --run $r --source sensor --res 18x14 --fov 55,42 --planner smooth --carrot 0.3 --w-max 2.0 $pl --out navS_tof18$sfx 2>&1 | grep -E "^navS"; done; }
navs g18_ret_rs025_s1
for r in g23_nav_rs025_s0 g23_nav_rs025_s1; do until grep -q "^done " $O/$r.out; do sleep 60; done; navs $r; cd ../..; experiments/mjx_port/.venv/Scripts/python.exe experiments/warp_ppo/eval_battery.py --run $r --dr-scales 1.0 1.75 2.5; cd experiments/warp_ppo; done
