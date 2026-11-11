cd /c/Users/Owner/Documents/GitHub/openballbot-rl
PY=experiments/mjx_port/.venv/Scripts/python.exe; R=experiments/warp_ppo/render_policy.py
G=g26_ret_rs025_mu01_s1
# corrected contact model: the heavy-ball design that used to "fall" standing (solver blow-up) now stands 20 s
$PY $R --run $G --kind flat --task vel --speed 0.0 --design kb4 --out c9_stand_kb4_fixed --seconds 20 2>&1 | tail -1
# a crossable step for the 9 cm ball (below the traction limit) and one above it
$PY $R --run $G --kind step_up --severity 0.018 --task vel --speed 0.3 --design nom --out c9_step18_cross --seconds 12 2>&1 | tail -1
$PY $R --run $G --kind step_up --severity 0.045 --task vel --speed 0.3 --design nom --out c9_step45_limit --seconds 12 2>&1 | tail -1
# routing around a block with the ToF head sensor, recommended checkpoint
cd experiments/warp_ppo
TRAIN_MOD=train_dev PUSH_SCALE=0 ../mjx_port/.venv/Scripts/python.exe eval_nav.py --run $G --source sensor --res 18x14 --fov 55,42 --planner smooth --carrot 0.3 --w-max 2.0 --aim target --n 2 --record 0 --out rec_tof18_smooth 2>&1 | grep -E "^rec_|Traceback" | cut -c1-120
../mjx_port/.venv/Scripts/python.exe render_nav.py --run $G --traj rec_tof18_smooth --out c9_nav_tof18_g26 --title "Routing around a block with a ToF sensor" --sub "Recommended controller, corrected contact model; ToF 18x14 on the head; smooth follower" 2>&1 | tail -1
echo ALLDONE
