cd /c/Users/Owner/Documents/GitHub/openballbot-rl
PY=experiments/mjx_port/.venv/Scripts/python.exe; R=experiments/warp_ppo/render_policy.py
# shot 3: pure learning stalls, residual charges (4.5 cm step, 20 s)
$PY $R --run g16_pure_ret100_s1 --kind step_up --severity 0.045 --task vel --speed 0.3 --design nom --out c8_step45_pure --seconds 20 2>&1 | tail -1
$PY $R --run g18_ret_rs025_s1 --kind step_up --severity 0.045 --task vel --speed 0.3 --design nom --out c8_step45_residual --seconds 20 2>&1 | tail -1
# shot 8: the scale family at each ball's traction limit and just above (scale policy)
$PY $R --run g24_scale_rs025_s0 --kind step_up --severity 0.0195 --task vel --speed 0.3 --design s0.75 --out c8_scale_s075_atlimit --seconds 12 2>&1 | tail -1
$PY $R --run g24_scale_rs025_s0 --kind step_up --severity 0.039 --task vel --speed 0.3 --design s1.5 --out c8_scale_s15_atlimit --seconds 12 2>&1 | tail -1
$PY $R --run g24_scale_rs025_s0 --kind step_up --severity 0.052 --task vel --speed 0.3 --design s2.0 --out c8_scale_s2_atlimit --seconds 12 2>&1 | tail -1
# shots 1 and 2: the derail in slow motion (stride 1 = half speed), same world, arrestors off and on
$PY $R --run g12_dr175_s0 --kind flat --task goal --design nom --out c8_derail_slow_nocage --seconds 6 --stride 1 2>&1 | tail -1
RETAINER=0,0,0,100 $PY $R --run g12_dr175_s0 --kind flat --task goal --design nom --out c8_derail_slow_cage --seconds 6 --stride 1 2>&1 | tail -1
# shot 6: smooth follower with the aimed ToF sensor
cd experiments/warp_ppo
TRAIN_MOD=train_dev PUSH_SCALE=0 ../mjx_port/.venv/Scripts/python.exe eval_nav.py --run g18_ret_rs025_s1 --source sensor --res 18x14 --fov 55,42 --planner smooth --carrot 0.3 --w-max 2.0 --aim target --n 2 --record 0 --out rec_tof18_smooth 2>&1 | grep -E "^rec_|Traceback" | cut -c1-120
../mjx_port/.venv/Scripts/python.exe render_nav.py --run g18_ret_rs025_s1 --traj rec_tof18_smooth --out c8_nav_tof18_smooth --title "Smooth follower, sensor aimed at the plan" --sub "ToF 18x14 on the head, aimed at the planner target; carrot 0.3 m, 2 rad/s; same controller" 2>&1 | tail -1
echo ALLDONE
