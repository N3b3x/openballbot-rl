cd /c/Users/Owner/Documents/GitHub/openballbot-rl/experiments/warp_ppo
until grep -q RECDONE ../../outputs/study/warp_ppo/render_c10.log; do sleep 20; done
../mjx_port/.venv/Scripts/python.exe render_nav.py --run g26_ret_rs025_mu01_s1 --traj rec_bump_3 --out c10_blind_escape --title "Blind robot: bump, remember, back off, re-plan" --sub "No terrain sensor; stall detection marks the obstacle, 1 s back-up, re-plan around it; recommended controller" 2>&1 | tail -1
echo RENDERDONE
