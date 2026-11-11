cd /c/Users/Owner/Documents/GitHub/openballbot-rl
export PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1
for j in climb2cm_robs_c climb3cm_robs_c tour_flat_robs_c push_robs_b perlin8_tour_res_c perlin12_tour_res_c perlin8_nav_robs_c square_lqr square_robs_b perlin8_tour_lqr; do
  python scripts/study/render_cinematic.py --job $j --dump 6,12 > outputs/study/cine/_logs/$j.log 2>&1
done
python scripts/study/render_cinematic.py --compare perlin8_tour_lqr perlin8_tour_res_c --dump 10,20 > outputs/study/cine/_logs/cmp1.log 2>&1
python scripts/study/render_cinematic.py --compare square_lqr square_robs_b --dump 10,20 > outputs/study/cine/_logs/cmp2.log 2>&1
echo ALLDONE > outputs/study/cine/_logs/done.txt
