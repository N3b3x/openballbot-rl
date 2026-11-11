import numpy as np
from warp_sim import *
import runner
m = load_cpu_model(design=Design("nom"))
sp = test_specs(4.021)["speed_0.4"]
be = WarpBallbot(m, 4)
print("--- balanced run begins")
r = runner.run_lqr(be, m, sp, design=Design("nom"))
print("--- balanced run ends; falls", r["fell"].sum())
sp = test_specs(4.021)["push_1.5"]
be = WarpBallbot(m, 16)
r = runner.run_lqr(be, m, sp, design=Design("nom"))
print("--- push run ends; falls", r["fell"].sum())
