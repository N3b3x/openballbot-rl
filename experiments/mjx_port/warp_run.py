"""Run the same test protocol on MuJoCo-Warp. Usage: python warp_run.py <xml> <tag> [nofix]"""
import sys
import numpy as np
from warp_sim import *   # noqa
import cpu_ref

if __name__ == "__main__":
    lowprio()
    xml, tag = sys.argv[1], sys.argv[2]
    fix = "nofix" not in sys.argv
    m = load_cpu_model(xml)
    specs = test_specs(total_mass(m))
    out = cpu_ref.run_all(lambda N: WarpBallbot(m, N, fix_frames=fix), m, specs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"traj_warp_{tag}.npz", **cpu_ref.flatten(out))
