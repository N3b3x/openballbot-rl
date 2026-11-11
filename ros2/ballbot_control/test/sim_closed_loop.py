"""Closed-loop software-in-the-loop check without ROS and without a GPU: CPU MuJoCo (the repo's ContactFrameStepper for the omni-wheel
contacts) plays the robot, the core plays the on-board computer through ControlLoop exactly as the controller node does:
simulated IMU (the model's gyro and accelerometer at imu_site) and wheel-encoder rates in, wheel torques out, 100 Hz,
the estimator in the loop (the controller never sees the simulator state).

Caveats: CPU MuJoCo is not the MuJoCo-Warp simulator the policy was trained in (different contact solver details, no arrestors),
there is no sensor noise, latency or actuator model; this checks the wiring, frames and signs end to end, not hardware readiness.

  python sim_closed_loop.py [--seconds 12] [--speed 0.3] [--lqr-only]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ballbot_control.core import ACTIVE, load_bundle  # noqa: E402
from ballbot_control.core.math3d import tilt_deg  # noqa: E402
from ballbot_control.core.runtime import ControlLoop, LoopConfig  # noqa: E402

REPO = HERE.parents[2]
RUN_DEFAULT = "g39_goalcentre_s0"      # recommended; g34_mref_rsad_db_s0 also exported


def bundle_file(run=RUN_DEFAULT):
    return HERE.parent / "config" / "bundle" / run / "bundle.json"


def make_plant(pair_mu1=0.01, tau=30.0):
    import mujoco
    spec = importlib.util.spec_from_file_location("cf", REPO / "ballbot_gym" / "utils" / "contact_frame.py")
    cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)
    m = mujoco.MjModel.from_xml_path(str(REPO / "ballbot_gym" / "models" / "variants" / "dz_nom.xml"))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * tau
    sel = (m.pair_friction[:, 0] < 0.01) & (m.pair_friction[:, 1] > 0.1)
    m.pair_friction[sel, 0] = pair_mu1          # wheel-ball free-rolling friction as in the g34 run (args.json pair_mu1)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] += 0.01; d.qpos[12] += 0.01        # common.reset_state
    mujoco.mj_forward(m, d)
    return m, d, cf.ContactFrameStepper(m)


def run(seconds=12.0, speed=0.3, residual=True, drive=(1.0, 6.0), verbose=False, run_name=RUN_DEFAULT):
    import mujoco
    b = load_bundle(bundle_file(run_name))
    m, d, stepper = make_plant()
    loop = ControlLoop(b, LoopConfig(residual=residual, cmd_vel_frame="odom", max_speed=0.6))
    gyro_adr = m.sensor_adr[m.sensor("imu_gyro").id]; acc_adr = m.sensor_adr[m.sensor("imu_accel").id]
    loop.enable()
    log = dict(t=[], tilt=[], mode=[], x=[], y=[], est_x=[], est_y=[])
    for k in range(int(seconds * 100)):
        now = k * 0.01
        if drive[0] <= now < drive[1]:
            loop.on_cmd_vel(now, speed, 0.0, 0.0)          # world-frame command, refreshed at 100 Hz like a Nav2 controller would at its rate
        gyro, acc, wr = d.sensordata[gyro_adr:gyro_adr + 3].copy(), d.sensordata[acc_adr:acc_adr + 3].copy(), d.qvel[6:9].copy()
        r = loop.tick(now, gyro, acc, wr, now)
        d.ctrl[:] = r.torque
        for _ in range(5):
            stepper.step(d)
        mujoco.mj_forward(m, d)                           # refresh sensordata for the next tick
        log["t"].append(now); log["tilt"].append(tilt_deg(d.qpos[3:7])); log["mode"].append(r.mode)
        c = d.geom_xpos[m.geom("the_ball").id]           # ball centre (the ball body origin wobbles as the ball rolls)
        log["x"].append(c[0]); log["y"].append(c[1])
        log["est_x"].append(r.odom.get("pos", [np.nan, np.nan])[0]); log["est_y"].append(r.odom.get("pos", [np.nan, np.nan])[1])
        if r.mode != ACTIVE and k > 0:
            break
    return {k: np.array(v) for k, v in log.items()}


def summary(L, drive=(1.0, 6.0), speed=0.3):
    t = L["t"]
    cruise = (t > drive[0] + 2.0) & (t < drive[1])
    stop = t > drive[1] + 3.0
    centre_x = L["x"]
    return dict(ticks=len(t), final_mode=str(L["mode"][-1]), max_tilt_deg=float(L["tilt"].max()),
                cruise_vx_mean=float(np.mean(np.diff(centre_x)[cruise[1:]]) / 0.01) if cruise.any() else None,
                distance_x=float(centre_x[-1] - centre_x[0]),
                est_minus_true_x=float(L["est_x"][-1] - L["x"][-1]),
                stopped_speed=float(np.hypot(np.diff(L["x"]), np.diff(L["y"]))[stop[1:]].mean() / 0.01) if stop.any() else None,
                final_lateral_y=float(L["y"][-1] - L["y"][0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0); ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--lqr-only", action="store_true"); ap.add_argument("--run", default=RUN_DEFAULT)
    a = ap.parse_args()
    L = run(a.seconds, a.speed, residual=not a.lqr_only, run_name=a.run)
    print(summary(L, speed=a.speed))
