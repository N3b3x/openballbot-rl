"""Replay a recorded simulator rollout (experiments/warp_ppo/record_parity.py) through the pure-Python core and measure the mismatch.

For every world the core runs its own TrackingController from the episode start; per tick it is given only what a robot would have:
the measured state (RobotState built from the logged qpos, qvel and base cvel), the desired velocity and yaw rate (velocity episodes)
or the goal (goal episodes). Everything else (motion-layer state, goal rule, history, velocity integral, LQR reference and integral,
residual authority, the policy forward pass) is computed by the core and compared with the simulator's logged values.
Ticks after a world's first fall are excluded (the simulator kept stepping a fallen robot; nothing there is meaningful).

For goal_centre runs (g39 and later) the goal point fed to the core is the ball centre, as the simulator's GpuBallbot.ball_xy();
for older runs it is the ball body origin. The bundle's goal_point section decides.

  python parity_replay.py [RUN]                   # RUN defaults to the recommended run; full rollout if present, else the fixture
  python parity_replay.py RUN --make-fixture      # rebuild test/data/parity_<RUN>_fixture.npz from the full rollout
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ballbot_control.core import RobotState, TrackingController, load_bundle  # noqa: E402
from ballbot_control.core.motion_ref import goal_velocity  # noqa: E402

REPO = HERE.parents[2]
RUNS = ("g39_goalcentre_s0", "g34_mref_rsad_db_s0")      # recommended first
DEFAULT_RUN = RUNS[0]


def bundle_path(run=DEFAULT_RUN):
    return HERE.parent / "config" / "bundle" / run / "bundle.json"


def full_rollout_path(run=DEFAULT_RUN):
    return REPO / "outputs" / "study" / "warp_ppo" / run / "parity_rollout.npz"


def fixture_path(run=DEFAULT_RUN):
    return HERE / "data" / f"parity_{run}_fixture.npz"


DEFAULT_BUNDLE = bundle_path()

KEYS = ("cmd", "goal_rule", "obs", "actor_x", "action", "lqr", "rs", "applied", "torque", "obs_goal", "torque_goal")      # *_goal: goal-mode ticks only


def replay(z, bundle, worlds=None, omega_from_gyro=False):
    """Returns ({quantity: array of the per-tick max abs error}, ticks compared)."""
    T, N = z["qpos"].shape[:2]
    designs = [str(d) for d in z["designs"]]
    fell = z["fell"].astype(bool)
    err = {k: [] for k in KEYS}
    n_ticks = 0
    for w in (range(N) if worlds is None else worlds):
        ctrl = TrackingController(bundle, design=designs[int(z["didx"][w])], snap_heading_at_rest=False)
        assert float(z["m_s0"][0, w]) == 0.0 and not bool(z["m_hold0"][0, w])
        ctrl.reset(heading=float(z["m_th0"][0, w]))
        tau, n_in = float(z["tau_w"][w]), ctrl.obs.actor_dim
        off = bundle.design(ctrl.design_name)["kinematics"]["ball_body_origin_from_centre_ball"]
        for t in range(T):
            if t > 0 and fell[:t, w].any():
                break
            s = RobotState.from_sim(z["qpos"][t, w], z["qvel"][t, w], z["cvel"][t, w], goal_centre=bundle.goal_centre, ball_origin_from_centre=off)
            if omega_from_gyro:
                s.omega_world = None
            on = bool(z["goal_on"][t, w])
            goal = z["goal"][t, w].astype(np.float64) if on else None
            if on:
                vg = goal_velocity(goal, s.ball_pos, ctrl.f, ctrl.v_max / ctrl.f, ctrl.goal_gain, ctrl.goal_d_zero)
                err["goal_rule"].append(float(np.abs(vg - z["v_des"][t, w]).max()))
            r = ctrl.step(s, v_des=z["v_des"][t, w], wz_des=float(z["cmd"][t, w, 2]), goal=goal)
            e = dict(cmd=r.cmd - z["cmd"][t, w], obs=r.obs - z["obs_clean"][t, w], actor_x=r.x - z["x"][t, w, :n_in],
                     action=r.action - z["act"][t, w], lqr=r.lqr - z["lqr"][t, w], rs=np.array([r.rs - z["rs"][t, w]]),
                     applied=r.applied - z["sent"][t, w])
            tq = float(z["tq"][t, w])
            if np.all(np.abs(r.applied * tq) < 1.0):      # sim ctrl = -clip(applied tau tq, +-tau); the core has no DR torque scale
                e["torque"] = r.torque * tq - z["ctrl"][t, w]
            if on:
                e["obs_goal"] = e["obs"]
                if "torque" in e:
                    e["torque_goal"] = e["torque"]
            for k, v in e.items():
                err[k].append(float(np.abs(v).max()))
            n_ticks += 1
    return {k: np.array(v) for k, v in err.items()}, n_ticks


def load(run=DEFAULT_RUN, path=None):
    p = Path(path) if path else (full_rollout_path(run) if full_rollout_path(run).exists() else fixture_path(run))
    return dict(np.load(p, allow_pickle=False)), p


def make_fixture(run=DEFAULT_RUN, ticks=300):
    """Small test fixture: one world per design (the one that stayed up longest), the first `ticks` ticks, actor columns only, plus
    1000 samples of the noisy-phase observations and actions."""
    src, dst = full_rollout_path(run), fixture_path(run)
    z = dict(np.load(src))
    fell = z["fell"]; T, N = fell.shape
    first = np.where(fell.any(0), fell.argmax(0), T)
    pick = [max((w for w in range(N) if z["didx"][w] == d), key=lambda w: (first[w], -w)) for d in range(len(z["designs"]))]
    keys = ["qpos", "qvel", "cvel", "cmd", "obs_clean", "act", "lqr", "rs", "sent", "ctrl", "tq", "v_des", "goal", "goal_on",
            "m_s0", "m_th0", "m_hold0", "fell"]
    out = {k: z[k][:ticks][:, pick] for k in keys}
    out["x"] = z["x"][:ticks][:, pick, :255].astype(np.float32)
    out.update({k: z[k][pick] for k in ("didx", "tau_w", "f_w")}, designs=z["designs"], worlds=np.array(pick),
               goal_centre=z.get("goal_centre", np.array(False)))
    xn, an = z["x_noisy"].reshape(-1, z["x_noisy"].shape[-1]), z["act_noisy"].reshape(-1, 3)
    sel = np.random.default_rng(0).choice(len(xn), 1000, replace=False)
    out["x_noisy"], out["act_noisy"] = xn[sel][:, :255].astype(np.float32), an[sel]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def report(run=DEFAULT_RUN):
    z, p = load(run)
    b = load_bundle(bundle_path(run))
    err, n = replay(z, b)
    gm = np.mean(z["goal_on"])
    print(f"{run}: rollout {p.name}, {n} ticks compared ({z['qpos'].shape[1]} worlds, {len(set(map(str, z['designs'])))} designs, "
          f"goal-mode share {gm:.2f}, {len(err['obs_goal'])} goal-mode ticks, goal point {b.track_point})")
    print(f"  {'quantity':10s} {'max':>10s} {'p99':>10s} {'median':>10s}   |core - sim| per tick (max over components)")
    for k in KEYS:
        print(f"  {k:10s} {err[k].max():10.2e} {np.percentile(err[k], 99):10.2e} {np.median(err[k]):10.2e}")
    err2, _ = replay(z, b, omega_from_gyro=True)
    print("  deployment-style angular velocity (R @ gyro instead of the simulator's 2 ms older cvel), median / p99 / max:")
    for k in ("obs", "action", "applied", "torque"):
        print(f"  {k:10s} {np.median(err2[k]):10.2e} {np.percentile(err2[k], 99):10.2e} {err2[k].max():10.2e}")
    return err


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = args[0] if args else DEFAULT_RUN
    if "--make-fixture" in sys.argv:
        make_fixture(run); print("wrote", fixture_path(run)); sys.exit(0)
    report(run)
