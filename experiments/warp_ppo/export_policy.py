"""Export a balance checkpoint (LQR base + learned residual, train_dev.py) for deployment: actor model files plus a JSON controller bundle.

Writes into --out (default ros2/ballbot_control/config/bundle/<run>/):
  actor.onnx           actor with the observation normalisation and both clips inside the graph: raw 255-D actor vector -> action in [-1, 1].
                       Written with torch.onnx when the onnx package is installed, otherwise with the small dependency-free writer below
                       (plain MatMul / Add / Tanh / Sub / Div / Clip graph, opset 17).
  actor.ts             TorchScript of the same function.
  actor_weights.npz    layer weights, mean and std as arrays (the ROS core runs this with numpy; no torch on the robot).
  bundle.json          everything else a deployment needs: control rate, observation layout (names, order, scales, clips, history
                       length and ordering), design vectors, action scaling to wheel torques, residual authority, LQR gains per design,
                       motion-layer parameters, goal rule, estimator calibration, kinematic offsets, wheel axes.
Verification (printed and stored in bundle.json "verification"): exported actor vs the torch actor (train_dev.Net, as loaded by
eval_motion.py / eval_tiles2.py) on recorded simulator observations (record_parity.py), max abs difference per backend. Backends whose
runtime is missing here (e.g. onnxruntime in the study venv) are skipped; re-run with --verify-only under another interpreter.

  experiments/mjx_port/.venv/Scripts/python.exe export_policy.py --run g34_mref_rsad_db_s0
  python export_policy.py --run g34_mref_rsad_db_s0 --verify-only          # e.g. an interpreter that has onnxruntime
Simulation-derived; nothing here was checked on hardware.
"""
import argparse, datetime, json, os, struct, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "outputs" / "study" / "warp_ppo"
OBS_DIM, ACT_DIM = 25, 3          # train_dev.OBS_DIM / ACT_DIM (not imported: train_dev needs Warp)

# ------------------------------------------------------------------------------------------------------------ actor
class Actor(nn.Module):
    """Deployment actor: normalisation + MLP + clip. Input: raw actor vector (B, 255) = training vector columns 0:255."""
    def __init__(self, linears, mean, std, clip_obs=10.0):
        super().__init__()
        self.lin = nn.ModuleList(linears)
        self.register_buffer("mean", mean.clone()); self.register_buffer("std", std.clone())
        self.clip_obs = float(clip_obs)

    def forward(self, x):
        h = ((x - self.mean) / self.std).clamp(-self.clip_obs, self.clip_obs)
        for i, l in enumerate(self.lin):
            h = l(h)
            if i < len(self.lin) - 1:
                h = torch.tanh(h)
        return h.clamp(-1.0, 1.0)


def build_actor(ck, n_in):
    sd = {k[len("pi.body."):]: v.float().cpu() for k, v in ck["ac"].items() if k.startswith("pi.body.")}
    idx = sorted({int(k.split(".")[0]) for k in sd})
    linears = []
    for i in idx:
        W, b = sd[f"{i}.weight"], sd[f"{i}.bias"]
        l = nn.Linear(W.shape[1], W.shape[0]); l.weight.data.copy_(W); l.bias.data.copy_(b); linears.append(l)
    assert linears[0].in_features == n_in, (linears[0].in_features, n_in)
    mean = ck["mean"][:n_in].float().cpu(); std = torch.sqrt(ck["var"][:n_in].float().cpu() + 1e-8)
    return Actor(linears, mean, std).eval()


def reference_policy(ck):
    """The torch actor exactly as the evaluation scripts run it (train_dev.Net + normalisation + clip), when train_dev imports here."""
    try:
        import train_dev as T
    except Exception as e:      # needs Warp; fall back to the reconstruction (same layers, same order)
        return None, f"train_dev not importable here ({type(e).__name__}); reference = reconstructed actor"
    net = T.Net(ck.get("sdim", 0), 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0),
                width=ck.get("width", 256), nlayers=ck.get("nlayers", 2))
    net.load_state_dict({k[3:]: v.cpu() for k, v in ck["ac"].items() if k.startswith("pi.")})
    net.eval()
    mean, var = ck["mean"].cpu(), ck["var"].cpu()
    return (lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)), "train_dev.Net (as in eval_motion.py)"


# ------------------------------------------------------------------------------------------------------------ minimal ONNX writer
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(field, wire, payload):
    key = _varint((field << 3) | wire)
    if wire == 0:
        return key + _varint(payload)
    return key + _varint(len(payload)) + payload      # wire 2: length-delimited


def _s(field, text):
    return _f(field, 2, text.encode())


def _tensor(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return b"".join(_f(1, 0, int(d)) for d in arr.shape) + _f(2, 0, 1) + _s(8, name) + _f(9, 2, arr.tobytes())


def _value_info(name, dims):
    dim = b"".join(_f(1, 2, (_s(2, d) if isinstance(d, str) else _f(1, 0, d))) for d in dims)
    tensor_type = _f(1, 0, 1) + _f(2, 2, dim)
    return _s(1, name) + _f(2, 2, _f(1, 2, tensor_type))


def _node(op, ins, outs, name):
    return b"".join(_s(1, i) for i in ins) + b"".join(_s(2, o) for o in outs) + _s(3, name) + _s(4, op)


def write_onnx_minimal(actor, path, n_in):
    """ONNX (opset 17, IR 8) of Actor.forward without the onnx package. Graph: Sub, Div, Clip, then (MatMul, Add, Tanh) per layer."""
    inits, nodes = [], []
    inits.append(_tensor("mean", actor.mean.numpy()[None])); inits.append(_tensor("std", actor.std.numpy()[None]))
    inits.append(_tensor("clip_lo", np.array(-actor.clip_obs))); inits.append(_tensor("clip_hi", np.array(actor.clip_obs)))
    inits.append(_tensor("act_lo", np.array(-1.0))); inits.append(_tensor("act_hi", np.array(1.0)))
    nodes.append(_node("Sub", ["x", "mean"], ["xc"], "sub")); nodes.append(_node("Div", ["xc", "std"], ["xn"], "div"))
    nodes.append(_node("Clip", ["xn", "clip_lo", "clip_hi"], ["h0"], "clip_obs"))
    h = "h0"
    for i, l in enumerate(actor.lin):
        inits.append(_tensor(f"W{i}", l.weight.detach().numpy().T)); inits.append(_tensor(f"b{i}", l.bias.detach().numpy()))
        nodes.append(_node("MatMul", [h, f"W{i}"], [f"m{i}"], f"matmul{i}")); nodes.append(_node("Add", [f"m{i}", f"b{i}"], [f"z{i}"], f"add{i}"))
        if i < len(actor.lin) - 1:
            nodes.append(_node("Tanh", [f"z{i}"], [f"t{i}"], f"tanh{i}")); h = f"t{i}"
        else:
            h = f"z{i}"
    nodes.append(_node("Clip", [h, "act_lo", "act_hi"], ["action"], "clip_act"))
    graph = (b"".join(_f(1, 2, n) for n in nodes) + _s(2, "ballbot_actor") + b"".join(_f(5, 2, t) for t in inits)
             + _f(11, 2, _value_info("x", ["batch", n_in])) + _f(12, 2, _value_info("action", ["batch", ACT_DIM])))
    model = _f(1, 0, 8) + _s(2, "openballbot export_policy.py") + _f(7, 2, graph) + _f(8, 2, _f(2, 0, 17))
    Path(path).write_bytes(model)


def write_onnx(actor, path, n_in):
    try:
        import onnx  # noqa: F401
        torch.onnx.export(actor, torch.zeros(1, n_in), str(path), input_names=["x"], output_names=["action"],
                          dynamic_axes={"x": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
        return "torch.onnx"
    except ImportError:
        write_onnx_minimal(actor, path, n_in)
        return "minimal writer (onnx package not installed)"


# ------------------------------------------------------------------------------------------------------------ physics-side tables
def design_tables(names, args):
    """Per design: tau, f, design vector (train_dev formula, reference = nom), LQR K/W/r_ball (lqr_gpu.design_tables), kinematic offsets."""
    if args.get("retainer"):
        os.environ["RETAINER"] = args["retainer"]
    if args.get("pair_mu1") is not None:
        os.environ["PAIR_MU1"] = f"{args['pair_mu1']:g}"
    sys.path.insert(0, str(ROOT / "experiments" / "mjx_port")); sys.path.insert(0, str(HERE))
    import mujoco
    from common import Design, load_cpu_model, reset_state
    from lqr_gpu import design_tables as lqr_tables
    K, W, Rb = lqr_tables(names, r=args.get("lqr_r", 1.0), dt=0.01, dev="cpu")
    ref_d = Design("nom"); ref_m = load_cpu_model(design=ref_d)
    out = {}
    for i, n in enumerate(names):
        d = Design(n); m = load_cpu_model(design=d)
        bi, ki = m.body("base").id, m.body("ball").id
        dvec = [np.log(d.f), np.log(d.m_total / ref_d.m_total), np.log(d.tau / ref_d.tau), np.log(m.body_mass[ki] / ref_m.body_mass[ki]),
                np.log(m.dof_armature[6] / ref_m.dof_armature[6]), np.log(m.body_mass[bi] / ref_m.body_mass[bi])]
        q0, _ = reset_state(m)
        if args.get("retainer"):      # retainer runs start from the loaded seat (train_dev.GpuBallbot)
            from retainer import seat_offset
            q0[2] = q0[12] + seat_offset(m)
        dd = mujoco.MjData(m); dd.qpos[:] = q0; mujoco.mj_forward(m, dd)
        gb = m.geom("the_ball").id
        c = dd.geom_xpos[gb].copy()
        Rbase = dd.xmat[bi].reshape(3, 3)
        axes = [(Rbase.T @ dd.xaxis[m.joint(f"wheel_joint_{k}").id]).tolist() for k in range(3)]
        Wm = np.array(json.load(open(d.json))["params"]["W"])
        out[n] = dict(tau=d.tau, f=d.f, dvec=[float(v) for v in dvec], m_total=d.m_total,
                      lqr=dict(K=K[i].tolist(), W=W[i].tolist(), r_ball=float(Rb[i])),
                      kinematics=dict(ball_radius=float(m.geom_size[gb][0]),
                                      base_origin_from_ball_centre_body=(Rbase.T @ (dd.xpos[bi] - c)).tolist(),
                                      base_com_from_ball_centre_body=(Rbase.T @ (dd.subtree_com[bi] - c)).tolist(),
                                      ball_body_origin_from_centre_ball=(-m.geom_pos[gb]).tolist(),
                                      wheel_axes_body=axes, encoder_C_planar=np.linalg.inv(Wm).tolist()))
    return out


def observation_layout(hist_k, ddim):
    L = [("rotvec", 0, 3, "rad", "rotation vector of the base quaternion (w,x,y,z) world<-body: axis * 2 atan2(|v|, w); includes yaw; no wrapping", None),
         ("ang_vel", 3, 3, "rad/s", "base angular velocity, world axes (sim: cvel[base][0:3])", 8.0),
         ("lin_vel", 6, 3, "m/s", "velocity of the base-subtree centre of mass, world axes (sim: cvel[base][3:6])", 8.0),
         ("motor", 9, 3, "1", "(v_origin_y, v_origin_z, omega_body_x) / tau: training-env quirk reading qvel[1:4] of the base free joint", 2.0),
         ("last_action", 12, 3, "1", "previous policy residual action (clipped to [-1,1])", 1.0),
         ("cmd", 15, 3, "m/s, m/s, rad/s", "shaped command (vx, vy world, wz) from the motion layer", None),
         ("heading", 18, 2, "1", "sin(yaw), cos(yaw) of the base", None),
         ("vint", 20, 2, "m", "velocity-error integral: clip(vint + (cmd_xy - lin_vel_xy) dt, +-1)", 1.0),
         ("goal", 22, 3, "1", "clip((goal - tracked_ball_point) / 2, +-2) * goal_on, goal_on", 2.0)]
    obs = [dict(name=n, start=s, size=z, units=u, description=dsc, clip=c) for n, s, z, u, dsc, c in L]
    hd = hist_k * (OBS_DIM + ACT_DIM)
    actor = [dict(name="obs", start=0, size=OBS_DIM),
             dict(name="history", start=OBS_DIM, size=hd, layout=f"{hist_k} entries x (obs {OBS_DIM} + action {ACT_DIM}), entry 0 = previous tick (newest first)"),
             dict(name="design_vector", start=OBS_DIM + hd, size=ddim)]
    return obs, actor


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="g34_mref_rsad_db_s0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rollout", default=None, help="recorded observations (record_parity.py npz); default <run>/parity_rollout.npz")
    ap.add_argument("--verify-only", action="store_true", help="do not export, only verify the files in --out (adds results to bundle.json)")
    a = ap.parse_args()
    run_dir = RUNS / a.run
    out = Path(a.out) if a.out else ROOT / "ros2" / "ballbot_control" / "config" / "bundle" / a.run
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run_dir / "model.pt", map_location="cpu")
    args = json.load(open(run_dir / "args.json"))
    hist_k, ddim = int(ck.get("hist", 0)), int(ck.get("ddim", 0))
    n_in = OBS_DIM + int(ck.get("sdim", 0)) + int(ck.get("hdim", 0)) + ddim
    assert ck.get("sdim", 0) == 0, "sensor (depth/scan) policies are not supported by this exporter"
    actor = build_actor(ck, n_in)

    if not a.verify_only:
        np.savez(out / "actor_weights.npz", n_layers=len(actor.lin), mean=actor.mean.numpy(), std=actor.std.numpy(),
                 **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(actor.lin)}, **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(actor.lin)})
        torch.jit.trace(actor, torch.zeros(1, n_in)).save(str(out / "actor.ts"))
        onnx_how = write_onnx(actor, out / "actor.onnx", n_in)
        names = ck.get("designs") or ["nom"]
        tables = design_tables(names, args)
        odom = RUNS / "odom" / "odom_results_flat.json"
        C_cal = json.load(open(odom))["C"] if odom.exists() else None
        obs_layout, actor_layout = observation_layout(hist_k, ddim)
        rsa = [float(v) for v in args["rs_adapt"].split(",")] if args.get("rs_adapt") else None
        goal_centre = bool(args.get("goal_centre"))
        track = "ball_centre" if goal_centre else "ball_body_origin"
        bundle = dict(
            format="ballbot_controller_bundle", version=1,
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            source=dict(run=a.run, checkpoint=str((run_dir / "model.pt").relative_to(ROOT)), train_steps=int(ck.get("steps", 0)), args=args,
                        note="simulation-trained (MuJoCo-Warp); not validated on hardware"),
            control=dict(rate_hz=100.0, dt=0.01, physics_dt=0.002, action_repeat=5,
                         note="one policy/LQR evaluation per 10 ms; the wheel torque is held constant until the next tick"),
            policy=dict(files=dict(onnx="actor.onnx", torchscript="actor.ts", weights="actor_weights.npz"), onnx_writer=onnx_how,
                        input_dim=n_in, output_dim=ACT_DIM, hidden=[l.out_features for l in actor.lin[:-1]], activation="tanh",
                        normalisation="(x - mean) / std, std = sqrt(var + 1e-8), then clip to +-clip_obs (inside every model file)",
                        clip_obs=10.0, action_clip=1.0, training_vector_dim=int(ck["mean"].shape[0]),
                        privileged_columns_unused_by_actor=[n_in, int(ck["mean"].shape[0])]),
            observation=dict(obs_dim=OBS_DIM, act_dim=ACT_DIM, hist_k=hist_k, layout=obs_layout, actor_input=actor_layout,
                             vel_clip=8.0, motor_clip=2.0, vint_clip=1.0, goal_scale=2.0, goal_clip=2.0,
                             reset_observation="first tick after a reset: ang_vel, lin_vel, motor = 0 and vint not integrated (GpuBallbot.reset_obs)",
                             noise_in_training=dict(rotvec_sd=0.0025, ang_vel_sd=0.02, lin_vel_sd=0.01, scaled_by_dr_scale=args.get("dr_scale", 1.0))),
            action=dict(rs=float(args.get("lqr_base", 0.0)), rs_adapt=None if rsa is None else dict(
                            lo=rsa[0], hi=rsa[1], v_db=rsa[2] if len(rsa) > 2 else 0.0, dc_scale=0.3, near0_frac=0.1, tilt0_deg=4.0,
                            tilt_span_deg=6.0, decay=0.98, w_init=1.0, dt=0.01),
                        combine="applied = clip(lqr + rs * a, -1, 1)",
                        torque="wheel_torque[k] = -clip(applied[k] * tau, -tau, tau)  (N m on wheel_joint_k about its axis; MuJoCo actuator ctrl, gear 1)"),
            lqr=dict(q=[1e-6, 50.0, 1.0, 1.0], r=float(args.get("lqr_r", 1.0)), ki=1.0, k_yaw=5.0, a_max=float(args.get("lqr_amax", 0.5)), dt=0.01, int_clip=2.0,
                     how="per design: discrete LQR (scipy cont2discrete + solve_discrete_are, Q*dt, R*dt) on the planar model A,B of "
                         "ballbot_gym/models/variants/dz_<design>.json; see experiments/warp_ppo/lqr_gpu.py design_tables"),
            motion_ref=dict(enabled=bool(args.get("motion_ref")), dt=0.01, a_acc=0.4, a_brk=0.5, a_lat=0.35, w_max=1.5, d_stop=0.03, d_hold=0.06,
                            d_exit=0.20, kp=0.8, v_hold=0.06, d_dead=0.015, note="speeds and accelerations are at the nominal size and scale with f"),
            goal_rule=dict(gain=1.5, d_zero=0.05, note="goal mode: v_des = unit(goal - p) * min(gain d, cmd_vmax f), zero within d_zero; wz = 0"),
            command_limits=dict(cmd_vmax=1.2, cmd_wmax=1.0, note="training range: |v| <= cmd_vmax f, |wz| <= cmd_wmax"),
            designs=dict(nominal="nom", table=tables, dvec_reference="nom",
                         dvec_names=["log f", "log m_total/m_total_nom", "log tau/tau_nom", "log m_ball/m_ball_nom", "log J_wheel/J_wheel_nom", "log m_base/m_base_nom"]),
            estimator=dict(C=C_cal if C_cal is not None else tables["nom"]["kinematics"]["encoder_C_planar"],
                           C_source="least-squares calibration of the nominal design in simulation (outputs/study/warp_ppo/odom)" if C_cal is not None else "planar model W^-1 (uncalibrated)",
                           k_acc=0.5, still_w=0.05, still_g=0.03, bias_tau=2.0, q_acc=0.5, r_enc=0.02, track_point=track,
                           note="wheel_rate = hinge rates of wheel_joint_0..2 (rad/s), gyro body rad/s, acc = specific force body m/s^2"),
            goal_point=dict(goal_centre=goal_centre, track_point=track,
                            note=("goal rule, goal observation and motion-layer arrival / hold use the true ball centre (train_dev --goal-centre, S118)"
                                  if goal_centre else "goal rule, goal observation and motion-layer arrival / hold use the ball BODY origin "
                                  "(ball centre + R_ball @ ball_body_origin_from_centre_ball), as in all runs before S118"),
                            authority_velocity_point="ball_body_origin",
                            authority_note="the adaptive-authority dead band reads the ball BODY origin velocity (qvel[9:11]) in every run"),
            frames=dict(world="odom: z up, fixed at estimator start", body="base_link = MuJoCo 'base' body frame (z along the tower)"),
        )
        json.dump(bundle, open(out / "bundle.json", "w"), indent=1)
        print(f"exported {a.run} -> {out} (onnx: {onnx_how})")

    # ---- verification on recorded simulator observations
    bundle = json.load(open(out / "bundle.json"))
    rp = Path(a.rollout) if a.rollout else run_dir / "parity_rollout.npz"
    ver = bundle.get("verification", {})
    if not rp.exists():
        print("no recorded rollout at", rp, "- run record_parity.py first; verification skipped")
        return
    z = np.load(rp)
    X = np.concatenate([z["x"].reshape(-1, z["x"].shape[-1]), z["x_noisy"].reshape(-1, z["x_noisy"].shape[-1])], 0).astype(np.float32)
    ref_fn, ref_name = reference_policy(ck)
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        ref = (ref_fn(Xt) if ref_fn is not None else actor(Xt[:, :n_in])).numpy().astype(np.float64)
        res = dict(reference=ref_name, samples=int(X.shape[0]), rollout=str(rp.relative_to(ROOT)))
        res["reconstructed_torch"] = float(np.abs(actor(Xt[:, :n_in]).numpy() - ref).max())
        rec = np.concatenate([z["act"].reshape(-1, 3), z["act_noisy"].reshape(-1, 3)], 0)
        res["recorded_sim_actions_vs_reference"] = float(np.abs(rec - ref).max())
        ts = torch.jit.load(str(out / "actor.ts"))
        res["torchscript"] = float(np.abs(ts(Xt[:, :n_in]).numpy() - ref).max())
    W = np.load(out / "actor_weights.npz")
    h = np.clip((X[:, :n_in].astype(np.float64) - W["mean"]) / W["std"], -10, 10)
    for i in range(int(W["n_layers"])):
        h = h @ W[f"W{i}"].T.astype(np.float64) + W[f"b{i}"]
        if i < int(W["n_layers"]) - 1:
            h = np.tanh(h)
    res["numpy_weights_float64"] = float(np.abs(np.clip(h, -1, 1) - ref).max())
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(out / "actor.onnx"), providers=["CPUExecutionProvider"])
        y = s.run(None, {"x": X[:, :n_in]})[0]
        res["onnx_onnxruntime"] = float(np.abs(y - ref).max()); res["onnxruntime_version"] = ort.__version__
    except ImportError:
        res["onnx_onnxruntime"] = None
        print("onnxruntime not installed in this interpreter: ONNX check skipped (re-run with --verify-only elsewhere)")
    res.update(python=sys.version.split()[0], torch=torch.__version__)
    ver["vs_train_dev_net" if ref_fn is not None else "vs_reconstructed_actor"] = res      # one entry per reference, so runs under two interpreters both stay
    bundle["verification"] = ver
    json.dump(bundle, open(out / "bundle.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
