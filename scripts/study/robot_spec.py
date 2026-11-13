"""Robot spec -> simulated ballbot: build a design from real dimensions, check it against physical limits, and (optionally)
measure it in simulation with the recommended controller. Simulation only; there is no physical robot yet.

A spec is a small YAML file in physical units (see specs/nominal.yaml):

  name: my_robot               # design name; files are written as dz_<name> (xml, LQR json, env yaml)
  ball_radius_m: 0.09
  ball_mass_kg: 0.168
  body_mass_kg: 3.85           # everything that is not the ball (frame, wheels, motors, battery, electronics, payload)
  body_com_height_m: 0.131     # body centre of mass above the ball centre
  motor_torque_limit: 30.0     # per-wheel limit in the simulator's actuator units (the nominal robot uses 30)
  rotor_inertia: 0.005         # wheel + motor rotor inertia [kg m^2] as seen at the wheel
  wheel_damping: 0.1           # viscous wheel damping [N m s/rad]
  contact_zenith_deg: 51.0     # angle of the wheel-ball contacts from the top of the ball (nominal 51)

The spec is mapped onto the parametric generator of design_space.py (scale s, ball mass factor, ballast factor, ballast
height, rotor inertia, torque, damping, contact angle); ball and body mass are matched exactly, the centre-of-mass height
by moving the ballast. The report has:
  - derived physics: mass ratio, COM height / radius, unstable pole p, non-minimum-phase zero z, lean per acceleration,
    torque authority (motor torque / m g l), acceleration limits from torque and from a 20 deg lean cap;
  - feasibility margins from design_constraints.py (>= 1 is satisfied): structure (z vs p), loop delay, traction, motor,
    lean, contact geometry, wheel-speed limit, and which one binds;
  - with --eval: simulated falls of the recommended controller (g34_mref_rsad_db_s0, design-conditioned, evaluated on this
    unseen design) on a flat map for: goal holding without disturbances, with short pushes, with half-strength sustained
    pulls (1.5 to 6 N), and driving at 0.3 m/s; plus goals on the standard terrain tiles. Each case uses --n worlds with
    the standard randomisation (mass, friction, damping, torque, latency, sensor noise).

  python scripts/study/robot_spec.py build specs/nominal.yaml            # files + physics + margins
  python scripts/study/robot_spec.py build specs/nominal.yaml --eval     # + simulation (GPU, a few minutes)
  python scripts/study/robot_spec.py sweep specs/sweep_example.yaml --eval   # many variants of a base spec

Results go to outputs/study/specs/<name>.json (and a summary CSV for sweeps).
Caveat for --eval: the controller was trained on 11 designs around the nominal robot; designs far outside that set are
extrapolation, so a poor simulated result can mean "outside the controller's training" as well as "physically hard".
The physics margins do not have that caveat.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "study"))
import design_space as ds  # noqa: E402
import design_constraints as dc  # noqa: E402

OUT = ROOT / "outputs" / "study" / "specs"
PY = ROOT / "experiments" / "mjx_port" / ".venv" / "Scripts" / "python.exe"
EVAL_RUN = os.environ.get("SPEC_RUN", "g34_mref_rsad_db_s0")      # controller used for --eval; S119 used g34; SPEC_RUN=g39_goalcentre_s0 for the S121 recommendation
KEYS = ("ball_radius_m", "ball_mass_kg", "body_mass_kg", "body_com_height_m", "motor_torque_limit", "rotor_inertia",
        "wheel_damping", "contact_zenith_deg")
NOMINAL_ZENITH = 51.0


def _metrics(design):
    """Analytic metrics of a generator design via a scratch xml next to the variants (mesh paths are relative), removed after."""
    p = ds.VAR_DIR / f"_probe_{os.getpid()}.xml"
    try:
        return ds.analytic(design, p)
    finally:
        p.unlink(missing_ok=True)


def spec_to_design(spec, tol=5e-4):
    """Solve the generator parameters that reproduce the spec's physical numbers."""
    s = spec["ball_radius_m"] / ds.BALL_R0
    d = dict(s=s, J=spec["rotor_inertia"] / s ** 5, tau=spec["motor_torque_limit"] / s ** 4,
             damp=spec["wheel_damping"] / s ** 2.5, dz=spec["contact_zenith_deg"] - NOMINAL_ZENITH)
    # ball mass: linear in kball
    m1 = _metrics({**d, "kball": 1.0})
    d["kball"] = spec["ball_mass_kg"] / m1["m_k"]
    # body mass: linear in the ballast factor (zb does not change it)
    a0 = _metrics({**d, "kbal": 0.0})["m_a"]; a1 = _metrics({**d, "kbal": 1.0})["m_a"]
    if spec["body_mass_kg"] < a0:
        raise ValueError(f"body_mass_kg {spec['body_mass_kg']:.3f} is below the frame-only mass {a0:.3f} kg at this size "
                         "(the generator removes ballast, not structure); use a larger body mass or a smaller ball")
    d["kbal"] = (spec["body_mass_kg"] - a0) / (a1 - a0)
    # COM height: monotone in the ballast height zb (bisection over a wide bracket)
    target = spec["body_com_height_m"]
    lo, hi = -0.3 * s, 0.6 * s
    flo, fhi = _metrics({**d, "zb": lo})["l"] - target, _metrics({**d, "zb": hi})["l"] - target
    if flo * fhi > 0:
        rng = (_metrics({**d, "zb": lo})["l"], _metrics({**d, "zb": hi})["l"])
        raise ValueError(f"body_com_height_m {target:.3f} is not reachable by moving the ballast "
                         f"(reachable {min(rng):.3f} to {max(rng):.3f} m with this mass split)")
    for _ in range(40):
        mid = 0.5 * (lo + hi); fm = _metrics({**d, "zb": mid})["l"] - target
        if abs(fm) < tol:
            break
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    d["zb"] = mid
    return d


def physics(design):
    m = _metrics(design)
    c = dc.constraints(design)
    keep = ("r", "m_k", "m_a", "total_mass", "l", "mass_ratio", "l_over_r", "p", "z", "z_over_p", "Ms_lb", "kappa_deg",
            "torque_authority", "a_torque", "a_lean", "a_max")
    out = {k: float(m[k]) for k in keep}
    out.update({k: (float(c[k]) if isinstance(c[k], (int, float, np.floating)) else c[k])
                for k in ("C_struct", "C_delay", "T_loop_max", "C_trac", "C_motor", "C_lean", "C_geom", "C_speed", "v_max",
                          "rec_tilt_trac_deg", "binding")})
    out["feasible"] = bool(c["feasible"])
    # static sustained-pull capacity: the steady lean that balances a horizontal force F at the body is about
    # F h = m_a g l sin(theta) with h ~ l, so F_max(theta) ~ m_a g sin(theta); reported for 5 and 10 deg of lean.
    # Measured in simulation (S116, nominal robot): pulls up to about 3 N are absorbed completely.
    out["pull_capacity_5deg_N"] = float(m["m_a"] * 9.81 * np.sin(np.radians(5.0)))
    out["pull_capacity_10deg_N"] = float(m["m_a"] * 9.81 * np.sin(np.radians(10.0)))
    return out


def _sim(name, n, extra_env, args):
    env = dict(os.environ, TRAIN_MOD="train_dev", **extra_env)
    cmd = [str(PY), "eval_tiles2.py", "--run", EVAL_RUN, "--eval-designs", f"{name},{name}", "--dr", "--surfaces", "--n", str(max(1, n // 2)), *args]
    # the design is listed twice: the policy input includes the design vector only when more than one design is loaded
    # (same trick as eval_battery.py's "nom,nom"); n // 2 per copy keeps n worlds in total
    p = subprocess.run(cmd, cwd=ROOT / "experiments" / "warp_ppo", env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith('{"fall"'):
            return json.loads(line.split(" | ")[0])
    raise RuntimeError(f"evaluation failed for {name}: {p.stdout[-800:]} {p.stderr[-1500:]}")


def simulate(name, n=256):
    """Recommended controller on the new design; flat map unless stated (TILES_FLAT=1: no terrain, no map edge)."""
    flat = dict(TILES_FLAT="1", EVAL_NOEXIT="1")
    res = {}
    cases = [("goal_still", dict(flat, PUSH_SCALE="0"), ["--task", "goal"]),
             ("goal_pushes", dict(flat, PUSH_SCALE="1"), ["--task", "goal"]),
             ("goal_pulls_half", dict(flat, PUSH_SCALE="0.5"), ["--task", "goal", "--pull"]),
             ("drive_0.3", dict(flat, PUSH_SCALE="0"), ["--task", "vel", "--speed", "0.3"]),
             ("goal_terrain", dict(PUSH_SCALE="0"), ["--task", "goal"])]
    for key, e, a in cases:
        r = _sim(name, n, e, a)
        res[key] = {k: r[k] for k in ("fall", "success", "track_err", "late_speed_ratio") if k in r}
    return res


def build(spec, do_eval=False, n=256):
    for k in KEYS:
        if k not in spec:
            raise KeyError(f"spec is missing {k}")
    name = spec["name"]
    design = spec_to_design(spec)
    files = ds.build(design, tag=f"dz_{name}")
    phys = physics(design)
    rep = dict(spec=spec, generator=design, physics=phys, files={k: str(v) for k, v in files.items() if k != "metrics"})
    if do_eval:
        rep["sim"] = simulate(name, n)
        rep["sim_protocol"] = dict(engine="MuJoCo-Warp float32 GPU", controller=EVAL_RUN, worlds_per_case=n, dr="on (standard ranges)",
                                   noise="on", note="flat map (TILES_FLAT) except goal_terrain; pulls 1.5 to 6 N sustained")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def summary_line(rep):
    p = rep["physics"]
    s = f"{rep['spec']['name']:<22} m {p['total_mass']:5.2f} kg  ball/body {p['m_k'] / p['m_a']:5.3f}  l/r {p['l_over_r']:4.2f}  " \
        f"p {p['p']:4.2f}  z/p {p['z_over_p']:4.2f}  auth {p['torque_authority']:5.1f}  lean {p['kappa_deg']:4.1f} deg/(m/s2)  " \
        f"binding {p['binding']}  feasible {p['feasible']}"
    if "sim" in rep:
        S = rep["sim"]
        s += "  | falls still {:.3f} push {:.3f} pull {:.3f} drive {:.3f} terrain {:.3f}".format(
            S["goal_still"]["fall"], S["goal_pushes"]["fall"], S["goal_pulls_half"]["fall"], S["drive_0.3"]["fall"], S["goal_terrain"]["fall"])
    return s


def sweep(cfg, do_eval, n):
    """cfg: base: <spec path or dict>, vary: {key: [values...]} (one key at a time, others at base)."""
    base = cfg["base"] if isinstance(cfg["base"], dict) else yaml.safe_load(open(ROOT / cfg["base"]))
    rows = []
    for key, vals in cfg["vary"].items():
        for v in vals:
            sp = copy.deepcopy(base); sp[key] = v
            sp["name"] = f"{cfg.get('prefix', 'sw')}_{key.split('_')[0]}{key.split('_')[1][:3]}_{v:g}".replace(".", "p")
            try:
                rep = build(sp, do_eval, n)
                print(summary_line(rep), flush=True)
                row = dict(name=sp["name"], vary=key, value=v, **{f"phys_{k}": rep["physics"][k] for k in rep["physics"]})
                for case, r in rep.get("sim", {}).items():
                    row[f"sim_{case}_fall"] = r["fall"]
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print(f"{sp['name']}: {e}", flush=True)
                rows.append(dict(name=sp["name"], vary=key, value=v, error=str(e)))
    import pandas as pd
    out = OUT / f"{cfg.get('prefix', 'sw')}_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def evaluate_built(prefix, n):
    """Run the simulation cases on every already-built spec <prefix>_* (from a physics-only sweep) and update its json."""
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob(f"{prefix}*.json")):
        rep = json.loads(f.read_text())
        if "sim" not in rep:
            rep["sim"] = simulate(rep["spec"]["name"], n)
            rep["sim_protocol"] = dict(engine="MuJoCo-Warp float32 GPU", controller=EVAL_RUN, worlds_per_case=n, dr="on (standard ranges)",
                                       noise="on", note="flat map (TILES_FLAT) except goal_terrain; pulls 1.5 to 6 N sustained")
            f.write_text(json.dumps(rep, indent=1, default=float))
        print(summary_line(rep), flush=True)
        rows.append(dict(name=rep["spec"]["name"], **{f"sim_{k}_fall": v["fall"] for k, v in rep["sim"].items()},
                         **{f"phys_{k}": v for k, v in rep["physics"].items()}))
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}_sim.csv", index=False)
    print("wrote", OUT / f"{prefix}_sim.csv")


# ---------------------------------------------------------------- one spec for the picture, the layout rules and the simulator design
LAYOUT_DEFAULTS = dict(wheels=3, drive=dict(type="direct", ratio=1.0, efficiency=1.0), guards=dict(reach_deg=112.0, gap_mm=1.0), arms=False)


def effective_spec(spec):
    """The spec as the simulator sees it. The spec's motor_torque_limit and rotor_inertia are MOTOR-side numbers; a transmission of ratio N and efficiency eta
    gives the wheel a torque limit N * eta * tau and a reflected rotor inertia N^2 * J. Direct drive is N = 1. (The simulator has no wheel speed limit, so the
    speed lost to the ratio is reported but not simulated.)"""
    d = {**LAYOUT_DEFAULTS["drive"], **(spec.get("drive") or {})}; N, eta = float(d["ratio"]), float(d["efficiency"])
    e = {k: spec[k] for k in KEYS}; e["name"] = spec["name"]
    e["motor_torque_limit"] = spec["motor_torque_limit"] * N * eta; e["rotor_inertia"] = spec["rotor_inertia"] * N ** 2
    return e, dict(type=d["type"], ratio=N, efficiency=eta)


def spec_to_layout(spec):
    """The same spec as a wheel layout (ballbot_layout.py) for the rules and the picture. Wheel and body sizes scale with the ball unless the spec gives them."""
    import ballbot_layout as BL
    k = spec["ball_radius_m"] / 0.09; _, drv = effective_spec(spec); guards = spec.get("guards", {}); g = {**LAYOUT_DEFAULTS["guards"], **(guards or {})}
    L = BL.family(spec["name"], int(spec.get("wheels", 3)), float(spec["contact_zenith_deg"]), spec.get("contact_zenith_b_deg"),
                  R=spec["ball_radius_m"], rw=spec.get("wheel_radius_m", 0.035 * k), ww=spec.get("wheel_width_m", 0.03 * k), body_mass=spec["body_mass_kg"], ball_mass=spec["ball_mass_kg"],
                  body_r=spec.get("body_half_width_m", 0.11 * k), body_h=spec.get("body_height_m", 0.28 * k), tau_max=spec["motor_torque_limit"], rotor_J=spec["rotor_inertia"],
                  drive="direct" if drv["type"] == "direct" else "geared", N=drv["ratio"], eta=drv["efficiency"], arr_theta=(None if guards is False else g["reach_deg"]), arr_gap_mm=g["gap_mm"])
    L["arms"] = bool(spec.get("arms", False))
    return L


def show(spec, out):
    """Picture + rules + simulator mapping of one spec, in one image; returns the report dict."""
    import ballbot_layout as BL, layout_render as LR
    from PIL import Image, ImageDraw
    L = spec_to_layout(spec); v = BL.validate(L); c = LR.checks(L); eff, drv = effective_spec(spec); sim_ok = int(spec.get("wheels", 3)) == 3 and spec.get("contact_zenith_b_deg") is None
    bad = [k for k, x in v["rules"].items() if not x] + (["motors do not fit above the wheels"] if c["motor_below"] else [])
    k = spec["ball_radius_m"] / 0.09
    a = LR.render(L, 760, 900, az=152, el=-8, dist=1.15 * k, look_z=0.27 * k); b = LR.render(L, 760, 900, az=152, el=-14, dist=0.64 * k, look_z=0.13 * k, cutaway=True)
    img = Image.new("RGB", (760 * 2 + 640, 900), (16, 22, 34)); img.paste(a, (0, 0)); img.paste(b, (760, 0)); d = ImageDraw.Draw(img); st = dict(y=16); x = 1540

    def line(t, col=(236, 241, 248), sz=15, bold=False, dy=24):
        d.text((x, st["y"]), t, fill=col, font=LR._f(sz, bold)); st["y"] += dy
    HEAD = (120, 170, 255); OKC = (94, 224, 138); BADC = (255, 107, 107); DIM = (150, 163, 182)
    line(spec["name"], sz=22, bold=True, dy=34); line("SPEC (what you set)", HEAD, 13, True)
    two = f" / {spec['contact_zenith_b_deg']:.0f} deg" if spec.get("contact_zenith_b_deg") else ""
    gtxt = "none" if L["arr_theta"] is None else f"to {L['arr_theta']:.0f} deg, gap {L['arr_gap_mm']:g} mm"
    for t in (f"ball: radius {spec['ball_radius_m'] * 100:.1f} cm, mass {spec['ball_mass_kg']:.3f} kg", f"body: mass {spec['body_mass_kg']:.2f} kg, COM {spec['body_com_height_m'] * 100:.1f} cm above ball centre",
              f"wheels: {L['n']} at {spec['contact_zenith_deg']:.0f} deg from the top{two}", f"motor: {spec['motor_torque_limit']:.1f} N m, rotor inertia {spec['rotor_inertia']:.4f} kg m2",
              f"drive: {drv['type']}, ratio {drv['ratio']:g}, efficiency {drv['efficiency']:.2f}", f"guards: {gtxt};  arms: {'yes (visual only)' if L['arms'] else 'no'}"): line(t)
    st["y"] += 8; line("AT THE WHEEL (what the physics uses)", HEAD, 13, True)
    line(f"torque limit {eff['motor_torque_limit']:.1f} N m, reflected inertia {eff['rotor_inertia']:.4f} kg m2"); line(f"speed is divided by {drv['ratio']:g} (not simulated: the simulator has no wheel speed limit)", DIM, 12)
    st["y"] += 8; line("LAYOUT RULES", HEAD, 13, True)
    for kname, okk in v["rules"].items(): line(("PASS  " if okk else "FAIL  ") + kname, OKC if okk else BADC, 13, dy=20)
    line(("PASS  " if not c["motor_below"] else "FAIL  ") + "motors fit above the wheels", OKC if not c["motor_below"] else BADC, 13, dy=20)
    line(f"wheel-ball gap {c['ball_gap_mm']:.2f} mm, wheel clearance {c['wheel_clear_mm']:.0f} mm", DIM, 12)
    st["y"] += 8; line("SIMULATOR", HEAD, 13, True); rep = dict(spec=spec, effective=eff, drive=drv, rules=v["rules"], checks=c, layout_ok=not bad, simulator_supported=sim_ok)
    if bad:
        line("not built: the layout fails the rules above", BADC)
    elif not sim_ok:
        line("picture and rules only: the simulator supports", (255, 180, 84)); line("3 wheels at one height today", (255, 180, 84))
    else:
        dsg = spec_to_design(eff); ph = physics(dsg); rep.update(design=dsg, physics=ph)
        line(f"design: scale {dsg['s']:.3f}, ball x{dsg['kball']:.2f}, ballast x{dsg['kbal']:.2f}", sz=13, dy=20); line(f"        ballast height {dsg['zb'] * 100:.1f} cm, wheel angle {dsg['dz']:+.0f} deg", sz=13, dy=20)
        line(f"total mass {ph['total_mass']:.2f} kg, COM {ph['l'] * 100:.1f} cm ({ph['l_over_r']:.2f} ball radii)", sz=13, dy=20)
        line(f"analytic check: {'feasible' if ph['feasible'] else 'NOT feasible'}, limit: {ph['binding']}", OKC if ph["feasible"] else BADC, 13, dy=20)
        line("physics model: validated crude geometry (not this picture)", DIM, 12)
    img.save(out); print("saved", out)
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("spec"); b.add_argument("--eval", action="store_true"); b.add_argument("--n", type=int, default=256)
    w = sub.add_parser("sweep"); w.add_argument("cfg"); w.add_argument("--eval", action="store_true"); w.add_argument("--n", type=int, default=256)
    e = sub.add_parser("evaluate"); e.add_argument("prefix"); e.add_argument("--n", type=int, default=256)
    sh = sub.add_parser("show"); sh.add_argument("spec"); sh.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "evaluate":
        evaluate_built(a.prefix, a.n); return
    if a.cmd == "show":
        rep = show(yaml.safe_load(open(a.spec)), a.out); print(json.dumps({k: rep[k] for k in ("effective", "drive", "layout_ok", "simulator_supported")}, indent=1, default=float)); return
    if a.cmd == "build":
        rep = build(yaml.safe_load(open(a.spec)), a.eval, a.n)
        print(summary_line(rep))
        print(json.dumps(rep["physics"], indent=1, default=float))
    else:
        sweep(yaml.safe_load(open(a.cfg)), a.eval, a.n)


if __name__ == "__main__":
    main()
