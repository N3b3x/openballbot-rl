"""Parametric ballbot layout: wheel count, contact heights, azimuths, drive type, arrestor. Geometry, kinematics, a feasibility screen and a to-scale design sheet.

A layout is a dict (see LAYOUTS below):
  n wheels, each with an azimuth phi_i [deg] and a polar angle theta_i [deg] measured from the top of the ball (nominal 51), so a wheel sits higher on the ball
  for a small theta and lower for a large theta; ball radius R, wheel radius rw and width ww; drive = "direct" (motor at the wheel) or "geared" (motor in the
  body, belt or chain down to the wheel, ratio N, efficiency eta); arrestor ring at polar angle arr_theta with a gap.

Model (stated assumptions, simulation only, not a hardware design):
  * contact point p_i = R (sin th cos ph, sin th sin ph, cos th) from the ball centre; the wheel pushes the ball along -p_i (inward);
  * an omni wheel drives the ball along the azimuthal tangent t_i = z x p_i / |z x p_i| (rollers free along the meridian), so a wheel speed w_i (surface speed) maps to the
    ball angular velocity w by  v_i = t_i . (w x p_i) = (p_i x t_i) . w   ->  v = J w with row i of J = p_i x t_i;
  * drive push in horizontal direction u: the largest ball torque about the axis z x u, sum_i F_i |(J u')_i| with the tangential force per wheel F_i = min(torque limit / rw, mu_wb N_i)
    (N_i = equilibrium wheel normals from non-negative least squares): grip, not motor torque, is the limit for strong motors;
  * top speed in direction u: the largest ball speed before some wheel passes its speed limit;
  * retention: the largest horizontal force F in direction u that the wheel normals N_i in [0, Nmax] can balance while carrying the body weight W through the ball (a linear program).
    This is a quasi-static contact check, not the full dynamics: it says whether the layout can hold the ball, not whether the controller can balance it.
  python scripts/study/ballbot_layout.py sheet OUT.png            # design sheet of the example layouts
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, nnls

G = 9.81
BASE = dict(R=0.09, rw=0.035, ww=0.03, body_mass=3.85, ball_mass=0.168, body_r=0.11, body_h=0.28, tau_max=30.0, rotor_J=0.005, wheel_speed_max=40.0,
            drive="direct", N=1.0, eta=1.0, arr_theta=None, arr_gap_mm=1.0, mu_wb=0.8, mu_roll=0.01)


def make(name, thetas, phis, **kw):
    d = dict(BASE); d.update(kw)
    d["name"] = name; d["theta"] = np.array(thetas, float); d["phi"] = np.array(phis, float); d["n"] = len(phis)
    return d


def even(n, theta=51.0, offset=0.0):
    return [theta] * n, [offset + 360.0 * i / n for i in range(n)]


LAYOUTS = [
    make("3 wheels, even, 51 deg (nominal)", *even(3)),
    make("3 wheels, higher on ball, 35 deg", *even(3, 35.0)),
    make("3 wheels, lower on ball, 65 deg", *even(3, 65.0)),
    make("4 wheels, even, 51 deg", *even(4)),
    make("3 wheels grouped (0, 50, 100 deg)", [51] * 3, [0, 50, 100]),
    make("4 wheels, alternating 40 / 62 deg", [40, 62, 40, 62], [0, 90, 180, 270]),
    make("3 wheels, geared body motors (N = 4)", *even(3), drive="geared", N=4.0, eta=0.85, tau_max=8.0, rotor_J=0.0004, wheel_speed_max=40.0),
]


def family(name, n, theta=51.0, theta_b=None, offset=0.0, **kw):
    """The valid layout families. n wheels evenly spaced in azimuth; one polar angle, or (theta, theta_b) alternating, which needs an even n of at least 6 so that
    the layout still has three-fold or higher rotational symmetry (two interleaved rings of n / 2 wheels)."""
    if theta_b is None:
        assert n >= 3, "a ballbot needs at least 3 wheels"
        return make(name, [theta] * n, [offset + 360.0 * i / n for i in range(n)], **kw)
    assert n % 2 == 0 and n >= 6, "alternating heights need an even number of wheels, at least 6 (three-fold symmetry of each ring)"
    return make(name, [theta if i % 2 == 0 else theta_b for i in range(n)], [offset + 360.0 * i / n for i in range(n)], **kw)


def validate(L, tol=0.02):
    """Rules a layout must pass before it is drawn, simulated or trained on. The LQR base treats the x and y balance problems as identical and uncoupled from yaw;
    that holds exactly when M = J^T J is diag(a, a, b): the wheels act equally in every horizontal direction (small motions) and do not mix roll, pitch and yaw.
    Any layout with three-fold or higher rotational symmetry satisfies it; wheels grouped on one side, or 4 wheels at alternating heights, do not."""
    p, t, J = geometry(L); M = J.T @ J; n = L["n"]; pn = p / np.linalg.norm(p, axis=1, keepdims=True)
    iso = abs(M[0, 0] - M[1, 1]) / (M[0, 0] + M[1, 1]); cpl = max(abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2])) / max(M[0, 0], 1e-12)
    bal = float(np.linalg.norm(pn[:, :2].sum(0)) / n)                                # equal wheel preloads must not push the ball sideways
    hubs = (L["R"] + L["rw"]) * pn; clr = min((np.linalg.norm(hubs[i] - hubs[j]) - 2 * L["rw"] for i in range(n) for j in range(i + 1, n)), default=1.0)
    _, _, _, r = authority_speed_retention(L, nang=36); W = (L["body_mass"] + L["ball_mass"]) * G
    rules = {"at least 3 wheels": n >= 3, "full control (rank 3)": int(np.linalg.matrix_rank(J, tol=1e-9)) == 3,
             "same in x and y (LQR symmetry)": iso < tol, "no roll/pitch/yaw coupling": cpl < tol, "balanced preload": bal < tol,
             "polar angle 30 to 70 deg": bool((L["theta"] >= 30).all() and (L["theta"] <= 70).all()), "wheels do not touch each other": clr > 0.005,
             "holds the ball in every direction": float(r.min() / W) > 0.2}
    return dict(ok=all(rules.values()), rules=rules, iso=float(iso), coupling=float(cpl), balance=bal, wheel_clear_mm=float(clr * 1000), ret_min=float(r.min() / W))


def geometry(L):
    th, ph = np.radians(L["theta"]), np.radians(L["phi"])
    p = L["R"] * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)       # contact points
    z = np.array([0, 0, 1.0]); t = np.cross(z, p); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    J = np.cross(p, t)                                                                               # v_i = J_i . w
    return p, t, J


def limits(L):
    """Effective per-wheel torque limit and speed limit at the wheel, after the transmission."""
    N, eta = L["N"], L["eta"]
    return L["tau_max"] * N * eta, L["wheel_speed_max"] / N, L["rotor_J"] * N ** 2


def authority_speed_retention(L, nang=72):
    p, t, J = geometry(L); tau_w, w_lim, _ = limits(L)
    ang = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    auth, vmax, ret = [], [], []
    # wheel torque T_i produces ball torque sum_i T_i * J_i / rw ... (power balance: tau_ball . w = sum T_i v_i / rw  ->  tau_ball = J^T T / rw)
    rw = L["rw"]; W = (L["body_mass"] + L["ball_mass"]) * G
    pn0 = -p / np.linalg.norm(p, axis=1, keepdims=True)                      # equilibrium wheel normals: carry W, no net horizontal force (non-negative least squares)
    N0 = _equilibrium_normals(p, W)
    Ft = np.minimum(tau_w / rw, L["mu_wb"] * N0)                              # tangential force per wheel: torque limit or grip limit, whichever is lower
    for a in ang:
        u = np.array([np.cos(a), np.sin(a), 0.0])            # horizontal direction of travel
        ax = np.cross(np.array([0, 0, 1.0]), u)               # ball rotates about z x u to roll toward u
        tau_ball = (Ft * np.abs(J @ ax)).sum()                  # max ball torque about ax (grip- or torque-limited)
        auth.append(tau_ball / L["R"])                          # horizontal push force on the floor [N]
        # speed: ball angular speed Om about ax: wheel surface speeds v_i = (J ax)_i Om; each |v_i| <= w_lim * rw
        s = np.abs(J @ ax); vmax.append(L["R"] * (w_lim * rw / max(s.max(), 1e-9)))
        # retention: normals N_i >= 0 along -p_i/|p|; vertical part sums to W (they push the ball down onto the floor), horizontal part F u; maximise F
        pn = -p / np.linalg.norm(p, axis=1, keepdims=True)
        n = len(p); c = np.zeros(n + 1); c[-1] = -1.0
        A = np.zeros((3, n + 1)); A[0, :n] = pn[:, 0] * 0 + (pn @ np.array([u[0], u[1], 0]))      # component along u
        A[1, :n] = pn @ np.array([-u[1], u[0], 0])                                                   # component across u must vanish
        A[2, :n] = pn[:, 2]
        A[0, -1] = -1.0
        b = np.array([0.0, 0.0, -W])
        res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, 40 * W)] * n + [(0, None)], method="highs")
        ret.append(res.x[-1] if res.status == 0 else 0.0)
    return ang, np.array(auth), np.array(vmax), np.array(ret)


def _equilibrium_normals(p, W):
    """Wheel normals that carry the weight W with no net horizontal force. More than three wheels are statically indeterminate: take the minimum-norm
    (evenly shared) solution, and fall back to non-negative least squares if that has a negative entry."""
    pn = -p / np.linalg.norm(p, axis=1, keepdims=True); A = pn.T; b = np.array([0.0, 0.0, -W])
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x if (x >= -1e-9).all() else nnls(A, b)[0]


def summary(L):
    p, t, J = geometry(L); sv = np.linalg.svd(J, compute_uv=False)
    ang, a, v, r = authority_speed_retention(L)
    _, _, Jr = limits(L)
    W = (L["body_mass"] + L["ball_mass"]) * G
    return dict(rank=int((sv > 1e-6 * sv.max()).sum()), cond=float(sv.max() / max(sv.min(), 1e-12)) if len(sv) == 3 else float("inf"),
                grip_torque=float(L["mu_wb"] * _equilibrium_normals(p, W).max() * L["rw"]),
                auth_min=float(a.min()), auth_max=float(a.max()), auth_aniso=float(a.min() / a.max()),
                vmax_min=float(v.min()), vmax_max=float(v.max()), ret_min=float(r.min() / W), ret_max=float(r.max() / W), refl_inertia=float(Jr),
                ok=bool(r.min() > 0.05 * W and a.min() > 0.2 * a.max()))


# ---------------------------------------------------------------- installation errors ("as built" against "as designed")
TOL = {   # 1-sigma errors of each installation tolerance class; angles in degrees, lengths in mm, gains as fractions
    "tight": dict(az=0.5, polar=0.5, rw_mm=0.1, gain=0.02, mu=0.05, com_xy_mm=1.0, imu_deg=0.2),
    "standard": dict(az=1.5, polar=1.0, rw_mm=0.3, gain=0.05, mu=0.15, com_xy_mm=3.0, imu_deg=0.5),
    "loose": dict(az=3.0, polar=2.0, rw_mm=0.8, gain=0.10, mu=0.30, com_xy_mm=8.0, imu_deg=1.5),
}


def as_built(L, rng, tol):
    """One random as-built robot: wheel azimuth and polar errors, wheel radius mismatch, per-wheel torque-gain mismatch, per-wheel wheel-ball friction,
    body COM offset in x and y, IMU mounting tilt. Returns the perturbed layout and the nuisance values the controller does not know."""
    A = dict(L); n = L["n"]
    A["phi"] = L["phi"] + rng.normal(0, tol["az"], n); A["theta"] = L["theta"] + rng.normal(0, tol["polar"], n)
    A["rw_i"] = L["rw"] + rng.normal(0, tol["rw_mm"] / 1000.0, n); A["gain_i"] = 1.0 + rng.normal(0, tol["gain"], n)
    A["mu_i"] = np.clip(L["mu_wb"] * (1.0 + rng.normal(0, tol["mu"], n)), 0.1, None)
    A["com_xy"] = rng.normal(0, tol["com_xy_mm"] / 1000.0, 2); A["imu_deg"] = rng.normal(0, tol["imu_deg"], 2)
    return A


def as_built_metrics(L, A):
    """Mismatch between what the controller assumes (ideal layout L) and what was built (A): coupling error of the ball-rate map, plus min retention and anisotropy of A,
    the resting lean the COM offset and IMU tilt force, and the weakest-wheel grip."""
    _, _, Jn = geometry(L); _, _, Ja = geometry(A)
    Ja = Ja * (A["gain_i"] * A["rw"] / A["rw_i"])[:, None]                                 # per-wheel gain and wheel-radius mismatch scale each row
    E = np.linalg.pinv(Jn) @ Ja; coupling = float(np.linalg.norm(E - np.eye(3), 2))
    ang, a, v, r = authority_speed_retention(A, nang=36); W = (A["body_mass"] + A["ball_mass"]) * G
    lean_com = float(np.degrees(np.arctan2(np.linalg.norm(A["com_xy"]), A["body_h"] / 2 + A["R"])))      # tilt at which the COM sits over the ball centre again
    return dict(coupling=coupling, aniso=float(a.min() / a.max()), ret_min=float(r.min() / W), lean_deg=float(np.hypot(lean_com, np.linalg.norm(A["imu_deg"]))))


def monte_carlo(L, tol, n=400, seed=0):
    rng = np.random.default_rng(seed); rows = [as_built_metrics(L, as_built(L, rng, tol)) for _ in range(n)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def tolerance_figure(out, layouts, n=300):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    metrics = [("coupling", "ball-rate coupling error (0 = as designed)"), ("aniso", "weakest / strongest push"), ("ret_min", "min retention [x body weight]"), ("lean_deg", "forced rest lean [deg]")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), dpi=130); cols = {"tight": "#2e86c1", "standard": "#e67e22", "loose": "#c0392b"}; table = {}
    for ax, (k, lab) in zip(axs, metrics):
        pos = 0; ticks = []; labs = []
        for L in layouts:
            for tn, tol in TOL.items():
                v = monte_carlo(L, tol, n)[k]; table[(L["name"], tn, k)] = (np.median(v), np.percentile(v, 5), np.percentile(v, 95))
                bp = ax.boxplot(v, positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
                for b in bp["boxes"]: b.set(facecolor=cols[tn], alpha=0.65)
                pos += 1
            ticks.append(pos - 2); labs.append(f"{L['n']} wheels"); pos += 1
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7); ax.set_title(lab, fontsize=8); ax.tick_params(labelsize=7)
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=c, alpha=0.65, label=f"{n_}: " + ", ".join(f"{a} {b}" for a, b in list(TOL[n_].items())[:3])) for n_, c in cols.items()], fontsize=5.5, loc="upper left")
    fig.suptitle("Installation errors (as built against as designed): Monte Carlo of 300 robots per box. Wheel azimuth, polar angle, wheel radius, torque gain, friction, COM offset, IMU tilt. Simulation-only screen.", fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("saved", out)
    return table


# ---------------------------------------------------------------- drawing (matplotlib, to scale in metres)
def draw_side(ax, L, view_az=0.0):
    import matplotlib.patches as mp
    R, rw, ww = L["R"], L["rw"], L["ww"]
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.115, 0.42)
    ax.add_patch(mp.Circle((0, 0), R, fc="#e8c9a0", ec="#7a5230", lw=1.5, zorder=2))
    ax.plot([-0.2, 0.2], [-R, -R], color="#444", lw=2)                       # floor
    th, ph = np.radians(L["theta"]), np.radians(L["phi"]); va = np.radians(view_az)
    # wheels: hub at (R + rw) along the contact direction, drawn as a disc (radius rw) seen edge-on as a rounded bar
    order = np.argsort(np.sin(ph - va))                                        # far to near
    for i in order:
        d = np.array([np.sin(th[i]) * np.cos(ph[i] - va), np.sin(th[i]) * np.sin(ph[i] - va), np.cos(th[i])])
        hub = (R + rw) * np.array([d[0] * 0 + np.sin(th[i]) * np.cos(ph[i] - va) * 1.0, d[2]])
        depth = np.sin(ph[i] - va); a = 0.95 if depth > -0.2 else 0.45
        ax.add_patch(mp.Circle((hub[0], hub[1]), rw, fc="#2b5f9e", ec="#14304f", alpha=a, zorder=3 if depth > 0 else 1))
        ax.add_patch(mp.Circle((hub[0], hub[1]), rw * 0.25, fc="#14304f", alpha=a, zorder=4 if depth > 0 else 1))
        top = hub + rw * 1.0 * np.array([np.cos(th[i]) * np.sign(hub[0] if hub[0] != 0 else 1) * 0, 0])
        # motor and transmission
        if L["drive"] == "direct":
            ax.add_patch(mp.Rectangle((hub[0] - 0.014, hub[1] + rw * 0.4), 0.028, 0.04, fc="#999", ec="#333", alpha=a, zorder=3 if depth > 0 else 1))
        else:
            mx, my = hub[0] * 0.55, 0.20                                          # motor up in the body
            ax.add_patch(mp.Rectangle((mx - 0.016, my - 0.03), 0.032, 0.06, fc="#999", ec="#333", alpha=a, zorder=3 if depth > 0 else 1))
            ax.plot([mx, hub[0]], [my - 0.03, hub[1] + rw * 0.5], color="#d2691e", lw=1.4, alpha=a, zorder=3 if depth > 0 else 1)
            ax.plot([mx + 0.01, hub[0] + 0.012], [my - 0.03, hub[1] + rw * 0.5], color="#d2691e", lw=1.4, alpha=a, zorder=3 if depth > 0 else 1)
    zb = max((R + rw) * np.cos(th).max() + rw + 0.01, 0.07)                      # body underside above the highest wheel
    ax.add_patch(mp.FancyBboxPatch((-L["body_r"], zb), 2 * L["body_r"], L["body_h"], boxstyle="round,pad=0,rounding_size=0.015", fc="#dfe6ee", ec="#4a5a6a", lw=1.2, zorder=5, alpha=0.96))
    ax.plot([0], [zb + 0.13], marker="x", color="#d62728", ms=7, zorder=6)
    ax.text(0.0, zb + 0.145, "COM", color="#d62728", ha="center", fontsize=6, zorder=6)
    if L["arr_theta"] is not None:
        at = np.radians(L["arr_theta"]); rr = R + L["arr_gap_mm"] / 1000.0
        for sgn in (-1, 1):
            ax.add_patch(mp.Circle((sgn * rr * np.sin(at), rr * np.cos(at)), 0.008, fc="#c0392b", ec="#7b241c", zorder=4))


def draw_top(ax, L):
    import matplotlib.patches as mp
    R = L["R"]; ax.set_aspect("equal"); ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.2, 0.2); ax.axis("off")
    ax.add_patch(mp.Circle((0, 0), R, fc="#e8c9a0", ec="#7a5230")); ax.add_patch(mp.Circle((0, 0), L["body_r"], fc="none", ec="#4a5a6a", ls="--"))
    p, t, J = geometry(L)
    for i in range(L["n"]):
        ax.add_patch(mp.FancyBboxPatch((p[i, 0] * 1.5 - 0.02, p[i, 1] * 1.5 - 0.012), 0.04, 0.024, boxstyle="round,pad=0,rounding_size=0.008", fc="#2b5f9e", ec="#14304f"))
        ax.annotate("", xy=(p[i, 0] + t[i, 0] * 0.05, p[i, 1] + t[i, 1] * 0.05), xytext=(p[i, 0], p[i, 1]), arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.4))
        ax.text(p[i, 0] * 2.15, p[i, 1] * 2.15, f"{L['phi'][i]:.0f}°\n{L['theta'][i]:.0f}°", ha="center", va="center", fontsize=6)


def polar(ax, ang, val, color, title, vmax=None):
    ax.plot(np.r_[ang, ang[0]], np.r_[val, val[0]], color=color, lw=1.8); ax.fill(np.r_[ang, ang[0]], np.r_[val, val[0]], color=color, alpha=0.18)
    ax.set_title(title, fontsize=7, pad=10); ax.tick_params(labelsize=5); ax.set_rlim(0, vmax if vmax else max(val.max() * 1.1, 1e-6)); ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["+x", "+y", "-x", "-y"], fontsize=5)


def sheet(out, layouts=LAYOUTS):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(layouts); fig = plt.figure(figsize=(3.1 * n, 14.5), dpi=130, facecolor="white")
    gs = fig.add_gridspec(6, n, height_ratios=[1.45, 1.0, 0.85, 0.85, 0.85, 0.9], hspace=0.62, wspace=0.12)
    S = [summary(L) for L in layouts]; allv = [authority_speed_retention(L) for L in layouts]
    amax = max(v[1].max() for v in allv); vmx = max(v[2].max() for v in allv); rmx = max(v[3].max() for v in allv)
    for k, (L, s, (ang, a, v, r)) in enumerate(zip(layouts, S, allv)):
        ax = fig.add_subplot(gs[0, k]); draw_side(ax, L); ax.axis("off")
        ax.set_title(L["name"], fontsize=8, fontweight="bold", color="#14304f" if s["ok"] else "#b03a2e", pad=4)
        draw_top(fig.add_subplot(gs[1, k]), L)
        polar(fig.add_subplot(gs[2, k], projection="polar"), ang, a, "#2b7a4b", f"drive push [N]  {a.min():.0f} to {a.max():.0f}", amax)
        polar(fig.add_subplot(gs[3, k], projection="polar"), ang, v, "#8e44ad", f"top speed [m/s]  {v.min():.1f} to {v.max():.1f}", vmx)
        polar(fig.add_subplot(gs[4, k], projection="polar"), ang, r / ((L["body_mass"] + L["ball_mass"]) * G), "#c0392b", f"retention [x body weight]  {s['ret_min']:.2f} to {s['ret_max']:.2f}", rmx / ((BASE["body_mass"] + BASE["ball_mass"]) * G))
        axt = fig.add_subplot(gs[5, k]); axt.axis("off")
        tau_w, w_lim, Jr = limits(L)
        rows = [("wheels", f"{L['n']}"), ("drive", "direct" if L["drive"] == "direct" else f"geared N={L['N']:g}, eta {L['eta']:.2f}"),
                ("wheel torque limit", f"{tau_w:.0f} N m"), ("torque at grip limit", f"{s['grip_torque']:.2f} N m"), ("reflected inertia", f"{Jr:.4f} kg m2"), ("rank / cond(J)", f"{s['rank']} / {s['cond']:.1f}"),
                ("anisotropy (min/max push)", f"{s['auth_aniso']:.2f}"), ("verdict", "holds the ball" if s["ok"] else "weak: see polar plots")]
        for j, (a1, b1) in enumerate(rows):
            axt.text(0.0, 0.95 - j * 0.14, a1, fontsize=6.5, color="#555", transform=axt.transAxes); axt.text(1.0, 0.95 - j * 0.14, b1, fontsize=6.5, ha="right", fontweight="bold", transform=axt.transAxes)
    fig.suptitle("Parametric ballbot layouts, to scale (ball R = 9 cm). Blue = omni wheel, orange arrow = drive direction, red dots = arrestor. Simulation-only geometry model, see module docstring.", fontsize=8.5, y=0.995)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)
    return S


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "sheet":
        S = sheet(sys.argv[2])
        for L, s in zip(LAYOUTS, S):
            print(f"{L['name']:44s} rank {s['rank']} cond {s['cond']:.1f} push {s['auth_min']:.0f}-{s['auth_max']:.0f} N  vmax {s['vmax_min']:.2f}-{s['vmax_max']:.2f}  retention {s['ret_min']:.2f}-{s['ret_max']:.2f}  ok {s['ok']}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "tolerance":
        T = tolerance_figure(sys.argv[2], [LAYOUTS[0], LAYOUTS[3]])
        for (nm, tn, k), (m, lo, hi) in T.items():
            print(f"{nm:34s} {tn:9s} {k:9s} median {m:.3f}  5-95%: {lo:.3f} to {hi:.3f}")
    else:
        print(__doc__)
