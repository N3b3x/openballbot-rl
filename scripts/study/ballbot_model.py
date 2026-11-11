"""First-principles ballbot model: parameters from the MuJoCo XML, omniwheel transmission, planar Lagrangian
dynamics (sympy), linearisation, structural properties (unstable pole, non-minimum-phase zero, lean/acceleration
law) and an LQR design. Also validates the derived model against the MuJoCo simulator.

  python scripts/study/ballbot_model.py            # print derivation, validation and LQR gains
  python scripts/study/ballbot_model.py --json out # also save the numbers (used by the paper and lqr controller)

Planar model (one vertical plane; the linearised 3-D ballbot decouples into two such planes plus yaw):
  coordinates q = (phi, theta): absolute ball rotation angle and absolute body tilt.
  ball position x = r phi (rolling without slip); body COM at (x + l sin theta, l cos theta).
  motor torque tau acts between body and ball; the wheels spin with the relative angle phi - theta.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
XML = ROOT / "ballbot_gym/models/ballbot.xml"
G = 9.81


# ----------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------
def extract_params(wheel_damping=None, armature=None, xml=None):
    m = mujoco.MjModel.from_xml_path(str(xml or XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base = m.body("base").id
    ball_g = m.geom("the_ball").id
    c_ball = d.geom_xpos[ball_g].copy()
    r = float(m.geom_size[ball_g][0])
    m_k = float(m.body_mass[m.body("ball").id])
    I_k = float(m.body_inertia[m.body("ball").id][0])
    # body = base subtree (base, cameras, wheels), rigid with the base except wheel spin
    ids = [i for i in range(m.nbody) if i != 0 and (i == base or m.body_parentid[i] == base)]
    m_a = float(sum(m.body_mass[i] for i in ids))
    com = sum(m.body_mass[i] * d.xipos[i] for i in ids) / m_a
    l = float(com[2] - c_ball[2])
    # inertia of the body about its COM around the world y axis (pitch plane), parallel-axis theorem
    I_a = 0.0
    for i in ids:
        R = d.ximat[i].reshape(3, 3)
        Iw = R @ np.diag(m.body_inertia[i]) @ R.T
        dx = d.xipos[i] - com
        I_a += Iw[1, 1] + m.body_mass[i] * (dx[0] ** 2 + dx[2] ** 2)
    # omniwheel transmission W: wheel spin rates = W @ (ball angular velocity relative to body)
    W, contacts = [], []
    for k in range(3):
        gw = m.geom(f"wheel_mesh_{k}").id
        j = m.joint(f"wheel_joint_{k}")
        a = d.xmat[j.bodyid].reshape(3, 3) @ m.jnt_axis[j.id]         # wheel spin axis (world)
        cw = d.geom_xpos[gw]
        u = (cw - c_ball) / np.linalg.norm(cw - c_ball)                # ball centre -> wheel centre
        p = r * u                                                       # contact point rel. ball centre
        r_w = float(np.linalg.norm(cw - c_ball) - r)                    # effective wheel radius at contact
        # rim velocity of the wheel at the contact = omega_w * (a x (p_c - c_w)); it must match the ball surface
        # velocity (Omega x p) along that driven direction. Rollers absorb the other tangent component.
        rim = np.cross(a, (c_ball + p) - cw)                            # = r_w * t_i
        t = rim / np.linalg.norm(rim)
        W.append(np.cross(p, t) / r_w)     # (Omega x p).t / r_w = Omega . (p x t) / r_w
        contacts.append(dict(p=p.tolist(), t=t.tolist(), axis=a.tolist(), r_w=r_w))
    W = np.array(W)
    J_w = float(m.dof_armature[m.joint("wheel_joint_0").dofadr[0]]) if armature is None else float(armature)
    dmp = float(m.dof_damping[m.joint("wheel_joint_0").dofadr[0]]) if wheel_damping is None else float(wheel_damping)
    zenith = float(np.degrees(np.arccos(np.clip(np.array(contacts[0]["p"])[2] / r, -1, 1))))
    return dict(r=r, m_k=m_k, I_k=I_k, m_a=m_a, l=l, I_a=float(I_a), W=W.tolist(), J_w=J_w, d_w=dmp,
                contacts=contacts, zenith_deg=zenith, tau_max_xml=float(m.actuator_ctrlrange[0, 1]))


# ----------------------------------------------------------------------------------------------
# planar Lagrangian
# ----------------------------------------------------------------------------------------------
def derive_planar():
    t = sp.symbols("t")
    r, m_k, I_k, m_a, l, I_a, J, g, dw = sp.symbols("r m_k I_k m_a l I_a J_eff g d_eff", positive=True)
    tau = sp.symbols("tau")
    phi, th = sp.Function("phi")(t), sp.Function("theta")(t)
    x = r * phi
    xc, zc = x + l * sp.sin(th), l * sp.cos(th)
    T = (sp.Rational(1, 2) * m_k * sp.diff(x, t) ** 2 + sp.Rational(1, 2) * I_k * sp.diff(phi, t) ** 2
         + sp.Rational(1, 2) * m_a * (sp.diff(xc, t) ** 2 + sp.diff(zc, t) ** 2)
         + sp.Rational(1, 2) * I_a * sp.diff(th, t) ** 2
         + sp.Rational(1, 2) * J * (sp.diff(phi, t) - sp.diff(th, t)) ** 2)
    V = m_a * g * zc
    L = T - V
    q = [phi, th]
    rel = sp.diff(phi, t) - sp.diff(th, t)
    Q = [tau - dw * rel, -(tau - dw * rel)]          # motor torque + wheel viscous damping act on phi - theta
    eqs = [sp.simplify(sp.diff(sp.diff(L, sp.diff(qi, t)), t) - sp.diff(L, qi) - Qi) for qi, Qi in zip(q, Q)]
    qdd = [sp.diff(qi, t, 2) for qi in q]
    Mq = sp.Matrix([[sp.expand(e).coeff(a) for a in qdd] for e in eqs])
    rest = sp.Matrix([sp.simplify(e - sum(Mq[i, k] * qdd[k] for k in range(2))) for i, e in enumerate(eqs)])
    # rest = C(q,qd) qd + G(q) - Q_tau ; split out gravity
    Gq = sp.Matrix([sp.diff(V, qi) for qi in q])
    syms = dict(r=r, m_k=m_k, I_k=I_k, m_a=m_a, l=l, I_a=I_a, J_eff=J, g=g, d_eff=dw, tau=tau)
    return dict(M=sp.simplify(Mq), rest=rest, G=Gq, eqs=eqs, q=q, syms=syms, t=t)


def linearize(der, P):
    """State s = (phi, theta, phidot, thetadot), input tau -> (A, B) at the upright equilibrium."""
    S = der["syms"]
    t = der["t"]
    phi, th = der["q"]
    a0, a1, a2, a3, u = sp.symbols("s0 s1 s2 s3 u")
    sub_state = {sp.diff(phi, t, 2): sp.Symbol("pdd"), sp.diff(th, t, 2): sp.Symbol("tdd")}
    eqs = [e.subs(sub_state) for e in der["eqs"]]
    rep = {sp.diff(phi, t): a2, sp.diff(th, t): a3}
    eqs = [e.subs(rep).subs({phi: a0, th: a1, S["tau"]: u}) for e in eqs]
    sol = sp.solve(eqs, [sp.Symbol("pdd"), sp.Symbol("tdd")], dict=True)[0]
    f = sp.Matrix([a2, a3, sol[sp.Symbol("pdd")], sol[sp.Symbol("tdd")]])
    s = sp.Matrix([a0, a1, a2, a3])
    num = {S["r"]: P["r"], S["m_k"]: P["m_k"], S["I_k"]: P["I_k"], S["m_a"]: P["m_a"], S["l"]: P["l"],
           S["I_a"]: P["I_a"], S["J_eff"]: P["J_eff"], S["g"]: G, S["d_eff"]: P["d_eff"]}
    at0 = {a0: 0, a1: 0, a2: 0, a3: 0, u: 0}
    A = np.array(f.jacobian(s).subs(at0).subs(num), dtype=float)
    B = np.array(f.jacobian(sp.Matrix([u])).subs(at0).subs(num), dtype=float)
    return A, B, f, s, u, num


def planar_effective(P):
    """Reduce the 3-wheel transmission to one vertical plane (pitch about world y)."""
    W = np.array(P["W"])
    # planar relative rotation Omega = (0, w, 0): wheel rates = W[:, 1] w ; J_eff = J_w * sum n_i^2
    n = W[:, 1]
    J_eff = P["J_w"] * float(n @ n)
    d_eff = P["d_w"] * float(n @ n)
    # planar torque on the ball from wheel torques tau_w (virtual power): tau = n . tau_w.
    # minimum-norm wheel torques for a planar torque tau: tau_w = n tau / |n|^2 -> |tau_w|_inf = |tau| max|n_i|/|n|^2
    tau_plane_max = lambda tau_w_max: tau_w_max * float(n @ n) / float(np.max(np.abs(n)))
    return n, J_eff, d_eff, tau_plane_max


def structural(A, B, P):
    """Unstable pole, x-output zero (non-minimum phase), lean-acceleration law."""
    poles = np.linalg.eigvals(A)
    # transfer function tau -> ball position x = r phi : C = [r,0,0,0]; zeros from det([[A - zI, B],[C, 0]])
    C = np.array([[P["r"], 0, 0, 0]])
    import scipy.signal as sig
    num, den = sig.ss2tf(A, B, C, np.zeros((1, 1)))
    zeros = np.roots(np.trim_zeros(num[0], "f"))
    # steady acceleration a (phidd = a/r, thetadd = 0) with constant lean theta*: from the theta equation
    #   (m_a l r cos th) phidd - m_a g l sin th = -tau ... solved jointly with the phi equation (linear):
    m_a, l, r, m_k, I_k, I_a, J = P["m_a"], P["l"], P["r"], P["m_k"], P["I_k"], P["I_a"], P["J_eff"]
    # sum of the two Lagrange equations eliminates tau:  ((m_k+m_a) r^2 + I_k + m_a l r) phidd = m_a g l theta (lin.)
    k_lean = ((m_k + m_a) * r ** 2 + I_k + m_a * l * r) / (m_a * g_l(l) * r)
    return dict(poles=poles, zeros=zeros, lean_per_accel_rad=k_lean)


def g_l(l):
    return G * l


def lqr(A, B, Q, R):
    import scipy.linalg as la
    X = la.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ X)
    return K, np.linalg.eigvals(A - B @ K)


# ----------------------------------------------------------------------------------------------
# validation against MuJoCo
# ----------------------------------------------------------------------------------------------
def mujoco_validation(P, J_eff, d_eff, n, wheel_damping):
    """Two checks of the derived model on the full simulator (contact-frame stepper, implicitfast at 500 Hz):
    (1) zero-torque fall from a small tilt -> fitted exponential rate vs the model's unstable pole;
    (2) initial accelerations under a planar torque step vs M^{-1} B."""
    from ballbot_gym.utils.contact_frame import ContactFrameStepper
    m = mujoco.MjModel.from_xml_path(str(XML))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if wheel_damping is not None:
        for k in range(3):
            m.dof_damping[m.joint(f"wheel_joint_{k}").dofadr[0]] = wheel_damping
    m.actuator_ctrlrange[:] = [-100, 100]
    st = ContactFrameStepper(m)
    d = mujoco.MjData(m)

    def settle():
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        gap = d.geom_xpos[m.geom("the_ball").id][2] - m.geom_size[m.geom("the_ball").id][0]  # ball bottom above floor
        d.qpos[m.joint("base_free_joint").qposadr[0] + 2] -= gap + 3e-4
        d.qpos[m.joint("ball_free_joint").qposadr[0] + 2] -= gap + 3e-4
        for _ in range(200):   # let contacts settle with the body held upright
            st.step(d)
            d.qvel[:] = 0
            bq = m.joint("base_free_joint").qposadr[0]
            d.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0

    out = {}
    # (1) fall from 1 deg tilt about world y, zero torque
    settle()
    bq = m.joint("base_free_joint").qposadr[0]
    th0 = np.radians(1.0)
    d.qpos[bq + 3:bq + 7] = [np.cos(th0 / 2), 0, np.sin(th0 / 2), 0]
    ts, ths = [], []
    for i in range(150):
        st.step(d)
        w, x, y, z = d.qpos[bq + 3:bq + 7]
        ths.append(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        ts.append((i + 1) * m.opt.timestep)
    ts, ths = np.array(ts), np.array(ths)
    ths = np.abs(ths)
    sel = (ths > np.radians(1.5)) & (ths < np.radians(8))
    rate = float(np.polyfit(ts[sel], np.log(ths[sel]), 1)[0]) if sel.sum() > 5 else float("nan")
    out["fall_rate_sim"] = rate
    # (2) planar torque step: wheel torques tau_w = n tau / |n|^2 (planar pitch torque tau)
    res = {}
    for tau in (0.5, -0.5):
        settle()
        tw = n * tau / float(n @ n)
        d.ctrl[:] = -tw          # the env applies ctrl = -command (wheel joint sign convention)
        st.forward(d)
        bd = m.joint("base_free_joint").dofadr[0]
        kd = m.joint("ball_free_joint").dofadr[0]
        res[tau] = dict(theta_dd=float(d.qacc[bd + 4]), ball_wy=float(d.qacc[kd + 4]), ball_ax=float(d.qacc[kd + 0]))
    out["torque_step"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel-damping", type=float, default=0.1)
    ap.add_argument("--armature", type=float, default=None, help="override wheel rotor inertia (model only)")
    ap.add_argument("--json")
    a = ap.parse_args()
    P = extract_params(a.wheel_damping, a.armature)
    n, J_eff, d_eff, tau_plane_max = planar_effective(P)
    P.update(J_eff=J_eff, d_eff=d_eff)
    print("== parameters")
    for k in ("r", "m_k", "I_k", "m_a", "l", "I_a", "J_w", "d_w", "zenith_deg"):
        print(f"  {k:10s} {P[k]:.5g}")
    print("  W (wheel rate per unit ball-relative angular velocity):\n", np.round(np.array(P["W"]), 3))
    print(f"  planar n = {np.round(n, 3)}, J_eff = J_w |n|^2 = {J_eff:.4f} kg m^2 (ball I_k = {P['I_k']:.5f})")
    der = derive_planar()
    print("== Lagrangian: M(q) =")
    sp.pprint(der["M"])
    print("   G(q) =")
    sp.pprint(der["G"])
    A, B, f, s, u, num = linearize(der, P)
    print("== linearised A =\n", np.round(A, 3), "\n   B =", np.round(B.ravel(), 3))
    S = structural(A, B, P)
    print("== open-loop poles", np.round(S["poles"], 3), " (unstable pole -> fall time constant",
          f"{1 / max(S['poles'].real):.3f} s)")
    print("== zeros of tau -> x:", np.round(S["zeros"], 3), "(positive real zero => non-minimum phase)")
    print(f"== steady lean per unit acceleration: {S['lean_per_accel_rad']:.4f} rad per m/s^2 "
          f"= {np.degrees(S['lean_per_accel_rad']):.2f} deg per m/s^2")
    for tw in (10.0, 30.0):
        tp = tau_plane_max(tw)
        print(f"   wheel torque limit {tw:.0f} N m -> planar torque {tp:.1f} N m")
    V = mujoco_validation(P, J_eff, d_eff, n, a.wheel_damping)
    lam_model = float(max(S["poles"].real))
    print(f"== validation: fall rate model {lam_model:.2f} 1/s, MuJoCo {V['fall_rate_sim']:.2f} 1/s")
    Minv_B = np.linalg.solve(np.array([[1, 0], [0, 1]]), np.array([B[2, 0], B[3, 0]]))
    for tau, rr in V["torque_step"].items():
        print(f"   tau={tau:+.1f}: model phidd={B[2, 0] * tau:+.3f} thetadd={B[3, 0] * tau:+.3f} | "
              f"MuJoCo ball wy_dd={rr['ball_wy']:+.3f} theta_dd={rr['theta_dd']:+.3f}")
    # LQR on (phi, theta, phidot, thetadot); position weight small (velocity tracking via reference shift)
    Q = np.diag([0.0, 50.0, 1.0, 1.0])
    R = np.array([[1.0]])
    Kq = None
    try:
        K, cl = lqr(A, B, Q + np.diag([1e-6, 0, 0, 0]), R)
        Kq = K
        print("== LQR K =", np.round(K, 3), "closed-loop poles", np.round(cl, 3))
    except Exception as e:  # noqa: BLE001
        print("LQR failed:", e)
    if a.json:
        outp = dict(params={k: v for k, v in P.items() if k not in ("contacts",)}, contacts=P["contacts"],
                    n=n.tolist(), A=A.tolist(), B=B.tolist(), poles=[complex(p).real for p in S["poles"]],
                    zeros=[complex(z).real for z in S["zeros"]], lean_per_accel=S["lean_per_accel_rad"],
                    K=None if Kq is None else Kq.tolist(), validation=V,
                    M_latex=sp.latex(der["M"]), G_latex=sp.latex(der["G"]))
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(outp, open(a.json, "w"), indent=1, default=str)
        print("saved", a.json)


if __name__ == "__main__":
    main()
