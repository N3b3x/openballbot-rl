// Planar ballbot physics for the course widgets: the model of scripts/study/ballbot_model.py and the deployed LQR law of
// ros2/ballbot_control/ballbot_control/core/lqr.py (one sagittal plane), in plain JavaScript.
//
// Coordinates q = (phi, theta): absolute ball rotation and absolute body tilt (theta > 0 leans toward +x).
// Ball position x = r phi (rolling without slip); body centre of mass at (x + l sin theta, l cos theta).
// Motor torque tau acts between body and ball; wheel damping d acts on the relative rate phi' - theta'.
//   M(theta) q'' = [tau - d(phi'-theta') + m_a l r sin(theta) theta'^2 + F r,  -(tau - d(phi'-theta')) + m_a g l sin(theta) + F l cos(theta)]
//   M = [[I_k + J + (m_a + m_k) r^2,  m_a l r cos(theta) - J], [m_a l r cos(theta) - J,  I_a + J + m_a l^2]]
// F is a horizontal force on the body centre of mass (a push). All parameters come from site/learn/data/learn.json.
export const G = 9.81;

export function paramsFrom(model) {
  const p = model.params;
  return { r: p.r, m_k: p.m_k, I_k: p.I_k, m_a: p.m_a, l: p.l, I_a: p.I_a, J: p.J_eff, d: p.d_eff };
}

export function accel(P, s, tau, F = 0) {
  const th = s[1], phd = s[2], thd = s[3];
  const c = Math.cos(th), sn = Math.sin(th);
  const a = P.I_k + P.J + (P.m_a + P.m_k) * P.r * P.r;
  const b = P.m_a * P.l * P.r * c - P.J;
  const dd = P.I_a + P.J + P.m_a * P.l * P.l;
  const tq = tau - P.d * (phd - thd);
  const f1 = tq + P.m_a * P.l * P.r * sn * thd * thd + F * P.r;
  const f2 = -tq + P.m_a * G * P.l * sn + F * P.l * c;
  const det = a * dd - b * b;
  return [(dd * f1 - b * f2) / det, (a * f2 - b * f1) / det];
}

/** Linearisation at the upright equilibrium: state (phi, theta, phi', theta'), input tau. */
export function linearize(P) {
  const a = P.I_k + P.J + (P.m_a + P.m_k) * P.r * P.r, b = P.m_a * P.l * P.r - P.J, dd = P.I_a + P.J + P.m_a * P.l * P.l;
  const det = a * dd - b * b;
  const Mi = [[dd / det, -b / det], [-b / det, a / det]];
  const g = P.m_a * G * P.l;
  const A = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 2; i++) {
    A[2 + i][1] = Mi[i][1] * g;
    A[2 + i][2] = -P.d * (Mi[i][0] - Mi[i][1]);
    A[2 + i][3] = P.d * (Mi[i][0] - Mi[i][1]);
  }
  const B = [0, 0, Mi[0][0] - Mi[0][1], Mi[1][0] - Mi[1][1]];
  return { A, B };
}
export const zeroNMP = (P) => Math.sqrt((P.m_a * G * P.l) / (P.I_a + P.m_a * P.l * P.l + P.m_a * P.l * P.r));
export const leanPerAccel = (P) => ((P.m_k + P.m_a) * P.r * P.r + P.I_k + P.m_a * P.l * P.r) / (P.m_a * G * P.l * P.r);
/** Rate of a plain inverted pendulum on a fixed pivot with the same body (for comparison). */
export const pivotPole = (P) => Math.sqrt((P.m_a * G * P.l) / (P.I_a + P.m_a * P.l * P.l));

// ---------------------------------------------------------------- small linear algebra
const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const madd = (A, B, c = 1) => A.map((r, i) => r.map((v, j) => v + c * B[i][j]));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));

export function expm(A) {
  const n = A.length;
  const nrm = Math.max(...A.map((r) => r.reduce((s, v) => s + Math.abs(v), 0)));
  const sq = Math.max(0, Math.ceil(Math.log2(nrm || 1)) + 1);
  const As = A.map((r) => r.map((v) => v / 2 ** sq));
  let E = eye(n), term = eye(n);
  for (let k = 1; k < 18; k++) { term = mm(term, As).map((r) => r.map((v) => v / k)); E = madd(E, term); }
  for (let i = 0; i < sq; i++) E = mm(E, E);
  return E;
}
/** Zero-order-hold discretisation of (A, B) at dt (B a column as a plain array). */
export function c2d(A, B, dt) {
  const n = A.length;
  const Mx = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i < n ? (j < n ? A[i][j] : B[i]) : 0) * dt));
  const E = expm(Mx);
  return { Ad: E.slice(0, n).map((r) => r.slice(0, n)), Bd: E.slice(0, n).map((r) => r[n]) };
}
/** Discrete LQR for a single input, as lqr_gpu.py design_tables: Q * dt, R * dt, Riccati iteration. Returns K (array). */
export function dlqr(A, B, q, r, dt = 0.01) {
  const { Ad, Bd } = c2d(A, B, dt);
  const n = Ad.length;
  const Q = q.map((v, i) => q.map((_, j) => (i === j ? v * dt : 0)));
  const R = r * dt;
  let X = Q.map((row) => row.slice());
  const AT = T(Ad);
  let K = new Array(n).fill(0);
  for (let it = 0; it < 20000; it++) {
    const XB = X.map((row) => row.reduce((s, v, k) => s + v * Bd[k], 0));            // X B
    const s = R + Bd.reduce((acc, v, i) => acc + v * XB[i], 0);                       // R + B'XB
    const BXA = Ad[0].map((_, j) => XB.reduce((acc, v, k) => acc + v * Ad[k][j], 0)); // B'XA
    K = BXA.map((v) => v / s);
    const XA = mm(X, Ad);
    const Xn = madd(madd(Q, mm(AT, XA)), BXA.map((vi) => BXA.map((vj) => vi * vj)), -1 / s);
    let diff = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) diff = Math.max(diff, Math.abs(Xn[i][j] - X[i][j]));
    X = Xn;
    if (diff < 1e-10 * (1 + Math.abs(X[1][1]))) break;
  }
  return K;
}

// ---------------------------------------------------------------- eigenvalues (characteristic polynomial + Durand-Kerner)
function charPoly(A) {           // Faddeev-LeVerrier: coefficients c0 = 1, c1, ..., cn of det(sI - A)
  const n = A.length;
  let M = eye(n).map((r) => r.map(() => 0));
  const c = [1];
  for (let k = 1; k <= n; k++) {
    M = madd(mm(A, M), eye(n), c[k - 1]);
    const AM = mm(A, M);
    let tr = 0; for (let i = 0; i < n; i++) tr += AM[i][i];
    c.push(-tr / k);
  }
  return c;
}
export function roots(c) {       // c[0] s^n + ... + c[n]; returns [[re, im], ...]
  const n = c.length - 1;
  const a = c.map((v) => v / c[0]);
  let z = Array.from({ length: n }, (_, k) => [Math.cos((2 * Math.PI * k) / n + 0.4) * 2, Math.sin((2 * Math.PI * k) / n + 0.4) * 2]);
  const mul = (p, q) => [p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]];
  const div = (p, q) => { const d = q[0] * q[0] + q[1] * q[1] || 1e-300; return [(p[0] * q[0] + p[1] * q[1]) / d, (p[1] * q[0] - p[0] * q[1]) / d]; };
  const scale = Math.max(1, ...a.slice(1).map((v, i) => Math.abs(v) ** (1 / (i + 1))));
  z = z.map((p) => [p[0] * scale, p[1] * scale]);
  for (let it = 0; it < 600; it++) {
    let moved = 0;
    z = z.map((zi, i) => {
      let num = [1, 0];
      for (let k = 1; k <= n; k++) num = [mul(num, zi)[0] + a[k], mul(num, zi)[1]];
      let den = [1, 0];
      z.forEach((zj, j) => { if (j !== i) den = mul(den, [zi[0] - zj[0], zi[1] - zj[1]]); });
      const dz = div(num, den);
      moved = Math.max(moved, Math.hypot(dz[0], dz[1]));
      return [zi[0] - dz[0], zi[1] - dz[1]];
    });
    if (moved < 1e-12 * scale) break;
  }
  return z.map(([re, im]) => [re, Math.abs(im) < 1e-7 * scale ? 0 : im]);
}
export const eig = (A) => roots(charPoly(A));
/** Closed-loop poles of A - B K (continuous model, for display). */
export function closedLoopPoles(A, B, K) {
  const Acl = A.map((r, i) => r.map((v, j) => v - B[i] * K[j]));
  return eig(Acl);
}

// ---------------------------------------------------------------- simulation
function rk4(P, s, tau, F, h) {
  const f = (x) => { const [a1, a2] = accel(P, x, tau, F); return [x[2], x[3], a1, a2]; };
  const k1 = f(s);
  const k2 = f(s.map((v, i) => v + 0.5 * h * k1[i]));
  const k3 = f(s.map((v, i) => v + 0.5 * h * k2[i]));
  const k4 = f(s.map((v, i) => v + h * k3[i]));
  return s.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/**
 * One sagittal plane of the deployed controller on the nonlinear planar model.
 * Control law per 10 ms tick (core/lqr.py): vref slews toward the command at a_max; int_v += (vref - v_com) dt (clipped);
 * phi'_ref = (vref + ki int_v) / r; tau = -K . [0, theta, phi' - phi'_ref, theta'] (zero-order hold, physics at 1 ms, RK4).
 * K = null runs the robot with the motors off.
 */
export class PlanarBot {
  constructor(P, o = {}) {
    this.P = P;
    this.K = o.K === undefined ? null : o.K;
    this.ki = o.ki ?? 1; this.a_max = o.a_max ?? 0.5; this.dt = o.dt ?? 0.01; this.int_clip = o.int_clip ?? 2;
    this.tauMax = o.tauMax ?? Infinity;
    this.sub = o.sub ?? 10;
    this.reset(o.theta0 || 0);
  }
  reset(theta0 = 0) {
    this.s = [0, theta0, 0, 0]; this.vref = 0; this.int_v = 0; this.t = 0; this.tau = 0; this.first = true; this.fallen = false;
  }
  get x() { return this.P.r * this.s[0]; }
  get theta() { return this.s[1]; }
  get vcom() { return this.P.r * this.s[2] + this.P.l * Math.cos(this.s[1]) * this.s[3]; }
  get vball() { return this.P.r * this.s[2]; }
  control(cmd) {
    if (!this.K) return 0;
    const dv = cmd - this.vref;
    const lim = this.a_max * this.dt;
    this.vref += Math.max(-lim, Math.min(lim, dv));
    const vm = this.first ? 0 : this.vcom;
    this.first = false;
    this.int_v = Math.max(-this.int_clip, Math.min(this.int_clip, this.int_v + (this.vref - vm) * this.dt));
    const phdRef = (this.vref + this.ki * this.int_v) / this.P.r;
    const st = [0, this.s[1], this.s[2] - phdRef, this.s[3]];
    let tau = -(this.K[0] * st[0] + this.K[1] * st[1] + this.K[2] * st[2] + this.K[3] * st[3]);
    return Math.max(-this.tauMax, Math.min(this.tauMax, tau));
  }
  /** Advance one control tick. cmd: commanded velocity (m/s); F: push force on the body (N) during this tick; tauExtra adds to tau. */
  tick(cmd = 0, F = 0, tauExtra = 0) {
    if (this.fallen) { this.t += this.dt; return; }
    this.tau = this.control(cmd) + tauExtra;
    if (Number.isFinite(this.tauMax)) this.tau = Math.max(-this.tauMax, Math.min(this.tauMax, this.tau));
    const h = this.dt / this.sub;
    for (let i = 0; i < this.sub; i++) this.s = rk4(this.P, this.s, this.tau, F, h);
    this.t += this.dt;
    if (Math.abs(this.s[1]) > (75 * Math.PI) / 180) { this.fallen = true; this.s[1] = Math.sign(this.s[1]) * (75 * Math.PI) / 180; this.s[3] = 0; this.s[2] = 0; }
  }
}

/** Planar torque available from per-wheel limit tau_w (ballbot_model.py planar_effective): tau_w |n|^2 / max|n_i|. */
export function planeTorqueMax(n, tauW) {
  const nn = n.reduce((s, v) => s + v * v, 0);
  return (tauW * nn) / Math.max(...n.map(Math.abs));
}

/** Run a velocity step on the closed loop; returns arrays t, x, theta, v, tau. */
export function stepResponse(P, K, { cmd = 0.5, T = 4, a_max = 0.5, ki = 1, tStep = 0.3, tauMax = Infinity } = {}) {
  const bot = new PlanarBot(P, { K, a_max, ki, tauMax });
  const out = { t: [], x: [], th: [], v: [], tau: [], vref: [] };
  const n = Math.round(T / bot.dt);
  for (let i = 0; i <= n; i++) {
    out.t.push(bot.t); out.x.push(bot.x); out.th.push(bot.theta); out.v.push(bot.vball); out.tau.push(bot.tau); out.vref.push(bot.vref);
    bot.tick(bot.t >= tStep ? cmd : 0);
  }
  return out;
}
