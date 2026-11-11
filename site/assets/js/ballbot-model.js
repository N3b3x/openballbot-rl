// Planar ballbot model, a faithful JavaScript port of scripts/study/ballbot_model.py, design_space.py and
// design_constraints.py (verified against design_sweep.csv / design_constraints.csv on math.html).
//
// Coordinates q = (phi, theta): absolute ball rotation angle and absolute body tilt (theta > 0 leans toward +x).
// Ball position x = r phi (rolling without slip); body COM at (x + l sin(theta), l cos(theta)).
// Motor torque tau acts between body and ball (equal and opposite); wheel damping acts on the relative rate phi' - theta'.
//
//   T = 1/2 m_k (r phi')^2 + 1/2 I_k phi'^2 + 1/2 m_a |d/dt (x + l sin th, l cos th)|^2 + 1/2 I_a th'^2 + 1/2 J (phi' - th')^2
//   V = m_a g l cos(th)
//   Euler-Lagrange:  M(q) q'' + C(q,q') + dV/dq = Q,   Q = [tau - d(phi'-th'), -(tau - d(phi'-th'))]
//   M = [[I_k + J + (m_a+m_k) r^2,      m_a l r cos th - J],
//        [m_a l r cos th - J,           I_a + J + m_a l^2 ]]
//   C = [ -m_a l r sin(th) th'^2 , 0 ]        dV/dq = [0, -m_a g l sin th]
export const G = 9.81;
export const NOMINAL = { s: 1, kball: 1, kbal: 1, zb: 0.002, J: 0.005, tau: 30, damp: 0.1, dz: 0 };
export const DEFAULTS = { mu: 1, tilt_rec_deg: 15, a_target: 1.5, v_target: 1, T_loop: 0.015, omega_max: 150, Ms_ref: 20 };

// ------------------------------------------------------------------------------------------------
// small dense linear algebra
// ------------------------------------------------------------------------------------------------
const zeros = (n, m = n) => Array.from({ length: n }, () => new Array(m).fill(0));
const eye = (n) => { const I = zeros(n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; };
const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const matvec = (A, x) => A.map((row) => row.reduce((s, a, k) => s + a * x[k], 0));
const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const add = (A, B, c = 1) => A.map((r, i) => r.map((v, j) => v + c * B[i][j]));
const scale = (A, c) => A.map((r) => r.map((v) => v * c));

/** Solve A X = B (Gaussian elimination with partial pivoting). B is a matrix; returns X. */
export function solve(A, B) {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    if (Math.abs(d) < 1e-300) throw new Error('singular matrix');
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f !== 0) for (let k = c; k < M[r].length; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r.slice(n).map((v) => v / M[i][i]));
}

/** Complex solve (A - sI) y = b for a real A and complex s; arrays [re, im]. Returns y as [[re,im],...]. */
function csolveShifted(A, s, b) {
  const n = A.length;
  const M = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) row.push([A[i][j] - (i === j ? s[0] : 0), i === j ? -s[1] : 0]);
    row.push([b[i], 0]);
    M.push(row);
  }
  const cmul = (a, c) => [a[0] * c[0] - a[1] * c[1], a[0] * c[1] + a[1] * c[0]];
  const cdiv = (a, c) => { const d = c[0] * c[0] + c[1] * c[1]; return [(a[0] * c[0] + a[1] * c[1]) / d, (a[1] * c[0] - a[0] * c[1]) / d]; };
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.hypot(...M[r][c]) > Math.hypot(...M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = cdiv(M[r][c], M[c][c]);
      for (let k = c; k <= n; k++) { const t = cmul(f, M[c][k]); M[r][k] = [M[r][k][0] - t[0], M[r][k][1] - t[1]]; }
    }
  }
  return M.map((r, i) => cdiv(r[n], r[i]));
}

/** Characteristic polynomial coefficients [1, c1, ..., cn] of A (Faddeev-LeVerrier). */
function charpoly(A) {
  const n = A.length;
  let Mk = zeros(n);
  const c = [1];
  for (let k = 1; k <= n; k++) {
    Mk = k === 1 ? eye(n) : add(matmul(A, Mk), eye(n), c[k - 1]);
    const AMk = matmul(A, Mk);
    c.push(-AMk.reduce((s, r, i) => s + r[i], 0) / k);
  }
  return c;
}

/** Roots of a real polynomial (descending coefficients) by Durand-Kerner + Newton polishing; returns [[re, im], ...]. */
export function polyroots(co) {
  while (co.length > 1 && Math.abs(co[0]) < 1e-300) co = co.slice(1);
  const n = co.length - 1;
  if (n < 1) return [];
  const a = co.map((v) => v / co[0]);
  const R = 1 + Math.max(...a.slice(1).map(Math.abs));
  let z = Array.from({ length: n }, (_, k) => [R * 0.5 * Math.cos(2 * Math.PI * k / n + 0.4), R * 0.5 * Math.sin(2 * Math.PI * k / n + 0.4)]);
  const cmul = (p, q) => [p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]];
  const peval = (x) => { let r = [a[0], 0]; for (let i = 1; i <= n; i++) { r = cmul(r, x); r[0] += a[i]; } return r; };
  for (let it = 0; it < 500; it++) {
    let md = 0;
    const nz = z.map((zi, i) => {
      let den = [1, 0];
      for (let j = 0; j < n; j++) if (j !== i) den = cmul(den, [zi[0] - z[j][0], zi[1] - z[j][1]]);
      const num = peval(zi);
      const d2 = den[0] * den[0] + den[1] * den[1] || 1e-300;
      const q = [(num[0] * den[0] + num[1] * den[1]) / d2, (num[1] * den[0] - num[0] * den[1]) / d2];
      md = Math.max(md, Math.hypot(q[0], q[1]));
      return [zi[0] - q[0], zi[1] - q[1]];
    });
    z = nz;
    if (md < 1e-15 * R) break;
  }
  // snap numerically-real roots
  return z.map(([re, im]) => (Math.abs(im) < 1e-9 * (1 + Math.abs(re)) ? [re, 0] : [re, im]));
}

export function eigenvalues(A) {
  const r = polyroots(charpoly(A));
  r.sort((p, q) => q[0] - p[0] || q[1] - p[1]);
  return r;
}

export function expm(A) {
  const n = A.length;
  const nrm = Math.max(...A.map((r) => r.reduce((s, v) => s + Math.abs(v), 0)));
  const sq = Math.max(0, Math.ceil(Math.log2(Math.max(nrm, 1e-12))) + 2);
  const As = scale(A, 1 / 2 ** sq);
  let E = eye(n), T = eye(n);
  for (let k = 1; k <= 20; k++) { T = scale(matmul(T, As), 1 / k); E = add(E, T); }
  for (let i = 0; i < sq; i++) E = matmul(E, E);
  return E;
}

// ------------------------------------------------------------------------------------------------
// parameters from a design
// ------------------------------------------------------------------------------------------------
/** Linear interpolation of the geometry-table rows over dz (the only knob that changes the wheel geometry). */
function interpRow(geo, dz) {
  const rows = geo.rows;
  const lo = rows[0].dz, hi = rows[rows.length - 1].dz;
  const x = Math.min(hi, Math.max(lo, dz));
  let i = rows.findIndex((r) => r.dz >= x);
  if (i <= 0) i = 1;
  const a = rows[i - 1], b = rows[i];
  const t = (x - a.dz) / (b.dz - a.dz);
  const mix = (u, v) => u + t * (v - u);
  const out = {};
  for (const k of Object.keys(a)) {
    if (k === 'S0') out.S0 = a.S0.map((u, j) => mix(u, b.S0[j]));
    else if (typeof a[k] === 'number') out[k] = mix(a[k], b[k]);
  }
  out.dz = x;
  return out;
}

/** Planar model parameters P for a design {s,kball,kbal,zb,J,tau,damp,dz}; geo = site/data/geometry_table.json. */
export function paramsFromDesign(design, geo) {
  const d = { ...NOMINAL, ...design };
  const row = interpRow(geo, d.dz);
  const zbShift = d.zb - geo.zb_nominal;
  const hb = row.hb + zbShift;
  const m_a1 = row.S0[0] + d.kbal * row.mb;
  const S1 = row.S0[1] + d.kbal * row.mb * hb;
  const S2 = row.S0[2] + d.kbal * (row.Ib + row.mb * hb * hb);
  const l1 = S1 / m_a1;
  const I_a1 = S2 - m_a1 * l1 * l1;
  const s = d.s;
  const P = {
    r: row.r * s, m_k: row.m_k * d.kball * s ** 3, I_k: row.I_k * d.kball * s ** 5,
    m_a: m_a1 * s ** 3, l: l1 * s, I_a: I_a1 * s ** 5,
    J_eff: d.J * s ** 5 * row.nn, d_eff: d.damp * s ** 2.5 * row.nn,
    zeta_deg: row.zeta_deg, r_w: row.r_w * s, fac: row.fac, gain: row.gain, nn: row.nn,
    tau_w_max: d.tau * s ** 4,
  };
  P.tau_plane_max = P.tau_w_max * P.fac;
  return P;
}

export function nominalParams(geo) { return paramsFromDesign({}, geo); }

// ------------------------------------------------------------------------------------------------
// Lagrangian dynamics
// ------------------------------------------------------------------------------------------------
export function massMatrix(P, th) {
  const { r, m_k, I_k, m_a, l, I_a, J_eff: J } = P;
  const M12 = m_a * l * r * Math.cos(th) - J;
  return [[I_k + J + (m_a + m_k) * r * r, M12], [M12, I_a + J + m_a * l * l]];
}
export const gravityVector = (P, th) => [0, -P.m_a * G * P.l * Math.sin(th)];   // dV/dq

/** Accelerations (phi'', theta'') for state (phi, th, phid, thd), motor torque tau and horizontal push force F on the body COM. */
export function accel(P, th, phid, thd, tau, F = 0) {
  const M = massMatrix(P, th);
  const te = tau - P.d_eff * (phid - thd);
  const dM12 = -P.m_a * P.l * P.r * Math.sin(th);
  const rhs1 = te + F * P.r - dM12 * thd * thd;
  const rhs2 = -te + F * P.l * Math.cos(th) + P.m_a * G * P.l * Math.sin(th);
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [(rhs1 * M[1][1] - M[0][1] * rhs2) / det, (M[0][0] * rhs2 - M[1][0] * rhs1) / det];
}

/** Linearisation at the upright equilibrium; state s = (phi, theta, phid, thetad), input tau. */
export function linearize(P) {
  const M = massMatrix(P, 0);
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const Mi = [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];
  const mgl = P.m_a * G * P.l, d = P.d_eff;
  // q'' = Mi ( [0, mgl theta] + [1,-1] (tau - d (phid - thd)) )
  const A = zeros(4);
  A[0][2] = 1; A[1][3] = 1;
  for (let i = 0; i < 2; i++) {
    A[2 + i][1] = Mi[i][1] * mgl;
    const u = Mi[i][0] - Mi[i][1];                     // Mi [1,-1]
    A[2 + i][2] = -d * u;
    A[2 + i][3] = d * u;
  }
  const B = [[0], [0], [Mi[0][0] - Mi[0][1]], [Mi[1][0] - Mi[1][1]]];
  return { A, B };
}

// ------------------------------------------------------------------------------------------------
// structural properties
// ------------------------------------------------------------------------------------------------
/** Closed-form NMP zero of tau -> ball position (Proposition 2 of the paper). */
export const zeroNMP = (P) => Math.sqrt(P.m_a * G * P.l / (P.I_a + P.m_a * P.l * P.l + P.m_a * P.l * P.r));
/** Undamped unstable pole lambda = sqrt(m_a g l M11 / det M) (Proposition 1). */
export function lambdaUndamped(P) {
  const M = massMatrix(P, 0);
  return Math.sqrt(P.m_a * G * P.l * M[0][0] / (M[0][0] * M[1][1] - M[0][1] * M[0][1]));
}
/** Lean-acceleration law: sin(theta) = kappa a (Proposition 3), kappa in rad per m/s^2. */
export const kappaLean = (P) => ((P.m_k + P.m_a) * P.r * P.r + P.I_k + P.m_a * P.l * P.r) / (P.m_a * G * P.l * P.r);
/** Skogestad lower bound on peak sensitivity for one RHP pole p and one RHP zero z, with loop delay T. */
export const skogestad = (z, p, T = 0) => (z + p) / Math.abs(z - p) * Math.exp(p * T);

export function poles(A) { return eigenvalues(A); }

/** Transfer function tau -> x = r phi at a complex point s = sr + j si (returns [re, im]). */
export function tfAt(P, lin, sr, si = 0) {
  const y = csolveShifted(lin.A, [sr, si], lin.B.map((b) => b[0]));
  return [-P.r * y[0][0], -P.r * y[0][1]];
}

/** Frequency response of tau -> x = r phi at s = j w: returns {re, im, mag, phase_deg}. */
export function freqResp(P, lin, w) {
  const y = csolveShifted(lin.A, [0, w], lin.B.map((b) => b[0]));
  // solve (A - jw I) y = B  ->  y = -(jw I - A)^-1 B ; G = C (jw I - A)^-1 B = -C y
  const re = -P.r * y[0][0], im = -P.r * y[0][1];
  return { re, im, mag: Math.hypot(re, im), phase_deg: Math.atan2(im, re) * 180 / Math.PI };
}

// ------------------------------------------------------------------------------------------------
// LQR: Kleinman (Newton) iteration on the algebraic Riccati equation, started from an Ackermann pole placement.
// Solves A'X + XA - X B R^-1 B' X + Q = 0, K = R^-1 B'X (u = -K s). Verified against scipy.linalg.solve_continuous_are.
// ------------------------------------------------------------------------------------------------
function ackermann(A, B, pls) {
  const n = A.length;
  const cols = [B.map((r) => r[0])];
  for (let k = 1; k < n; k++) cols.push(matvec(A, cols[k - 1]));
  const Co = transpose(cols);                                    // n x n controllability matrix
  // phi(A) = prod (A - p_i I)
  let PhiA = eye(n);
  for (const p of pls) PhiA = matmul(PhiA, add(A, eye(n), -p));
  const e = [[...Array(n - 1).fill(0), 1]];
  // K = e Co^-1 phi(A):  solve Co^T y = e^T  -> y^T = e Co^-1
  const y = solve(transpose(Co), transpose(e)).map((r) => r[0]);
  return matmul([y], PhiA)[0];
}

function lyap(Ac, Qm) {
  const n = Ac.length, N = n * n;
  const M = zeros(N);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const row = i * n + j;
    for (let k = 0; k < n; k++) {
      M[row][k * n + j] += Ac[k][i];            // (Ac' X)_ij = sum_k Ac_ki X_kj
      M[row][i * n + k] += Ac[k][j];            // (X Ac)_ij  = sum_k X_ik Ac_kj
    }
  }
  const rhs = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) rhs.push([-Qm[i][j]]);
  const x = solve(M, rhs).map((r) => r[0]);
  return Array.from({ length: n }, (_, i) => x.slice(i * n, i * n + n));
}

export function lqr(A, B, Q, Rv) {
  const n = A.length;
  const p0 = Math.max(...eigenvalues(A).map((e) => Math.abs(e[0])), 1);
  let K = null;
  for (const mult of [3, 2, 5, 1.5, 8]) {
    const pls = [1, 1.3, 1.7, 2.2].slice(0, n).map((f) => -mult * f * p0);
    try {
      const K0 = ackermann(A, B, pls);
      const Acl = A.map((r, i) => r.map((v, j) => v - B[i][0] * K0[j]));
      if (Math.max(...eigenvalues(Acl).map((e) => e[0])) < -1e-9) { K = K0; break; }
    } catch (e) { /* try next */ }
  }
  if (!K) throw new Error('no stabilising gain found');
  let X = null;
  for (let it = 0; it < 200; it++) {
    const Acl = A.map((r, i) => r.map((v, j) => v - B[i][0] * K[j]));
    const Qk = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Qk[i][j] = Q[i][j] + K[i] * Rv * K[j];
    const Xn = lyap(Acl, Qk);
    const Kn = Xn.map((_, j) => 0).map((_, j) => (B.reduce((s, b, i) => s + b[0] * Xn[i][j], 0)) / Rv);
    const dK = Math.max(...Kn.map((v, j) => Math.abs(v - K[j]))) / (1 + Math.max(...Kn.map(Math.abs)));
    X = Xn; K = Kn;
    if (dK < 1e-13 && it > 0) break;
  }
  const Acl = A.map((r, i) => r.map((v, j) => v - B[i][0] * K[j]));
  return { K, X, poles: eigenvalues(Acl) };
}

export const LQR_WEIGHTS = { Q: [1e-6, 50, 1, 1], R: 1 };
export function designLQR(lin, w = LQR_WEIGHTS) {
  return lqr(lin.A, lin.B, w.Q.map((q, i) => w.Q.map((_, j) => (i === j ? q : 0))), w.R);
}

// ------------------------------------------------------------------------------------------------
// design metrics (design_space.analytic + design_constraints.constraints)
// ------------------------------------------------------------------------------------------------
export function metricsFromP(P, extra = {}) {
  const c = { ...DEFAULTS, ...extra };
  const lin = linearize(P);
  const pl = eigenvalues(lin.A);
  const p = Math.max(...pl.map((e) => e[0]));
  const z = zeroNMP(P);
  const Ms_lb = skogestad(z, p, 0);
  const tau_max = P.tau_plane_max;
  const mgl = P.m_a * G * P.l;
  let peak = NaN, lam_cl = NaN, K = null, clPoles = null;
  try {
    const L = designLQR(lin);
    K = L.K; clPoles = L.poles;
    lam_cl = Math.max(...L.poles.map((e) => e[0]));
    const Acl = lin.A.map((r, i) => r.map((v, j) => v - lin.B[i][0] * K[j]));
    const dt = 1.5 / 299;
    const E = expm(scale(Acl, dt));
    let x = [0, 3 * Math.PI / 180, 0, 0];
    peak = 0;
    for (let i = 0; i < 300; i++) {
      peak = Math.max(peak, Math.abs(K.reduce((s, kj, j) => s + kj * x[j], 0)));
      x = matvec(E, x);
    }
  } catch (e) { /* leave NaN */ }
  const tilt_domain_deg = peak > 0 ? 3 * tau_max / peak : NaN;
  const Mx = [[lin.A[2][1], lin.B[2][0]], [lin.A[3][1], lin.B[3][0]]];
  const sol = solve(Mx, [[1 / P.r], [0]]);
  const th_a = sol[0][0], u_a = sol[1][0];
  const a_torque = tau_max / Math.abs(u_a);
  const a_lean = (20 * Math.PI / 180) / Math.abs(th_a);
  const kappa_deg = Math.abs(th_a) * 180 / Math.PI;
  const out = {
    r: P.r, m_k: P.m_k, m_a: P.m_a, l: P.l, I_a: P.I_a, J_eff: P.J_eff, mass_ratio: P.m_a / P.m_k, l_over_r: P.l / P.r,
    p, z, z_over_p: z / p, Ms_lb, kappa_deg, kappa_closed_deg: kappaLean(P) * 180 / Math.PI, tau_plane_max: tau_max, mgl,
    torque_authority: tau_max / mgl, lqr_peak_3deg: peak, tilt_domain_deg, a_torque, a_lean, a_max: Math.min(a_torque, a_lean),
    lam_cl, total_mass: P.m_a + P.m_k, K, clPoles, poles: pl,
  };
  // constraints
  if (P.zeta_deg !== undefined) {
    const zeta = P.zeta_deg * Math.PI / 180;
    const N = P.m_a * G / (3 * Math.cos(zeta));
    const tw_trac = c.mu * N * P.r_w;
    const tau_trac = tw_trac * P.fac;
    const mgl_sin = mgl * Math.sin(c.tilt_rec_deg * Math.PI / 180);
    const v_max = P.r * c.omega_max / P.gain;
    const floor = Ms_lb * Math.exp(p * c.T_loop);
    const zd = P.zeta_deg;
    Object.assign(out, {
      zeta_deg: zd, r_w: P.r_w, N_wheel: N, tau_w_trac: tw_trac, tau_plane_trac: tau_trac, tau_plane_motor: tau_max,
      Ms_floor: floor, C_struct: c.Ms_ref / floor, C_delay: Math.LN2 / p / c.T_loop, T_loop_max: Math.LN2 / p,
      C_trac: tau_trac / mgl_sin, C_motor: tau_max / mgl_sin,
      C_lean: (20 * Math.PI / 180) / (Math.abs(th_a) * c.a_target),
      C_geom: zd >= 35 && zd <= 75 ? Math.min((zd - 35) / 10, (75 - zd) / 10) + 1 : 0,
      C_speed: v_max / c.v_target, v_max,
      rec_tilt_trac_deg: Math.asin(Math.min(1, tau_trac / mgl)) * 180 / Math.PI,
      a_trac: c.mu * G,
    });
    const keys = ['C_struct', 'C_delay', 'C_trac', 'C_motor', 'C_lean', 'C_geom', 'C_speed'];
    out.binding = keys.reduce((a, b) => (out[b] < out[a] ? b : a));
    out.feasible = keys.every((k) => out[k] >= 1);
  }
  return out;
}

export function metrics(design, geo, extra = {}) {
  return { ...metricsFromP(paramsFromDesign(design, geo), extra), design: { ...NOMINAL, ...design } };
}

// ------------------------------------------------------------------------------------------------
// nonlinear simulator: RK4 on the Lagrangian dynamics, LQR + velocity-reference governor (ballbot_gym/controllers/lqr.py)
// ------------------------------------------------------------------------------------------------
export class PlanarSim {
  /**
   * opts: dtc control period [s] (ZOH), sub RK4 substeps per period, aMax governor accel [m/s^2], ki integral gain,
   * tauLim torque cap [N m] (Infinity = none), failDeg tilt failure threshold, governor (false = raw step in v_ref).
   */
  constructor(P, opts = {}) {
    this.P = P;
    this.o = { dtc: 0.01, sub: 10, aMax: 0.5, ki: 1, tauLim: Infinity, failDeg: 30, governor: true, ...opts };
    const lin = linearize(P);
    const L = designLQR(lin);
    this.K = L.K;
    this.lin = lin;
    this.reset();
  }
  reset(theta0 = 0) {
    this.s = [0, theta0, 0, 0];               // phi, theta, phid, thetad
    this.t = 0; this.vref = 0; this.cmd = 0; this.intv = 0; this.u = 0; this.fell = false;
    this.pushT = 0; this.pushF = 0; this.pushLeft = 0;
    this.xHist = []; this.log = [];
  }
  get x() { return this.P.r * this.s[0]; }
  get v() { return this.P.r * this.s[2]; }
  get theta() { return this.s[1]; }
  setCommand(v) { this.cmd = v; }
  push(F, dur = 0.1) { this.pushF = F; this.pushLeft = dur; }
  controller() {
    const { dtc, aMax, ki } = this.o;
    const dv = this.cmd - this.vref;
    if (this.o.governor) { const lim = aMax * dtc; this.vref += Math.abs(dv) > lim ? Math.sign(dv) * lim : dv; } else this.vref = this.cmd;
    this.intv = Math.max(-2, Math.min(2, this.intv + (this.vref - this.v) * dtc));
    const phdRef = (this.vref + ki * this.intv) / this.P.r;
    const [, th, phd, thd] = this.s;
    const st = [0, th, phd - phdRef, thd];
    let u = -this.K.reduce((a, k, j) => a + k * st[j], 0);
    const lim = this.o.tauLim;
    if (u > lim) u = lim; else if (u < -lim) u = -lim;
    return u;
  }
  deriv(s, u, F) {
    const [a, b] = accel(this.P, s[1], s[2], s[3], u, F);
    return [s[2], s[3], a, b];
  }
  /** Advance one control period; returns the current sample {t, x, theta, v, u, vref}. */
  step() {
    if (this.fell) return this.sample();
    const u = this.controller();
    this.u = u;
    const h = this.o.dtc / this.o.sub;
    for (let i = 0; i < this.o.sub; i++) {
      const F = this.pushLeft > 0 ? this.pushF : 0;
      const f = (s) => this.deriv(s, u, F);
      const k1 = f(this.s);
      const k2 = f(this.s.map((v, j) => v + 0.5 * h * k1[j]));
      const k3 = f(this.s.map((v, j) => v + 0.5 * h * k2[j]));
      const k4 = f(this.s.map((v, j) => v + h * k3[j]));
      this.s = this.s.map((v, j) => v + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
      this.t += h;
      if (this.pushLeft > 0) this.pushLeft -= h;
    }
    if (Math.abs(this.s[1]) > this.o.failDeg * Math.PI / 180 || !Number.isFinite(this.s[1])) this.fell = true;
    return this.sample();
  }
  sample() { return { t: this.t, x: this.x, theta: this.s[1], v: this.v, u: this.u, vref: this.vref, fell: this.fell }; }
}

/** Headless run helper (used for verification and the "initial reversal" readout): returns arrays. */
export function runStep(P, vcmd, T = 3, opts = {}) {
  const sim = new PlanarSim(P, opts);
  sim.setCommand(vcmd);
  const out = [];
  for (let i = 0; i < Math.round(T / sim.o.dtc); i++) out.push(sim.step());
  return out;
}

/** Initial reversal depth [m] of the ball position in a command step (most negative x before the final forward motion). */
export function reversalDepth(samples, sign = 1) {
  let mn = 0;
  for (const s of samples) mn = Math.min(mn, sign * s.x);
  return -mn;
}

// ------------------------------------------------------------------------------------------------
// tiny SVG plotting helper (pure string output, theme-aware through CSS variables; no DOM needed)
// ------------------------------------------------------------------------------------------------
let _plotId = 0;
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
export const fmtNum = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : +v.toPrecision(3) + '');
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
export class Plot {
  constructor(o) {
    this.o = { w: 640, h: 300, ml: 54, mr: 14, mt: 14, mb: 42, xlim: [0, 1], ylim: [0, 1], logx: false, logy: false, xlabel: '', ylabel: '', grid: true, ...o };
    this.id = 'pl' + (++_plotId);
    this.under = []; this.over = [];
  }
  X(x) { const { xlim: [a, b], logx, ml, w, mr } = this.o; const t = logx ? (Math.log10(x) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (x - a) / (b - a); return ml + t * (w - ml - mr); }
  Y(y) { const { ylim: [a, b], logy, mt, h, mb } = this.o; const t = logy ? (Math.log10(y) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (y - a) / (b - a); return h - mb - t * (h - mt - mb); }
  rect(x0, x1, y0, y1, { fill = 'var(--warn)', opacity = 0.12 } = {}) {
    const xa = this.X(x0), xb = this.X(x1), ya = this.Y(y1), yb = this.Y(y0);
    this.under.push(`<rect x="${Math.min(xa, xb)}" y="${Math.min(ya, yb)}" width="${Math.abs(xb - xa)}" height="${Math.abs(yb - ya)}" style="fill:${fill};opacity:${opacity}"/>`);
  }
  line(pts, { color = 'var(--model)', width = 2, dash = '', opacity = 1 } = {}) {
    const d = pts.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])).map((p, i) => `${i ? 'L' : 'M'}${this.X(p[0]).toFixed(1)},${this.Y(p[1]).toFixed(1)}`).join('');
    this.over.push(`<path d="${d}" clip-path="url(#${this.id})" style="fill:none;stroke:${color};stroke-width:${width};stroke-dasharray:${dash};opacity:${opacity}"/>`);
  }
  points(pts, { color = 'var(--model)', r = 3.5, opacity = 0.8, stroke = 'none', title = null } = {}) {
    for (const p of pts) {
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
      this.over.push(`<circle cx="${this.X(p[0]).toFixed(1)}" cy="${this.Y(p[1]).toFixed(1)}" r="${r}" style="fill:${color};opacity:${opacity};stroke:${stroke}" clip-path="url(#${this.id})">${title ? `<title>${esc(p[2] ?? title)}</title>` : ''}</circle>`);
    }
  }
  vline(x, { color = 'var(--text-3)', dash = '4 3', label = '', width = 1.5, anchor = 'start', dy = 12 } = {}) {
    const X = this.X(x);
    this.over.push(`<line x1="${X}" x2="${X}" y1="${this.o.mt}" y2="${this.o.h - this.o.mb}" style="stroke:${color};stroke-width:${width};stroke-dasharray:${dash}"/>`);
    if (label) this.text(X + (anchor === 'end' ? -4 : 4), this.o.mt + dy, label, { color, anchor, px: true });
  }
  hline(y, { color = 'var(--text-3)', dash = '4 3', label = '', width = 1.5 } = {}) {
    const Y = this.Y(y);
    this.over.push(`<line y1="${Y}" y2="${Y}" x1="${this.o.ml}" x2="${this.o.w - this.o.mr}" style="stroke:${color};stroke-width:${width};stroke-dasharray:${dash}"/>`);
    if (label) this.text(this.o.w - this.o.mr - 4, Y - 4, label, { color, anchor: 'end', px: true });
  }
  text(x, y, s, { color = 'var(--text-2)', anchor = 'start', size = 11, px = false, weight = 400 } = {}) {
    const X = px ? x : this.X(x), Y = px ? y : this.Y(y);
    this.over.push(`<text x="${X}" y="${Y}" text-anchor="${anchor}" style="fill:${color};font:${weight} ${size}px var(--font-mono)">${esc(s)}</text>`);
  }
  marker(x, y, kind, { color = 'var(--model)', size = 6, width = 2.2 } = {}) {
    const X = this.X(x), Y = this.Y(y), s = size;
    if (kind === 'x') this.over.push(`<path d="M${X - s},${Y - s}L${X + s},${Y + s}M${X - s},${Y + s}L${X + s},${Y - s}" style="stroke:${color};stroke-width:${width};fill:none"/>`);
    else if (kind === 'o') this.over.push(`<circle cx="${X}" cy="${Y}" r="${s}" style="stroke:${color};stroke-width:${width};fill:none"/>`);
    else if (kind === 'star') this.over.push(`<path d="M${X},${Y - s * 1.5}L${X + s * 0.45},${Y - s * 0.45}L${X + s * 1.5},${Y}L${X + s * 0.45},${Y + s * 0.45}L${X},${Y + s * 1.5}L${X - s * 0.45},${Y + s * 0.45}L${X - s * 1.5},${Y}L${X - s * 0.45},${Y - s * 0.45}Z" style="fill:${color};stroke:var(--bg);stroke-width:1.5"/>`);
  }
  toString(label = '') {
    const { w, h, ml, mr, mt, mb, xlim, ylim, logx, logy, xlabel, ylabel, grid } = this.o;
    const ticks = (lim, log) => {
      if (this.o.xticks && lim === xlim) return this.o.xticks;
      if (this.o.yticks && lim === ylim) return this.o.yticks;
      if (log) { const t = []; for (let e = Math.ceil(Math.log10(lim[0]) - 1e-9); e <= Math.floor(Math.log10(lim[1]) + 1e-9); e++) t.push(10 ** e); return t; }
      return niceTicks(lim[0], lim[1], 5);
    };
    let g = '';
    for (const v of ticks(xlim, logx)) {
      const X = this.X(v);
      g += `${grid ? `<line x1="${X}" x2="${X}" y1="${mt}" y2="${h - mb}" style="stroke:var(--line);stroke-width:1"/>` : ''}<text x="${X}" y="${h - mb + 15}" text-anchor="middle" style="fill:var(--text-3);font:11px var(--font-mono)">${fmtNum(v)}</text>`;
    }
    for (const v of ticks(ylim, logy)) {
      const Y = this.Y(v);
      g += `${grid ? `<line y1="${Y}" y2="${Y}" x1="${ml}" x2="${w - mr}" style="stroke:var(--line);stroke-width:1"/>` : ''}<text x="${ml - 6}" y="${Y + 4}" text-anchor="end" style="fill:var(--text-3);font:11px var(--font-mono)">${fmtNum(v)}</text>`;
    }
    const axes = `<rect x="${ml}" y="${mt}" width="${w - ml - mr}" height="${h - mt - mb}" style="fill:none;stroke:var(--line-2)"/>` +
      (xlabel ? `<text x="${(ml + w - mr) / 2}" y="${h - 6}" text-anchor="middle" style="fill:var(--text-2);font:12px var(--font-sans)">${esc(xlabel)}</text>` : '') +
      (ylabel ? `<text transform="translate(13 ${(mt + h - mb) / 2}) rotate(-90)" text-anchor="middle" style="fill:var(--text-2);font:12px var(--font-sans)">${esc(ylabel)}</text>` : '');
    return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}" style="width:100%;height:auto;display:block"><defs><clipPath id="${this.id}"><rect x="${ml}" y="${mt}" width="${w - ml - mr}" height="${h - mt - mb}"/></clipPath></defs>${this.under.join('')}${g}${axes}${this.over.join('')}</svg>`;
  }
}
