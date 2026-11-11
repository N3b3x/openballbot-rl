// Lesson 7 figures: how dead-reckoning errors turn into position drift, a Kalman filter on encoder and accelerometer data, and the S108 numbers.
import { el, canvas, chart, slider, seg, button, readouts, legend, clock, palette, fmt } from './ui.js';

const DEG = Math.PI / 180;
function mulberry(seed) { let a = seed >>> 0; return () => { a += 0x6d2b79f5; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

// ------------------------------------------------------------------------------------------------ drift
export function drift(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const est = ctx.data.odometry.estimator;
  const S = { bias: 0.005, scale: 0, zupt: false, stopEvery: 20, speed: 0.4 };
  let res = null;
  // true path: a lap-and-a-half of a figure-of-eight, travelled at constant speed (arc-length parametrised numerically)
  const T = 60, dt = 0.05;
  const raw = Array.from({ length: 4001 }, (_, i) => { const u = (i / 4000) * 2 * Math.PI; return [2.0 * Math.sin(u), 1.1 * Math.sin(2 * u)]; });
  const arc = [0]; for (let i = 1; i < raw.length; i++) arc.push(arc[i - 1] + Math.hypot(raw[i][0] - raw[i - 1][0], raw[i][1] - raw[i - 1][1]));
  const at = (s) => { s = s % arc[arc.length - 1]; let lo = 0, hi = arc.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; arc[m] <= s ? lo = m : hi = m; } const f = (s - arc[lo]) / (arc[hi] - arc[lo] || 1); return [raw[lo][0] + f * (raw[hi][0] - raw[lo][0]), raw[lo][1] + f * (raw[hi][1] - raw[lo][1]), Math.atan2(raw[hi][1] - raw[lo][1], raw[hi][0] - raw[lo][0])]; };
  function simulate() {
    const rnd = mulberry(3);
    let s = 0, p0 = at(0), px = 0, py = 0, bEst = 0, yawErr = 0;
    const o = { t: [], tx: [], ty: [], ex: [], ey: [], err: [], yaw: [], dist: [] };
    let prev = at(0), ex = p0[0], ey = p0[1], moving = true, nextStop = S.stopEvery, stopLeft = 0;
    for (let t = 0; t <= T + 1e-9; t += dt) {
      if (S.zupt && moving && t >= nextStop) { moving = false; stopLeft = 3; }
      if (!moving) { stopLeft -= dt; bEst += (S.bias - bEst) * (dt / est.bias_tau); if (stopLeft <= 0) { moving = true; nextStop += S.stopEvery; } }
      if (moving) s += S.speed * dt;
      const cur = at(s);
      const ds = Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);
      yawErr += (S.bias - (S.zupt ? bEst : 0)) * dt;
      const h = Math.atan2(cur[1] - prev[1], cur[0] - prev[0]);
      if (ds > 1e-9) { ex += (1 + S.scale) * ds * Math.cos(h + yawErr); ey += (1 + S.scale) * ds * Math.sin(h + yawErr); }
      prev = cur;
      o.t.push(t); o.tx.push(cur[0]); o.ty.push(cur[1]); o.ex.push(ex); o.ey.push(ey); o.err.push(Math.hypot(ex - cur[0], ey - cur[1])); o.yaw.push(yawErr / DEG); o.dist.push(s);
    }
    return o;
  }
  let t = T;
  const cv = canvas(stage, (w) => Math.min(340, Math.max(220, w * 0.5)), (g, w, h) => {
    if (!res) return;
    const pal = palette(); g.clearRect(0, 0, w, h);
    const xs = [...res.tx, ...res.ex], ys = [...res.ty, ...res.ey];
    const x0 = Math.min(...xs, -2.3), x1 = Math.max(...xs, 2.3), y0 = Math.min(...ys, -1.3), y1 = Math.max(...ys, 1.3);
    const sc = Math.min((w - 20) / (x1 - x0), (h - 24) / (y1 - y0)), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const X = (x) => w / 2 + (x - cx) * sc, Y = (y) => h / 2 - (y - cy) * sc;
    g.strokeStyle = pal.line; g.lineWidth = 1; g.fillStyle = pal.text3; g.font = '10px ' + getComputedStyle(document.body).fontFamily;
    for (let m = Math.ceil(x0); m <= x1; m++) { g.beginPath(); g.moveTo(X(m), 8); g.lineTo(X(m), h - 8); g.stroke(); g.fillText(`${m} m`, X(m) + 2, h - 10); }
    for (let m = Math.ceil(y0); m <= y1; m++) { g.beginPath(); g.moveTo(8, Y(m)); g.lineTo(w - 8, Y(m)); g.stroke(); }
    const i = Math.min(res.t.length - 1, Math.round(t / dt));
    const draw = (xa, ya, col, wd, dash) => { g.strokeStyle = col; g.lineWidth = wd; g.setLineDash(dash || []); g.beginPath(); for (let k = 0; k <= i; k++) k ? g.lineTo(X(xa[k]), Y(ya[k])) : g.moveTo(X(xa[k]), Y(ya[k])); g.stroke(); g.setLineDash([]); };
    draw(res.tx, res.ty, pal.model, 2.6); draw(res.ex, res.ey, pal.bad, 2, [6, 4]);
    for (const [xa, ya, col] of [[res.tx, res.ty, pal.model], [res.ex, res.ey, pal.bad]]) { g.fillStyle = col; g.beginPath(); g.arc(X(xa[i]), Y(ya[i]), 5, 0, 6.3); g.fill(); }
    g.strokeStyle = pal.warn; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(res.tx[i]), Y(res.ty[i])); g.lineTo(X(res.ex[i]), Y(res.ey[i])); g.stroke();
    g.fillStyle = pal.text3; g.fillText(`t = ${res.t[i].toFixed(0)} s`, 8, 14);
  }, ctx);
  cv.c.setAttribute('role', 'img'); cv.c.setAttribute('aria-label', 'Top view of the true path (a figure of eight) and the path the robot believes it drove from dead reckoning; the amber line joins the two current positions.');
  legend(body, [['--model', 'true path'], ['--bad', 'dead-reckoned path', true], ['--warn', 'current error']]);
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Gyro bias (study: 0.005 rad/s standard deviation)', min: 0, max: 0.03, step: 0.0005, value: S.bias, fmt: (v) => `${v.toFixed(4)} rad/s`, onInput: (v) => { S.bias = v; refresh(); } });
  slider(ctl, { label: 'Speed scale error (wheel calibration)', min: -0.1, max: 0.1, step: 0.005, value: S.scale, fmt: (v) => `${(v * 100).toFixed(1)} %`, onInput: (v) => { S.scale = v; refresh(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  seg(bt, [[false, 'Never stops'], [true, 'Stops 3 s every 20 s (zero-velocity bias update)']], false, (v) => { S.zupt = v; refresh(); }, 'stopping');
  button(bt, 'Play', () => { t = 0; if (ctx.reduced) { t = T; cv.redraw(); } else { playing = true; clk.start(); } }, 'primary');
  let playing = false;
  const ro = readouts(body, [['err', 'position error at 60 s', 'bad'], ['dist', 'distance driven', 'model'], ['pct', 'error as share of distance', 'warn'], ['yaw', 'heading error at 60 s', 'hybrid']]);
  const ec = chart(body, { x: [0, T], y: [0, 2], xlabel: 'time (s)', ylabel: 'position error (m)', height: 170, crosshair: true, desc: 'Position error against time.' });
  function refresh() {
    res = simulate();
    const n = res.t.length - 1;
    ro.set('err', `${res.err[n].toFixed(2)} m`); ro.set('dist', `${res.dist[n].toFixed(1)} m`); ro.set('pct', `${(100 * res.err[n] / res.dist[n]).toFixed(1)} %`); ro.set('yaw', `${res.yaw[n].toFixed(1)} deg`);
    ec.update({ y: [0, Math.max(0.5, Math.ceil(Math.max(...res.err) * 2) / 2)], series: [{ data: res.t.map((tt, i) => [tt, res.err[i]]).filter((_, i) => i % 4 === 0), color: '--bad', label: 'error (m)' }] });
    t = T; cv.redraw();
  }
  const clk = clock(stage, (dt2) => { if (!playing) return false; t += dt2 * 6; if (t >= T) { t = T; playing = false; } cv.redraw(); return playing; });
  refresh();
}

// ------------------------------------------------------------------------------------------------ Kalman filter in one dimension
export function kalman(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const est = ctx.data.odometry.estimator;
  const S = { renc: 0.05, qacc: est.q_acc, rfilt: est.r_enc, accN: 0.05, accB: 0.03, seed: 1 };
  const dt = 0.01, T = 12;
  const vTrue = (t) => (t < 1 ? 0 : t < 3 ? 0.2 * (t - 1) : t < 8 ? 0.4 : t < 10 ? 0.4 - 0.2 * (t - 8) : 0);
  const aTrue = (t) => (t < 1 ? 0 : t < 3 ? 0.2 : t < 8 ? 0 : t < 10 ? -0.2 : 0);
  legend(stage, [['--model', 'true speed'], ['--text-3', 'encoder reading'], ['--warn', 'accelerometer integrated', true], ['--hybrid', 'Kalman estimate']]);
  const c1 = chart(stage, { x: [0, T], y: [-0.2, 0.9], xlabel: 'time (s)', ylabel: 'speed (m/s)', height: 220, crosshair: true, desc: 'True speed, noisy encoder speed, integrated accelerometer and Kalman estimate.' });
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Encoder speed noise (sensor)', min: 0, max: 0.2, step: 0.005, value: S.renc, fmt: (v) => `${v.toFixed(3)} m/s`, onInput: (v) => { S.renc = v; run(); } });
  slider(ctl, { label: 'Accelerometer noise (study: 0.05)', min: 0, max: 0.5, step: 0.01, value: S.accN, fmt: (v) => `${v.toFixed(2)} m/s²`, onInput: (v) => { S.accN = v; run(); } });
  slider(ctl, { label: 'Accelerometer bias (study: 0.03)', min: 0, max: 0.3, step: 0.01, value: S.accB, fmt: (v) => `${v.toFixed(2)} m/s²`, onInput: (v) => { S.accB = v; run(); } });
  slider(ctl, { label: 'Filter: trust in encoder (R as sd)', min: 0.005, max: 0.3, step: 0.005, value: S.rfilt, fmt: (v) => `${v.toFixed(3)} m/s`, onInput: (v) => { S.rfilt = v; run(); } });
  slider(ctl, { label: 'Filter: trust in accelerometer (process noise)', min: 0.05, max: 3, step: 0.05, value: S.qacc, fmt: (v) => `${v.toFixed(2)} m/s²`, onInput: (v) => { S.qacc = v; run(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  button(bt, 'New noise draw', () => { S.seed++; run(); });
  const ro = readouts(body, [['e1', 'encoder alone: RMS error', ''], ['e2', 'accelerometer alone: error at 12 s', 'warn'], ['e3', 'Kalman: RMS error', 'hybrid'], ['k', 'Kalman gain on speed (end)', 'model']]);
  function run() {
    const rnd = mulberry(S.seed);
    // state (x, v), predict with measured acceleration, update with the encoder speed: the structure of BallbotOdometry in 1-D
    let x = 0, v = 0, P = [[1e-6, 0], [0, 1e-2]], vi = 0;
    const o = { v: [], enc: [], ker: [], acc: [] }; let se1 = 0, se3 = 0, K1 = 0, n = 0;
    const Q = (qa) => { const b = [0.5 * dt * dt, dt]; return [[qa * qa * b[0] * b[0], qa * qa * b[0] * b[1]], [qa * qa * b[1] * b[0], qa * qa * b[1] * b[1]]]; };
    for (let t = 0; t <= T + 1e-9; t += dt) {
      const vt = vTrue(t), a = aTrue(t) + S.accB + gauss(rnd) * S.accN, enc = vt + gauss(rnd) * S.renc;
      x += v * dt + 0.5 * a * dt * dt; v += a * dt;
      const F = [[1, dt], [0, 1]];
      const FP = [[F[0][0] * P[0][0] + F[0][1] * P[1][0], F[0][0] * P[0][1] + F[0][1] * P[1][1]], [F[1][0] * P[0][0] + F[1][1] * P[1][0], F[1][0] * P[0][1] + F[1][1] * P[1][1]]];
      const FPF = [[FP[0][0] * F[0][0] + FP[0][1] * F[0][1], FP[0][0] * F[1][0] + FP[0][1] * F[1][1]], [FP[1][0] * F[0][0] + FP[1][1] * F[0][1], FP[1][0] * F[1][0] + FP[1][1] * F[1][1]]];
      const q = Q(S.qacc); P = [[FPF[0][0] + q[0][0], FPF[0][1] + q[0][1]], [FPF[1][0] + q[1][0], FPF[1][1] + q[1][1]]];
      const Sv = P[1][1] + S.rfilt * S.rfilt, k0 = P[0][1] / Sv, k1 = P[1][1] / Sv, inn = enc - v;
      x += k0 * inn; v += k1 * inn;
      P = [[P[0][0] - k0 * P[1][0], P[0][1] - k0 * P[1][1]], [P[1][0] - k1 * P[1][0], P[1][1] - k1 * P[1][1]]];
      vi += a * dt; K1 = k1;
      o.v.push([t, vt]); o.enc.push([t, enc]); o.ker.push([t, v]); o.acc.push([t, vi]);
      se1 += (enc - vt) ** 2; se3 += (v - vt) ** 2; n++;
    }
    c1.update({ y: [-0.3, Math.max(0.9, Math.min(1.6, Math.max(...o.acc.map((p) => p[1])) * 1.05))], series: [
      { data: o.enc.filter((_, i) => i % 3 === 0), color: '--text-3', width: 1, opacity: 0.7, label: 'encoder' },
      { data: o.acc.filter((_, i) => i % 3 === 0), color: '--warn', dash: '5 4', width: 1.4, label: 'accel. integral' },
      { data: o.v, color: '--model', width: 2.4, label: 'true' },
      { data: o.ker.filter((_, i) => i % 3 === 0), color: '--hybrid', width: 2, label: 'Kalman' }] });
    ro.set('e1', `${Math.sqrt(se1 / n).toFixed(3)} m/s`); ro.set('e2', `${(vi - vTrue(T)).toFixed(2)} m/s`); ro.set('e3', `${Math.sqrt(se3 / n).toFixed(3)} m/s`); ro.set('k', K1.toFixed(3));
  }
  run();
}

// ------------------------------------------------------------------------------------------------ S108 results
export function errors(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const rows = ctx.data.odometry.rows;
  const short = ['ideal sensors', 'realistic', 'radius 2% off', 'uncalibrated'];
  let metric = 'err';
  const hz = ['10 s', '30 s', '60 s'];
  legend(stage, [['--good', 'after 10 s'], ['--warn', 'after 30 s'], ['--bad', 'after 60 s']]);
  const ch = chart(stage, { x: [-0.6, rows.length - 0.4], y: [0, 0.5], xticks: rows.map((_, i) => i), xfmt: (v) => short[v] || '', ylabel: 'position error, median (m)', height: 250, ml: 48, desc: 'Median position error for four sensor configurations at three horizons.' });
  const sg = el('div', { class: 'btns' }); body.append(sg);
  seg(sg, [['err', 'Position error (m)'], ['pct', 'Error as percent of distance'], ['yaw', 'Heading error (deg)']], metric, (m) => { metric = m; draw(); }, 'metric');
  const ro = readouts(body, [['dist', 'median distance driven at 10 / 30 / 60 s', 'model'], ['note', 'reading', '']]);
  const title = { err: 'position error, median (m)', pct: 'error, percent of distance driven', yaw: 'heading error, median (deg)' };
  function draw() {
    const key = { err: 'err_median', pct: 'pct', yaw: 'yaw_deg' }[metric];
    const vals = rows.flatMap((r) => r[key]);
    const cols = ['--good', '--warn', '--bad'];
    ch.update({ y: [0, Math.max(...vals) * 1.15], ylabel: title[metric],
      bars: rows.flatMap((r, i) => r[key].map((v, k) => ({ x: i, dx: (k - 1) * 22, w: 19, y: v, color: cols[k], label: `${short[i]}, ${hz[k]}: ${v}` }))) });
    ro.set('dist', rows[0].dist.map((d) => d.toFixed(1)).join(' / ') + ' m');
    ro.set('note', metric === 'err' ? 'Realistic sensors: 5 cm after 1.8 m, 22 cm after 12.7 m.' : metric === 'pct' ? 'About 2 percent of distance driven with calibrated encoders; 13 percent over the first 2 m if the wheel model is used uncalibrated.' : 'Heading is the error that grows: 12 degrees after a minute, with no absolute reference.');
  }
  draw();
}
