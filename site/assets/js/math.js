// Math lab: live widgets built on ballbot-model.js (JS port of scripts/study/ballbot_model.py).
import * as M from './ballbot-model.js';

const $ = (s, r = document) => r.querySelector(s);
const [geo, tables] = await Promise.all([
  fetch('data/geometry_table.json').then((r) => r.json()),
  fetch('data/designlab_tables.json').then((r) => r.json()),
]);
const root = document.documentElement;
const css = (n) => getComputedStyle(root).getPropertyValue(n).trim();
const f = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '-');

const P0 = M.nominalParams(geo);
const KG2 = P0.I_a / P0.m_a;                       // body radius of gyration squared at nominal (kept constant)
const NOM = { r: P0.r, l: P0.l, m_a: P0.m_a, m_k: P0.m_k, J: P0.J_eff };
const st = { ...NOM };

function buildP() {
  return {
    ...P0, r: st.r, l: st.l, m_a: st.m_a, m_k: st.m_k, I_k: 0.4 * st.m_k * st.r * st.r, I_a: st.m_a * KG2, J_eff: st.J,
    r_w: P0.r_w * (st.r / P0.r),
  };
}

// ------------------------------------------------------------------ controls
const ctl = {
  r: { el: $('#c-r'), val: $('#v-r'), to: (v) => v / 100, from: (v) => v * 100, fmt: (v) => `${(v * 100).toFixed(1)} cm` },
  l: { el: $('#c-l'), val: $('#v-l'), to: (v) => v / 100, from: (v) => v * 100, fmt: (v) => `${(v * 100).toFixed(1)} cm` },
  m_a: { el: $('#c-ma'), val: $('#v-ma'), to: (v) => v, from: (v) => v, fmt: (v) => `${v.toFixed(2)} kg` },
  m_k: { el: $('#c-mk'), val: $('#v-mk'), to: (v) => v, from: (v) => v, fmt: (v) => `${v.toFixed(3)} kg` },
  J: { el: $('#c-J'), val: $('#v-J'), to: (v) => v, from: (v) => v, fmt: (v) => `${v.toFixed(3)} kg m²` },
};
function syncControls() { for (const [k, c] of Object.entries(ctl)) { c.el.value = c.from(st[k]); c.val.textContent = c.fmt(st[k]); } }
for (const [k, c] of Object.entries(ctl)) {
  c.el.addEventListener('input', () => { st[k] = c.to(parseFloat(c.el.value)); c.val.textContent = c.fmt(st[k]); schedule(); });
}
const PRESETS = {
  nominal: {}, low: { l: 0.087 }, high: { l: 0.38 }, lightball: { m_k: 0.04 }, heavyball: { m_k: 1.0 },
};
document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
  Object.assign(st, NOM, PRESETS[b.dataset.preset]); syncControls(); schedule();
}));

// ------------------------------------------------------------------ math rendering
function tex(el, s) {
  if (window.katex) window.katex.render(s, el, { throwOnError: false, displayMode: true });
  else el.textContent = s;
}
const mat = (rows, d = 2) => `\\begin{pmatrix}${rows.map((r) => r.map((v) => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(d))).join(' & ')).join(' \\\\ ')}\\end{pmatrix}`;
function renderStatic() {
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
  }
}

// ------------------------------------------------------------------ model state
let P, lin, met, lqrD, win;
const live = (k, v) => document.querySelectorAll(`[data-live="${k}"]`).forEach((e) => { e.textContent = v; });

function recompute() {
  P = buildP();
  lin = M.linearize(P);
  met = M.metricsFromP(P);
  lqrD = { K: met.K, poles: met.clPoles };
  const zp = met.z / met.p;
  win = met.z / 2 > met.p;
  live('lambda', f(M.lambdaUndamped(P)));
  live('p', f(met.p)); live('tau_fall', f(1 / met.p, 2));
  live('kappa', f(met.kappa_deg, 1)); live('amax', f(Math.min(met.a_lean), 1));
  live('zp', f(zp, 2));
  live('window', win ? `open (from p = ${f(met.p)} to z/2 = ${f(met.z / 2)} 1/s)` : `empty (z/2 = ${f(met.z / 2)} is below p = ${f(met.p)} 1/s)`);
  live('ms0', f(met.Ms_lb, 1)); live('msdb', f(20 * Math.log10(met.Ms_lb), 0)); live('ms', f(M.skogestad(met.z, met.p, 0.015), 1));
  const sw = tables.sweep.map((r) => r.z_over_p);
  live('nsweep', String(sw.length)); live('zpmin', f(Math.min(...sw), 2)); live('zpmax', f(Math.max(...sw), 2));
  $('#o-p').textContent = f(met.p) + ' /s'; $('#o-z').textContent = f(met.z) + ' /s'; $('#o-zp').textContent = f(zp, 2);
  $('#o-ms').textContent = f(M.skogestad(met.z, met.p, 0.015), 1); $('#o-k').textContent = f(met.kappa_deg, 1) + '°';
  tex($('#mat-A'), 'A=' + mat(lin.A));
  tex($('#mat-B'), 'B=' + mat(lin.B) + '\\;\\;[\\mathrm{s^{-2}/(N\\,m)}]');
}

// ------------------------------------------------------------------ pole-zero, bode, floor, lean plots
const cw = (sel) => Math.max(300, Math.min(760, $(sel).clientWidth || 640));
function drawPZ() {
  const pl = M.eigenvalues(lin.A);
  const cl = lqrD.poles;
  const z = met.z;
  const xs = [...pl.map((a) => a[0]), ...cl.map((a) => a[0]), z, -z];
  const xmin = Math.min(-8, Math.min(...xs) * 1.1), xmax = Math.max(8, Math.max(...xs) * 1.2);
  const ymax = Math.max(4, ...pl.map((a) => Math.abs(a[1])), ...cl.map((a) => Math.abs(a[1]))) * 1.2;
  const p = new M.Plot({ w: cw('#pz'), h: 260, ml: 44, xlim: [xmin, xmax], ylim: [-ymax, ymax], xlabel: 'Re(s)  [1/s]', ylabel: 'Im(s)' });
  p.rect(0, xmax, -ymax, ymax, { fill: 'var(--bad)', opacity: 0.06 });
  p.vline(0, { color: 'var(--text-3)', dash: '' });
  p.text(xmax * 0.97, -ymax * 0.88, 'right half plane: unstable', { anchor: 'end', color: 'var(--bad)', size: 10 });
  for (const [re, im] of cl) p.marker(re, im, 'x', { color: 'var(--model)', size: 9, width: 3 });
  for (const [re, im] of pl) p.marker(re, im, 'x', { color: re > 1e-6 ? 'var(--bad)' : 'var(--model)', size: 6 });
  p.marker(z, 0, 'o', { color: 'var(--warn)', size: 6 }); p.marker(-z, 0, 'o', { color: 'var(--warn)', size: 6 });
  p.text(z, ymax * 0.18, `+z ${f(z)}`, { anchor: 'middle', color: 'var(--warn)' });
  p.text(-z, ymax * 0.18, `-z`, { anchor: 'middle', color: 'var(--warn)' });
  p.text(met.p, -ymax * 0.2, `p ${f(met.p)}`, { anchor: 'middle', color: 'var(--bad)' });
  $('#pz').innerHTML = p.toString('Pole-zero map of the planar ballbot');
}

function drawBode() {
  const ws = Array.from({ length: 220 }, (_, i) => 10 ** (-0.7 + 3.4 * i / 219));
  const mag = [], ph = [];
  let prev = null, off = 0;
  for (const w of ws) {
    const g = M.freqResp(P, lin, w);
    let a = g.phase_deg + off;
    if (prev !== null) { while (a - prev > 180) { a -= 360; off -= 360; } while (a - prev < -180) { a += 360; off += 360; } }
    prev = a;
    mag.push([w, 20 * Math.log10(g.mag)]); ph.push([w, a]);
  }
  const mlo = Math.floor(Math.min(...mag.map((m) => m[1])) / 20) * 20, mhi = Math.ceil(Math.max(...mag.map((m) => m[1])) / 20) * 20;
  const plo = Math.floor(Math.min(...ph.map((m) => m[1])) / 90) * 90, phi = Math.ceil(Math.max(...ph.map((m) => m[1])) / 90) * 90;
  const mk = (data, ylim, ylabel, xlabel, label, h) => {
    const p = new M.Plot({ w: cw('#bode-m'), h, xlim: [ws[0], ws[ws.length - 1]], ylim, logx: true, ylabel, xlabel, mb: xlabel ? 42 : 10, xticks: [0.3, 1, 3, 10, 30, 100] });
    if (win) p.rect(met.p, met.z / 2, ylim[0], ylim[1], { fill: 'var(--good)', opacity: 0.14 });
    p.vline(met.p, { color: 'var(--bad)', dash: '', label: 'p' });
    p.vline(met.z / 2, { color: 'var(--warn)', label: 'z/2', anchor: 'end' });
    p.vline(met.z, { color: 'var(--warn)', dash: '', label: 'z' });
    p.line(data, { color: 'var(--model)', width: 2.4 });
    return p.toString(label);
  };
  $('#bode-m').innerHTML = mk(mag, [mlo, mhi], '|G| [dB]', '', 'Magnitude of the torque to ball position response', 190);
  $('#bode-p').innerHTML = mk(ph, [plo, phi], 'phase [deg]', 'ω  [rad/s]', 'Phase of the torque to ball position response', 220);
}

function drawFloor() {
  const p = new M.Plot({ w: cw('#floor'), h: 280, xlim: [1, 3.5], ylim: [1, 100], logy: true, xlabel: 'z / p', ylabel: 'sensitivity floor Ms' });
  const rs = Array.from({ length: 250 }, (_, i) => 1.003 + (3.5 - 1.003) * (i / 249) ** 1.5);
  p.line(rs.map((r) => [r, (r + 1) / Math.abs(r - 1)]), { color: 'var(--model)' });
  p.line(rs.map((r) => [r, (r + 1) / Math.abs(r - 1) * Math.exp(met.p * 0.015)]), { color: 'var(--warn)', dash: '5 3' });
  p.vline(2, { color: 'var(--good)', label: 'z = 2p: window opens', dy: 14 });
  p.points(tables.sweep.map((r) => [r.z_over_p, r.Ms_lb, r.name]), { color: 'var(--text-3)', r: 3, opacity: 0.7, title: 'sweep' });
  p.marker(met.z / met.p, M.skogestad(met.z, met.p, 0.015), 'star', { color: 'var(--model)', size: 5 });
  $('#floor').innerHTML = p.toString('Peak sensitivity floor versus the ratio of the zero to the pole');
}

function drawLean() {
  const k = met.kappa_deg;
  const amax = Math.max(3, 20 / k * 1.15);
  const p = new M.Plot({ w: cw('#lean'), h: 240, xlim: [0, amax], ylim: [0, 30], xlabel: 'sustained acceleration a  [m/s²]', ylabel: 'steady lean  [deg]' });
  p.line([[0, 0], [amax, k * amax]], { color: 'var(--model)', width: 2.4 });
  p.hline(20, { color: 'var(--warn)', label: '20° lean budget' });
  p.vline(20 / k, { color: 'var(--warn)', label: `a max ${f(20 / k, 2)}`, dy: 30 });
  p.text(amax * 0.55, k * amax * 0.55 - 2, `${f(k, 1)}° per m/s²`, { color: 'var(--model)', anchor: 'middle' });
  $('#lean').innerHTML = p.toString('Steady lean required as a function of sustained acceleration');
}

// ------------------------------------------------------------------ simulator
const cv = $('#sim'), sv = $('#strip');
const sim = { run: true, gov: true, cap: false, ctrl: true, obj: null, hist: [], acc: 0, last: 0, rev: null, F: 15, cmd: 0 };
function dprResize(c) {
  const d = Math.min(2, window.devicePixelRatio || 1), w = c.clientWidth || 720;
  const h = c === cv ? Math.max(210, Math.round(w * 0.36)) : Math.max(250, Math.round(w * 0.32));
  if (c.width !== Math.round(w * d)) { c.width = Math.round(w * d); c.height = Math.round(h * d); c.style.height = h + 'px'; }
  return { ctx: c.getContext('2d'), w, h, d };
}
function newSim(theta0 = 0) {
  const tauLim = !sim.ctrl ? 0 : sim.cap ? met.tau_plane_trac : Infinity;
  sim.obj = new M.PlanarSim(P, { governor: sim.gov, tauLim });
  sim.obj.reset(theta0);
  sim.obj.setCommand(sim.cmd);
  sim.hist = []; sim.rev = null; sim.acc = 0;
}
function applyLim() { if (sim.obj) sim.obj.o.tauLim = !sim.ctrl ? 0 : sim.cap ? met.tau_plane_trac : Infinity; }
function setCmd(v) {
  v = Math.max(-1, Math.min(1, v));
  const old = sim.cmd; sim.cmd = v;
  $('#s-cmd').value = v; $('#s-cmdv').textContent = v.toFixed(2) + ' m/s';
  if (sim.obj) {
    sim.obj.setCommand(v);
    if (v !== old) sim.rev = { x0: sim.obj.x, sign: Math.sign(v - old), depth: 0, done: false, t0: sim.obj.t };
  }
}
function drawSim() {
  const { ctx, w, h, d } = dprResize(cv);
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const o = sim.obj, ppm = Math.min(330, w / 1.7) * (w < 420 ? 0.9 : 1);
  const gy = h - 34, cx = w * 0.42;
  const col = { line: css('--line-2'), t3: css('--text-3'), t2: css('--text-2'), text: css('--text'), model: css('--model'), warn: css('--warn'), bad: css('--bad'), bg3: css('--bg-3') };
  ctx.font = '11px ' + css('--font-mono');
  // floor + ticks
  ctx.strokeStyle = col.line; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, gy + 0.5); ctx.lineTo(w, gy + 0.5); ctx.stroke();
  const x0 = o.x, m0 = Math.floor((x0 - cx / ppm) / 0.25) * 0.25;
  ctx.fillStyle = col.t3; ctx.strokeStyle = col.line; ctx.lineWidth = 1;
  for (let m = m0; m < x0 + (w - cx) / ppm + 0.25; m += 0.25) {
    const px = cx + (m - x0) * ppm;
    ctx.beginPath(); ctx.moveTo(px, gy); ctx.lineTo(px, gy + (Math.abs(m / 0.5 - Math.round(m / 0.5)) < 1e-6 ? 8 : 4)); ctx.stroke();
    if (Math.abs(m / 0.5 - Math.round(m / 0.5)) < 1e-6) ctx.fillText(m.toFixed(1) + ' m', px - 12, gy + 21);
  }
  // origin flag
  const ox = cx + (0 - x0) * ppm;
  ctx.strokeStyle = col.t3; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(ox, gy - 90); ctx.lineTo(ox, gy); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = col.t3; ctx.fillText('start', ox + 4, gy - 80);
  if (sim.rev) {
    const rx = cx + (sim.rev.x0 - x0) * ppm;
    ctx.strokeStyle = col.warn; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(rx, gy - 60); ctx.lineTo(rx, gy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col.warn; ctx.fillText('command', rx + 4, gy - 50);
  }
  // ball
  const R = P.r * ppm, by = gy - R, ph = o.s[0];
  ctx.strokeStyle = col.model; ctx.lineWidth = 2.5; ctx.fillStyle = col.bg3;
  ctx.beginPath(); ctx.arc(cx, by, R, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.lineWidth = 1.5;
  for (let i = 0; i < 4; i++) { const a = ph + i * Math.PI / 2; ctx.beginPath(); ctx.moveTo(cx, by); ctx.lineTo(cx + R * Math.cos(a), by + R * Math.sin(a)); ctx.stroke(); }
  // body (schematic): rounded bar rotated by theta about the ball centre; theta > 0 leans toward +x
  const th = o.s[1], L = P.l * ppm, bw = Math.max(12, 0.07 * ppm);
  ctx.save(); ctx.translate(cx, by); ctx.rotate(th);
  ctx.fillStyle = col.bg3; ctx.strokeStyle = o.fell ? col.bad : col.model; ctx.lineWidth = 2.5;
  const top = L + 0.07 * ppm;
  ctx.beginPath(); ctx.roundRect(-bw, -top, 2 * bw, top - R * 0.4, 6); ctx.fill(); ctx.stroke();
  ctx.fillStyle = col.warn; ctx.beginPath(); ctx.arc(0, -L, 5, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
  ctx.fillStyle = col.warn; ctx.fillText('COM', cx + bw + 8 + Math.sin(th) * L, by - Math.cos(th) * L + 4);
  // command arrow
  ctx.fillStyle = col.t2; ctx.fillText(`cmd ${sim.cmd.toFixed(2)} m/s`, 10, 16);
  if (Math.abs(sim.cmd) > 0.01) { ctx.strokeStyle = col.model; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(w - 100, 12); ctx.lineTo(w - 100 + sim.cmd * 70, 12); ctx.stroke(); }
  if (o.fell) { ctx.fillStyle = col.bad; ctx.font = '600 14px ' + css('--font-sans'); ctx.fillText('FELL: tilt beyond 30°. Press Reset.', 10, 38); }
}
function drawStrip() {
  const { ctx, w, h, d } = dprResize(sv);
  ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, w, h);
  const hist = sim.hist; if (hist.length < 2) return;
  const col = { line: css('--line-2'), t3: css('--text-3'), t2: css('--text-2'), model: css('--model'), warn: css('--warn') };
  const rows = [['ball x [m]', (s) => s.x, col.model, 0.05], ['tilt [deg]', (s) => s.theta * 180 / Math.PI, col.t2, 2], ['torque [N m]', (s) => s.u, col.warn, 1]];
  const rh = (h - 12) / 3, t1 = hist[hist.length - 1].t, t0 = Math.max(0, t1 - 6);
  ctx.font = '10px ' + css('--font-mono');
  rows.forEach(([name, fn, c, minspan], i) => {
    const y0 = 6 + i * rh, y1 = y0 + rh - 8;
    const vis = hist.filter((s) => s.t >= t0);
    const vals = vis.map(fn); let lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
    if (hi - lo < minspan) { hi += minspan / 2; lo -= minspan / 2; }
    const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
    const X = (t) => 46 + (t - t0) / 6 * (w - 56), Y = (v) => y1 - (v - lo) / (hi - lo) * (y1 - y0);
    ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.strokeRect(46, y0, w - 56, y1 - y0);
    ctx.beginPath(); ctx.moveTo(46, Y(0)); ctx.lineTo(w - 10, Y(0)); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col.t3; ctx.fillText(name, 52, y0 + 12); ctx.fillText(hi.toFixed(2), 3, y0 + 10); ctx.fillText(lo.toFixed(2), 3, y1 - 2);
    ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
    vis.forEach((s, k) => { const px = X(s.t), py = Y(fn(s)); if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py); }); ctx.stroke();
    if (sim.rev && i === 0 && sim.rev.t0 >= t0) { ctx.strokeStyle = col.warn; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(sim.rev.t0), y0); ctx.lineTo(X(sim.rev.t0), y1); ctx.stroke(); ctx.setLineDash([]); }
  });
}
function readouts() {
  const o = sim.obj, s = o.sample();
  $('#s-x').textContent = s.x.toFixed(2) + ' m'; $('#s-v').textContent = s.v.toFixed(2) + ' m/s';
  $('#s-th').textContent = (s.theta * 180 / Math.PI).toFixed(1) + '°'; $('#s-u').textContent = s.u.toFixed(1) + ' N m';
  const r = sim.rev;
  $('#s-rev').textContent = r ? (r.depth > 5e-4 ? `${(r.depth * 1000).toFixed(1)} mm ${r.sign > 0 ? 'back' : 'forward'}` : (r.done ? 'none' : '...')) : 'issue a command';
}
function tick(ts) {
  if (!sim.last) sim.last = ts;
  const dt = Math.min(0.1, (ts - sim.last) / 1000); sim.last = ts;
  if (sim.run && sim.obj && !sim.obj.fell) {
    sim.acc += dt;
    let n = 0;
    while (sim.acc >= sim.obj.o.dtc && n < 8) {
      const s = sim.obj.step(); sim.acc -= sim.obj.o.dtc; n++;
      sim.hist.push(s); if (sim.hist.length > 900) sim.hist.shift();
      const r = sim.rev;
      if (r && !r.done) {
        const dd = r.sign * (s.x - r.x0);
        r.depth = Math.max(r.depth, -dd);
        if (dd > 0.003) r.done = true;
      }
    }
  }
  if (sim.obj) { drawSim(); drawStrip(); readouts(); }
  requestAnimationFrame(tick);
}
const tog = (id, key, labels) => $(id).addEventListener('click', (e) => {
  sim[key] = !sim[key]; e.target.setAttribute('aria-pressed', sim[key]); e.target.classList.toggle('on', sim[key]); e.target.textContent = labels[sim[key] ? 0 : 1];
  if (key === 'gov') newSim(); else applyLim();
});
tog('#b-gov', 'gov', ['Governor: on', 'Governor: off']);
tog('#b-cap', 'cap', ['Torque cap: traction', 'Torque cap: none']);
tog('#b-ctrl', 'ctrl', ['LQR: on', 'LQR: off']);
$('#b-run').addEventListener('click', (e) => { sim.run = !sim.run; e.target.textContent = sim.run ? 'Pause' : 'Run'; e.target.setAttribute('aria-pressed', sim.run); });
$('#b-reset').addEventListener('click', () => { setCmd(0); newSim(); });
$('#b-drop').addEventListener('click', () => { setCmd(0); newSim(5 * Math.PI / 180); });
$('#b-pf').addEventListener('click', () => sim.obj.push(sim.F, 0.1));
$('#b-pb').addEventListener('click', () => sim.obj.push(-sim.F, 0.1));
$('#b-go').addEventListener('click', () => setCmd(0.5));
$('#b-stop').addEventListener('click', () => setCmd(0));
$('#s-cmd').addEventListener('input', (e) => setCmd(parseFloat(e.target.value)));
$('#s-F').addEventListener('input', (e) => { sim.F = parseFloat(e.target.value); $('#s-Fv').textContent = sim.F + ' N'; });
if (matchMedia('(prefers-reduced-motion: reduce)').matches) { sim.run = false; $('#b-run').textContent = 'Run'; $('#b-run').setAttribute('aria-pressed', 'false'); }
$('#b-gov').classList.add('on'); $('#b-ctrl').classList.add('on');

// ------------------------------------------------------------------ verification
function verify() {
  const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
  const toD = (r) => ({ s: r.s, kball: r.kball, kbal: r.kbal, zb: r.zb, J: r.J, tau: r.tau, damp: r.damp });
  const qs = [['p', 'p [1/s]', 3], ['z', 'z [1/s]', 3], ['kappa_deg', 'κ [deg per m/s²]', 2], ['Ms_lb', 'Ms floor', 2]];
  const byName = Object.fromEntries(tables.sweep.map((r) => [r.name, r]));
  const rowsWanted = [['d_nominal', 'nominal'], ['d_s0.5', 'half scale'], ['d_s1.5', 'scale 1.5'], ['d_kball8', 'ball mass x8'], ['d_kbal4', 'body mass x4'], ['d_zb0.3', 'ballast at 0.3 m'], ['d_J0.02', 'rotor J 0.02']];
  const tb = $('#vtab tbody');
  $('#vtab thead').innerHTML = `<tr><th>Design (design_sweep.csv)</th>${qs.map(([, l]) => `<th class="num">${l}<br><small>JS / Python</small></th>`).join('')}<th class="num">max rel. error</th><th>Status</th></tr>`;
  tb.innerHTML = '';
  let ok = true;
  for (const [nm, label] of rowsWanted) {
    const py = byName[nm]; const js = M.metrics(toD(py), geo);
    const errs = qs.map(([k]) => rel(js[k], py[k])); const me = Math.max(...errs);
    ok = ok && me < 0.01;
    tb.insertAdjacentHTML('beforeend', `<tr><td>${label} <code>${nm}</code></td>${qs.map(([k, , d]) => `<td class="num">${js[k].toFixed(d)} / ${py[k].toFixed(d)}</td>`).join('')}<td class="num">${me.toExponential(1)}</td><td class="${me < 0.01 ? 'pass' : 'fail'}">${me < 0.01 ? 'PASS' : 'FAIL'}</td></tr>`);
  }
  // all rows
  let worst = 0, worstName = '';
  const allKeys = ['p', 'z', 'kappa_deg', 'Ms_lb', 'mgl', 'tau_plane_max', 'lqr_peak_3deg', 'tilt_domain_deg', 'a_max', 'l', 'm_a', 'I_a', 'J_eff'];
  for (const py of tables.sweep) {
    const js = M.metrics(toD(py), geo);
    for (const k of allKeys) { const e = rel(js[k], py[k]); if (e > worst) { worst = e; worstName = `${py.name}/${k}`; } }
  }
  const nom = byName.d_nominal;
  $('#vsummary').innerHTML = `Across all <b>${tables.sweep.length}</b> rows of <code>design_sweep.csv</code> and ${allKeys.length} quantities each, the largest relative difference is <b>${worst.toExponential(1)}</b> (${worstName}); the CSV itself is rounded to 10 digits. The nominal values quoted in <code>docs/research/design_rules.md</code> are p = 4.79, z = 5.78, &kappa; = 10.1&deg; and Ms floor 10.7, matching the live values ${f(M.metrics({}, geo).p)}, ${f(M.metrics({}, geo).z)}, ${f(M.metrics({}, geo).kappa_deg, 1)}, ${f(M.metrics({}, geo).Ms_lb, 1)}. Overall: <span class="${ok && worst < 0.01 ? 'pass' : 'fail'}">${ok && worst < 0.01 ? 'PASS (all within 1%)' : 'FAIL'}</span>.`;
  // extra identities at nominal
  const pn = M.nominalParams(geo), ln = M.linearize(pn), mn = M.metricsFromP(pn);
  const lam = M.lambdaUndamped(pn);
  const Gz = M.tfAt(pn, ln, mn.z, 0), Gh = M.tfAt(pn, ln, mn.z / 2, 0);
  const cl = mn.clPoles.map((a) => a[0]).sort((a, b) => a - b);
  const extra = [
    ['A[3,2]  (theta -> phi acceleration)', ln.A[2][1], 11.53, 'paper_v2 matrix A', 2e-3],
    ['A[3,3]', ln.A[2][2], -12.35, 'paper_v2 matrix A', 2e-3],
    ['A[4,2]', ln.A[3][1], 27.31, 'paper_v2 matrix A', 2e-3],
    ['A[4,3]', ln.A[3][2], 6.56, 'paper_v2 matrix A', 2e-3],
    ['B[3]', ln.B[2][0], 6.02, 'paper_v2 matrix B', 2e-3],
    ['B[4]', ln.B[3][0], -3.20, 'paper_v2 matrix B', 2e-3],
    ['undamped unstable pole lambda [1/s]', lam, 5.22, 'paper_v2 Prop. 1 text', 2e-3],
    ['LQR closed-loop pole 1 [1/s]', cl[0], -20.3, 'paper_v2 LQR section', 5e-3],
    ['LQR closed-loop pole 2', cl[1], -5.02, 'paper_v2 LQR section', 5e-3],
    ['LQR closed-loop pole 3', cl[2], -4.50, 'paper_v2 LQR section', 5e-3],
    ['LQR |K| on theta', Math.abs(mn.K[1]), 76.1, 'paper_v2 K = (0, 76.1, 4.33, 11.6); sign convention differs', 5e-3],
    ['LQR |K| on phi-dot', Math.abs(mn.K[2]), 4.33, 'paper_v2', 5e-3],
    ['LQR |K| on theta-dot', Math.abs(mn.K[3]), 11.6, 'paper_v2 (rounded to 3 digits)', 5e-3],
    ['LQR peak torque for 3 deg start [N m]', mn.lqr_peak_3deg, nom.lqr_peak_3deg, 'design_sweep.csv (scipy CARE)', 1e-6],
    ['LQR slow closed-loop pole [1/s]', mn.lam_cl, nom.lam_cl, 'design_sweep.csv (scipy CARE)', 1e-3],
  ];
  const tb2 = $('#vtab2 tbody'); tb2.innerHTML = '';
  let ok2 = true;
  for (const [name, js, py, src, tol] of extra) {
    const e = name.includes('slow') ? Math.abs(js - py) / 1e-3 : rel(js, py); const pass = e < Math.max(tol, 0);
    ok2 = ok2 && pass;
    tb2.insertAdjacentHTML('beforeend', `<tr><td>${name}</td><td class="num">${js.toPrecision(5)}</td><td class="num">${py.toPrecision(5)}</td><td>${src}</td><td class="num">${e.toExponential(1)}</td><td class="${pass ? 'pass' : 'fail'}">${pass ? 'PASS' : 'FAIL'}</td></tr>`);
  }
  const zr = Math.hypot(...Gz) / Math.hypot(...Gh);
  tb2.insertAdjacentHTML('beforeend', `<tr><td>|G_x(s=z)| / |G_x(s=z/2)|  (zero is at +z)</td><td class="num">${zr.toExponential(1)}</td><td class="num">0</td><td>Proposition 2 (exact zero)</td><td class="num">${zr.toExponential(1)}</td><td class="${zr < 1e-9 ? 'pass' : 'fail'}">${zr < 1e-9 ? 'PASS' : 'FAIL'}</td></tr>`);
  window.__verify = { ok, worst, ok2, zr };
  document.documentElement.dataset.verify = ok && worst < 0.01 && ok2 && zr < 1e-9 ? 'pass' : 'fail';
}

// ------------------------------------------------------------------ inline the static SVG diagrams so CSS variables (theme) apply
async function inlineSVGs() {
  await Promise.all([...document.querySelectorAll('[data-svg]')].map(async (host) => {
    try {
      const txt = await fetch(host.dataset.svg, { cache: 'no-cache' }).then((r) => (r.ok ? r.text() : Promise.reject(r.status)));
      const doc = new DOMParser().parseFromString(txt, 'image/svg+xml');
      const svg = doc.documentElement;
      if (svg.nodeName !== 'svg') return;
      const img = host.querySelector('img');
      if (img && img.alt && !svg.getAttribute('aria-label')) svg.setAttribute('aria-label', img.alt);
      host.replaceChildren(document.importNode(svg, true));
    } catch (e) { /* keep the <img> fallback */ }
  }));
}

// ------------------------------------------------------------------ 5. authority dial (S82, S83)
const AUTH = [ // rho, flat tracking (mid of range), goal disturbed (mid of range)
  [0.15, 1.015, 0.35], [0.25, 1.0, 0.425], [0.35, 0.945, 0.45], [0.5, 0.85, 0.475], [1.0, 0.745, 0.50],
];
const lerpRow = (x) => {
  if (x <= AUTH[0][0]) return AUTH[0];
  for (let i = 1; i < AUTH.length; i++) if (x <= AUTH[i][0]) { const [a, b] = [AUTH[i - 1], AUTH[i]], t = (x - a[0]) / (b[0] - a[0]); return [x, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]; }
  return AUTH[AUTH.length - 1];
};
function drawAuthority() {
  const el = $('#rho'); if (!el) return;
  const rho = parseFloat(el.value); $('#rho-v').textContent = rho.toFixed(2);
  const aL = 0.3, aP = -1, out = Math.max(-1, Math.min(1, aL + rho * aP));
  const bar = (label, v, col) => `<div class="rw" style="display:grid;grid-template-columns:7.5em 1fr 3.4em;gap:8px;align-items:center;font:400 .74rem var(--font-mono);color:var(--text-2);margin:4px 0"><span>${label}</span><span style="position:relative;height:14px;background:var(--bg-3);border-radius:7px;overflow:hidden"><i style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line-2)"></i><i style="position:absolute;top:2px;bottom:2px;border-radius:5px;background:${col};left:${v >= 0 ? 50 : 50 + v * 50}%;width:${Math.abs(v) * 50}%"></i></span><b style="text-align:right;color:var(--text);font-weight:500">${v.toFixed(2)}</b></div>`;
  $('#rho-bars').innerHTML = bar('a_LQR', aL, 'var(--model)') + bar('rho x a_pi', rho * aP, 'var(--learned)') + bar('applied', out, 'var(--hybrid)');
  const p = new M.Plot({ w: cw('#rho-plot'), h: 230, xlim: [0.1, 1.05], ylim: [0.3, 1.1], xlabel: 'learned authority rho (1 = pure learning)', ylabel: 'ratio', xticks: [0.15, 0.25, 0.35, 0.5, 1] });
  p.hline(1, { color: 'var(--text-3)', dash: '2 4' });
  p.line(AUTH.map((r) => [r[0], r[1]]), { color: 'var(--model)', width: 2.2 });
  p.points(AUTH.map((r) => [r[0], r[1], `rho ${r[0]}: tracking ${r[1]}`]), { color: 'var(--model)', r: 4, title: 'tracking' });
  p.line(AUTH.map((r) => [r[0], r[2]]), { color: 'var(--good)', width: 2.2 });
  p.points(AUTH.map((r) => [r[0], r[2], `rho ${r[0]}: goal success with pushes ${r[2]}`]), { color: 'var(--good)', r: 4, title: 'robust' });
  const cur = lerpRow(rho);
  p.marker(cur[0], cur[1], 'o', { color: 'var(--hybrid)', size: 7 }); p.marker(cur[0], cur[2], 'o', { color: 'var(--hybrid)', size: 7 });
  p.text(0.12, 1.06, 'flat tracking (achieved / command)', { color: 'var(--model)', size: 10 });
  p.text(0.12, 0.34, 'goal success, pushes + pulls', { color: 'var(--good)', size: 10 });
  p.vline(0.25, { color: 'var(--hybrid)', label: 'knee', dy: 24 });
  $('#rho-plot').innerHTML = p.toString('Measured tracking and robustness against learned authority, S82 and S83');
}

// ------------------------------------------------------------------ 6. Froude scaling widget
function drawFroude() {
  const el = $('#fr-s'); if (!el) return;
  const s = parseFloat(el.value); $('#fr-sv').textContent = s.toFixed(2);
  const fsq = Math.sqrt(s);
  const Ps = { ...P, r: P.r * s, l: P.l * s, m_a: P.m_a * s ** 3, m_k: P.m_k * s ** 3, I_a: P.I_a * s ** 5, I_k: P.I_k * s ** 5, J_eff: P.J_eff * s ** 5, d_eff: P.d_eff * s ** 2.5 };
  const ps = Math.max(...M.eigenvalues(M.linearize(Ps).A).map((e) => e[0]));
  const rows = [
    ['ball radius', `${(P.r * s * 100).toFixed(1)} cm`], ['total mass', `${((P.m_a + P.m_k) * s ** 3).toFixed(2)} kg`],
    ['command range (x sqrt s)', `0 to ${(1.2 * fsq).toFixed(2)} m/s`], ['pushes (x s^3)', `${(5 * s ** 3).toFixed(1)} to ${(20 * s ** 3).toFixed(1)} N`],
    ['unstable pole', `${ps.toFixed(2)} 1/s (x ${(ps / met.p).toFixed(2)})`], ['lean per m/s^2', `${(M.kappaLean(Ps) * 180 / Math.PI).toFixed(1)} deg (unchanged)`],
  ];
  $('#fr-out').innerHTML = rows.map(([k, v]) => `<div><small>${k}</small><b style="font-size:.9rem">${v}</b></div>`).join('');
}

// ------------------------------------------------------------------ 7. traction limit at a step edge
function drawTraction() {
  const el = $('#tr-mu'); if (!el) return;
  const mu = parseFloat(el.value), r = parseFloat($('#tr-r').value) / 100;
  $('#tr-muv').textContent = mu.toFixed(2); $('#tr-rv').textContent = (r * 100).toFixed(1) + ' cm';
  const th = Math.atan(mu), hmax = r * (1 - Math.cos(th));
  // drawing: floor, step of height hmax, ball touching the edge
  const W = cw('#tr-draw'), H = 230, px = Math.min(700, (H - 60) / (2 * r + hmax)); // px per metre, fit the height
  const R = r * px, hp = hmax * px, gy = H - 24, ex = W * 0.55;
  const cx = ex - R * Math.sin(th), cy = gy - hp - R * Math.cos(th);
  const sv = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Ball of radius ${(r * 100).toFixed(1)} cm at a step of height ${(hmax * 100).toFixed(1)} cm, the largest it can climb with friction ${mu.toFixed(2)}" style="width:100%;height:auto;display:block">
    <path d="M0 ${gy} H${ex} V${gy - hp} H${W}" fill="none" style="stroke:var(--line-2);stroke-width:2"/>
    <rect x="${ex}" y="${gy - hp}" width="${W - ex}" height="${hp + 24}" style="fill:var(--bg-3);opacity:.7"/>
    <circle cx="${cx}" cy="${cy}" r="${R}" style="fill:var(--bg-3);stroke:var(--model);stroke-width:2.5"/>
    <line x1="${cx}" y1="${cy}" x2="${ex}" y2="${gy - hp}" style="stroke:var(--line-2);stroke-dasharray:4 3"/>
    <line x1="${cx}" y1="${cy}" x2="${cx}" y2="${gy}" style="stroke:var(--line-2);stroke-dasharray:2 5"/>
    <path d="M${cx} ${cy + R * 0.45} A${R * 0.45} ${R * 0.45} 0 0 0 ${cx + R * 0.45 * Math.sin(th)} ${cy + R * 0.45 * Math.cos(th)}" fill="none" style="stroke:var(--warn);stroke-width:1.6"/>
    <text x="${cx - 6}" y="${cy + R * 0.45 + 14}" text-anchor="end" style="fill:var(--warn);font:11px var(--font-mono)">theta ${(th * 180 / Math.PI).toFixed(0)}°</text>
    <line x1="${ex}" y1="${gy - hp}" x2="${ex - 0.42 * R * Math.cos(th)}" y2="${gy - hp - 0.42 * R * Math.sin(th)}" style="stroke:var(--good);stroke-width:2.6"/>
    <line x1="${ex}" y1="${gy - hp}" x2="${ex + 0.42 * R * mu * Math.sin(th)}" y2="${gy - hp - 0.42 * R * mu * Math.cos(th)}" style="stroke:var(--bad);stroke-width:2.6"/>
    <text x="${ex - 0.42 * R * Math.cos(th) - 14}" y="${gy - hp - 0.42 * R * Math.sin(th) - 6}" style="fill:var(--good);font:11px var(--font-mono)">N</text>
    <text x="${ex + 0.42 * R * mu * Math.sin(th) + 6}" y="${gy - hp - 0.42 * R * mu * Math.cos(th)}" style="fill:var(--bad);font:11px var(--font-mono)">F = mu N</text>
    <text x="${W - 8}" y="${H - 7}" text-anchor="end" style="fill:var(--warn);font:11px var(--font-mono)">h max ${(hmax * 100).toFixed(1)} cm</text>
    <line x1="${W - 30}" y1="${gy}" x2="${W - 30}" y2="${gy - hp}" style="stroke:var(--warn);stroke-width:1.4"/>
  </svg>`;
  $('#tr-draw').innerHTML = sv;
  $('#tr-out').innerHTML = [['edge angle', `${(th * 180 / Math.PI).toFixed(1)}°`], ['h max / r', (1 - Math.cos(th)).toFixed(3)], ['h max', `${(hmax * 100).toFixed(1)} cm`]]
    .map(([k, v]) => `<div><small>${k}</small><b class="warn">${v}</b></div>`).join('');
}

// ------------------------------------------------------------------ 9. wavefront toy
const WV = { n: 21, occ: null, infl: true, robot: [18, 3], goal: [2, 17], drag: null };
function wvField() {
  const n = WV.n, INF = 1e9, k = WV.infl ? 1 : 0;
  const blocked = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
    for (let a = -k; a <= k; a++) for (let b = -k; b <= k; b++) { const y = i + a, x = j + b; if (y >= 0 && y < n && x >= 0 && x < n && WV.occ[y][x]) return true; }
    return false;
  }));
  const d = Array.from({ length: n }, () => new Array(n).fill(INF));
  const [gx, gy] = WV.goal; d[gy][gx] = 0; blocked[gy][gx] = false;
  for (let it = 0; it < 2 * n; it++) {
    let ch = false;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (blocked[i][j]) continue;
      let m = d[i][j];
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const y = i + a, x = j + b; if (y >= 0 && y < n && x >= 0 && x < n && !(a === 0 && b === 0)) m = Math.min(m, d[y][x] + 1); }
      if (m < d[i][j]) { d[i][j] = m; ch = true; }
    }
    if (!ch) break;
  }
  return { d, blocked };
}
function drawWave() {
  const c = $('#wave'); if (!c) return;
  const { d, blocked } = wvField();
  const n = WV.n, dpr = Math.min(2, window.devicePixelRatio || 1), size = c.clientWidth || 420;
  if (c.width !== Math.round(size * dpr)) { c.width = c.height = Math.round(size * dpr); c.style.height = size + 'px'; }
  const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, size, size);
  const cs = size / n, INF = 1e9;
  const fin = d.flat().filter((v) => v < INF), dmax = Math.max(1, ...fin);
  const bg3 = css('--bg-3'), warn = css('--warn'), good = css('--good'), hyb = css('--hybrid'), model = css('--model'), t3 = css('--text-3');
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = j * cs, y = i * cs;
    if (WV.occ[i][j]) { ctx.fillStyle = warn; ctx.globalAlpha = 0.85; }
    else if (blocked[i][j]) { ctx.fillStyle = warn; ctx.globalAlpha = 0.25; }
    else if (d[i][j] >= INF) { ctx.fillStyle = t3; ctx.globalAlpha = 0.5; }
    else { ctx.fillStyle = model; ctx.globalAlpha = 0.12 + 0.7 * (d[i][j] / dmax); }
    ctx.fillRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
  }
  ctx.globalAlpha = 1;
  // goal
  ctx.fillStyle = good; ctx.fillRect(WV.goal[0] * cs + 1, WV.goal[1] * cs + 1, cs - 2, cs - 2);
  // carrot descent: 6 steps, 8 directions
  let [px, py] = WV.robot; const path = [[px, py]];
  for (let s = 0; s < 6; s++) {
    let best = d[py][px], bx = px, by = py;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const y = py + a, x = px + b; if (y >= 0 && y < n && x >= 0 && x < n && d[y][x] < best) { best = d[y][x]; bx = x; by = y; } }
    if (bx === px && by === py) break;
    px = bx; py = by; path.push([px, py]);
  }
  ctx.strokeStyle = hyb; ctx.lineWidth = 2.5; ctx.setLineDash([4, 4]); ctx.beginPath();
  path.forEach(([x, y], k) => { const X = (x + 0.5) * cs, Y = (y + 0.5) * cs; if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = hyb; ctx.beginPath(); ctx.arc((px + 0.5) * cs, (py + 0.5) * cs, cs * 0.3, 0, 2 * Math.PI); ctx.fill();
  // robot
  ctx.fillStyle = bg3; ctx.strokeStyle = hyb; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc((WV.robot[0] + 0.5) * cs, (WV.robot[1] + 0.5) * cs, cs * 0.45, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  // heading arrow toward the carrot
  if (path.length > 1) {
    const ang = Math.atan2(py - WV.robot[1], px - WV.robot[0]);
    const X = (WV.robot[0] + 0.5) * cs, Y = (WV.robot[1] + 0.5) * cs;
    ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + Math.cos(ang) * cs * 1.6, Y + Math.sin(ang) * cs * 1.6); ctx.stroke();
  }
  if (d[WV.robot[1]][WV.robot[0]] >= INF) { ctx.fillStyle = css('--bad'); ctx.font = '600 12px ' + css('--font-sans'); ctx.fillText('robot inside the margin or cut off: no path', 8, 16); }
}
function initWave() {
  const c = $('#wave'); if (!c) return;
  const n = WV.n;
  const demo = () => { WV.occ = Array.from({ length: n }, () => new Array(n).fill(false)); for (let i = 6; i <= 14; i++) { WV.occ[i][9] = WV.occ[i][10] = true; } for (let j = 3; j <= 6; j++) WV.occ[14][j] = true; };
  demo();
  const cell = (e) => { const r = c.getBoundingClientRect(); const cs = r.width / n; return [Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / cs))), Math.max(0, Math.min(n - 1, Math.floor((e.clientY - r.top) / cs)))]; };
  c.addEventListener('pointerdown', (e) => {
    const [x, y] = cell(e);
    if (x === WV.robot[0] && y === WV.robot[1]) WV.drag = 'robot';
    else if (x === WV.goal[0] && y === WV.goal[1]) WV.drag = 'goal';
    else { WV.drag = WV.occ[y][x] ? 'erase' : 'paint'; WV.occ[y][x] = WV.drag === 'paint'; }
    c.setPointerCapture(e.pointerId); drawWave();
  });
  c.addEventListener('pointermove', (e) => {
    if (!WV.drag) return; const [x, y] = cell(e);
    if (WV.drag === 'robot') WV.robot = [x, y]; else if (WV.drag === 'goal') WV.goal = [x, y]; else WV.occ[y][x] = WV.drag === 'paint';
    drawWave();
  });
  c.addEventListener('pointerup', () => { WV.drag = null; });
  $('#wv-clear').addEventListener('click', () => { WV.occ = Array.from({ length: n }, () => new Array(n).fill(false)); drawWave(); });
  $('#wv-demo').addEventListener('click', () => { demo(); drawWave(); });
  $('#wv-infl').addEventListener('click', (e) => { WV.infl = !WV.infl; e.target.setAttribute('aria-pressed', WV.infl); e.target.textContent = `Inflation: ${WV.infl ? 'on' : 'off'}`; drawWave(); });
  drawWave();
}
for (const id of ['#rho']) $(id)?.addEventListener('input', drawAuthority);
$('#fr-s')?.addEventListener('input', drawFroude);
$('#tr-mu')?.addEventListener('input', drawTraction); $('#tr-r')?.addEventListener('input', drawTraction);

// ------------------------------------------------------------------ orchestration
let pend = false;
function schedule() { if (pend) return; pend = true; requestAnimationFrame(() => { pend = false; render(); }); }
function render() {
  recompute(); drawPZ(); drawBode(); drawFloor(); drawLean(); drawFroude();
  const th0 = 0;
  newSim(th0);
}
const extras = () => { drawAuthority(); drawTraction(); drawWave(); };
function init() {
  renderStatic();
  syncControls(); recompute();
  newSim(); render(); verify(); initWave(); extras();
  inlineSVGs();
  requestAnimationFrame(tick);
}
document.addEventListener('themechange', () => { drawPZ(); drawBode(); drawFloor(); drawLean(); extras(); });
let rz = 0;
window.addEventListener('resize', () => { if (sim.obj) { drawSim(); drawStrip(); } clearTimeout(rz); rz = setTimeout(() => { if (met) { drawPZ(); drawBode(); drawFloor(); drawLean(); extras(); } }, 150); });
const wait = () => (window.katex && window.renderMathInElement ? init() : setTimeout(wait, 60));
if (document.readyState === 'complete') wait(); else window.addEventListener('load', wait);
