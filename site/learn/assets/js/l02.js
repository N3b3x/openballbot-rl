// Lesson 2 figures: the velocity step response (backward dip, lean, torque) and the lean-per-acceleration law across designs.
import { paramsFrom, linearize, dlqr, stepResponse, leanPerAccel, zeroNMP, PlanarBot } from './physics.js';
import { el, canvas, chart, slider, seg, button, readouts, legend, clock, palette, drawBot, floor, fmt } from './ui.js';

const DEG = Math.PI / 180;

export function step(root, ctx) {
  const stage = root.querySelector('.stage');
  const body = root.querySelector('.body');
  const P = paramsFrom(ctx.data.model);
  const { A, B } = linearize(P);
  const K = dlqr(A, B, ctx.data.lqr.q, ctx.data.lqr.r, 0.01);
  const kappa = leanPerAccel(P) / DEG;       // deg per m/s^2
  const S = { cmd: 0.5, a_max: 1, ki: 1 };
  let res = null, t = 0, playing = false;
  const T = 5;
  function compute() { res = stepResponse(P, K, { cmd: S.cmd, T, a_max: S.a_max, ki: S.ki, tStep: 0.3 }); }
  compute();
  const idxAt = (tt) => Math.min(res.t.length - 1, Math.round(tt / 0.01));
  const cv = canvas(stage, (w) => Math.min(260, Math.max(190, w * 0.3)), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const sc = Math.min(w / 2.0, 300), fy = h - 26, i = idxAt(t);
    floor(g, w, fy, pal, w / 2 - res.x[i] * sc, sc);
    drawBot(g, res.x[i], res.th[i], { pal, scale: sc, ox: w / 2 - res.x[i] * sc, oy: fy, l: P.l, color: pal.model });
    g.fillStyle = pal.text3; g.font = '11px ' + getComputedStyle(document.body).fontFamily;
    g.fillText(`t = ${t.toFixed(2)} s   x = ${(res.x[i] * 100).toFixed(1)} cm   tilt = ${(res.th[i] / DEG).toFixed(1)} deg`, 10, 16);
    g.fillStyle = pal.text3;
  }, ctx);
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Side view of the ballbot following a velocity command. The plots below give the same run as numbers.');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Speed command', min: 0.1, max: 1, step: 0.05, value: S.cmd, fmt: (v) => `${v.toFixed(2)} m/s`, onInput: (v) => { S.cmd = v; refresh(); } });
  slider(ctl, { label: 'Acceleration limit of the command governor (a_max)', min: 0.2, max: 4, step: 0.1, value: S.a_max, fmt: (v) => `${v.toFixed(1)} m/s²`, onInput: (v) => { S.a_max = v; refresh(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  button(bt, 'Play', () => { t = 0; if (ctx.reduced) { t = T; cv.redraw(); marker(); } else { playing = true; clk.start(); } }, 'primary');
  seg(bt, [[1, 'Velocity integral on'], [0, 'off']], 1, (v) => { S.ki = v; refresh(); }, 'velocity integral');
  const ro = readouts(body, [['dip', 'backward dip of the ball', 'bad'], ['tdip', 'dip, seconds after the command', ''], ['peak', 'peak tilt', 'warn'], ['lean', 'peak of the lean law', 'model'], ['kap', 'lean per acceleration, κ', 'model'], ['tq', 'peak plane torque', 'hybrid']]);
  const c1 = chart(body, { x: [0, T], y: [-0.05, 2.5], xlabel: 'time (s)', ylabel: 'ball position (m)', height: 170, crosshair: true, desc: 'Ball position against time after a velocity command at 0.3 s.' });
  const c1b = chart(body, { x: [0.2, 1.4], y: [-0.06, 0.06], xlabel: 'time (s), zoom on the first second', ylabel: 'ball position (m)', height: 150, desc: 'Zoom on the first second: the ball moves backward before it moves forward.', crosshair: true });
  legend(body, [['--model', 'tilt (deg)'], ['--warn', 'lean law: κ a + tilt acceleration / z² (deg)', true], ['--learned', 'plane torque / 10 (N m)']]);
  const c2 = chart(body, { x: [0, T], y: [-10, 30], xlabel: 'time (s)', ylabel: 'deg', height: 190, crosshair: true, desc: 'Body tilt and the lean predicted from the ball acceleration, with the motor torque.' });
  function series() {
    const n = res.t.length;
    const xs = res.t.map((tt, i) => [tt, res.x[i]]);
    const ths = res.t.map((tt, i) => [tt, res.th[i] / DEG]);
    const z2 = zeroNMP(P) ** 2;
    const acc = res.t.map((tt, i) => { const ok = i > 2 && i < n - 3; const a = ok ? (res.v[i + 3] - res.v[i - 3]) / 0.06 : 0; const d2 = ok ? (res.th[i + 3] - 2 * res.th[i] + res.th[i - 3]) / 0.0009 : 0; return [tt, kappa * a + (d2 / z2) / DEG]; });
    const tqs = res.t.map((tt, i) => [tt, res.tau[i] / 10]);
    return { xs, ths, acc, tqs };
  }
  function refresh() {
    compute();
    const s = series();
    const min = Math.min(...res.x), imin = res.x.indexOf(min);
    const ymax = Math.max(0.5, Math.min(5, Math.max(...res.x) * 1.05));
    c1.update({ y: [Math.min(-0.05, min * 1.3), ymax], series: [{ data: s.xs, color: '--model', label: 'x (m)' }, { data: res.t.map((tt, i) => [tt, res.vref[i] * 0.0]), color: '--line-2' }] });
    const zy = Math.max(0.02, Math.abs(min) * 2.2);
    c1b.update({ y: [-zy, zy], series: [{ data: s.xs.filter((p) => p[0] >= 0.2 && p[0] <= 1.4), color: '--model', label: 'x (m)' }], hlines: [{ y: min, color: '--bad', label: `dip ${(min * 100).toFixed(1)} cm` }] });
    const pk = Math.max(...res.th) / DEG, mn = Math.min(...res.th) / DEG;
    const lo = Math.min(-5, Math.floor(mn / 5) * 5 - 0), hi = Math.max(15, Math.ceil(Math.max(pk, ...s.acc.map((p) => p[1])) / 5) * 5);
    c2.update({ y: [lo, hi], series: [{ data: s.ths, color: '--model', label: 'tilt' }, { data: s.acc, color: '--warn', dash: '5 4', label: 'κ a' }, { data: s.tqs, color: '--learned', width: 1.4, label: 'torque/10' }] });
    ro.set('dip', `${(Math.min(0, min) * 100).toFixed(1)} cm`); ro.set('tdip', `${(res.t[imin] - 0.3).toFixed(2)} s`);
    ro.set('peak', `${pk.toFixed(1)} deg`); ro.set('lean', `${Math.max(...s.acc.map((p) => p[1])).toFixed(1)} deg`); ro.set('kap', `${kappa.toFixed(1)} deg/(m/s²)`);
    ro.set('tq', `${(Math.max(...res.tau.map(Math.abs))).toFixed(1)} N m`);
    cv.redraw(); marker();
  }
  function marker() {
    const tt = Math.min(t, T);
    [c1, c1b, c2].forEach((c) => c.update({ vlines: [{ x: tt, color: '--text-3', dash: '2 3' }] }));
  }
  const clk = clock(stage, (dt) => {
    if (!playing) return false;
    t += dt; if (t >= T) { t = T; playing = false; }
    cv.redraw(); marker();
    return playing;
  });
  refresh();
}

export function lean(root, ctx) {
  const stage = root.querySelector('.stage');
  const body = root.querySelector('.body');
  const rows = ctx.data.designs.rows;
  const nomK = ctx.data.model.lean_deg;
  const sw = ctx.data.sweep.groups.flatMap((g) => g.rows).filter((r) => r.nominal)[0];
  const bars = rows.map((r, i) => ({ x: i, y: r.lean_deg, color: r.name === 'nom' ? '--hybrid' : '--model', label: `${r.label}: ${r.lean_deg.toFixed(1)} deg per m/s²` }));
  const ch = chart(stage, {
    x: [-0.7, rows.length - 0.3], y: [0, 14], xticks: rows.map((_, i) => i), xfmt: (v) => rows[v] ? rows[v].name : '', ylabel: 'deg per m/s²', height: 230,
    bars, hlines: [{ y: nomK, color: '--hybrid', label: `nominal ${nomK.toFixed(1)}` }], desc: 'Lean angle needed per unit of acceleration for each of the 11 training designs.', ml: 40,
  });
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const ro = readouts(body, [['lean', 'lean needed', 'warn'], ['lim', 'lean-limited max acceleration (nominal)', 'model'], ['t', 'tilt at that limit', '']]);
  slider(ctl, { label: 'Acceleration you want (nominal robot)', min: 0.1, max: 3, step: 0.05, value: 1, fmt: (v) => `${v.toFixed(2)} m/s²`, onInput: upd });
  function upd(a = 1) {
    ro.set('lean', `${(a * nomK).toFixed(1)} deg`);
    ro.set('lim', `${sw.phys.a_max.toFixed(2)} m/s²`);
    ro.set('t', `${(sw.phys.a_max * nomK).toFixed(1)} deg`);
  }
  upd(1);
}
