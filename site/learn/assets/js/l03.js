// Lesson 3 figure: the LQR loop on the planar model. Change the design weights or scale each gain, push the robot, watch poles and time traces.
import { paramsFrom, linearize, dlqr, c2d, eig, PlanarBot, planeTorqueMax } from './physics.js';
import { el, canvas, chart, slider, button, readouts, legend, clock, palette, drawBot, floor, fmt, seg } from './ui.js';

const DEG = Math.PI / 180;

export function loop(root, ctx) {
  const stage = root.querySelector('.stage');
  const body = root.querySelector('.body');
  const P = paramsFrom(ctx.data.model);
  const { A, B } = linearize(P);
  const q0 = ctx.data.lqr.q, dt = 0.01;
  const S = { qth: q0[1], R: ctx.data.lqr.r, m: [1, 1, 1], tauW: 30, push: 15, cmd: 0 };
  const nvec = ctx.data.model.n;
  let K0 = ctx.data.lqr.K_nom.slice(), K = K0.slice(), tr = null, t = 0, playing = false;
  const T = 4;
  function design() {
    K0 = dlqr(A, B, [q0[0], S.qth, q0[2], q0[3]], S.R, dt);
    K = K0.map((k, i) => k * (i === 0 ? 1 : S.m[i - 1]));
  }
  function poles() {
    const { Ad, Bd } = c2d(A, B, dt);
    const Acl = Ad.map((r, i) => r.map((v, j) => v - Bd[i] * K[j]));
    const z = eig(Acl);
    return z.map(([re, im]) => { const m = Math.hypot(re, im) || 1e-12; return { s: [Math.log(m) / dt, Math.atan2(im, re) / dt], m }; });
  }
  function simulate() {
    const tauMax = planeTorqueMax(nvec, S.tauW);
    const bot = new PlanarBot(P, { K, tauMax, ki: 1, a_max: 0.5 });
    tr = { t: [], x: [], th: [], tau: [], sat: false, fell: false };
    const n = Math.round(T / dt);
    for (let i = 0; i <= n; i++) {
      tr.t.push(bot.t); tr.x.push(bot.x); tr.th.push(bot.theta); tr.tau.push(bot.tau);
      if (Math.abs(bot.tau) >= tauMax - 1e-6) tr.sat = true;
      const F = bot.t >= 0.5 && bot.t < 0.6 ? S.push : 0;
      bot.tick(bot.t >= 2.2 ? S.cmd : 0, F);
      if (bot.fallen) tr.fell = true;
    }
  }
  design(); simulate();
  const cv = canvas(stage, (w) => Math.min(250, Math.max(190, w * 0.3)), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const i = Math.min(tr.t.length - 1, Math.round(t / dt));
    const sc = Math.min(w / 2.0, 300), fy = h - 26, x = tr.x[i];
    floor(g, w, fy, pal, w / 2 - x * sc, sc);
    const pushing = tr.t[i] >= 0.5 && tr.t[i] < 0.6 && S.push !== 0;
    drawBot(g, x, tr.th[i], { pal, scale: sc, ox: w / 2 - x * sc, oy: fy, l: P.l, color: pal.model, push: pushing ? S.push : 0 });
    g.fillStyle = pal.text3; g.font = '11px ' + getComputedStyle(document.body).fontFamily;
    g.fillText(`t = ${t.toFixed(2)} s   tilt = ${(tr.th[i] / DEG).toFixed(1)} deg   torque = ${tr.tau[i].toFixed(1)} N m`, 10, 16);
  }, ctx);
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Side view of the balanced robot being pushed at 0.5 seconds. Numeric read-outs and plots below.');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const sl = {};
  sl.qth = slider(ctl, { label: 'Tilt weight Q (design)', min: 0, max: 3, step: 0.05, value: Math.log10(S.qth), fmt: (v) => `${(10 ** v).toFixed(v < 1 ? 1 : 0)}`, onInput: (v) => { S.qth = 10 ** v; refresh(); } });
  sl.R = slider(ctl, { label: 'Effort weight R (design)', min: -1, max: 1, step: 0.05, value: Math.log10(S.R), fmt: (v) => `${(10 ** v).toFixed(2)}`, onInput: (v) => { S.R = 10 ** v; refresh(); } });
  ['tilt gain K₂', 'ball-rate gain K₃', 'tilt-rate gain K₄'].forEach((lab, i) => {
    sl['m' + i] = slider(ctl, { label: `${lab} (scale)`, min: 0, max: 2.5, step: 0.05, value: 1, fmt: (v) => `×${v.toFixed(2)}`, onInput: (v) => { S.m[i] = v; refresh(); } });
  });
  sl.tau = slider(ctl, { label: 'Wheel torque limit', min: 1, max: 30, step: 1, value: S.tauW, fmt: (v) => `${v} N m per wheel`, onInput: (v) => { S.tauW = v; refresh(); } });
  sl.push = slider(ctl, { label: 'Push at 0.5 s (0.1 s long)', min: -40, max: 40, step: 1, value: S.push, fmt: (v) => `${v} N`, onInput: (v) => { S.push = v; refresh(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  button(bt, 'Replay', () => { t = 0; if (ctx.reduced) { t = T; cv.redraw(); } else { playing = true; clk.start(); } }, 'primary');
  button(bt, 'Reset gains', () => { S.qth = q0[1]; S.R = ctx.data.lqr.r; S.m = [1, 1, 1]; S.tauW = 30; S.push = 15; sl.qth.set(Math.log10(S.qth)); sl.R.set(0); [0, 1, 2].forEach((i) => sl['m' + i].set(1)); sl.tau.set(30); sl.push.set(15); refresh(); });
  seg(bt, [[0, 'Hold still'], [0.5, 'Then drive 0.5 m/s']], 0, (v) => { S.cmd = v; refresh(); }, 'after the push');
  const ro = readouts(body, [['st', 'closed loop', ''], ['K', 'gains [K₂ K₃ K₄]', 'model'], ['peak', 'peak tilt', 'warn'], ['tq', 'peak torque', 'hybrid'], ['set', 'recovered (tilt under 1 deg) after', '']]);
  const st = el('p', { class: 'status' }); body.append(st);
  legend(body, [['--model', 'tilt (deg)'], ['--hybrid', 'torque / 10 (N m)'], ['--warn', 'ball position (m x 10)', true]]);
  const c1 = chart(body, { x: [0, T], y: [-12, 12], xlabel: 'time (s)', ylabel: 'deg', height: 190, crosshair: true, desc: 'Tilt, torque and ball position against time.' });
  legend(body, [['--model', 'closed-loop poles (1/s)'], ['--bad', 'open-loop unstable pole', true]]);
  const c2 = chart(body, { x: [-40, 10], y: [-30, 30], xlabel: 'real part (1/s)', ylabel: 'imaginary part (1/s)', height: 230, desc: 'Closed-loop poles in the complex plane. Stable when all are left of the vertical line.', ml: 50 });
  function refresh() {
    design(); simulate();
    const pl = poles();
    const stable = pl.every((p) => p.m < 1 + 1e-7);
    const pkth = Math.max(...tr.th.map((v) => Math.abs(v))) / DEG;
    const pktq = Math.max(...tr.tau.map((v) => Math.abs(v)));
    const lastBad = (() => { for (let i = tr.t.length - 1; i >= 0; i--) if (Math.abs(tr.th[i]) > 1 * DEG && tr.t[i] > 0.6 && (S.cmd === 0 || tr.t[i] < 2.2)) return tr.t[i]; return 0.6; })();
    ro.set('st', stable ? 'stable' : 'UNSTABLE'); ro.box.querySelector('.ro:first-child').className = 'ro ' + (stable ? 'good' : 'bad');
    ro.set('K', K.slice(1).map((v) => v.toFixed(1)).join('  '));
    ro.set('peak', `${pkth.toFixed(1)} deg`); ro.set('tq', `${pktq.toFixed(1)} N m`);
    ro.set('set', stable && !tr.fell ? `${Math.max(0, lastBad - 0.6).toFixed(2)} s` : 'never');
    st.className = 'status ' + (tr.fell ? 'bad' : tr.sat ? 'warn' : 'good');
    st.textContent = tr.fell ? 'The robot falls over.' : !stable ? 'Poles outside the unit circle: the linear loop is unstable.' : tr.sat ? 'The motors saturate: the torque limit is shaping the response.' : 'Balanced.';
    const maxT = Math.max(12, Math.ceil(Math.max(pkth, pktq / 10, Math.max(...tr.x.map(Math.abs)) * 10) / 4) * 4);
    c1.update({ y: [-maxT, maxT], series: [
      { data: tr.t.map((tt, i) => [tt, tr.th[i] / DEG]), color: '--model', label: 'tilt' },
      { data: tr.t.map((tt, i) => [tt, tr.tau[i] / 10]), color: '--hybrid', width: 1.5, label: 'torque/10' },
      { data: tr.t.map((tt, i) => [tt, tr.x[i] * 10]), color: '--warn', dash: '5 4', width: 1.5, label: 'x*10' }],
      bands: [{ x0: 0.5, x1: 0.6, color: '--bad', opacity: 0.15 }] });
    const ys = pl.map((p) => Math.abs(p.s[1]));
    const ymax = Math.max(30, Math.ceil(Math.max(...ys) / 10) * 10);
    const xs = pl.map((p) => p.s[0]);
    const xmin = Math.min(-40, Math.floor(Math.min(...xs) / 10) * 10), xmax = Math.max(10, Math.ceil(Math.max(...xs) / 5) * 5);
    c2.update({ x: [xmin, xmax], y: [-ymax, ymax], vlines: [{ x: 0, color: '--bad', dash: '', label: 'unstable →' }],
      dots: [{ x: ctx.data.model.pole, y: 0, color: '--bad', hollow: true, r: 6, label: `open-loop pole ${ctx.data.model.pole.toFixed(2)} 1/s` },
        ...pl.map((p) => ({ x: p.s[0], y: p.s[1], color: '--model', label: `pole ${p.s[0].toFixed(1)} ${p.s[1] >= 0 ? '+' : '-'} ${Math.abs(p.s[1]).toFixed(1)}j 1/s` }))] });
    t = Math.min(t, T); cv.redraw();
  }
  const clk = clock(stage, (d) => { if (!playing) return false; t += d; if (t >= T) { t = T; playing = false; } cv.redraw(); c1.update({ vlines: [{ x: t, color: '--text-3', dash: '2 3' }] }); return playing; });
  refresh();
}
