// Lesson 5 figures: the motion-reference layer (a port of experiments/warp_ppo/motion_ref.py) shaping raw commands, and its effect on navigation.
import { el, canvas, chart, slider, seg, button, readouts, legend, clock, palette, fmt, arrow } from './ui.js';

const wrap = (a) => { const t = (a + Math.PI) % (2 * Math.PI); return (t < 0 ? t + 2 * Math.PI : t) - Math.PI; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

class MotionRef {
  constructor(c) { Object.assign(this, c); this.s = 0; this.th = 0; this.hold = false; }
  step(vdes, pos, goal, goalOn, f = 1) {
    const dt = this.dt;
    let sd = Math.hypot(vdes[0], vdes[1]);
    let hd = Math.atan2(vdes[1], vdes[0]);
    if (sd <= 1e-4) hd = this.th;
    let e = null, d = 0;
    if (goal) {
      e = [goal[0] - pos[0], goal[1] - pos[1]]; d = Math.hypot(e[0], e[1]);
      const cap = Math.sqrt(2 * this.a_brk * f * Math.max(0, d - this.d_stop * f));
      if (goalOn) sd = Math.min(sd, cap);
      const enter = goalOn && d < this.d_hold * f && this.s < 0.1 * f;
      const leave = !goalOn || d > this.d_exit * f;
      this.hold = (this.hold || enter) && !leave;
    }
    const dh = wrap(hd - this.th);
    sd *= Math.max(0, Math.cos(dh));
    const wl = Math.min(this.w_max, (this.a_lat * f) / Math.max(this.s, 0.1 * f));
    this.th = wrap(this.th + clamp(dh, -wl * dt, wl * dt));
    this.s = Math.max(0, this.s + clamp(sd - this.s, -this.a_brk * f * dt, this.a_acc * f * dt));
    let cmd = [this.s * Math.cos(this.th), this.s * Math.sin(this.th)];
    if (goal) {
      let ph = [this.kp * e[0], this.kp * e[1]];
      const pn = Math.hypot(ph[0], ph[1]);
      const k = Math.min(1, (this.v_hold * f) / Math.max(pn, 1e-9));
      ph = [ph[0] * k, ph[1] * k];
      if (d < this.d_dead * f) ph = [0, 0];
      if (this.hold) {
        cmd = ph; this.s = Math.hypot(ph[0], ph[1]);
        if (pn > 1e-6) this.th = Math.atan2(ph[1], ph[0]);
      }
    }
    return cmd;
  }
}

const SCEN = {
  goal: { label: 'Arrive at a goal 1.2 m away', T: 10, goal: [1.2, 0.45] },
  zigzag: { label: 'Zigzag: 60 degree switches every 2 s', T: 12 },
  turn: { label: 'Turn around (180 degree reversal)', T: 10 },
  curve: { label: 'Steady curve, 0.6 rad/s', T: 12 },
};
const VMAX = 0.5;

export function compare(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const mr = ctx.data.motion_ref;
  const S = { sc: 'goal', a_acc: mr.a_acc, a_brk: mr.a_brk, a_lat: mr.a_lat, w_max: mr.w_max, tau: 0.3 };
  let runs = null, t = 0, playing = false;

  function desired(sc, t, pos) {
    if (sc === 'goal') {
      const g = SCEN.goal.goal, e = [g[0] - pos[0], g[1] - pos[1]], d = Math.hypot(e[0], e[1]);
      if (d < ctx.data.goal_rule.d_zero) return [0, 0];
      const sp = Math.min(ctx.data.goal_rule.gain * d, VMAX);
      return [(e[0] / d) * sp, (e[1] / d) * sp];
    }
    if (sc === 'zigzag') { const h = (Math.floor(t / 2) % 2 ? -1 : 1) * (Math.PI / 6); return t < 0.5 ? [0, 0] : [0.3 * Math.cos(h), 0.3 * Math.sin(h)]; }
    if (sc === 'turn') return t < 0.5 ? [0, 0] : t < 5 ? [0.4, 0] : [-0.4, 0];
    const h = 0.6 * Math.max(0, t - 0.5); return t < 0.5 ? [0, 0] : [0.3 * Math.cos(h), 0.3 * Math.sin(h)];
  }
  function simulate(useLayer) {
    const dt = mr.dt, T = SCEN[S.sc].T, n = Math.round(T / dt);
    const m = new MotionRef({ ...mr, a_acc: S.a_acc, a_brk: S.a_brk, a_lat: S.a_lat, w_max: S.w_max });
    let pos = [0, 0], vel = [0, 0];
    const out = { t: [], pos: [], cmd: [], sp: [], hr: [], d: [] };
    let prevH = null;
    const goal = SCEN[S.sc].goal || null;
    for (let i = 0; i <= n; i++) {
      const tt = i * dt;
      const v0 = desired(S.sc, tt, pos);
      const cmd = useLayer ? m.step(v0, pos, goal, !!goal) : v0;
      const sp = Math.hypot(cmd[0], cmd[1]);
      let hr = 0;
      if (sp > 0.02) { const h = Math.atan2(cmd[1], cmd[0]); if (prevH !== null) hr = wrap(h - prevH) / dt; prevH = h; } else prevH = null;
      out.t.push(tt); out.pos.push(pos.slice()); out.cmd.push(cmd); out.sp.push(sp); out.hr.push(hr);
      out.d.push(goal ? Math.hypot(goal[0] - pos[0], goal[1] - pos[1]) : 0);
      const a = dt / S.tau;
      vel = [vel[0] + (cmd[0] - vel[0]) * a, vel[1] + (cmd[1] - vel[1]) * a];
      pos = [pos[0] + vel[0] * dt, pos[1] + vel[1] * dt];
    }
    return out;
  }
  function metrics(o) {
    const hs = o.hr.filter((v, i) => o.sp[i] > 0.02);
    const rms = hs.length ? Math.sqrt(hs.reduce((s, v) => s + v * v, 0) / hs.length) : 0;
    let rev = 0; const k = 10;
    for (let i = k; i < o.cmd.length; i += k) { const a = o.cmd[i], b = o.cmd[i - k]; if (Math.hypot(...a) > 0.05 && Math.hypot(...b) > 0.05 && a[0] * b[0] + a[1] * b[1] < 0) rev++; }
    const g = SCEN[S.sc].goal; let over = 0, hold = 0;
    if (g) { over = Math.max(0, Math.max(...o.pos.map((p) => p[0])) - g[0]); const tail = o.d.slice(-300); hold = Math.sqrt(tail.reduce((s, v) => s + v * v, 0) / tail.length); }
    return { rms, rev, over, hold };
  }
  function compute() { runs = { raw: simulate(false), lay: simulate(true) }; }

  compute();
  const cv = canvas(stage, (w) => Math.min(340, Math.max(220, w * 0.42)), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const all = [...runs.raw.pos, ...runs.lay.pos, ...(SCEN[S.sc].goal ? [SCEN[S.sc].goal, [0, 0]] : [])];
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    let x0 = Math.min(...xs) - 0.15, x1 = Math.max(...xs) + 0.15, y0 = Math.min(...ys) - 0.15, y1 = Math.max(...ys) + 0.15;
    const sc = Math.min((w - 20) / (x1 - x0), (h - 20) / (y1 - y0));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const X = (x) => w / 2 + (x - cx) * sc, Y = (y) => h / 2 - (y - cy) * sc;
    g.strokeStyle = pal.line; g.lineWidth = 1; g.fillStyle = pal.text3; g.font = '10px ' + getComputedStyle(document.body).fontFamily;
    for (let m = Math.ceil(x0 * 2) / 2; m <= x1; m += 0.5) { g.beginPath(); g.moveTo(X(m), 6); g.lineTo(X(m), h - 6); g.stroke(); g.fillText(`${m.toFixed(1)} m`, X(m) + 2, h - 8); }
    for (let m = Math.ceil(y0 * 2) / 2; m <= y1; m += 0.5) { g.beginPath(); g.moveTo(6, Y(m)); g.lineTo(w - 6, Y(m)); g.stroke(); }
    const goal = SCEN[S.sc].goal;
    if (goal) { g.strokeStyle = pal.good; g.lineWidth = 2; g.beginPath(); g.arc(X(goal[0]), Y(goal[1]), 0.2 * sc, 0, 6.3); g.stroke(); g.fillStyle = pal.good; g.beginPath(); g.arc(X(goal[0]), Y(goal[1]), 4, 0, 6.3); g.fill(); }
    const i = Math.min(runs.raw.t.length - 1, Math.round(t / mr.dt));
    for (const [r, col] of [[runs.raw, pal.warn], [runs.lay, pal.hybrid]]) {
      g.strokeStyle = col; g.lineWidth = 2.2; g.lineJoin = 'round'; g.beginPath();
      for (let k = 0; k <= i; k += 2) { const p = r.pos[k]; k ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])); }
      g.stroke();
      const p = r.pos[i]; g.fillStyle = col; g.beginPath(); g.arc(X(p[0]), Y(p[1]), 6, 0, 6.3); g.fill(); g.strokeStyle = pal.bg; g.lineWidth = 2; g.stroke();
      const c = r.cmd[i]; if (Math.hypot(...c) > 0.03) arrow(g, X(p[0]), Y(p[1]), X(p[0]) + c[0] * sc * 0.9, Y(p[1]) - c[1] * sc * 0.9, col, 1.6);
    }
    g.fillStyle = pal.text3; g.fillText(`t = ${t.toFixed(1)} s`, 10, 14);
  }, ctx);
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Top-down view of two paths followed by an illustrative robot: one driven by raw commands, one through the motion-reference layer. Numbers below.');
  legend(body, [['--warn', 'raw command'], ['--hybrid', 'through the layer']]);
  const sg = el('div', { class: 'btns' }); body.append(sg);
  seg(sg, Object.entries(SCEN).map(([k, v]) => [k, v.label]), S.sc, (k) => { S.sc = k; refresh(true); }, 'scenario');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const mk = (key, label, min, max, step, unit) => slider(ctl, { label, min, max, step, value: S[key], fmt: (v) => `${v.toFixed(2)} ${unit}`, onInput: (v) => { S[key] = v; refresh(); } });
  const sl = [mk('a_acc', 'Acceleration limit', 0.1, 1.5, 0.05, 'm/s²'), mk('a_brk', 'Braking limit', 0.1, 1.5, 0.05, 'm/s²'), mk('a_lat', 'Lateral acceleration limit', 0.05, 1, 0.05, 'm/s²'), mk('w_max', 'Turn-rate limit', 0.2, 4, 0.1, 'rad/s'), mk('tau', 'Follower lag (illustrative)', 0.1, 1.2, 0.05, 's')];
  const bt = el('div', { class: 'btns' }); body.append(bt);
  button(bt, 'Play', () => { t = 0; if (ctx.reduced) { t = SCEN[S.sc].T; cv.redraw(); } else { playing = true; clk.start(); } }, 'primary');
  button(bt, 'Defaults', () => { S.a_acc = mr.a_acc; S.a_brk = mr.a_brk; S.a_lat = mr.a_lat; S.w_max = mr.w_max; S.tau = 0.3; sl.forEach((s, i) => s.set([mr.a_acc, mr.a_brk, mr.a_lat, mr.w_max, 0.3][i])); refresh(); });
  const ro = readouts(body, [['hr0', 'raw: command heading-rate RMS', 'warn'], ['hr1', 'layer: command heading-rate RMS', 'hybrid'], ['rv0', 'raw: command reversals', 'warn'], ['rv1', 'layer: command reversals', 'hybrid'], ['ov', 'goal overshoot (raw / layer)', ''], ['hd', 'final-3-s distance to goal (raw / layer)', '']]);
  const c1 = chart(body, { x: [0, 10], y: [0, 0.6], xlabel: 'time (s)', ylabel: 'speed (m/s)', height: 150, crosshair: true, desc: 'Command speed over time, raw and through the layer.' });
  const c2 = chart(body, { x: [0, 10], y: [-6, 6], xlabel: 'time (s)', ylabel: 'heading rate (rad/s)', height: 150, crosshair: true, desc: 'Heading rate of the command over time.' });
  const note = el('p', { class: 'hint', text: 'The follower is a first-order lag standing in for the balance controller; it is not the trained policy. What the figure shows is exactly what the layer changes: the command. How well the real robot follows it is measured on the right (S106, S110, S114).' });
  body.append(note);
  function refresh(reset) {
    compute();
    const m0 = metrics(runs.raw), m1 = metrics(runs.lay), T = SCEN[S.sc].T;
    const sm = (a) => a.map((v, i) => [runs.raw.t[i], v]).filter((_, i) => i % 5 === 0);
    c1.update({ x: [0, T], series: [{ data: sm(runs.raw.sp), color: '--warn', label: 'raw', width: 1.8 }, { data: sm(runs.lay.sp), color: '--hybrid', label: 'layer', width: 2.4 }] });
    const lim = clamp(Math.max(...runs.raw.hr.map(Math.abs), ...runs.lay.hr.map(Math.abs)), 2, 12);
    c2.update({ x: [0, T], y: [-Math.ceil(lim), Math.ceil(lim)], series: [{ data: sm(runs.raw.hr), color: '--warn', label: 'raw', width: 1.4 }, { data: sm(runs.lay.hr), color: '--hybrid', label: 'layer', width: 2.4 }] });
    ro.set('hr0', `${m0.rms.toFixed(2)} rad/s`); ro.set('hr1', `${m1.rms.toFixed(2)} rad/s`); ro.set('rv0', `${m0.rev}`); ro.set('rv1', `${m1.rev}`);
    ro.set('ov', SCEN[S.sc].goal ? `${(m0.over * 100).toFixed(0)} / ${(m1.over * 100).toFixed(0)} cm` : 'n/a'); ro.set('hd', SCEN[S.sc].goal ? `${(m0.hold * 100).toFixed(0)} / ${(m1.hold * 100).toFixed(0)} cm` : 'n/a');
    if (reset || t > T) t = T;
    cv.redraw();
  }
  const clk = clock(stage, (dt) => { if (!playing) return false; t += dt * 1.5; const T = SCEN[S.sc].T; if (t >= T) { t = T; playing = false; } cv.redraw(); return playing; });
  refresh(true);
}

export function navlayer(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const rows = ctx.data.motion_nav.rows;
  let sel = 0;
  legend(stage, [['--hybrid', 'path wobble: heading-rate RMS of the planner command (rad/s), lower is smoother']]);
  const ch = chart(stage, { x: [-0.6, rows.length - 0.4], y: [0, 2], xticks: rows.map((_, i) => i), xfmt: (v) => (rows[v] ? rows[v].short : ''), ylabel: 'rad/s', height: 230, ml: 38, desc: 'Heading-rate RMS of the navigation path for several layer settings.' });
  const sg = el('div', { class: 'btns' }); body.append(sg);
  seg(sg, rows.map((r, i) => [i, r.short]), 0, (i) => { sel = i; draw(); }, 'layer setting');
  const ro = readouts(body, [['d', 'setting', ''], ['s', 'reached the goal', 'good'], ['f', 'fell', 'bad'], ['k', 'stuck', 'warn'], ['t', 'median time to goal', 'hybrid']]);
  function draw() {
    ch.update({ bars: rows.map((r, i) => ({ x: i, y: r.rms, color: i === sel ? '--hybrid' : '--model', opacity: i === sel ? 1 : 0.6, label: `${r.short}: ${r.rms} rad/s` })) });
    const r = rows[sel];
    ro.set('d', r.long); ro.set('s', r.success == null ? 'n/a' : r.success.toFixed(3)); ro.set('f', r.fall == null ? 'n/a' : r.fall.toFixed(3)); ro.set('k', r.stuck == null ? 'n/a' : r.stuck.toFixed(3)); ro.set('t', r.time == null ? 'n/a' : `${r.time} s`);
  }
  draw();
}
