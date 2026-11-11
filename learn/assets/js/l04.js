// Lesson 4 figures: residual authority, the adaptive-authority rule on a simulated run, per-design falls and the seed spread.
import { paramsFrom, linearize, dlqr, PlanarBot, planeTorqueMax } from './physics.js';
import { el, chart, slider, seg, button, readouts, legend, palette, fmt } from './ui.js';

const DEG = Math.PI / 180;
const mid = (a) => (a[0] + a[1]) / 2;

// ---------------------------------------------------------------- applied = clip(lqr + rs * residual), and the authority dial
export function authority(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const rows = ctx.data.authority_curve.rows;
  let sel = 1;
  legend(stage, [['--model', 'tracking: achieved / commanded speed (bar = range over runs)'], ['--learned', 'robustness: goal success with pushes and pulls']]);
  const ch = chart(stage, { x: [-0.5, rows.length - 0.5], y: [0, 1.15], xticks: rows.map((_, i) => i), xfmt: (v) => (rows[v] ? rows[v].label : ''), ylabel: '', height: 250, xlabel: 'learned authority rs (how much the policy may change)', desc: 'Tracking accuracy and robustness against residual authority.', ml: 36 });
  const ro = readouts(body, [['rs', 'authority', ''], ['trk', 'speed achieved / commanded (flat, S82, S83)', 'model'], ['rob', 'goal success with pushes and pulls (DR 1)', 'learned']]);
  const sg = el('div', { class: 'btns' }); body.append(sg);
  seg(sg, rows.map((r, i) => [i, r.label]), sel, (i) => { sel = i; draw(); }, 'authority');
  const txt = el('p', { class: 'status' }); body.append(txt);
  const notes = ['Tracks speed exactly (1.01 to 1.02) but is the least robust: 0.35 with pushes and pulls. It is the cruising level of the deployed adaptive rule.', 'The knee of the fixed-authority curve: exact tracking (0.99 to 1.01) and most of the robustness gain. Earlier recipes used it as a fixed level; the deployed controller schedules 0.15 to 0.3 (next sections).', 'More robustness, tracking starts to slip (0.92 to 0.97).', 'Robustness saturates (0.46 to 0.49) while tracking falls to 0.80 to 0.90.', 'A policy with no LQR base: most robust on its training designs (0.50), but only 0.66 to 0.83 of the commanded speed, and it does not transfer to unseen robots (see below).'];
  function draw() {
    const r = rows[sel];
    ch.update({
      bars: rows.flatMap((q, i) => [{ x: i, dx: -12, w: 18, y: mid(q.robust), color: '--learned', opacity: i === sel ? 1 : 0.55, label: `${q.label}: robustness ${q.robust.map((v) => v.toFixed(2)).join(' to ')}` }]),
      dots: rows.flatMap((q, i) => [{ x: i + 0.0, y: mid(q.tracking), color: '--model', r: i === sel ? 7 : 5, label: `${q.label}: tracking ${q.tracking.map((v) => v.toFixed(2)).join(' to ')}` }]),
      hlines: [{ y: 1, color: '--text-3', label: 'command' }],
      series: rows.map((q, i) => ({ data: [[i, q.tracking[0]], [i, q.tracking[1]]], color: '--model', width: 3 })),
    });
    ro.set('rs', r.rs >= 1 ? 'pure learning' : r.rs.toFixed(2));
    ro.set('trk', r.tracking.map((v) => v.toFixed(2)).join(' to '));
    ro.set('rob', r.robust.map((v) => v.toFixed(2)).join(' to '));
    txt.textContent = notes[sel];
  }
  draw();
}

// ---------------------------------------------------------------- toy: how authority scales the learned part
export function mix(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  let rs = 0.25;
  const sig = (t) => 0.35 * Math.sin(2 * Math.PI * 0.8 * t) + 0.18 * Math.sin(2 * Math.PI * 2.1 * t + 1);
  const res = (t) => 1.6 * Math.sin(2 * Math.PI * 1.4 * t + 0.5) + 0.6 * Math.sin(2 * Math.PI * 3.3 * t);
  const ts = Array.from({ length: 301 }, (_, i) => i * 0.01);
  legend(stage, [['--model', 'LQR action u'], ['--learned', 'learned output a (illustrative signal)', true], ['--hybrid', 'applied = clip(u + rs a, -1, 1)']]);
  const ch = chart(stage, { x: [0, 3], y: [-1.2, 1.2], xlabel: 'time (s)', ylabel: 'normalised action', height: 200, crosshair: true, desc: 'LQR action, an illustrative learned output and the applied action.' });
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Residual authority rs', min: 0, max: 1, step: 0.01, value: rs, fmt: (v) => v.toFixed(2), onInput: (v) => { rs = v; draw(); } });
  const ro = readouts(body, [['sh', 'share of the action that is learned (RMS)', 'learned']]);
  function draw() {
    const rms = (f) => Math.sqrt(ts.reduce((s, t) => s + f(t) ** 2, 0) / ts.length);
    ch.update({ series: [
      { data: ts.map((t) => [t, sig(t)]), color: '--model', label: 'u' },
      { data: ts.map((t) => [t, Math.max(-1.2, Math.min(1.2, res(t)))]), color: '--learned', dash: '5 4', width: 1.4, label: 'a', opacity: 0.8 },
      { data: ts.map((t) => [t, Math.max(-1, Math.min(1, sig(t) + rs * res(t)))]), color: '--hybrid', width: 2.4, label: 'applied' }] });
    const a = rms((t) => rs * res(t)), u = rms(sig);
    ro.set('sh', `${(100 * a / (a + u)).toFixed(0)} percent`);
  }
  draw();
}

// ---------------------------------------------------------------- adaptive authority, the real rule on a simulated run
export function adaptive(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const cfg = ctx.data.authority;
  const P = paramsFrom(ctx.data.model);
  const { A, B } = linearize(P);
  const K = dlqr(A, B, ctx.data.lqr.q, ctx.data.lqr.r, 0.01);
  const tauMax = planeTorqueMax(ctx.data.model.n, 30);
  const S = { vdb: cfg.v_db, push: 0, stop: true };
  legend(stage, [['--hybrid', 'authority rs'], ['--model', 'command (m/s)'], ['--warn', 'ball speed (m/s)', true]]);
  const ch = chart(stage, { x: [0, 12], y: [0, 0.35], xlabel: 'time (s)', ylabel: 'authority rs', height: 210, crosshair: true, desc: 'Residual authority over a simulated run, with command, ball speed and tilt.' });
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Hold dead band (S113): near-zero term counts only above', min: 0, max: 0.2, step: 0.01, value: S.vdb, fmt: (v) => (v === 0 ? 'off' : `${v.toFixed(2)} m/s`), onInput: (v) => { S.vdb = v; draw(); } });
  slider(ctl, { label: 'Push at 9 s', min: 0, max: 40, step: 1, value: S.push, fmt: (v) => `${v} N`, onInput: (v) => { S.push = v; draw(); } });
  const ro = readouts(body, [['hi', 'time at the high level', 'learned'], ['lo', 'time at the low level', 'model'], ['why', 'what raised it first', '']]);
  const cmdAt = (t) => (t < 1 ? 0 : t < 3 ? 0.4 : t < 6 ? 0.9 : t < 7 ? 0.0 : 0);   // cruise, speed up, stop at 6 s
  function run() {
    const bot = new PlanarBot(P, { K, tauMax, ki: 1, a_max: 0.5 });
    let w = cfg.w_init, prev = null;
    const out = { t: [], rs: [], cmd: [], v: [], th: [], why: [] };
    const first = {};
    for (let i = 0; i <= 1200; i++) {
      const t = i * 0.01, c = cmdAt(t);
      const dc = prev === null ? 0 : Math.abs(c - prev) / 0.01; prev = c;
      const tilt = Math.abs(bot.theta) / DEG;
      let near0 = Math.min(1, Math.max(0, 1 - Math.abs(c) / (cfg.near0_frac * 1)));
      if (S.vdb > 0) near0 *= +(Math.abs(bot.vball) > S.vdb);
      const nd = Math.min(1, dc / cfg.dc_scale), nt = Math.min(1, Math.max(0, (tilt - cfg.tilt0_deg) / cfg.tilt_span_deg));
      const need = Math.max(nd, near0, nt);
      w = Math.max(need, w * cfg.decay);
      const rs = cfg.lo + (cfg.hi - cfg.lo) * w;
      const why = need > 0.5 ? (nd >= near0 && nd >= nt ? 'command change' : nt >= near0 ? 'tilt' : 'near-zero command') : '';
      if (why && t > 0.5 && !first[why]) first[why] = t;
      out.t.push(t); out.rs.push(rs); out.cmd.push(c); out.v.push(Math.abs(bot.vball)); out.th.push(tilt / 10); out.why.push(why);
      bot.tick(c, t >= 9 && t < 9.1 ? S.push : 0);
    }
    return { out, first };
  }
  function draw() {
    const { out } = run();
    const hi = out.rs.filter((v) => v > cfg.hi - 0.02).length / 100, lo = out.rs.filter((v) => v < cfg.lo + 0.02).length / 100;
    const firstWhy = out.why.find((x, i) => x && out.t[i] > 0.5);
    ch.update({ y: [0, 1], series: [
      { data: out.t.map((t, i) => [t, out.rs[i]]), color: '--hybrid', width: 2.6, label: 'rs' },
      { data: out.t.map((t, i) => [t, out.cmd[i]]), color: '--model', width: 1.4, label: 'command' },
      { data: out.t.map((t, i) => [t, Math.min(1, out.v[i])]), color: '--warn', dash: '5 4', width: 1.4, label: 'speed' },
    ],
      hlines: [{ y: cfg.lo, color: '--text-3', label: `cruise ${cfg.lo}` }, { y: cfg.hi, color: '--text-3', label: `max ${cfg.hi}` }] });
    ro.set('hi', `${hi.toFixed(1)} s of 12`); ro.set('lo', `${lo.toFixed(1)} s of 12`); ro.set('why', firstWhy || 'none');
  }
  draw();
}

// ---------------------------------------------------------------- per-design falls, full controller against LQR alone
export function designs(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const rows = ctx.data.designs.rows, fg = ctx.data.flat_goal;
  legend(stage, [['--hybrid', 'LQR + residual (g34)'], ['--model', 'LQR alone']]);
  const ch = chart(stage, { x: [-0.6, rows.length - 0.4], y: [0, 1], xticks: rows.map((_, i) => i), xfmt: (v) => (rows[v] ? rows[v].name : ''), ylabel: 'fall fraction', height: 250, desc: 'Fall fraction on the flat goal test per design, for the full controller and for the LQR alone.', ml: 40 });
  const bars = [];
  rows.forEach((r, i) => {
    bars.push({ x: i, dx: -7, w: 12, y: fg.lqr[r.name], color: '--model', label: `${r.label} (${r.name}), LQR alone: ${fg.lqr[r.name].toFixed(2)} fell` });
    bars.push({ x: i, dx: 7, w: 12, y: fg.full[r.name], color: '--hybrid', label: `${r.label} (${r.name}), full controller: ${fg.full[r.name].toFixed(2)} fell` });
  });
  ch.update({ bars });
  const ro = readouts(body, [['f', 'all designs: full controller', 'hybrid'], ['l', 'all designs: LQR alone', 'model']]);
  ro.set('f', `${fg.full_all.toFixed(3)} fell`); ro.set('l', `${fg.lqr_all.toFixed(3)} fell`);
  const sel = el('div', { class: 'ctls' }); body.append(sel);
  const tb = el('table'); const wrap = el('div', { class: 'tablewrap' }, tb); body.append(wrap);
  body.insertBefore(el('p', { class: 'cap', style: { padding: '0', margin: '.6em 0 0' }, text: 'What each design is (same names as the bars). 48 worlds per design, 15 s, flat map.' }), wrap);
  tb.append(el('tr', {}, el('th', { text: 'design' }), el('th', { text: 'what differs' }), el('th', { text: 'mass kg' })));
  rows.forEach((r) => tb.append(el('tr', {}, el('td', { text: `${r.label} (${r.name})` }), el('td', { text: r.what }), el('td', { class: 'n', text: r.m_total.toFixed(2) }))));
}

// ---------------------------------------------------------------- seed spread of goal hunting
export function seeds(root, ctx) {
  const stage = root.querySelector('.stage');
  const rows = ctx.data.seeds.rows;
  legend(stage, [['--learned', 'g32 (no dead band)'], ['--hybrid', 'g33 (tracking weight 3)'], ['--model', 'g34 (dead band 0.05 m/s)']]);
  chart(stage, {
    x: [-0.6, rows.length - 0.4], y: [0, 5.5], xticks: rows.map((_, i) => i), xfmt: (v) => (rows[v] ? rows[v][0] : ''), ylabel: 'command reversals per world', height: 230, ml: 44,
    bars: rows.map((r, i) => ({ x: i, y: r[1], color: r[0].startsWith('g32') ? '--learned' : r[0].startsWith('g33') ? '--hybrid' : '--model', label: `${r[0]}: ${r[1]} reversals per world` })),
    desc: 'Command reversals per world in the flat goal test for each seed.',
  });
}

// ---------------------------------------------------------------- held-out robot
export function heldout(root, ctx) {
  const stage = root.querySelector('.stage');
  legend(stage, [['--hybrid', 'LQR + residual 0.25'], ['--learned', 'pure learning']]);
  const names = ['undisturbed', 'pushes + pulls'];
  chart(stage, {
    x: [-0.6, 1.6], y: [0, 1], xticks: [0, 1], xfmt: (v) => names[v] || '', ylabel: 'goal success', height: 220, ml: 40,
    bars: [
      { x: 0, dx: -26, w: 40, y: 0.71, color: '--hybrid', label: 'LQR + residual, undisturbed: 0.71' }, { x: 0, dx: 26, w: 40, y: 0.09, color: '--learned', label: 'pure learning, undisturbed: 0.09' },
      { x: 1, dx: -26, w: 40, y: 0.66, color: '--hybrid', label: 'LQR + residual, pushes and pulls: 0.66' }, { x: 1, dx: 26, w: 40, y: 0.15, color: '--learned', label: 'pure learning, pushes and pulls: 0.15' }],
    desc: 'Goal success on an unseen 1.5 times larger robot.',
  });
}
