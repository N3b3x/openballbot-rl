// Lesson 8 figures: the design explorer over the S119 sweep, a guideline checker, and the sustained-pull limit.
import { el, chart, slider, seg, button, readouts, legend, fmt } from './ui.js';

const G = 9.81;
const LABEL = { ball_mass_kg: 'Ball mass', motor_torque_limit: 'Motor torque limit', rotor_inertia: 'Rotor inertia', body_com_height_m: 'Body COM height', body_mass_kg: 'Body mass', contact_zenith_deg: 'Wheel contact angle', ball_radius_m: 'Ball radius' };
const UNIT = { ball_mass_kg: 'kg', motor_torque_limit: 'N m per wheel', rotor_inertia: 'kg m²', body_com_height_m: 'm', body_mass_kg: 'kg', contact_zenith_deg: 'deg', ball_radius_m: 'm' };
const METRICS = [['goal_still', 'standing / holding a goal', '--good'], ['goal_pushes', 'with short pushes', '--model'], ['goal_pulls_half', 'with half-strength pulls', '--warn'], ['drive_0.3', 'driving at 0.3 m/s', '--learned'], ['goal_terrain', 'goals on terrain', '--bad']];

export function explorer(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const groups = ctx.data.sweep.groups;
  const facts = ctx.data.facts.find((f) => f.what === 'LQR-only check').values;
  let gi = 0, selRow = Math.max(0, groups[0].rows.findIndex((r) => r.nominal)), on = new Set(['goal_still', 'goal_pulls_half', 'goal_terrain']);
  const sg = el('div', { class: 'btns', style: { paddingBottom: 0 } }); stage.append(sg);
  seg(sg, groups.map((g, i) => [i, LABEL[g.vary] || g.vary]), 0, (i) => { gi = i; selRow = Math.max(0, groups[gi].rows.findIndex((r) => r.nominal)); draw(); }, 'property to vary');
  const lg = el('div', { class: 'btns', style: { paddingBottom: 0, paddingTop: 0 } }); stage.append(lg);
  METRICS.forEach(([k, lab, col]) => {
    const b = el('button', { type: 'button', class: 'btn', 'aria-pressed': String(on.has(k)), text: lab, style: { borderColor: `var(${col})` } });
    b.onclick = () => { on.has(k) ? on.delete(k) : on.add(k); b.setAttribute('aria-pressed', String(on.has(k))); draw(); };
    lg.append(b);
  });
  const ch = chart(stage, { x: [-0.5, 5], y: [0, 1], height: 260, ylabel: 'fraction of robots that fall', ml: 44, desc: 'Fall fractions against the varied property.' });
  const sel = el('div', { class: 'btns' }); body.append(sel);
  const ro = readouts(body, [['v', 'selected value', ''], ['p', 'unstable pole p (planar)', 'model'], ['z', 'zero z', 'model'], ['k', 'lean per m/s²', 'model'], ['mr', 'ball mass / body mass', 'warn'], ['pc', 'static pull capacity', 'hybrid'], ['bd', 'binding analytic limit', ''], ['am', 'lean-limited max acceleration', '']]);
  const note = el('p', { class: 'status warn' }); body.append(note);
  function draw() {
    const g = groups[gi], rows = g.rows;
    sel.replaceChildren();
    seg(sel, rows.map((r, i) => [i, `${+r.value.toPrecision(3)}${r.nominal ? ' (nominal)' : ''}`]), selRow, (i) => { selRow = i; draw2(); }, 'value');
    ch.update({
      x: [-0.5, rows.length - 0.5], xticks: rows.map((_, i) => i), xfmt: (v) => (rows[v] ? `${+rows[v].value.toPrecision(3)}` : ''), xlabel: `${LABEL[g.vary]} (${UNIT[g.vary]})`,
      y: [0, Math.min(1, Math.max(0.2, Math.ceil(Math.max(...METRICS.filter((m) => on.has(m[0])).flatMap((m) => rows.map((r) => r.sim[m[0]]))) * 10) / 10 + 0.05))],
      series: METRICS.filter((m) => on.has(m[0])).map((m) => ({ data: rows.map((r, i) => [i, r.sim[m[0]]]), color: m[2], width: 2, label: m[1] })),
      dots: METRICS.filter((m) => on.has(m[0])).flatMap((m) => rows.map((r, i) => ({ x: i, y: r.sim[m[0]], color: m[2], r: r.nominal ? 6 : 4, hollow: r.nominal, label: `${LABEL[g.vary]} ${+r.value.toPrecision(3)}: ${m[1]} ${r.sim[m[0]].toFixed(3)} fell` }))),
      vlines: [{ x: selRow, color: '--text-3', dash: '2 3' }],
    });
    draw2();
  }
  function draw2() {
    const g = groups[gi]; selRow = Math.min(selRow, g.rows.length - 1);
    const r = g.rows[selRow], ph = r.phys;
    ro.set('v', `${+r.value.toPrecision(3)} ${UNIT[g.vary]}`); ro.set('p', `${ph.p.toFixed(2)} 1/s`); ro.set('z', `${ph.z.toFixed(2)} 1/s`); ro.set('k', `${ph.kappa_deg.toFixed(1)} deg`);
    ro.set('mr', `${(100 * ph.m_k / ph.m_a).toFixed(1)} %`); ro.set('pc', `${ph.pull_capacity_5deg_N.toFixed(1)} N`); ro.set('bd', `${ph.binding}${ph.feasible ? '' : ' (flagged infeasible)'}`); ro.set('am', `${ph.a_max.toFixed(2)} m/s²`);
    let msg = '';
    if (g.vary === 'ball_radius_m') msg = `Extrapolation warning: the controller never trained on another ball size. LQR alone vs the full controller falls: 12 cm ball ${facts.r12[1]} vs ${facts.r12[0]}, 7 cm ball ${facts.r7[1]} vs ${facts.r7[0]}. The 12 cm ball is physically easy; the learned part fails on it.`;
    else if (g.vary === 'rotor_inertia' && r.value < 0.002) msg = 'The LQR alone fails equally (0.45 against 0.49), so this is a physical limit of the build, not a training gap (S119).';
    else if (g.vary === 'motor_torque_limit' && r.value < 6) msg = 'LQR alone falls as often (3 N m: 0.48 against 0.46): a physical limit of the build (S119).';
    else if (g.vary === 'body_com_height_m' && r.phys.feasible !== 1 && r.phys.feasible !== true) msg = 'The analytic traction check flags this height infeasible by asking for recovery from a 15 degree tilt; the controller never reaches it, so the check is too conservative here (S119).';
    note.textContent = msg; note.style.display = msg ? '' : 'none';
  }
  draw();
}

// ------------------------------------------------------------------------------------------------ checker
const RULES = (v) => {
  const ratio = v.ball / v.body, lr = v.com / v.r, pull = v.body * G * Math.sin(5 * Math.PI / 180);
  const st = (ok, warn) => (ok ? 'good' : warn ? 'warn' : 'bad');
  return [
    { k: 'ball_mass_kg', name: 'Ball mass', val: `${(100 * ratio).toFixed(1)} percent of body mass`, s: st(ratio >= 0.04, ratio >= 0.02), tip: 'Rule 1: at least 4 percent, 8 to 15 percent better. 1 percent: 41 percent of robots fall standing; 2 percent: 5 percent; 4.4 percent: 0.7 percent; 9 to 17 percent: none (S119).', x: v.ball },
    { k: 'motor_torque_limit', name: 'Motor torque', val: `${v.tau} N m per wheel peak`, s: st(v.tau >= 15, v.tau >= 10), tip: 'Rule 2: at least 15 N m peak for a 4 kg robot; 10: six times more falls; 6: 13 percent fall; 3: cannot stand (S119).', x: v.tau },
    { k: 'rotor_inertia', name: 'Rotor inertia', val: `${v.J} kg m²`, s: st(v.J >= 0.005, v.J >= 0.003), tip: 'Rule 3: not below 0.005; 0.02 was the best variant; 0.001: 43 percent fall and the LQR alone falls as often (S119).', x: v.J },
    { k: 'body_com_height_m', name: 'COM height', val: `${lr.toFixed(2)} ball radii`, s: st(lr >= 1.2 && lr <= 2.6, lr >= 0.9), tip: 'Rule 4: at least the nominal 1.4 radii, higher is easier up to the tested 2.4 (S119); this checker passes 1.2 to 2.6.', x: v.com },
    { k: 'body_mass_kg', name: 'Body mass', val: `${v.body} kg, pull capacity ${pull.toFixed(1)} N`, s: st(v.body >= 3 && v.body <= 5, v.body >= 2 && v.body <= 6.5), tip: 'Rule 5: 3 to 5 kg; below 3 kg steady pulls topple it (2 kg: 24 percent under half pulls); above 5 kg standing falls rise slightly (S119).', x: v.body },
    { k: 'contact_zenith_deg', name: 'Wheel contact angle', val: `${v.zen} deg from the top`, s: st(Math.abs(v.zen - 51) <= 4, Math.abs(v.zen - 51) <= 11), tip: 'Rule 6: about 50 degrees; 41 and 61 are somewhat worse (S119).', x: v.zen },
    { k: 'ball_radius_m', name: 'Ball size', val: `${(v.r * 100).toFixed(1)} cm radius`, s: st(v.r >= 0.09 && v.r <= 0.1, v.r >= 0.07 && v.r <= 0.12), tip: 'Rule 7: 9 cm or larger; a larger ball needs the controller retrained; 7 cm is genuinely harder (S119).', x: v.r },
  ];
};

export function checker(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const groups = Object.fromEntries(ctx.data.sweep.groups.map((g) => [g.vary, g]));
  const V = { ball: 0.168, body: 3.85, com: 0.131, tau: 30, J: 0.005, zen: 51, r: 0.09 };
  const ctl = el('div', { class: 'ctls' }); stage.append(ctl);
  const mk = (key, label, min, max, step, f) => slider(ctl, { label, min, max, step, value: V[key], fmt: f, onInput: (x) => { V[key] = x; draw(); } });
  mk('r', 'Ball radius', 0.05, 0.14, 0.005, (v) => `${(v * 100).toFixed(1)} cm`);
  mk('ball', 'Ball mass', 0.03, 0.8, 0.005, (v) => `${v.toFixed(3)} kg`);
  mk('body', 'Body mass', 1, 8, 0.1, (v) => `${v.toFixed(1)} kg`);
  mk('com', 'COM above the ball centre', 0.06, 0.26, 0.005, (v) => `${(v * 100).toFixed(1)} cm`);
  mk('tau', 'Peak motor torque per wheel', 2, 40, 1, (v) => `${v} N m`);
  mk('J', 'Rotor inertia', 0.001, 0.03, 0.001, (v) => `${v.toFixed(3)} kg m²`);
  mk('zen', 'Wheel contact angle', 35, 70, 1, (v) => `${v} deg`);
  const bt = el('div', { class: 'btns' }); stage.append(bt);
  button(bt, 'Reset to nominal', () => location.reload());
  const list = el('ul', { style: { listStyle: 'none', padding: '0 18px 12px', margin: 0, display: 'grid', gap: '8px' } }); body.append(list);
  const sum = el('p', { class: 'status' }); body.append(sum);
  function nearest(k, x) { const rows = groups[k].rows; return rows.reduce((b, r) => (Math.abs(Math.log(r.value / x)) < Math.abs(Math.log(b.value / x)) ? r : b)); }
  function draw() {
    const rules = RULES(V);
    list.replaceChildren(...rules.map((r) => {
      const n = nearest(r.k, r.x);
      const col = { good: 'var(--good)', warn: 'var(--warn)', bad: 'var(--bad)' }[r.s];
      return el('li', { style: { background: 'var(--bg-3)', borderLeft: `4px solid ${col}`, borderRadius: '0 10px 10px 0', padding: '8px 12px' } },
        el('b', { text: `${r.name}: ` }), el('span', { class: 'mono', style: { color: col }, text: `${r.val} (${{ good: 'within the guideline', warn: 'borderline', bad: 'outside the guideline' }[r.s]})` }),
        el('div', { style: { fontSize: '.8rem', color: 'var(--text-3)' }, text: r.tip }),
        el('div', { style: { fontSize: '.8rem', color: 'var(--text-2)' }, text: `Closest simulated variant: ${+n.value.toPrecision(3)} ${UNIT[r.k]}, falls while standing ${n.sim.goal_still.toFixed(3)}, on terrain goals ${n.sim.goal_terrain.toFixed(2)}.` }));
    }));
    const bad = rules.filter((r) => r.s === 'bad').length, warn = rules.filter((r) => r.s === 'warn').length;
    sum.className = 'status ' + (bad ? 'bad' : warn ? 'warn' : 'good');
    sum.textContent = bad ? `${bad} guideline${bad > 1 ? 's' : ''} violated.` : warn ? `No violation, ${warn} borderline.` : 'All seven guidelines met.';
    sum.textContent += ' Guidelines come from one-at-a-time changes; combinations were not tested (S119).';
  }
  draw();
}

// ------------------------------------------------------------------------------------------------ pulls
export function pulls(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const f = ctx.data.facts.find((x) => x.what === 'pull-strength sweep').values.ranges;
  const bm = ctx.data.sweep.groups.find((g) => g.vary === 'body_mass_kg').rows;
  let mass = 3.85;
  legend(stage, [['--bad', 'fell, nominal robot, sustained pulls of this strength (S116)']]);
  chart(stage, {
    x: [-0.5, f.length - 0.5], y: [0, 0.45], xticks: f.map((_, i) => i), xfmt: (v) => (f[v] ? `${f[v][0]} to ${f[v][1]} N` : ''), xlabel: 'sustained pull strength', ylabel: 'fall fraction', height: 220, ml: 44,
    bars: f.map((r, i) => ({ x: i, y: r[2], color: r[1] <= 3 ? '--good' : '--bad', label: `${r[0]} to ${r[1]} N: ${r[2]}` })), desc: 'Falls under sustained pulls on flat ground.',
  });
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const ro = readouts(body, [['cap', 'static pull capacity m g sin 5°', 'hybrid'], ['near', 'closest simulated body mass: falls under half pulls', 'warn']]);
  slider(ctl, { label: 'Body mass', min: 1, max: 8, step: 0.1, value: mass, fmt: (v) => `${v.toFixed(1)} kg`, onInput: (v) => { mass = v; upd(); } });
  function upd() {
    ro.set('cap', `${(mass * G * Math.sin(5 * Math.PI / 180)).toFixed(1)} N`);
    const r = bm.reduce((b, x) => (Math.abs(x.value - mass) < Math.abs(b.value - mass) ? x : b));
    ro.set('near', `${r.value} kg: ${r.sim.goal_pulls_half.toFixed(3)}`);
  }
  upd();
}
