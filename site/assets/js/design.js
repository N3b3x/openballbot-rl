// Design-space explorer: live constraint gauges (design_constraints.py), sweep scatter (design_sweep.csv) and the
// simulation evidence (dz_lqr.csv), kept separate from the model predictions.
import * as M from './ballbot-model.js';

const $ = (s, r = document) => r.querySelector(s);
const [geo, T] = await Promise.all([
  fetch('data/geometry_table.json').then((r) => r.json()),
  fetch('data/designlab_tables.json').then((r) => r.json()),
]);
const f = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '-');
const NOM = { ...M.NOMINAL };
const st = { ...NOM };
const asm = { ...M.DEFAULTS };
const g = (v) => (v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3)).replace(/\.?0+$/, '');

// ------------------------------------------------------------------ sliders
const SL = {
  s: { min: 0.5, max: 2, step: 0.05, fmt: (v) => `${v.toFixed(2)}x` },
  kball: { log: true, min: 0.25, max: 8, fmt: (v) => `x${g(v)}` },
  kbal: { log: true, min: 0.25, max: 4, fmt: (v) => `x${g(v)}` },
  zb: { min: -0.05, max: 0.3, step: 0.005, fmt: (v) => `${(v * 100).toFixed(1)} cm above base` },
  J: { log: true, min: 0.0005, max: 0.02, fmt: (v) => `${v.toPrecision(2)} kg m²` },
  tau: { log: true, min: 1.5, max: 30, fmt: (v) => `${g(v)} N m` },
  dz: { min: -15, max: 24, step: 1, fmt: (v) => `${(51.170175 + v).toFixed(1)}°` },
};
for (const [k, c] of Object.entries(SL)) {
  const el = $('#d-' + k);
  el.min = c.log ? Math.log2(c.min) : c.min; el.max = c.log ? Math.log2(c.max) : c.max; el.step = c.log ? 0.02 : c.step;
  el.addEventListener('input', () => { st[k] = c.log ? 2 ** parseFloat(el.value) : parseFloat(el.value); schedule(); });
}
function syncSliders() {
  for (const [k, c] of Object.entries(SL)) $('#d-' + k).value = c.log ? Math.log2(st[k]) : st[k];
  const A = { mu: [(v) => v.toFixed(2)], tilt: [(v) => `${v}°`], acc: [(v) => `${v.toFixed(1)} m/s²`], vt: [(v) => `${v.toFixed(1)} m/s`], T: [(v) => `${v} ms`], ms: [(v) => `${v}`] };
  const map = { mu: 'mu', tilt: 'tilt_rec_deg', acc: 'a_target', vt: 'v_target', T: 'T_loop', ms: 'Ms_ref' };
  for (const [id, key] of Object.entries(map)) {
    const val = key === 'T_loop' ? asm[key] * 1000 : asm[key];
    $('#a-' + id).value = val; $('#av-' + id).textContent = A[id][0](val);
  }
}
for (const [id, key, mult] of [['mu', 'mu', 1], ['tilt', 'tilt_rec_deg', 1], ['acc', 'a_target', 1], ['vt', 'v_target', 1], ['T', 'T_loop', 0.001], ['ms', 'Ms_ref', 1]]) {
  $('#a-' + id).addEventListener('input', (e) => { asm[key] = parseFloat(e.target.value) * mult; syncSliders(); schedule(); });
}
$('#a-reset').addEventListener('click', () => { Object.assign(asm, M.DEFAULTS); syncSliders(); schedule(); });
const PRESETS = {
  nominal: {}, half: { s: 0.5 }, highcom: { zb: 0.3 }, lightball: { kball: 0.25 }, steep: { dz: 15 }, weak: { tau: 1.5 },
};
document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
  Object.assign(st, NOM, PRESETS[b.dataset.preset]); syncSliders(); schedule();
}));

// ------------------------------------------------------------------ gauges
const GAUGES = [
  ['C_struct', 'Structural floor', (m) => `Sensitivity floor Ms >= ${f(m.Ms_floor, 1)} (zero z ${f(m.z)} vs pole p ${f(m.p)} 1/s, with delay) against a ceiling of ${asm.Ms_ref}. Soft limit: diverges as z approaches p.`],
  ['C_delay', 'Delay budget', (m) => `Loop delay may be at most ln2/p = ${f(m.T_loop_max * 1000, 0)} ms for a factor-two floor inflation; assumed ${f(asm.T_loop * 1000, 0)} ms (sample, compute, sensing, actuator lag).`],
  ['C_trac', 'Traction', (m) => `Slip-free planar torque ${f(m.tau_plane_trac)} N m (mu N r_w, N = ${f(m.N_wheel, 1)} N per wheel) against ${f(m.mgl * Math.sin(asm.tilt_rec_deg * Math.PI / 180))} N m needed to recover ${asm.tilt_rec_deg}°. Static recovery tops out at ${f(m.rec_tilt_trac_deg, 1)}°.`],
  ['C_motor', 'Torque authority', (m) => `Motor planar torque ${f(m.tau_plane_motor, 1)} N m against the same ${f(m.mgl * Math.sin(asm.tilt_rec_deg * Math.PI / 180))} N m recovery need.`],
  ['C_lean', 'Lean ceiling', (m) => `Accelerating at ${asm.a_target} m/s² needs ${f(m.kappa_deg * asm.a_target, 1)}° of lean (kappa = ${f(m.kappa_deg, 1)}° per m/s²); the limit is 20°, so a_max = ${f(m.a_lean, 2)} m/s².`],
  ['C_geom', 'Contact geometry', (m) => `Wheel-ball contact zenith ${f(m.zeta_deg, 1)}° must lie within 35° to 75° (margin is 1 at either edge, 0 outside).`],
  ['C_speed', 'Wheel speed limit', (m) => `v_max = r omega_max / max|W| = ${f(m.v_max, 2)} m/s at ${asm.omega_max} rad/s wheel speed, against a target of ${asm.v_target} m/s.`],
];
$('#gauges').innerHTML = GAUGES.map(([k, name]) => `<div class="gauge" id="g-${k}"><div class="gh"><span>${name}</span><b></b></div><div class="gbar"><i></i><u style="left:${100 * Math.log10(1 / 0.25) / Math.log10(64)}%"></u></div><div class="gd"></div></div>`).join('');
function drawGauges(m) {
  for (const [k, , desc] of GAUGES) {
    const v = m[k], el = $('#g-' + k);
    const pos = v > 0 ? Math.min(1, Math.max(0.01, Math.log10(v / 0.25) / Math.log10(64))) : 0.01;
    el.className = 'gauge ' + (v < 1 ? 'red' : v < 1.25 ? 'amber' : 'ok') + (m.binding === k ? ' bind' : '');
    el.querySelector('b').textContent = v >= 100 ? `x${v.toFixed(0)}` : `x${v.toFixed(2)}`;
    el.querySelector('i').style.width = (pos * 100) + '%';
    el.querySelector('.gd').textContent = desc(m);
    el.setAttribute('aria-label', `${k} margin ${v.toFixed(2)}: ${v < 1 ? 'violated' : 'satisfied'}`);
  }
}

// ------------------------------------------------------------------ metrics / status / hints
function drawMetrics(m) {
  const cells = [
    ['unstable pole p', `${f(m.p)} /s`], ['NMP zero z', `${f(m.z)} /s`], ['z / p', f(m.z_over_p, 2)], ['Ms floor (no delay)', f(m.Ms_lb, 1)],
    ['lean per m/s²', `${f(m.kappa_deg, 1)}°`], ['a_max (lean, torque)', `${f(m.a_max, 2)} m/s²`],
    ['COM height l', `${f(m.l * 100, 1)} cm`], ['total mass', `${f(m.total_mass, 2)} kg`], ['body : ball mass', f(m.mass_ratio, 1)],
    ['m_a g l', `${f(m.mgl, 2)} N m`], ['LQR peak torque, 3° start', `${f(m.lqr_peak_3deg, 2)} N m`], ['tilt domain under motor limit', `${f(m.tilt_domain_deg, 0)}°`],
  ];
  $('#metrics').innerHTML = cells.map(([a, b]) => `<div><small>${a}</small><b class="model">${b}</b></div>`).join('');
  const bad = GAUGES.filter(([k]) => m[k] < 1).map(([, n]) => n);
  const el = $('#status');
  el.className = 'callout' + (bad.length ? ' bad' : '');
  const nm = GAUGES.find(([k]) => k === m.binding)[1];
  el.innerHTML = bad.length
    ? `<strong>Model says: infeasible.</strong> Violated: ${bad.join(', ')}. Thinnest margin: ${nm} (x${f(m[m.binding], 2)}).`
    : `<strong>Model says: all seven constraints satisfied.</strong> Thinnest margin: ${nm} (x${f(m[m.binding], 2)}). This is a prediction of the planar model, not a guarantee: see the simulation evidence below for where it failed.`;
}
function drawHints(m) {
  const P = M.paramsFromDesign(st, geo);
  $('#dv-s').textContent = SL.s.fmt(st.s);
  for (const k of Object.keys(SL)) if (k !== 's') $('#dv-' + k).textContent = SL[k].fmt(st[k]);
  $('#dv-s').textContent = SL.s.fmt(st.s);
  $('#dh-kball').textContent = `ball mass ${f(P.m_k, 3)} kg, body : ball ratio ${f(m.mass_ratio, 1)}`;
  $('#dh-kbal').textContent = `body ${f(P.m_a, 2)} kg, total ${f(m.total_mass, 2)} kg`;
  $('#dh-zb').textContent = `COM height l = ${f(P.l * 100, 1)} cm (nominal 13.1 cm at s = 1), l/r = ${f(m.l_over_r, 2)}`;
  $('#dh-J').textContent = `reflected J_eff = ${f(P.J_eff, 4)} kg m² (ball inertia ${f(P.I_k, 5)})`;
  $('#dh-tau').textContent = `planar torque ceiling ${f(m.tau_plane_motor, 1)} N m; wheel torque x s⁴ = ${f(P.tau_w_max, 2)} N m`;
  $('#dh-dz').textContent = `zenith = 51.2° + ${st.dz}°; the model barely changes with this knob (see note)`;
  const sim = Object.fromEntries(T.sim_lqr.filter((r) => r.run === 'lqr' || r.run === 'lqrd').map((r) => [r.design, r]));
  const nomM = M.metrics({}, geo, asm);
  $('#zetanote').innerHTML = `<strong>Model versus simulation on this knob.</strong> The model's slip-free planar torque is ${f(m.tau_plane_trac, 2)} N m here and ${f(nomM.tau_plane_trac, 2)} N m at the nominal 51°: it is almost flat in &zeta;. In MuJoCo (LQR, a push of 1 x body weight) the fraction of trials that fell was ${(sim.zeta41['pushfall_1.0'] * 100).toFixed(0)}% at 41°, ${(sim.nom['pushfall_1.0'] * 100).toFixed(0)}% at 51° and ${(sim.zeta66['pushfall_1.0'] * 100).toFixed(0)}% at 66°. The planar model misses whatever makes steep contact better; <a href="#evidence">details below</a>.`;
}

// ------------------------------------------------------------------ rule cards
function ruleCards(m) {
  const by = Object.fromEntries(T.sweep.map((r) => [r.name, r]));
  const lo = by['d_zb-0.05'], hi = by['d_zb0.3'], nom = by.d_nominal;
  const zr = T.sweep.map((r) => r.z_over_p);
  const nomC = Object.fromEntries(T.constraints.map((r) => [r.name, r]));
  const cards = [
    ['1. Structural', ['MODEL'], `One unstable pole p and one non-minimum-phase zero z. Any linear controller using ball position has a sensitivity peak of at least (z+p)/|z-p| e^(pT). The usual crossover window p &lt; &omega;c &lt; z/2 is empty because z/p ranges only ${f(Math.min(...zr), 2)} to ${f(Math.max(...zr), 2)} over all ${zr.length} swept designs. Moving the COM down helps (floor ${f(lo.Ms_lb, 1)} at l = ${f(lo.l * 100, 1)} cm), up hurts (${f(hi.Ms_lb, 1)} at l = ${f(hi.l * 100, 1)} cm; nominal ${f(nom.Ms_lb, 1)}).`, `now: floor ${f(m.Ms_lb, 1)} (no delay), ${f(m.Ms_floor, 1)} with ${f(asm.T_loop * 1000, 0)} ms`],
    ['2. Traction', ['MODEL', 'LIT'], `Each wheel transmits at most &mu; N r_w with N = m_a g / (3 cos &zeta;). Nominal: ${f(nomC.d_nominal.N_wheel, 0)} N and 15 mm give 0.31 N m per wheel, about ${f(nomC.d_nominal.tau_plane_trac, 2)} N m planar, against m_a g l = ${f(nomC.d_nominal.mgl, 1)} N m: slip-free tilt recovery tops out near ${f(nomC.d_nominal.rec_tilt_trac_deg, 0)}&deg;. The 30 N m motor limit has a ${f(nomC.d_nominal.C_motor, 0)}x margin over the recovery need: size motors by traction, not peak torque.`, `now: ${f(m.tau_plane_trac, 2)} N m, recovery to ${f(m.rec_tilt_trac_deg, 1)}°`],
    ['3. Torque authority', ['MODEL'], `The planar motor torque over m_a g l should be about 7 or more for LQR to recover a 20&deg; tilt within the actuator limit. It binds only for weak motors (below about 6 N m per wheel at s = 1); otherwise traction binds first.`, `now: ratio ${f(m.torque_authority, 1)}`],
    ['4. Lean budget', ['MODEL'], `Steady acceleration a needs lean &kappa;a with &kappa; about 7 to 15&deg; per m/s&sup2; across the sweep, so a 20&deg; lean budget gives an acceleration ceiling of about 2 m/s&sup2; for every design.`, `now: &kappa; ${f(m.kappa_deg, 1)}°, a_max ${f(m.a_lean, 2)} m/s²`],
    ['5. Scaling (exact)', ['MODEL'], `Geometric scaling s: p ~ s^(-1/2), torques ~ s&#8308;, the traction-limited tilt is scale invariant (about 21&deg;), and the delay budget ln2/p grows as s^(1/2). The JavaScript model reproduces these laws to machine precision (checked in <code>scripts/site/geometry_table.py</code>).`, `now: p ${f(m.p)}, ln2/p ${f(m.T_loop_max * 1000, 0)} ms`],
    ['6. Delay', ['MODEL', 'SIM'], `For a factor-two floor inflation the loop delay must stay under ln2/p (145 ms nominal). A 100 Hz loop uses about 10% of it. In simulation the LQR at 100, 250 and 500 Hz behaves identically (same falls at 1 m/s), so loop rate is not the limit.`, `now: budget ${f(m.T_loop_max * 1000, 0)} ms vs ${f(asm.T_loop * 1000, 0)} ms`],
  ];
  $('#rule-cards').innerHTML = cards.map(([h, tags, body, live]) => `<div class="card rule-card"><div class="tagrow">${tags.map((t) => `<span class="chip ${t === 'MODEL' ? 'model' : t === 'SIM' ? 'sim' : ''}">${t}</span>`).join('')}</div><h3>${h}</h3><p>${body}</p><div class="live" data-live-rule>${live}</div></div>`).join('');
}
function staticCards() {
  const sim = Object.fromEntries(T.sim_lqr.map((r) => [r.run + ':' + r.design, r]));
  const n = sim['lqr:nom'], z41 = sim['lqrd:zeta41'], z66 = sim['lqrd:zeta66'];
  const pc = (v) => `${(v * 100).toFixed(0)}%`;
  $('#refuted-cards').innerHTML = [
    ['Wheel contact angle', `The model says the traction-limited planar torque does not depend on &zeta;: the extra normal force 1/cos &zeta; is cancelled by the shorter lever arm. Simulation (LQR, pushes of 1 x body weight) disagrees: the fraction of trials that fell was ${pc(z41['pushfall_1.0'])} at 41&deg;, ${pc(n['pushfall_1.0'])} at 51&deg; and ${pc(z66['pushfall_1.0'])} at 66&deg; (design_rules.md quotes 6/6 survived at 66&deg;, 1/8 at 51&deg;, 0/6 at 41&deg;). The mechanism is unidentified: 3-D force distribution, the lower reflected rotor inertia (0.043 vs 0.103 kg m&sup2;) or non-planar traction. Treat 60 to 70&deg; as beneficial, bounded near 70&deg; by geometry.`],
    ['LQR limit near 1 m/s and the half-scale failure', `Neither is loop rate (100 to 500 Hz identical) nor wheel friction (x2, x4 and roll friction x10 change nothing). PPO policies do reach 1 m/s (ratio 0.91 to 0.95), so these are limits of the linear controller or the planar model, which is the argument for the learned gate on the <a href="story.html">story page</a>. In dz_lqr.csv every design, including nominal, fell in all four headings at 1.1 x sqrt(s) m/s; the half-scale robot fell in all four at every speed.`],
    ['Mass ratio', `The body : ball mass ratio is 23 here; the literature summaries cite roughly 2 to 12 (LIT, not re-verified). The planar model is least faithful for light balls: the 0.25 x ball fails under LQR in simulation (fell in all four headings at 0.4 m/s) although the model rates it feasible. A character-scale build should use a ball of at least a few hundred grams.`],
  ].map(([h, b]) => `<div class="card rule-card" style="border-color:color-mix(in srgb,var(--bad) 45%,var(--line))"><div class="tagrow"><span class="chip bad">REFUTED</span><span class="chip sim">SIM</span></div><h3>${h}</h3><p>${b}</p></div>`).join('');
  const rows = [
    ['COM as low as the mechanics allow (l at most 0.18 m at r = 0.09 m)', 'lowers the sensitivity floor and keeps the traction constraint satisfied', 'MODEL'],
    ['Wheel-ball contact zenith 60 to 70 degrees', 'pushes survived in simulation; lower reflected inertia; upper bound by geometry', 'SIM'],
    ['High-friction ball skin (urethane, rubber), mu at least 1', 'traction is the binding limit (also for step climbing: F ~ W tan alpha)', 'MODEL + SIM + LIT'],
    ['Motors sized by traction: wheel torque ~ mu N r_w with 2x margin, back-drivable, low Coulomb friction', 'peak torque beyond that is wasted; the literature says low friction beats more torque', 'MODEL + LIT'],
    ['IMU at 1 kHz or faster, encoders on all wheels', 'observability of tilt rate and ball velocity', 'suggestion'],
    ['Inner stabiliser 500 Hz to 2 kHz, policy 100 to 200 Hz', 'the model says rate is not binding; a faster inner loop is cheap insurance', 'MODEL + SIM'],
    ['Body : ball mass ratio nearer 5 to 10 than 23', 'model fidelity and contact stiffness', 'SIM + LIT'],
  ];
  $('#guide-body').innerHTML = rows.map(([a, b, c]) => `<tr><td style="white-space:normal">${a}</td><td style="white-space:normal">${b}</td><td><span class="chip">${c}</span></td></tr>`).join('');
}

// ------------------------------------------------------------------ scatter
const AX = {
  l: ['COM height l [m]', 0], l_over_r: ['l / r', 0], z_over_p: ['z / p', 0], Ms_lb: ['Ms floor (no delay)', 1], p: ['unstable pole p [1/s]', 0], z: ['NMP zero z [1/s]', 0],
  kappa_deg: ['lean per m/s² [deg]', 0], a_max: ['a_max [m/s²]', 0], torque_authority: ['torque authority tau/(m_a g l)', 1], tilt_domain_deg: ['tilt domain [deg]', 1],
  mass_ratio: ['body : ball mass', 1], total_mass: ['total mass [kg]', 1], mgl: ['m_a g l [N m]', 1], J_eff: ['J_eff [kg m²]', 1],
};
const famOf = (r) => {
  const ch = ['s', 'kball', 'kbal', 'zb', 'J', 'tau'].filter((k) => Math.abs(r[k] - NOM[k]) > 1e-12);
  if (ch.length === 0) return 'nominal';
  if (ch.length === 2 && ch.includes('zb') && ch.includes('kball')) return 'grid: COM height x ball mass';
  return ch.length === 1 ? { s: 'scale', kball: 'ball mass', kbal: 'ballast mass', zb: 'ballast height', J: 'rotor inertia', tau: 'motor torque' }[ch[0]] : 'other';
};
const fams = ['all', ...new Set(T.sweep.map(famOf))];
$('#nrows').textContent = T.sweep.length;
$('#sx').innerHTML = Object.entries(AX).map(([k, [l]]) => `<option value="${k}">${l}</option>`).join('');
$('#sy').innerHTML = $('#sx').innerHTML;
$('#sf').innerHTML = fams.map((x) => `<option>${x}</option>`).join('');
$('#sx').value = 'l'; $('#sy').value = 'Ms_lb'; $('#ly').checked = true;
let lastM = null;
function drawScatter() {
  const kx = $('#sx').value, ky = $('#sy').value, lx = $('#lx').checked, ly = $('#ly').checked, fam = $('#sf').value;
  const ok = (v, lg) => Number.isFinite(v) && (!lg || v > 0);
  const pts = T.sweep.filter((r) => ok(r[kx], lx) && ok(r[ky], ly));
  const ext = (k, lg) => { const v = pts.map((r) => r[k]); if (lastM && ok(lastM[k], lg)) v.push(lastM[k]); let lo = Math.min(...v), hi = Math.max(...v); if (lg) { lo = 10 ** Math.floor(Math.log10(lo) - 0.05); hi = 10 ** Math.ceil(Math.log10(hi) + 0.05); } else { const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad; } return [lo, hi]; };
  const W = Math.max(300, Math.min(760, $('#scatter').clientWidth || 760));
  const p = new M.Plot({ w: W, h: W < 520 ? 340 : 400, xlim: ext(kx, lx), ylim: ext(ky, ly), logx: lx, logy: ly, xlabel: AX[kx][0], ylabel: AX[ky][0], ml: 62 });
  const other = pts.filter((r) => fam !== 'all' && famOf(r) !== fam);
  const mine = pts.filter((r) => fam === 'all' || famOf(r) === fam);
  p.points(other.map((r) => [r[kx], r[ky], `${r.name}  (${AX[kx][0]} = ${g(r[kx])}, ${AX[ky][0]} = ${g(r[ky])})`]), { color: 'var(--text-3)', r: 3.5, opacity: 0.45, title: 'x' });
  p.points(mine.map((r) => [r[kx], r[ky], `${r.name}  (${AX[kx][0]} = ${g(r[kx])}, ${AX[ky][0]} = ${g(r[ky])})`]), { color: 'var(--model)', r: 5, opacity: 0.85, title: 'x', stroke: 'var(--bg)' });
  if (lastM && ok(lastM[kx], lx) && ok(lastM[ky], ly)) p.marker(lastM[kx], lastM[ky], 'star', { color: 'var(--text)', size: 6 });
  $('#scatter').innerHTML = p.toString(`Scatter of the design sweep: ${AX[ky][0]} versus ${AX[kx][0]}`);
  $('#scatter-leg').innerHTML = `<span><i style="background:var(--model)"></i>${fam === 'all' ? 'sweep designs' : fam}</span>${fam === 'all' ? '' : '<span><i style="background:var(--text-3)"></i>other designs</span>'}<span>&#9733; design in the explorer (live)</span><span>Hover a dot for its name and values.</span>`;
}
['#sx', '#sy', '#sf', '#lx', '#ly'].forEach((s) => $(s).addEventListener('input', drawScatter));
$('#sx').addEventListener('input', () => { $('#lx').checked = !!AX[$('#sx').value][1]; drawScatter(); });
$('#sy').addEventListener('input', () => { $('#ly').checked = !!AX[$('#sy').value][1]; drawScatter(); });

// ------------------------------------------------------------------ simulation evidence table
const LABEL = {
  nom: 'Nominal', 's0.5': 'Half scale (s = 0.5)', 's1.5': 'Scale 1.5', zlow: 'Ballast low (zb = -5 cm)', zhigh: 'Ballast high (zb = 15 cm)',
  kb4: 'Ball mass x4', kb025: 'Ball mass x0.25', bal025: 'Ballast mass x0.25', bal4: 'Ballast mass x4', tau3: 'Wheel torque limit 3 N m',
  tau6: 'Wheel torque limit 6 N m', J02: 'Rotor inertia 0.02', J0005: 'Rotor inertia 0.0005', zeta41: 'Contact angle 41°', zeta66: 'Contact angle 66°',
};
function evidence() {
  const rows = T.sim_lqr.filter((r) => r.design !== 'nom_motor');
  const head = `<thead><tr class="grp"><th rowspan="2">Design</th><th colspan="5" class="m" style="text-align:center;border-left:2px solid var(--model)">MODEL (planar linear model, live)</th><th colspan="6" style="text-align:center;border-left:2px solid var(--text-3)">SIMULATION (MuJoCo, LQR, copied from dz_lqr.csv)</th><th rowspan="2">Model vs simulation</th></tr>
  <tr><th class="num" style="border-left:2px solid var(--model)">p [1/s]</th><th class="num">z [1/s]</th><th class="num">Ms floor</th><th class="num">all constraints</th><th>thinnest</th>
  <th class="num" style="border-left:2px solid var(--text-3)">speed ratio at 0.4&radic;s <small>(falls /4)</small></th><th class="num">0.8&radic;s</th><th class="num">1.1&radic;s</th><th class="num">push 0.5 W: fell</th><th class="num">push 1.0 W: fell</th><th class="num">push 1.5 W: fell</th></tr></thead>`;
  const spd = (r, k) => (r[`falls_${k}`] >= 4 ? `<td class="num no">fell 4/4</td>` : `<td class="num ${r[`falls_${k}`] ? '' : 'ok'}">${f(r[`ratio_${k}`], 2)} (${r[`falls_${k}`]}/4)</td>`);
  const psh = (r, k) => { const v = r[`pushfall_${k}`]; return `<td class="num ${v === 0 ? 'ok' : v >= 0.5 ? 'no' : ''}">${(v * 100).toFixed(0)}%</td>`; };
  const body = rows.map((r, i) => {
    const m = M.metrics(r.design_params, geo);
    const refuted = m.feasible && r['falls_0.4'] >= 4;
    const other = !m.feasible && r['falls_0.4'] === 0 && r['falls_0.8'] === 0;
    const flag = refuted ? '<span class="mism">REFUTED: model feasible, simulation fell in all 4 headings at 0.4 sqrt(s)</span>' : other ? `<span class="mism">REFUTED the other way: model says ${m.binding.replace('C_', '')} violated, simulation tracked 0.4 and 0.8 with no falls</span>` : (r.design === 'zeta66' || r.design === 'zeta41' ? '<span class="mism">contact angle: model predicts no change in traction</span>' : '');
    const first = r.run === 'lqrd' && i > 0 && rows[i - 1].run === 'lqr' ? ' style="border-top:2px solid var(--line-2)"' : '';
    return `<tr${first}><td>${LABEL[r.design] || r.design}</td><td class="num" style="border-left:2px solid var(--model)">${f(m.p)}</td><td class="num">${f(m.z)}</td><td class="num">${f(m.Ms_floor, 1)}</td><td class="num ${m.feasible ? 'ok' : 'no'}">${m.feasible ? 'satisfied' : 'violated'}</td><td>${m.binding.replace('C_', '')}</td>${spd(r, '0.4').replace('<td', '<td style="border-left:2px solid var(--text-3)"')}${spd(r, '0.8')}${spd(r, '1.1')}${psh(r, '0.5')}${psh(r, '1.0')}${psh(r, '1.5')}<td style="white-space:normal;min-width:230px">${flag}</td></tr>`;
  }).join('');
  $('#simtab').innerHTML = head + `<tbody>${body}</tbody>`;
  $('#protocol').innerHTML = `Protocol (<code>eval_design.py</code>): speeds are 0.4, 0.8 and 1.1 times sqrt(s) m/s (Froude scaling), four headings, 4 s each; ratio is achieved over commanded speed for the runs that did not fall. Push tests: one 50 ms horizontal push of alpha x body weight in a random direction while driving at 0.4 sqrt(s) m/s (8 trials; 6 for the 41&deg; contact angle); the entry is the fraction of trials that fell. Pushes at 1 x weight sit on the edge (the nominal robot fell in 88% of trials), so single-design differences there are weak evidence. Rows below the rule are the contact-angle variants (run label lqrd).`;
  const nm = Object.fromEntries(T.sim_lqr.map((r) => [r.design, r]));
  const m3 = M.metrics({ tau: 3 }, geo), m6 = M.metrics({ tau: 6 }, geo), mn = M.metrics({}, geo);
  const need = (v) => mn.zeta_deg !== undefined ? (M.paramsFromDesign({}, geo).d_eff * v / M.paramsFromDesign({}, geo).r) : NaN;
  $('#evid-cards').innerHTML = [
    ['REFUTED', 'bad', 'Half scale and light ball', `The model puts the half-scale robot (s = 0.5) and the 0.25 x ball inside every constraint (thinnest margin x${f(M.metrics({ s: 0.5 }, geo)[M.metrics({ s: 0.5 }, geo).binding], 2)} and x${f(M.metrics({ kball: 0.25 }, geo)[M.metrics({ kball: 0.25 }, geo).binding], 2)}). In simulation both fell in all four headings at 0.4 sqrt(s) m/s, and the half-scale robot fell at every speed and push.`],
    ['REFUTED', 'bad', 'Contact angle', `Model: traction torque ${f(M.metrics({ dz: -10 }, geo).tau_plane_trac, 3)} / ${f(mn.tau_plane_trac, 3)} / ${f(M.metrics({ dz: 15 }, geo).tau_plane_trac, 3)} N m at 41 / 51 / 66 degrees (flat). Simulation: pushes of 1 x weight felled ${(nm.zeta41['pushfall_1.0'] * 100).toFixed(0)}%, ${(nm.nom['pushfall_1.0'] * 100).toFixed(0)}% and ${(nm.zeta66['pushfall_1.0'] * 100).toFixed(0)}% of trials. The model was wrong about this knob.`],
    ['CONSISTENT once one more check is added', '', 'Torque limit of 3 N m', `design_constraints.py rates the 3 N m design as comfortable (torque authority x${f(m3.C_motor, 1)}) although it fell in all four headings at 0.8 sqrt(s). The model does contain the reason, but not in that constraint: holding a steady 0.8 m/s needs d_eff v / r = ${f(need(0.8), 1)} N m of planar torque (paper_v2, constant-velocity remark), above the ${f(m3.tau_plane_max, 1)} N m ceiling of the 3 N m design and below the ${f(m6.tau_plane_max, 1)} N m ceiling of the 6 N m design, which tracked 0.8 sqrt(s) without falls. This extra check is ours, computed here, not in the CSVs.`],
  ].map(([tag, cls, h, b]) => `<div class="card rule-card"${cls ? ' style="border-color:color-mix(in srgb,var(--bad) 45%,var(--line))"' : ''}><div class="tagrow"><span class="chip ${cls}">${tag}</span></div><h3>${h}</h3><p>${b}</p></div>`).join('');
  const mo = nm.nom_motor, nmr = nm.nom;
  const mm = M.metrics({ tau: 3 }, geo);
  $('#motor-cards').innerHTML = `
    <div class="card rule-card" style="border-left:3px solid var(--model)"><div class="tagrow"><span class="chip model">MODEL</span></div><h3>What the model can say</h3><p>The motor model in <code>configs/env/dz_nom_motor.yaml</code> adds a speed-torque curve (no-load speed 150 rad/s, stall torque 3.0 N m, Coulomb friction 0.05, 3 ms lag). The planar model has no motor dynamics, so the nearest it can offer is the stall torque as a limit: ${f(mm.tau_plane_max, 1)} N m planar, the same as the 3 N m design above, so it can predict only a failure at the higher speeds, from the steady torque d_eff v / r (${f(need(0.8), 1)} N m at 0.8 m/s). It cannot predict anything about the lag or the friction.</p></div>
    <div class="card rule-card" style="border-left:3px solid var(--text-3)"><div class="tagrow"><span class="chip sim">SIM</span></div><h3>What simulation did (dz_lqr_motor.csv)</h3><p>With the motor model the LQR still tracked 0.4 sqrt(s) m/s (ratio ${f(mo['ratio_0.4'], 2)}, ${mo['falls_0.4']}/4 falls, nominal ${f(nmr['ratio_0.4'], 2)}), but fell in all four headings at 0.8 (nominal: ${nmr['falls_0.8']}/4 falls), as the stall-torque reading suggests. Push result: the fraction of trials that fell was ${(mo['pushfall_0.5'] * 100).toFixed(0)}% / ${(mo['pushfall_1.0'] * 100).toFixed(0)}% / ${(mo['pushfall_1.5'] * 100).toFixed(0)}% at 0.5 / 1 / 1.5 x weight versus ${(nmr['pushfall_0.5'] * 100).toFixed(0)}% / ${(nmr['pushfall_1.0'] * 100).toFixed(0)}% / ${(nmr['pushfall_1.5'] * 100).toFixed(0)}% for the ideal-torque nominal robot: pushes were survived <em>more</em> often with the motor model. Nothing in the model predicts that, and it is a single run, so we treat it as an open question rather than a finding.</p></div>`;
}

// ------------------------------------------------------------------ orchestration
let pend = false;
function schedule() { if (pend) return; pend = true; requestAnimationFrame(() => { pend = false; render(); }); }
function render() {
  const m = M.metrics(st, geo, asm);
  lastM = m;
  drawGauges(m); drawMetrics(m); drawHints(m); ruleCards(m); drawScatter();
  window.__design = { m };
}
syncSliders(); staticCards(); evidence(); render();
document.addEventListener('themechange', drawScatter);
let rz = 0;
window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(drawScatter, 150); });
// deep-link support: #evidence etc. works natively; expose a tiny self-check for verification
document.documentElement.dataset.ready = '1';
