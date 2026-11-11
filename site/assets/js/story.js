// story.html behaviour: progress rail, mini charts (all numbers cited in the page to docs/research/tracking_study_log.md),
// decision-ledger filter, expand-all toggle. Charts are HTML/CSS or SVG using theme variables, so they follow light/dark.
document.documentElement.classList.add('sjs');

/* ---------- chart specs: values copied from docs/research/tracking_study_log.md ---------- */
const VIZ = {
  ch2: { type: 'bars', title: 'Best speed ratio (achieved / commanded) at each stage', max: 1, rows: [
    { l: 'S0 baseline limits', v: 0.06, t: '0.06', c: 'neutral' },
    { l: 'S2 on the ball', v: 0.33, t: '0.21-0.33', c: 'learned' },
    { l: 'S4 lean reward', v: 0.36, t: '0.36', c: 'learned' },
    { l: 'S5 error reward', v: 0.31, t: '0.15-0.31', c: 'learned' },
    { l: 'S6 spin capped (trk_f)', v: 0.91, t: '0.91', c: 'good', sub: 'ratio at 1 m/s' }] },
  ch3a: { type: 'bars', title: 'Speed ratio at 1 m/s before and after capping the spin command', max: 1, rows: [
    { l: 'S5 best, spin to 3 rad/s', v: 0.31, t: '0.31', c: 'bad' },
    { l: 'Continue a timid policy, spin 1', v: 0.59, t: '0.59', c: 'warn' },
    { l: 'From scratch, spin 1 (trk_f)', v: 0.91, t: '0.91', c: 'good' },
    { l: 'Second seed (trk_f_s1)', v: 0.95, t: '0.95', c: 'good' }] },
  ch3b: { type: 'groups', title: 'S8 spin curriculum: ratio at 1 m/s, before vs after widening spin to 1.5 rad/s', max: 1,
    series: [{ n: 'before (spin 1)', c: 'neutral' }, { n: 'after (spin 1.5)', c: 'warn' }], groups: [
      { l: 'from trk_g', vals: [0.87, 0.87] }, { l: 'from trk_h', vals: [0.81, 0.70] }, { l: 'from trk_f_s1', vals: [0.95, 0.75] }] },
  ch4a: { type: 'bars', title: 'Heading test, 0.5 m/s from random start yaw', max: 0.4, rows: [
    { l: 'trk_h velocity error (m/s)', v: 0.31, t: '0.31', c: 'learned' },
    { l: 'trk_g velocity error (m/s)', v: 0.32, t: '0.32', c: 'learned' },
    { l: 'LQR velocity error (m/s)', v: 0.02, t: '0.02', c: 'model', sub: 'chapter 6' }] },
  ch4b: { type: 'bars', title: 'Cost of one control step (ms)', max: 3.5, rows: [
    { l: 'RK4 contact-frame stepper', v: 3.26, t: '3.26', c: 'neutral' },
    { l: 'implicitfast, same fix', v: 1.10, t: '1.10', c: 'good' }] },
  ch5a: { type: 'polezero' },
  ch5b: { type: 'bars', title: 'Lean law recovered by each controller (% of the derived kappa = 0.176 rad per m/s^2)', max: 100, rows: [
    { l: 'LQR (model-based)', v: 95, t: '95%', c: 'model', sub: '0.168' },
    { l: 'trk_g (PPO)', v: 91, t: '91%', c: 'learned', sub: '0.160' },
    { l: 'trk_h (PPO)', v: 83, t: '83%', c: 'learned', sub: '0.146' }] },
  ch6a: { type: 'bars', title: 'Highest speed reached by the discrete LQR before falling, by rotor inertia J_w (kg m^2)', max: 1.1, rows: [
    { l: 'J_w = 0.0005', v: 0.84, t: '0.84 m/s', c: 'model' },
    { l: 'J_w = 0.002', v: 0.93, t: '0.93 m/s', c: 'model' },
    { l: 'J_w = 0.005', v: 1.0, t: '1.0 m/s', c: 'model' }] },
  ch7a: { type: 'bars', title: 'Yaw rate achieved when commanded 1 rad/s (higher is better)', max: 1.1, rows: [
    { l: 'LQR, negative spin', v: 0.57, t: '0.57', c: 'model' },
    { l: 'LQR, positive spin', v: 0.57, t: '0.57', c: 'model' },
    { l: 'res_c, negative spin', v: 1.01, t: '1.01', c: 'hybrid' },
    { l: 'res_c, positive spin', v: 0.93, t: '0.93', c: 'hybrid' }] },
  ch7b: { type: 'bars', title: 'Falls in 30 random-command episodes', max: 30, rows: [
    { l: 'res_c (zero-init, beta 0.1)', v: 3, t: '3/30', c: 'hybrid' },
    { l: 'res_d (beta 0.2, DR, pushes)', v: 15, t: '15/30', c: 'bad' }] },
  ch8a: { type: 'bars', title: 'Trials that fall in 10 (trk_g, randomised physics, one parameter at a time)', max: 10, rows: [
    { l: 'wheel damping', v: 5, t: '5/10', c: 'bad' },
    { l: 'body mass', v: 1, t: '1/10', c: 'warn' },
    { l: 'friction', v: 0, t: '0/10', c: 'good' },
    { l: 'torque limit', v: 0, t: '0/10', c: 'good' },
    { l: 'action latency', v: 0, t: '0/10', c: 'good' },
    { l: 'sensor noise', v: 0, t: '0/10', c: 'good' }] },
  ch8b: { type: 'bars', title: 'What robustness cost: speed ratio at 1 m/s', max: 1, rows: [
    { l: 'trk_g, nominal training', v: 0.87, t: '0.87', c: 'learned' },
    { l: 'rob_a, strong DR fine-tune', v: 0.35, t: '0.35', c: 'warn' },
    { l: 'rob_b, strong DR fine-tune', v: 0.27, t: '0.27', c: 'bad' },
    { l: 'robs_a, DR from scratch', v: 0.35, t: '~0.35', c: 'warn' },
    { l: 'robs_b, narrow damping, scratch', v: 0.50, t: '~0.50', c: 'learned' }] },
  ch9: { type: 'lines' },
  ch10a: { type: 'groups', title: 'Step climbing at 0.7 m/s: successes in 10 trials', max: 10,
    series: [{ n: 'robs_c @ ki=1', c: 'learned' }, { n: 'LQR', c: 'model' }, { n: 'step-trained climb_a', c: 'warn' }],
    groups: [{ l: '2 cm step', vals: [10, 10, 8.5], txt: ['10', '10', '8-9'] },
             { l: '3 cm step', vals: [3, 0, 2], txt: ['3', '0', '2'] },
             { l: '4 cm step', vals: [null, null, 0], txt: ['n/a', 'n/a', '0'] }] },
  ch10b: { type: 'bars', title: 'Traction hypothesis: horizontal force to roll the ball over the step vs friction available (N)', max: 70, rows: [
    { l: '3 cm step needs', v: 44, t: '44 N', c: 'bad' },
    { l: '4 cm step needs', v: 60, t: '60 N', c: 'bad' },
    { l: 'wheels can supply', v: 52, t: '~52 N', c: 'good', sub: '3 wheels x 17-18 N x mu 1' }] },
  ch10c: { type: 'groups', title: 'Scaling wheel-ball cross-friction, robs_c @ ki=1 at 0.7 m/s: successes in 4 trials', max: 4,
    series: [{ n: '3 cm step', c: 'learned' }, { n: '4 cm step', c: 'warn' }],
    groups: [{ l: 'friction x1', vals: [1, 0] }, { l: 'friction x2', vals: [4, 1] }, { l: 'friction x4', vals: [4, 1] }] },
  ch11: { type: 'bars', title: 'LQR under a push of 1 x body weight, by wheel contact zenith angle', max: 1, rows: [
    { l: 'zenith 41 deg', v: 0, t: '0/6', c: 'bad' },
    { l: 'zenith 51 deg (nominal)', v: 1 / 8, t: '1/8', c: 'warn' },
    { l: 'zenith 66 deg', v: 1, t: '6/6', c: 'good' }], note: 'bar = fraction of pushes survived' },
  ch12a: { type: 'bars', title: 'Structural sensitivity floor ||S||inf >= (z+p)/|z-p| (lower is better)', max: 28, rows: [
    { l: 'COM l = 8.7 cm (low)', v: 9.0, t: '9.0', c: 'good' },
    { l: 'nominal', v: 10.7, t: '10.7', c: 'model' },
    { l: '4x ballast (heavy body)', v: 14.3, t: '14.3', c: 'warn' },
    { l: 'COM l = 38 cm (high)', v: 26.5, t: '26.5', c: 'bad' }] },
  ch12b: { type: 'bars', title: 'From-scratch PPO on six designs, halfway (2.5M of 5M steps): mean episode length', max: 700, rows: [
    { l: 'tau3 (strong motors)', v: 683, t: '683', c: 'learned' },
    { l: 's0.5 (half scale)', v: 564, t: '564', c: 'learned', sub: 'LQR cannot balance this' },
    { l: 'zhigh', v: 557, t: '557', c: 'learned' },
    { l: 'nom', v: 407, t: '407', c: 'learned' },
    { l: 'kb4 (4x ball)', v: 407, t: '407', c: 'learned' },
    { l: 's1.5 (1.5x scale)', v: 82, t: '82', c: 'bad', sub: 'struggling' }] },
  gate: { type: 'gate' },
};

/* ---------- renderers ---------- */
const el = (t, a = {}, h = '') => { const e = document.createElement(t); for (const k in a) e.setAttribute(k, a[k]); e.innerHTML = h; return e; };
const fmt = (v) => (Number.isInteger(v) ? String(v) : String(v));

function bars(s) {
  const rows = s.rows.map((r) => `<div class="brow"><div class="bl">${r.l}${r.sub ? `<small>${r.sub}</small>` : ''}</div>
    <div class="btrack"><span class="bfill ${r.c || 'neutral'}" style="--w:${Math.max(0, Math.min(100, (r.v / s.max) * 100)).toFixed(1)}%"></span></div><div class="bv">${r.t ?? fmt(r.v)}</div></div>`).join('');
  return `<div class="vt">${s.title}</div><div class="bars">${rows}</div>${s.note ? `<div class="legend">${s.note}</div>` : ''}`;
}
function groups(s) {
  const g = s.groups.map((gr) => `<div class="bgroup"><div class="gl">${gr.l}</div>${gr.vals.map((v, i) => v == null
    ? `<div class="brow"><div class="bl">${s.series[i].n}</div><div class="btrack"></div><div class="bv">${gr.txt ? gr.txt[i] : 'n/a'}</div></div>`
    : `<div class="brow"><div class="bl">${s.series[i].n}</div><div class="btrack"><span class="bfill ${s.series[i].c}" style="--w:${((v / s.max) * 100).toFixed(1)}%"></span></div><div class="bv">${gr.txt ? gr.txt[i] : fmt(v)}</div></div>`).join('')}</div>`).join('');
  return `<div class="vt">${s.title}</div><div class="bars">${g}</div>`;
}
function polezero() {
  // number line in 1/s: unstable pole 4.79, non-minimum-phase zero 5.78 (log: First-principles model)
  const x = (v) => 40 + (v - 3) * 92;
  return `<div class="vt">The two numbers that shape everything, in 1/s</div>
  <svg viewBox="0 0 440 150" role="img" aria-label="Number line from 3 to 7 per second with the unstable pole at 4.79 and the non-minimum-phase zero at 5.78">
    <line class="ax" x1="30" x2="430" y1="80" y2="80"/>
    ${[3, 4, 5, 6, 7].map((v) => `<line class="ax" x1="${x(v)}" x2="${x(v)}" y1="76" y2="84"/><text x="${x(v) - 4}" y="100">${v}</text>`).join('')}
    <rect x="${x(4.79)}" y="62" width="${x(5.78) - x(4.79)}" height="36" fill="var(--bad)" opacity=".16"/>
    <circle cx="${x(4.79)}" cy="80" r="7" fill="var(--model)"/>
    <path d="M${x(5.78) - 7} 73 l14 14 M${x(5.78) + 7} 73 l-14 14" stroke="var(--model)" stroke-width="3" stroke-linecap="round"/>
    <text class="big" x="${x(4.79) - 38}" y="50">pole p = 4.79</text>
    <text class="big" x="${x(5.78) - 8}" y="50">zero z = 5.78</text>
    <text x="${x(4.79) - 22}" y="132">z/p = 1.21: the empty window</text>
  </svg>
  <div class="legend">Usual rule: set the loop bandwidth between p and z/2. Here z/2 is below p, so there is no such window.</div>`;
}
function lines() {
  const W = 460, H = 300, L = 46, B = 40, T = 14, R = 14;
  const px = (v) => L + (v / 1.0) * (W - L - R), py = (v) => H - B - (v / 1.0) * (H - B - T);
  const xs = [0.25, 0.5, 0.75, 1.0];
  const ser = [
    { n: 'ideal tracking', pts: xs, col: 'var(--text-3)', dash: '2 5' },
    { n: 'robs_b, no integral (ratio ~0.50)', pts: xs.map((v) => v * 0.5), col: 'var(--warn)', dash: '6 4' },
    { n: 'robs_b + integral (ki 1)', pts: [0.25, 0.51, 0.73, 0.79], col: 'var(--learned)' },
    { n: 'robs_c + integral (ki 1)', pts: [0.25, 0.51, 0.76, 0.93], col: 'var(--hybrid)' },
  ];
  const grid = [0, 0.25, 0.5, 0.75, 1.0].map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}"/><text x="6" y="${py(v) + 4}">${v}</text><text x="${px(v) - 8}" y="${H - B + 18}">${v}</text>`).join('');
  const paths = ser.map((s) => `<polyline fill="none" stroke="${s.col}" stroke-width="2.4" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} points="${s.pts.map((v, i) => `${px(xs[i])},${py(v)}`).join(' ')}"/>${s.dash ? '' : s.pts.map((v, i) => `<circle cx="${px(xs[i])}" cy="${py(v)}" r="3.5" fill="${s.col}"/>`).join('')}`).join('');
  return `<div class="vt">Achieved speed vs commanded speed (m/s)</div>
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Line chart of achieved versus commanded speed. Without the outer integral the policy reaches about half the command; with it the policy reaches 0.79 to 0.93 at 1 metre per second.">
    ${grid}<line class="ax" x1="${L}" x2="${W - R}" y1="${py(0)}" y2="${py(0)}"/><line class="ax" x1="${L}" x2="${L}" y1="${T}" y2="${py(0)}"/>
    ${paths}<text x="${W / 2 - 40}" y="${H - 6}">commanded speed</text></svg>
  <div class="legend">${ser.map((s) => `<span><i style="background:${s.col}"></i>${s.n}</span>`).join('')}</div>`;
}
function gate() {
  return `<div class="vt">The interleaved controller, one step</div>
  <svg viewBox="0 0 440 210" role="img" aria-label="Diagram: the LQR action and the PPO action are blended by a learned gate g between 0 and 1.">
    <defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="var(--text-3)"/></marker></defs>
    <rect x="12" y="20" width="110" height="46" rx="10" fill="none" stroke="var(--model)" stroke-width="2"/><text class="big" x="30" y="48" style="fill:var(--model)">LQR a_L</text>
    <rect x="12" y="130" width="110" height="46" rx="10" fill="none" stroke="var(--learned)" stroke-width="2"/><text class="big" x="30" y="158" style="fill:var(--learned)">PPO a_pi</text>
    <rect x="165" y="75" width="110" height="50" rx="10" fill="none" stroke="var(--hybrid)" stroke-width="2"/><text class="big" x="176" y="97" style="fill:var(--hybrid)">gate g</text><text x="176" y="114">(1+clip x)/2</text>
    <rect x="318" y="75" width="110" height="50" rx="10" fill="none" stroke="var(--hybrid)" stroke-width="2.5"/><text class="big" x="330" y="97" style="fill:var(--hybrid)">wheels</text><text x="330" y="114">3 torques</text>
    <path d="M122 43 L165 88" stroke="var(--text-3)" stroke-width="1.5" marker-end="url(#ah)"/>
    <path d="M122 153 L165 112" stroke="var(--text-3)" stroke-width="1.5" marker-end="url(#ah)"/>
    <path d="M275 100 L316 100" stroke="var(--text-3)" stroke-width="1.5" marker-end="url(#ah)"/>
    <text x="14" y="198">a = (1 - g) a_L + g a_pi      g = 0: pure LQR      g = 1: pure PPO</text>
  </svg>`;
}
const R = { bars, groups, polezero, lines, gate };

document.querySelectorAll('figure.viz[data-viz]').forEach((f) => {
  const spec = VIZ[f.dataset.viz];
  if (!spec) return;
  const holder = el('div', { class: 'vbody' }, R[spec.type](spec));
  f.prepend(holder);
});

/* ---------- reveal bars when scrolled into view ---------- */
const vio = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); vio.unobserve(e.target); } }), { threshold: 0.25 });
document.querySelectorAll('figure.viz').forEach((f) => vio.observe(f));

/* ---------- chapter rail + progress ---------- */
const chapters = [...document.querySelectorAll('article.chapter')];
const rail = document.getElementById('railList');
if (rail) {
  rail.innerHTML = chapters.map((c, i) => `<li><a href="#${c.id}" data-id="${c.id}"><span class="n">${String(i + 1).padStart(2, '0')}</span><span class="t">${c.dataset.short || c.querySelector('h2').textContent}</span></a></li>`).join('');
}
const links = rail ? [...rail.querySelectorAll('a')] : [];
const seen = new Set();
const cio = new IntersectionObserver((es) => {
  es.forEach((e) => { if (e.isIntersecting) seen.add(e.target.id); });
  const vis = es.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!vis) return;
  links.forEach((a) => { a.classList.toggle('active', a.dataset.id === vis.target.id); a.classList.toggle('seen', seen.has(a.dataset.id)); });
  const act = links.find((a) => a.dataset.id === vis.target.id);
  if (act && window.innerWidth <= 900 && act.parentElement.parentElement.scrollTo) act.parentElement.parentElement.scrollTo({ left: act.offsetLeft - 40, behavior: 'smooth' });
}, { rootMargin: '-25% 0px -55% 0px', threshold: [0, 0.01, 0.5] });
chapters.forEach((c) => cio.observe(c));

const bar = document.getElementById('progress');
const onScroll = () => {
  const h = document.documentElement.scrollHeight - innerHeight;
  if (bar) bar.style.transform = `scaleX(${h > 0 ? Math.min(1, scrollY / h) : 0})`;
};
addEventListener('scroll', onScroll, { passive: true }); onScroll();

/* ---------- ledger filter ---------- */
const q = document.getElementById('ledgerQ');
if (q) q.addEventListener('input', () => {
  const s = q.value.trim().toLowerCase();
  let n = 0;
  document.querySelectorAll('#ledgerTable tbody tr').forEach((r) => { const ok = !s || r.textContent.toLowerCase().includes(s); r.hidden = !ok; if (ok) n++; });
  const c = document.getElementById('ledgerCount'); if (c) c.textContent = `${n} decisions`;
});

/* ---------- expand / collapse all details ---------- */
const tg = document.getElementById('toggleAll');
if (tg) tg.addEventListener('click', () => {
  const ds = [...document.querySelectorAll('details.more')];
  const open = !ds.every((d) => d.open);
  ds.forEach((d) => { d.open = open; });
  tg.textContent = open ? 'Collapse all details' : 'Expand all details';
  tg.setAttribute('aria-pressed', String(open));
});
// open a details element if the URL hash points inside it
if (location.hash) { const t = document.querySelector(location.hash); const d = t && t.closest('details'); if (d) d.open = true; }
