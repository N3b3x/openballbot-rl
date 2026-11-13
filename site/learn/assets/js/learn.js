// Course chrome for site/learn/: top bar, theme toggle, progress (localStorage), quiz buttons, lesson pager and lazy loading of
// each lesson's interactive figures. Pages include it as <script type="module" src="assets/js/learn.js"></script>.
// A figure is <figure class="widget" data-widget="l03:lqr">; the module ./l03.js is imported only when the figure comes near
// the viewport, and its export `lqr(el, ctx)` builds it. ctx = { data, reduced, onTheme }.

export const LESSONS = [
  ['01-ballbot.html', 'What a ballbot is and why it is hard', 8],
  ['02-lean.html', 'Lean to accelerate', 7],
  ['03-lqr.html', 'LQR balance control', 8],
  ['04-learning.html', 'Adding learning', 10],
  ['05-commands.html', 'Commands that feel good', 8],
  ['06-navigation.html', 'Sensing and navigation', 9],
  ['07-odometry.html', 'Knowing where you are', 8],
  ['08-design.html', 'Designing the robot', 8],
  ['09-lessons.html', 'Lessons from the study', 9],
  ['10-ros2.html', 'From simulation to ROS 2', 9],
];

// Short prerequisite and teaser lines, shown under the goals and in the pager (index = lesson number - 1).
const PREREQ = [
  'Prerequisites: none beyond high-school physics. Unfamiliar word? Try the <a href="../glossary.html">glossary</a>.',
  'Prerequisites: lesson 1 (the pole, the zero and the planar model).',
  'Prerequisites: lessons 1 and 2. Matrices are not needed; each equation comes with a plain reading.',
  'Prerequisites: lesson 3 (the LQR loop). Terms such as PPO, domain randomisation and seed are explained where they first appear, and in the <a href="../glossary.html">glossary</a>.',
  'Prerequisites: lessons 2 (why leaning forces a backward move first) and 4 (the residual policy).',
  'Prerequisites: lesson 5 (the motion layer the planner feeds). Lesson 4 helps for the results.',
  'Prerequisites: lesson 6 is useful, not required. Basic ideas of noise and averaging are enough.',
  'Prerequisites: lessons 1 to 3 for the poles and the lean law; lesson 4 for the design names (kb025, bal4, tau6 and so on).',
  'Prerequisites: lessons 4 to 8 supply the numbers being examined here.',
  'Prerequisites: lessons 3 to 7 (this lesson maps each to a module). No ROS knowledge is needed.',
];
const TEASE = [
  'Why does the robot lean, and why does it move backwards first?',
  'How are the gains of a balance controller chosen?',
  'What does a learned correction add, and why is it kept small?',
  'How do you make commands the robot can actually follow?',
  'How does it see obstacles and plan around them?',
  'Where does it think it is, and how fast does that estimate drift?',
  'What should the robot itself look like?',
  'Which of these numbers can you trust, and how was that checked?',
  'How does all of this leave the simulator?',
  '',
];
const KEY = 'ballbot-learn-v1';
export function loadProgress() {
  try { const v = JSON.parse(localStorage.getItem(KEY) || '{}'); return v && typeof v === 'object' ? v : {}; } catch (e) { return {}; }
}
export function saveProgress(p) {
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch (e) { /* storage blocked: progress is not kept */ }
}

const here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
const idx = LESSONS.findIndex(([f]) => f === here);
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- theme (same key as the rest of the site) ----------
let theme = 'dark';
try { theme = localStorage.getItem('theme') || 'dark'; } catch (e) { /* ignore */ }
document.documentElement.dataset.theme = theme;
const icon = (t) => (t === 'dark' ? '☾' : '☀');

// ---------- skip link + top bar ----------
const main = document.querySelector('main');
if (main && !main.id) main.id = 'main';
const bar = document.createElement('header');
bar.className = 'lnav';
const crumb = idx >= 0
  ? `<a href="index.html">Course</a><span class="sep">/</span><span class="opt">Lesson ${idx + 1} of ${LESSONS.length}</span>`
  : '<a href="index.html">Course</a>';
bar.innerHTML = `<div class="wrap">
  <a class="brand" href="../index.html" title="Ballbot study home"><svg width="24" height="24" viewBox="0 0 26 26" aria-hidden="true"><circle cx="13" cy="17" r="7" fill="none" stroke="var(--hybrid)" stroke-width="2"/><rect x="8" y="2" width="10" height="9" rx="3" fill="var(--model)"/></svg>Ballbot</a>
  <span class="crumb"><span class="sep">/</span>${crumb}</span>
  <span class="prog" aria-live="polite"><span class="bar" aria-hidden="true"><i></i></span><span class="pt"></span></span>
  <a class="site" href="../index.html">Main site</a>
  <button type="button" class="theme" aria-label="Switch to ${theme === 'dark' ? 'light' : 'dark'} theme">${icon(theme)}</button>
</div>`;
document.body.prepend(bar);
const skip = document.createElement('a');
skip.className = 'skip'; skip.href = '#' + (main ? main.id : 'main'); skip.textContent = 'Skip to content';
document.body.prepend(skip);

const tbtn = bar.querySelector('button.theme');
tbtn.onclick = () => {
  const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('theme', t); } catch (err) { /* ignore */ }
  tbtn.textContent = icon(t);
  tbtn.setAttribute('aria-label', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
  document.dispatchEvent(new CustomEvent('themechange'));
};

// ---------- prerequisites line under the goals ----------
if (idx >= 0) {
  const gl = document.querySelector('.goals');
  if (gl && PREREQ[idx]) { const pr = document.createElement('p'); pr.className = 'prereq'; pr.innerHTML = PREREQ[idx]; gl.append(pr); }
}

// ---------- progress ----------
function paintProgress() {
  const p = loadProgress();
  const done = LESSONS.filter(([f]) => p[f] && p[f].done).length;
  bar.querySelector('.bar i').style.width = `${(100 * done) / LESSONS.length}%`;
  bar.querySelector('.pt').textContent = `${done}/${LESSONS.length} done`;
  document.querySelectorAll('[data-progress-bar]').forEach((el) => { el.style.width = `${(100 * done) / LESSONS.length}%`; });
  document.querySelectorAll('[data-progress-text]').forEach((el) => { el.textContent = `${done} of ${LESSONS.length} lessons completed`; });
  document.querySelectorAll('.lcard[data-lesson]').forEach((c) => {
    const f = c.dataset.lesson; const st = c.querySelector('.st');
    const d = p[f] && p[f].done; const v = p[f] && p[f].visited;
    c.classList.toggle('done', !!d);
    if (st) st.textContent = d ? 'Completed' : v ? 'Started' : 'Not started';
  });
  const cont = document.querySelector('[data-continue]');
  if (cont) {
    const next = LESSONS.find(([f]) => !(p[f] && p[f].done));
    if (next) { cont.href = next[0]; cont.textContent = done ? `Continue: lesson ${LESSONS.indexOf(next) + 1}` : 'Start lesson 1'; }
    else { cont.href = LESSONS[0][0]; cont.textContent = 'All done: start again'; }
  }
  const btn = document.querySelector('.lpager .done button');
  if (btn && idx >= 0) {
    const d = p[here] && p[here].done;
    btn.setAttribute('aria-pressed', String(!!d));
    btn.textContent = d ? 'Completed (undo)' : 'Mark lesson complete';
  }
}
if (idx >= 0) { const p = loadProgress(); p[here] = { ...(p[here] || {}), visited: Date.now() }; saveProgress(p); }

// ---------- lesson pager + footer ----------
if (idx >= 0 && main) {
  const prev = LESSONS[idx - 1], next = LESSONS[idx + 1];
  const pg = document.createElement('nav');
  pg.className = 'lpager'; pg.setAttribute('aria-label', 'Lessons');
  pg.innerHTML = `<div class="wrap">
    ${prev ? `<a class="pn prev" href="${prev[0]}"><span>Previous lesson ${idx}</span>${prev[1]}</a>` : '<a class="pn prev" href="index.html"><span>Back to</span>Course overview</a>'}
    <div class="done"><button type="button" class="btn primary">Mark lesson complete</button></div>
    ${next ? `<a class="pn next" href="${next[0]}"><span>Next lesson ${idx + 2}</span>${next[1]}</a>` : '<a class="pn next" href="index.html"><span>Finished</span>Course overview</a>'}
  </div>`;
  main.after(pg);
  const nl = pg.querySelector('a.next');
  if (nl && next && TEASE[idx]) { const t = document.createElement('em'); t.className = 'tease'; t.textContent = TEASE[idx]; nl.append(t); }
  pg.querySelector('.done button').onclick = () => {
    const p = loadProgress(); const d = !(p[here] && p[here].done);
    p[here] = { ...(p[here] || {}), done: d, at: Date.now() }; saveProgress(p); paintProgress();
  };
}
const foot = document.createElement('footer');
foot.className = 'lfoot';
foot.innerHTML = '<div class="wrap"><p>Ballbot course. Every result is from simulation (MuJoCo and MuJoCo-Warp); there is no physical robot. Numbers are read from the repository and cited to the study log by entry number. Source on <a href="https://github.com/N3b3x/openballbot-rl">GitHub</a>. <a href="../index.html">Main site</a> &middot; <a href="../glossary.html">Glossary</a>.</p></div>';
(document.querySelector('.lpager') || main || document.body.lastElementChild).after(foot);
document.querySelectorAll('[data-reset]').forEach((b) => b.addEventListener('click', () => { saveProgress({}); paintProgress(); }));
paintProgress();
addEventListener('storage', paintProgress);

// ---------- reading progress line ----------
if (idx >= 0) {
  const rb = document.createElement('div'); rb.className = 'readbar'; rb.setAttribute('aria-hidden', 'true'); document.body.append(rb);
  let raf = 0;
  const upd = () => { raf = 0; const h = document.documentElement.scrollHeight - innerHeight; rb.style.width = `${h > 0 ? Math.min(100, (100 * scrollY) / h) : 0}%`; };
  addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(upd); }, { passive: true });
  upd();
}

// ---------- multiple-choice questions: <div class="opts" data-q> <button data-ok>..</button> ... </div> ----------
document.querySelectorAll('.opts').forEach((box) => {
  const ans = box.parentElement.querySelector('details.answer');
  box.querySelectorAll('button').forEach((b) => {
    b.type = 'button';
    b.addEventListener('click', () => {
      const ok = b.hasAttribute('data-ok');
      b.classList.add(ok ? 'right' : 'wrong');
      if (ok) { box.querySelectorAll('button[data-ok]').forEach((x) => x.classList.add('right')); if (ans) ans.open = true; }
    });
  });
});

// ---------- data + lazy widgets ----------
let dataP = null;
export function getData() {
  if (!dataP) dataP = fetch('data/learn.json').then((r) => { if (!r.ok) throw new Error('learn.json ' + r.status); return r.json(); });
  return dataP;
}
const themeSubs = new Set();
document.addEventListener('themechange', () => themeSubs.forEach((f) => { try { f(); } catch (e) { console.warn(e); } }));
const ctxBase = { reduced, onTheme: (f) => themeSubs.add(f) };

async function mount(el) {
  if (el.dataset.mounted) return;
  el.dataset.mounted = '1';
  const [mod, name] = el.dataset.widget.split(':');
  try {
    const [m, data] = await Promise.all([import(`./${mod}.js`), getData()]);
    const fn = m[name];
    if (typeof fn !== 'function') throw new Error(`widget ${el.dataset.widget} not found`);
    el.querySelectorAll('.stage.ph').forEach((s) => s.classList.remove('ph'));
    await fn(el, { ...ctxBase, data });
  } catch (e) {
    console.error(e);
    const st = el.querySelector('.stage') || el;
    st.classList.remove('ph');
    const p = document.createElement('p'); p.className = 'status bad'; p.textContent = 'This figure could not load. The text around it explains the same idea.';
    st.append(p);
  }
}
const ws = [...document.querySelectorAll('[data-widget]')];
if ('IntersectionObserver' in window) {
  const mk = (margin) => new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.__io.unobserve(e.target); mount(e.target); } }), { rootMargin: margin });
  const near = mk('400px 0px'), exact = mk('0px');   // the 3D figure (three.js, 1.2 MB) loads only once it is actually on screen
  ws.forEach((w) => { const o = /robot3d/.test(w.dataset.widget) ? exact : near; w.__io = o; o.observe(w); });
} else ws.forEach(mount);
