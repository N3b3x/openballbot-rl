// Shared chrome for every page: skip link, nav (primary links, Learn, Explore menu), theme toggle, prev/next pager, footer,
// scroll reveal, sticky in-page TOC (nav.pagetoc), reading-time chips (.readtime), glossary hover cards (.term[data-def])
// and click-to-play videos (.vplayer > button.vph). Every page: <script type="module" src="assets/js/layout.js"></script>.
// Links are resolved against the site root (derived from this file's URL), so pages in sub-folders can use the same chrome.
// Components are documented in site/DESIGN.md.

const ROOT = new URL('../../', import.meta.url);
const href = (p) => new URL(p, ROOT).href;
const ICONS = href('assets/img/icons.svg');
const ico = (id, cls = 'ico') => `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="${ICONS}#${id}"/></svg>`;
const REPO = 'https://github.com/N3b3x/openballbot-rl';

// Top-level links. [file, label, icon, long label (wide screens), extra class]
const PRIMARY = [
  ['index.html', 'Overview', 'i-home', '', ''],
  ['learn/index.html', 'Learn', 'i-learn', '', 'learn'],
  ['generalist.html', 'Results', 'i-results', '', ''],
  ['history.html', 'History', 'i-clock', '', ''],
  ['videos.html', 'Videos', 'i-video', '', ''],
  ['design.html', 'Guidelines', 'i-ruler', 'Design guidelines', ''],
  ['deploy.html', 'ROS 2', 'i-ros', '', ''],
];
// Explore menu: [group title, [file, label, one-line description, icon, reading-order number or '']]
const GROUPS = [
  ['Read in order', [
    ['math.html', 'The robot and its physics', 'the derived model, Froude scaling, the LQR', 'i-ball', 1],
    ['architecture.html', 'The controller', 'how LQR, residual, arrestors and planner talk', 'i-layers', 2],
    ['generalist.html', 'Results, chapter by chapter', 'the journey, with the honest nulls', 'i-results', 3],
    ['bounds.html', 'Rules and bounds', 'what a ballbot can and cannot do', 'i-shield', 4],
    ['lessons.html', 'Lessons', 'what to carry forward', 'i-book', 5],
  ]],
  ['Build and run', [
    ['design.html', 'Design guidelines', 'which robot properties the controller needs', 'i-ruler', ''],
    ['deploy.html', 'ROS 2 deployment', 'the exported controller and its parity tests', 'i-ros', ''],
    ['design-space.html', 'Design space explorer', 'bigger, lighter or built differently', 'i-design', ''],
    ['live.html', 'Live policy', 'a trained policy in your browser', 'i-ppo', ''],
    ['reproduce.html', 'Reproduce', 'setup, commands, how to cite', 'i-code', ''],
  ]],
  ['Reference', [
    ['history.html', 'History', 'the timeline, S0 to S138, with every turning point', 'i-clock', ''],
    ['story.html', 'The early journey', 'S0 to S16 as a narrative, dead ends included', 'i-route', ''],
    ['method.html', 'Setup and method', 'stack, reward, randomisation, protocols', 'i-eval', ''],
    ['results.html', 'Data tables', 'charts straight from the CSVs', 'i-grid', ''],
    ['engines.html', 'Engines', 'CPU against GPU physics', 'i-gpu', ''],
    ['literature.html', 'Literature', 'prior ballbot work', 'i-source', ''],
    ['glossary.html', 'Glossary', 'every defined term', 'i-book', ''],
  ]],
];
// Prev/next pager order. [file, pager label]
const ORDER = [
  ['index.html', 'Overview: the study in one page'],
  ['math.html', 'The robot and its physics'],
  ['architecture.html', 'The controller'],
  ['generalist.html', 'Results, chapter by chapter'],
  ['bounds.html', 'Rules and bounds'],
  ['lessons.html', 'Lessons'],
  ['design.html', 'Design guidelines'],
  ['deploy.html', 'ROS 2 deployment'],
  ['videos.html', 'Videos'],
];

const norm = (u) => { const p = new URL(u, location.href).pathname.replace(/\/$/, '/index.html'); return p.toLowerCase(); };
const herePath = norm(location.href);
const isHere = (f) => norm(href(f)) === herePath;
const cur = (f) => (isHere(f) ? ' aria-current="page"' : '');
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

// ---------- theme: stored choice, else the system preference ----------
const mqLight = matchMedia('(prefers-color-scheme: light)');
let stored = null;
try { stored = localStorage.getItem('theme'); } catch (e) { /* storage may be blocked */ }
const applyTheme = (t) => { document.documentElement.dataset.theme = t; document.dispatchEvent(new CustomEvent('themechange')); };
document.documentElement.dataset.theme = stored || (mqLight.matches ? 'light' : 'dark');
mqLight.addEventListener('change', (e) => { if (!stored) applyTheme(e.matches ? 'light' : 'dark'); });

// ---------- skip link + nav ----------
const main = document.querySelector('main');
if (main && !main.id) main.id = 'main';
const skip = document.createElement('a');
skip.className = 'skip';
skip.href = '#' + (main ? main.id : 'main');
skip.textContent = 'Skip to content';

const brandMark = '<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><rect x="7.5" y="1.5" width="11" height="10" rx="3.2" fill="var(--model)"/><path d="M9.5 11.5l-2 4.5M16.5 11.5l2 4.5" stroke="var(--text-3)" stroke-width="1.6" stroke-linecap="round"/><circle cx="13" cy="18" r="6.5" fill="none" stroke="var(--hybrid)" stroke-width="2.2"/></svg>';
const nav = document.createElement('header');
nav.className = 'nav';
const groupOpen = GROUPS.some(([, items]) => items.some(([f]) => isHere(f)));
nav.innerHTML = `<div class="wrap">
  <a class="brand" href="${href('index.html')}" aria-label="Ballbot study, overview">${brandMark}Ballbot <small>sim study</small></a>
  <div class="panel" id="site-menu">
    <nav class="links" aria-label="Main">${PRIMARY.map(([f, t, i, long, cls]) => `<a href="${href(f)}"${cur(f)}${cls ? ` class="${cls}"` : ''}>${ico(i)}<span>${t}</span>${cls === 'learn' ? '<span class="new">new</span>' : ''}</a>`).join('')}</nav>
    <details class="more-menu"><summary${groupOpen ? ' class="here"' : ''}>Explore</summary><div class="mega" role="group" aria-label="All pages">${GROUPS.map(([g, items]) => `<section><h2>${g}</h2><ul>${items.map(([f, t, d, i, n]) => `<li><a href="${href(f)}"${cur(f)}>${ico(i)}<span>${n ? `<span class="n">${n}</span>` : ''}${t}</span><span class="d">${esc(d)}</span></a></li>`).join('')}</ul></section>`).join('')}</div></details>
  </div>
  <div class="tools">
    <a class="iconbtn gh" href="${REPO}" aria-label="Source code on GitHub" title="Source on GitHub">${ico('i-github')}</a>
    <button type="button" class="iconbtn theme" aria-label="Switch colour theme" title="Light or dark theme">${ico('i-moon', 'ico i-moon')}${ico('i-sun', 'ico i-sun')}</button>
    <button type="button" class="iconbtn menu" aria-expanded="false" aria-controls="site-menu" aria-label="Open menu">${ico('i-bars', 'ico i-bars')}${ico('i-close', 'ico i-close')}</button>
  </div>
</div>`;
document.body.prepend(nav);
document.body.prepend(skip); // skip link stays first in DOM order

const themeBtn = nav.querySelector('.theme');
const syncThemeLabel = () => themeBtn.setAttribute('aria-label', `Switch to ${document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);
syncThemeLabel();
themeBtn.onclick = () => {
  const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  stored = t;
  try { localStorage.setItem('theme', t); } catch (err) { /* ignore */ }
  applyTheme(t);
  syncThemeLabel();
};

const menuBtn = nav.querySelector('.menu');
const more = nav.querySelector('.more-menu');
const narrow = matchMedia('(max-width:1100px)');
const setMenu = (open, focusBack = false) => {
  nav.classList.toggle('open', open);
  document.body.classList.toggle('menu-open', open);
  menuBtn.setAttribute('aria-expanded', String(open));
  menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  if (open) { more.open = true; const first = nav.querySelector('.links a'); if (first) first.focus(); }
  if (!open && focusBack) menuBtn.focus();
};
menuBtn.onclick = () => setMenu(!nav.classList.contains('open'));
document.addEventListener('click', (e) => {
  if (!nav.contains(e.target)) { if (nav.classList.contains('open')) setMenu(false); if (!narrow.matches) more.open = false; }
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (nav.classList.contains('open')) setMenu(false, true);
  else if (more.open) { more.open = false; more.querySelector('summary').focus(); }
});
// keep keyboard focus inside the open mobile menu
nav.addEventListener('keydown', (e) => {
  if (e.key !== 'Tab' || !nav.classList.contains('open')) return;
  const f = [...nav.querySelectorAll('a[href], button')].filter((x) => x.offsetParent !== null);
  if (!f.length) return;
  if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
  else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
});
narrow.addEventListener('change', (e) => { if (!e.matches) setMenu(false); more.open = false; });
nav.querySelectorAll('.panel a').forEach((a) => a.addEventListener('click', () => { if (narrow.matches) setMenu(false); }));

// ---------- prev / next pager + footer ----------
const idx = ORDER.findIndex(([f]) => isHere(f));
if (idx >= 0) {
  const pager = document.createElement('nav');
  pager.className = 'pager';
  pager.setAttribute('aria-label', 'Reading order');
  const prev = ORDER[idx - 1];
  const next = ORDER[idx + 1];
  pager.innerHTML = `<div class="wrap">${prev ? `<a class="prev" href="${href(prev[0])}"><span>Previous</span>${prev[1]}</a>` : '<span></span>'}${next ? `<a class="next" href="${href(next[0])}"><span>Next</span>${next[1]}</a>` : `<a class="next" href="${href('learn/index.html')}"><span>Keep going</span>Learn it step by step</a>`}</div>`;
  document.body.append(pager);
}
const f = document.createElement('footer');
f.className = 'site';
const col = (title, items) => `<div><h2>${title}</h2><ul>${items.map(([p, t]) => `<li><a href="${p.startsWith('http') ? p : href(p)}">${t}</a></li>`).join('')}</ul></div>`;
f.innerHTML = `<div class="wrap"><div class="cols">
  <div class="about"><div class="brand">${brandMark}Ballbot</div><p>A simulation study of a robot that balances on a ball: derived physics, a learned residual, passive ball arrestors, navigation and a ROS 2 bridge. Every result on this site is from MuJoCo simulation; there is no physical robot and no hardware claim.</p></div>
  ${col('Study', [['index.html', 'Overview'], ['learn/index.html', 'Learn'], ['generalist.html', 'Results'], ['videos.html', 'Videos'], ['lessons.html', 'Lessons']])}
  ${col('Build', [['design.html', 'Design guidelines'], ['deploy.html', 'ROS 2 deployment'], ['reproduce.html', 'Reproduce'], ['design-space.html', 'Design space explorer']])}
  ${col('Reference', [['method.html', 'Setup and method'], ['glossary.html', 'Glossary'], ['literature.html', 'Literature'], [REPO, 'Source on GitHub']])}
</div>
<div class="legal"><span>Numbers are generated from the result files by <code>site/build_data.py</code> and cite study-log entries (S-numbers).</span><span>Simulation only.</span></div></div>`;
document.body.append(f);

// ---------- scroll reveal ----------
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduce || !('IntersectionObserver' in window)) document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
else {
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  // Safety net: never leave content hidden (print, full-page capture, observers that never fire).
  setTimeout(() => document.querySelectorAll('.reveal:not(.in)').forEach((el) => el.classList.add('in')), 2500);
  addEventListener('beforeprint', () => document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in')));
}

// ---------- reading time: fills every .readtime (220 words per minute, counted over <main>) ----------
if (main) {
  const chips = document.querySelectorAll('.readtime');
  if (chips.length) {
    const words = (main.textContent || '').trim().split(/\s+/).length;
    const mins = Math.max(1, Math.round(words / 220));
    chips.forEach((c) => { if (!c.textContent.trim()) c.textContent = `${mins} min read`; });
  }
}

// ---------- sticky in-page TOC: <nav class="pagetoc" aria-label="On this page"></nav> ----------
// Filled from main h2[id] (or, if data-level="h3", from h2 and h3). Headings without ids get one. Pre-filled TOCs are left alone.
const toc = document.querySelector('nav.pagetoc');
if (toc && main) {
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'section';
  const sel = toc.dataset.level === 'h3' ? 'h2, h3' : 'h2';
  const heads = [...main.querySelectorAll(sel)].filter((h) => !h.closest('.tldr, .keynum, .card, details, .pagetoc') && h.textContent.trim());
  if (!toc.querySelector('a')) {
    const used = new Set();
    heads.forEach((h) => {
      if (!h.id) { let id = slug(h.textContent); let k = 2; while (used.has(id) || document.getElementById(id)) id = `${slug(h.textContent)}-${k++}`; h.id = id; }
      used.add(h.id);
    });
    const ol = document.createElement('ol');
    ol.innerHTML = heads.map((h, i) => `<li><a href="#${h.id}"${h.tagName === 'H3' ? ' class="sub"' : ''}><span class="n">${i + 1}</span>${esc(h.textContent.trim().replace(/\s+/g, ' ').slice(0, 48))}</a></li>`).join('');
    toc.append(ol);
  }
  const links = [...toc.querySelectorAll('a[href^="#"]')];
  const byId = new Map(links.map((a) => [decodeURIComponent(a.getAttribute('href').slice(1)), a]));
  const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  if (targets.length) {
    let active = null;
    const mark = (id) => {
      if (active) active.classList.remove('on');
      active = byId.get(id) || null;
      if (active) {
        active.classList.add('on');
        const ol = active.closest('ol');   // scroll the chip row only, never the page
        if (ol) ol.scrollTo({ left: active.offsetLeft - ol.clientWidth / 2 + active.offsetWidth / 2, behavior: reduce ? 'auto' : 'smooth' });
      }
    };
    const pick = () => {
      const y = window.scrollY + 140;
      let best = targets[0];
      for (const t of targets) if (t.getBoundingClientRect().top + window.scrollY <= y) best = t;
      if (!active || active !== byId.get(best.id)) mark(best.id);
    };
    let raf = 0;
    addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; pick(); }); }, { passive: true });
    pick();
  }
}

// ---------- defined terms: shift the hover card so it stays inside the viewport (16 px gutters) ----------
document.querySelectorAll('.term[data-def]').forEach((el) => {
  const place = () => {
    const r = el.getBoundingClientRect();
    const w = Math.min(320, window.innerWidth - 32);
    const left = Math.max(16, Math.min(r.left, window.innerWidth - 16 - w));
    el.style.setProperty('--tip-x', `${Math.round(left - r.left)}px`);
  };
  el.addEventListener('mouseenter', place);
  el.addEventListener('focus', place);
  // Touch: first tap shows the card, second tap follows the link to the glossary.
  el.addEventListener('touchend', (e) => {
    if (!el.classList.contains('show')) { place(); el.classList.add('show'); e.preventDefault(); setTimeout(() => el.classList.remove('show'), 4000); }
  }, { passive: false });
});

// ---------- lazy video posters: <video data-poster> gets its poster only when it is within 500 px of the viewport ----------
{
  const vs = [...document.querySelectorAll('video[data-poster]')];
  const set = (v) => { v.poster = v.dataset.poster; };
  if (vs.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { set(e.target); io.unobserve(e.target); } }), { rootMargin: '500px 0px' });
    vs.forEach((v) => io.observe(v));
  } else vs.forEach(set);
}

// ---------- click-to-play videos: <div class="vplayer" data-src data-poster data-title><button class="vph">poster img</button></div> ----------
// Nothing is downloaded until the reader asks for it; [data-vpair] with a button.vboth plays its players in sync from the start.
function mount(player, { autoplay = true, muted = false } = {}) {
  const have = player.querySelector('video');
  if (have) return have;
  const v = document.createElement('video');
  v.controls = true; v.playsInline = true; v.preload = 'metadata'; v.muted = muted;
  v.poster = player.dataset.poster; v.src = player.dataset.src; v.setAttribute('aria-label', player.dataset.title || 'Simulation clip');
  player.replaceChildren(v);
  if (autoplay) v.play().catch(() => {});
  return v;
}
document.querySelectorAll('.vplayer .vph').forEach((b) => b.addEventListener('click', () => mount(b.closest('.vplayer'))));
document.querySelectorAll('[data-vpair]').forEach((pair) => {
  const btn = pair.querySelector('button.vboth');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const vids = [...pair.querySelectorAll('.vplayer')].map((p) => mount(p, { autoplay: false, muted: true }));
    vids.forEach((v) => { v.pause(); v.currentTime = 0; });
    Promise.all(vids.map((v) => new Promise((res) => (v.readyState >= 1 ? res() : v.addEventListener('loadedmetadata', res, { once: true })))))
      .then(() => vids.forEach((v) => v.play().catch(() => {})));
    btn.textContent = 'Replay both together';
  });
});
