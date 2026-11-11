// Shared chrome for every page: skip link, nav (reading order + More menu), theme toggle, prev/next pager, footer,
// scroll reveal, sticky in-page TOC (nav.pagetoc), reading-time chips (.readtime) and glossary hover cards (.term[data-def]).
// Every page: <script type="module" src="assets/js/layout.js"></script>. Components are documented in site/DESIGN.md.

// Reading order from BRIEF_2026-11-08.md. [file, nav label, pager label]
const ORDER = [
  ['index.html', 'Start here', 'Start here: the study in one page'],
  ['math.html', 'Physics', 'The robot and its physics'],
  ['architecture.html', 'Controller', 'The controller'],
  ['generalist.html', 'Results', 'Results, chapter by chapter'],
  ['bounds.html', 'Rules and bounds', 'Rules and bounds'],
  ['lessons.html', 'Lessons', 'Lessons'],
  ['videos.html', 'Videos', 'Videos'],
];
// Everything else lives under "More". [file, label, one-line description]
const MORE = [
  ['design-space.html', 'Design space explorer', 'what changes when the robot is bigger, lighter or built differently'],
  ['live.html', 'Live policy', 'a trained policy running in your browser'],
  ['literature.html', 'Literature', 'prior ballbot control and learning work'],
  ['reproduce.html', 'Reproduce', 'setup, commands, protocols, how to cite'],
  null,
  ['story.html', 'The journey', 'the narrative version, dead ends included'],
  ['method.html', 'Setup and method', 'software stack, reward, randomisation, protocols'],
  ['results.html', 'Data tables', 'tracking, pushes, steps and designs from the CSVs'],
  ['engines.html', 'Engines', 'CPU against GPU physics engines'],
  ['glossary.html', 'Glossary', 'every defined term in one place'],
];

const here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
const cur = (h) => (h === here ? ' aria-current="page"' : '');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

// ---------- theme ----------
let theme = 'dark';
try { theme = localStorage.getItem('theme') || 'dark'; } catch (e) { /* storage may be blocked */ }
document.documentElement.dataset.theme = theme;
const themeIcon = (t) => (t === 'dark' ? '☾' : '☀');

// ---------- skip link + nav ----------
const main = document.querySelector('main');
if (main && !main.id) main.id = 'main';
const skip = document.createElement('a');
skip.className = 'skip';
skip.href = '#' + (main ? main.id : 'main');
skip.textContent = 'Skip to content';
document.body.prepend(skip);

const nav = document.createElement('header');
nav.className = 'nav';
nav.innerHTML = `<div class="wrap">
  <a class="brand" href="index.html"><svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><circle cx="13" cy="17" r="7" fill="none" stroke="var(--hybrid)" stroke-width="2"/><rect x="8" y="2" width="10" height="9" rx="3" fill="var(--model)"/></svg>Ballbot</a>
  <div class="panel" id="site-menu">
    <nav class="links" aria-label="Reading order">${ORDER.map(([h, t], i) => `<a href="${h}"${cur(h)}><span class="n" aria-hidden="true">${i + 1}</span>${t}</a>`).join('')}</nav>
    <details class="more-menu"><summary aria-label="More pages">More</summary><ul>${MORE.map((m) => (m ? `<li><a href="${m[0]}"${cur(m[0])}>${m[1]}<span class="d">${esc(m[2])}</span></a></li>` : '<li class="sep" role="separator"></li>')).join('')}</ul></details>
  </div>
  <div class="tools">
    <button type="button" id="theme" aria-label="Switch to ${theme === 'dark' ? 'light' : 'dark'} theme" title="Theme">${themeIcon(theme)}</button>
    <button type="button" class="menu" aria-expanded="false" aria-controls="site-menu"><span class="bars" aria-hidden="true"></span>Menu</button>
  </div>
</div>`;
document.body.prepend(nav);
document.body.prepend(skip); // skip link stays first in DOM order

const themeBtn = nav.querySelector('#theme');
themeBtn.onclick = () => {
  const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('theme', t); } catch (err) { /* ignore */ }
  themeBtn.textContent = themeIcon(t);
  themeBtn.setAttribute('aria-label', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
  document.dispatchEvent(new CustomEvent('themechange'));
};

const menuBtn = nav.querySelector('.menu');
const more = nav.querySelector(".more-menu");
const setMenu = (open) => { nav.classList.toggle('open', open); menuBtn.setAttribute('aria-expanded', String(open)); if (open) more.open = true; };
menuBtn.onclick = () => setMenu(!nav.classList.contains('open'));
document.addEventListener('click', (e) => {
  if (!nav.contains(e.target)) { setMenu(false); if (window.innerWidth > 960) more.open = false; }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setMenu(false); more.open = false; } });
matchMedia('(max-width:960px)').addEventListener('change', (e) => { if (!e.matches) setMenu(false); more.open = false; });

// ---------- prev / next pager + footer ----------
const idx = ORDER.findIndex(([h]) => h === here);
if (idx >= 0) {
  const pager = document.createElement('nav');
  pager.className = 'pager';
  pager.setAttribute('aria-label', 'Reading order');
  const prev = ORDER[idx - 1];
  const next = ORDER[idx + 1];
  pager.innerHTML = `<div class="wrap">${prev ? `<a class="prev" href="${prev[0]}"><span>Previous</span>${prev[2]}</a>` : '<span></span>'}${next ? `<a class="next" href="${next[0]}"><span>Next in the reading order</span>${next[2]}</a>` : '<a class="next" href="index.html"><span>Back to</span>Start here</a>'}</div>`;
  document.body.append(pager);
}
const f = document.createElement('footer');
f.className = 'site';
f.innerHTML = '<div class="wrap">Ballbot control research: derived physics, learned policies and their interleaving. Simulation results only; no hardware claims. Source on <a href="https://github.com/N3b3x/openballbot-rl">GitHub</a>. <a href="glossary.html">Glossary</a>.</div>';
document.body.append(f);

// ---------- scroll reveal ----------
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduce) document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
else {
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add('in')), { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  // Safety net: never leave content hidden (print, full-page capture, observers that never fire).
  setTimeout(() => document.querySelectorAll('.reveal:not(.in)').forEach((el) => el.classList.add('in')), 2500);
  addEventListener('beforeprint', () => document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in')));
}

// ---------- reading time: fills every .readtime (220 words per minute, counted over <main>) ----------
if (main) {
  const chips = document.querySelectorAll('.readtime');
  if (chips.length) {
    const words = (main.innerText || main.textContent || '').trim().split(/\s+/).length;
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
      if (active) { active.classList.add('on'); active.scrollIntoView({ block: 'nearest', inline: 'center', behavior: reduce ? 'auto' : 'smooth' }); }
    };
    const pick = () => {
      const y = window.scrollY + 130;
      let best = targets[0];
      for (const t of targets) if (t.offsetTop <= y) best = t;
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
