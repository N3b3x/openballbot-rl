// Hand-built SVG charts for results.html (corrected-engine study only). No dependencies.
// Colours come from CSS variables so the charts follow the light/dark theme; every chart is also redrawn on the
// `themechange` event and on resize. Data: site/data/study.json (built by site/build_data.py). The obstacle-size
// chart carries its numbers inline because build_data.py does not export them; each value is cited to the study log
// (S87) and to the evaluation file it was read from.
const NS = 'http://www.w3.org/2000/svg';
const SVGT = new Set(['svg', 'g', 'path', 'circle', 'rect', 'line', 'text', 'title', 'desc', 'polygon']);
const C = { model: 'var(--model)', learned: 'var(--learned)', hybrid: 'var(--hybrid)', good: 'var(--good)', bad: 'var(--bad)', warn: 'var(--warn)', mute: 'var(--text-3)' };

export function h(tag, attrs, ...kids) {
  const e = SVGT.has(tag) ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(9)) { if (c == null || c === false) continue; e.append(c.nodeType ? c : document.createTextNode(String(c))); }
  return e;
}
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const f2 = (v) => (Math.round(v * 100) / 100).toFixed(2);
const range = (vals) => { const a = vals.filter((v) => v != null); const lo = Math.min(...a), hi = Math.max(...a); return { lo, hi, mean: a.reduce((s, v) => s + v, 0) / a.length, text: f2(lo) === f2(hi) ? f2(lo) : `${f2(lo)} to ${f2(hi)}` }; };
// study.json is written by Python and can contain bare NaN tokens (not valid JSON): read as text and map them to null.
const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + ' ' + r.status); return r.text(); }).then((t) => JSON.parse(t.replace(/\bNaN\b/g, 'null')));

// -------------------------------------------------------------------------------- shared
function tooltip(host) {
  const t = h('div', { class: 'ctip', role: 'tooltip', hidden: true });
  host.append(t);
  return {
    show(anchor, html) {
      t.innerHTML = html; t.hidden = false;
      const hb = host.getBoundingClientRect(), ab = anchor.getBoundingClientRect();
      const w = t.offsetWidth, hh = t.offsetHeight;
      let x = ab.left - hb.left + ab.width / 2 - w / 2, y = ab.top - hb.top - hh - 8;
      if (y < 0) y = ab.bottom - hb.top + 8;
      x = Math.max(4, Math.min(hb.width - w - 4, x));
      t.style.left = x + 'px'; t.style.top = y + 'px';
    },
    hide() { t.hidden = true; },
  };
}
function shell(host, build) {
  host.classList.add('chart-host');
  let lastW = 0;
  const draw = () => { lastW = host.clientWidth; host.innerHTML = ''; build(host, tooltip(host)); };
  draw();
  if ('ResizeObserver' in window) new ResizeObserver(() => { if (Math.abs(host.clientWidth - lastW) > 8) draw(); }).observe(host);
  document.addEventListener('themechange', draw);
}
function focusable(g, tipc, aria, tip) {
  g.setAttribute('tabindex', 0); g.setAttribute('role', 'img'); g.setAttribute('aria-label', aria); g.classList.add('pt');
  const show = () => tipc.show(g, tip);
  g.addEventListener('mouseenter', show); g.addEventListener('focus', show);
  g.addEventListener('mouseleave', () => tipc.hide()); g.addEventListener('blur', () => tipc.hide());
  g.addEventListener('keydown', (e) => { if (e.key === 'Escape') tipc.hide(); });
  return g;
}
function keyLegend(items) {
  return h('div', { class: 'klg', 'aria-hidden': 'true' }, items.map(([sw, label]) => h('span', null, h('i', { style: sw }), label)));
}

// Horizontal bars. rows: [{label, small, bars:[{v, lo, hi, fill, op, text, aria, tip}]}]
function hbars(host, o, tipc) {
  const W = Math.max(280, host.clientWidth), narrow = W < 600;
  const lw = narrow ? 0 : Math.min(250, Math.round(W * 0.36));
  const barH = o.barH || 13, gap = 4, labelH = narrow ? 20 : 0, rowPad = 11;
  const x0 = narrow ? 4 : lw + 10, x1 = W - (o.valW || 88);
  const xs = lin(0, o.xmax, x0, x1);
  const rowH = (r) => labelH + r.bars.length * (barH + gap) + rowPad;
  const top = 6, H = top + o.rows.reduce((s, r) => s + rowH(r), 0) + 34;
  const svg = h('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-labelledby': o.id + '-t ' + o.id + '-d' },
    h('title', { id: o.id + '-t' }, o.title), h('desc', { id: o.id + '-d' }, o.desc));
  o.ticks.forEach((t) => svg.append(h('line', { x1: xs(t), x2: xs(t), y1: top, y2: H - 30, class: t === o.ref ? 'ref' : 'grid' }),
    h('text', { x: xs(t), y: H - 14, class: 'tick', 'text-anchor': 'middle' }, (o.fmt || f2)(t))));
  svg.append(h('text', { x: (x0 + x1) / 2, y: H - 1, class: 'axl', 'text-anchor': 'middle' }, narrow && o.xshort ? o.xshort : o.xlabel));
  let y = top;
  o.rows.forEach((r) => {
    if (narrow) {
      svg.append(h('text', { x: x0, y: y + 13, class: 'val' }, r.label), r.small ? h('text', { x: x0 + 6 + r.label.length * 7.2, y: y + 13, class: 'tick' }, r.small) : null);
    } else {
      svg.append(h('text', { x: lw, y: y + 12 + (r.bars.length > 1 ? (barH + gap) * (r.bars.length - 1) / 2 : 0) - (r.small ? 6 : 0), class: 'val', 'text-anchor': 'end' }, r.label),
        r.small ? h('text', { x: lw, y: y + 12 + (r.bars.length > 1 ? (barH + gap) * (r.bars.length - 1) / 2 : 0) + 8, class: 'tick', 'text-anchor': 'end' }, r.small) : null);
    }
    let by = y + labelH;
    r.bars.forEach((b) => {
      const g = h('g');
      g.append(h('rect', { x: xs(0), y: by, width: Math.max(2, xs(b.v) - xs(0)), height: barH, rx: 3, fill: b.fill, opacity: b.op ?? 0.95 }));
      if (b.lo != null && b.hi != null && b.hi > b.lo) g.append(h('line', { x1: xs(b.lo), x2: xs(b.hi), y1: by + barH / 2, y2: by + barH / 2, stroke: 'var(--text)', 'stroke-width': 2, 'stroke-linecap': 'round', opacity: 0.85 }),
        h('line', { x1: xs(b.lo), x2: xs(b.lo), y1: by + 2, y2: by + barH - 2, stroke: 'var(--text)', 'stroke-width': 1.5 }), h('line', { x1: xs(b.hi), x2: xs(b.hi), y1: by + 2, y2: by + barH - 2, stroke: 'var(--text)', 'stroke-width': 1.5 }));
      g.append(h('text', { x: Math.max(xs(b.v), xs(b.hi ?? b.v)) + 6, y: by + barH - 2, class: 'val', style: 'font-size:11px' }, b.text));
      g.append(h('rect', { x: xs(0), y: by - 2, width: x1 - x0 + 60, height: barH + 4, fill: 'transparent' }));
      svg.append(focusable(g, tipc, b.aria, b.tip));
      by += barH + gap;
    });
    y += rowH(r);
  });
  return svg;
}

// -------------------------------------------------------------------------------- 1. the journey
const RECIPES = [
  { label: 'Pure learning', small: 'no arrestors', runs: ['g12_pure_s0', 'g12_pure_s1'], fill: C.learned },
  { label: 'LQR + residual 0.5', small: 'no arrestors', runs: ['g12_dr175_s0', 'g12_dr175_s1'], fill: C.hybrid },
  { label: 'LQR + residual 0.5', small: 'with arrestors', runs: ['g16_ret100_s0', 'g16_ret100_s1'], fill: C.hybrid },
  { label: 'Pure learning', small: 'with arrestors', runs: ['g16_pure_ret100_s0', 'g16_pure_ret100_s1'], fill: C.learned },
  { label: 'LQR + residual 0.25', small: 'with arrestors, then-recommended', runs: ['g18_ret_rs025_s0', 'g18_ret_rs025_s1', 'g18_ret_rs025_s2'], fill: C.hybrid },
];
function chartJourney(host, d) {
  shell(host, (host, tipc) => {
    host.append(keyLegend([['background:var(--hybrid)', 'LQR + learned residual'], ['background:var(--learned)', 'pure learning'], ['background:var(--text-2);opacity:.95', 'solid: undisturbed'], ['background:var(--text-2);opacity:.4', 'faded: pushes and pulls']]));
    const rows = RECIPES.map((r) => {
      const u = range(r.runs.map((k) => d.goal[k]?.u.success)), dd = range(r.runs.map((k) => d.goal[k]?.d.success));
      const seeds = r.runs.map((k) => k.replace(/.*_s(\d)$/, 's$1')).join(', ');
      return { label: r.label, small: r.small, bars: [
        { v: u.mean, lo: u.lo, hi: u.hi, fill: r.fill, text: u.text, aria: `${r.label}, ${r.small}, undisturbed: goal success ${u.text} over ${r.runs.length} seed(s)`, tip: `<b>${r.label}, ${r.small}</b><br>undisturbed goal success <b>${u.text}</b><br>seeds ${seeds}, 1760 worlds each` },
        { v: dd.mean, lo: dd.lo, hi: dd.hi, fill: r.fill, op: 0.4, text: dd.text, aria: `${r.label}, ${r.small}, with pushes and pulls: goal success ${dd.text}`, tip: `<b>${r.label}, ${r.small}</b><br>with pushes and pulls <b>${dd.text}</b><br>seeds ${seeds}, 1760 worlds each` },
      ] };
    });
    host.append(hbars(host, { id: 'jr', title: 'Goal success per recipe', desc: 'Share of 1760 worlds that reached a goal 1.1 m away across ten terrain kinds, undisturbed and with pushes and pulls, for five controller recipes; whiskers span the seeds.',
      rows, xmax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'goal success (share of worlds)', valW: 96 }, tipc));
  });
}

// -------------------------------------------------------------------------------- 2. the authority dial
const DIAL_LABELS = [
  ['no arrestors (reference)', 'no arrestors', 0.997, 0.262, 'middle'], ['authority 0.15', '0.15', 1.015, 0.335, 'middle'], ['authority 0.25 (then-recommended)', '0.25, then-recommended', 0.992, 0.396, 'end'],
  ['authority 0.35', '0.35', 0.942, 0.470, 'middle'], ['authority 0.5', '0.5', 0.856, 0.506, 'middle'], ['soft LQR base, authority 0.5', 'soft LQR', 0.719, 0.526, 'middle'], ['pure learning (no LQR)', 'pure learning', 0.727, 0.478, 'middle'],
];
function chartDial(host, d) {
  shell(host, (host, tipc) => {
    const W = Math.max(280, host.clientWidth), narrow = W < 520, H = narrow ? 320 : 380, m = { l: 48, r: 16, t: 14, b: 42 };
    const xd = [0.6, 1.06], yd = [0.2, 0.56];
    const xs = lin(xd[0], xd[1], m.l, W - m.r), ys = lin(yd[0], yd[1], H - m.b, m.t);
    host.append(keyLegend([['background:var(--hybrid)', 'LQR + residual, arrestors'], ['background:var(--learned)', 'pure learning, arrestors'], ['border:2px solid var(--text-3);background:transparent', 'no arrestors'], ['border:2px solid var(--hybrid);background:transparent', 'arrestors added at evaluation only']]));
    const svg = h('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-labelledby': 'dl-t dl-d' }, h('title', { id: 'dl-t' }, 'Tracking accuracy against disturbance robustness'),
      h('desc', { id: 'dl-d' }, 'Each point is one trained policy: achieved over commanded speed on flat ground (x) against goal success with pushes and pulls (y). Robustness rises and tracking falls as the learned authority grows; authority 0.25 keeps tracking at 1.0 with most of the robustness.'));
    [0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => svg.append(h('line', { x1: xs(t), x2: xs(t), y1: m.t, y2: H - m.b, class: t === 1 ? 'ref' : 'grid' }), h('text', { x: xs(t), y: H - m.b + 16, class: 'tick', 'text-anchor': 'middle' }, t.toFixed(1))));
    [0.2, 0.3, 0.4, 0.5].forEach((t) => svg.append(h('line', { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), class: 'grid' }), h('text', { x: m.l - 6, y: ys(t) + 4, class: 'tick', 'text-anchor': 'end' }, t.toFixed(1))));
    svg.append(h('text', { x: (m.l + W - m.r) / 2, y: H - 6, class: 'axl', 'text-anchor': 'middle' }, narrow ? 'achieved / commanded speed (flat)' : 'achieved / commanded speed on flat ground (1.0 = exact)'),
      h('text', { x: 12, y: (m.t + H - m.b) / 2, class: 'axl', 'text-anchor': 'middle', transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, narrow ? 'goal success, pushes + pulls' : 'goal success with pushes and pulls'),
      h('text', { x: xs(1.0) - 4, y: m.t + 10, class: 'axl', 'text-anchor': 'end' }, 'exact tracking'));
    DIAL_LABELS.forEach(([, text, x, y, anchor]) => svg.append(h('text', { x: xs(x), y: ys(y), class: 'axl', 'text-anchor': anchor, style: 'font-size:10.5px;fill:var(--text-2)' }, text)));
    d.tradeoff.forEach((p) => {
      const col = p.kind === 'pure' ? C.learned : p.kind === 'ref' ? C.mute : C.hybrid, hollow = p.kind === 'ref' || p.kind === 'zs';
      const g = h('g');
      if (/then-recommended/.test(p.label)) g.append(h('circle', { cx: xs(p.ach), cy: ys(p.d), r: 11, fill: 'none', stroke: C.hybrid, 'stroke-width': 1.5, 'stroke-dasharray': '3 3', opacity: 0.8 }));
      g.append(h('circle', { cx: xs(p.ach), cy: ys(p.d), r: 6, fill: hollow ? 'var(--bg-2)' : col, stroke: col, 'stroke-width': 2 }), h('circle', { cx: xs(p.ach), cy: ys(p.d), r: 13, fill: 'transparent' }));
      svg.append(focusable(g, tipc, `${p.label} (${p.run}): tracking ${f2(p.ach)}, goal success with pushes and pulls ${f2(p.d)}`, `<b>${p.label}</b><br>run ${p.run}<br>achieved / commanded <b>${f2(p.ach)}</b><br>goal success, pushes + pulls <b>${f2(p.d)}</b>`));
    });
    host.append(svg);
  });
}

// -------------------------------------------------------------------------------- 3. falls by obstacle size
// Values from docs/research/tracking_study_log.md S87 (Table 15 of generalist.html), read from
// outputs/study/warp_ppo/g18_ret_rs025_s1/speed_long_0.3_sev.json (by_severity). Velocity task 0.3 m/s, 20 s, undisturbed.
const SIZE = [
  { k: 'step up', unit: 'cm', lo: 1.1, split: 2.3, hi: 5.9, s: 0.40, l: 0.95 },
  { k: 'bump', unit: 'cm', lo: 1.4, split: 2.4, hi: 3.6, s: 0.59, l: 0.90 },
  { k: 'stairs (riser)', unit: 'cm', lo: 1.2, split: 1.4, hi: 3.2, s: 0.27, l: 0.70 },
  { k: 'large drop', unit: 'cm', lo: 10.9, split: 15.1, hi: 18.3, s: 0.61, l: 0.77 },
  { k: 'small drop', unit: 'cm', lo: 3.1, split: 4.4, hi: 5.8, s: 0.23, l: 0.34 },
  { k: 'step down', unit: 'cm', lo: 1.2, split: 2.6, hi: 5.5, s: 0.12, l: 0.21 },
];
function chartSize(host) {
  shell(host, (host, tipc) => {
    host.append(keyLegend([['background:var(--warn)', 'smaller half of the feature range'], ['background:var(--bad)', 'larger half']]));
    const rows = SIZE.map((e) => ({ label: e.k, small: `${e.lo} to ${e.hi} ${e.unit}, split ${e.split}`, bars: [
      { v: e.s, fill: C.warn, text: `${f2(e.s)}  (${e.lo} to ${e.split} ${e.unit})`, aria: `${e.k}, smaller half ${e.lo} to ${e.split} ${e.unit}: falls ${f2(e.s)}`, tip: `<b>${e.k}</b>, ${e.lo} to ${e.split} ${e.unit}<br>falls <b>${f2(e.s)}</b>` },
      { v: e.l, fill: C.bad, text: `${f2(e.l)}  (${e.split} to ${e.hi} ${e.unit})`, aria: `${e.k}, larger half ${e.split} to ${e.hi} ${e.unit}: falls ${f2(e.l)}`, tip: `<b>${e.k}</b>, ${e.split} to ${e.hi} ${e.unit}<br>falls <b>${f2(e.l)}</b>` },
    ] }));
    host.append(hbars(host, { id: 'sz', title: 'Falls by obstacle size', desc: 'Fall rate of the then-recommended controller on each terrain feature, split at the median drawn size into a smaller and a larger half. Up-steps and bumps above about 2.4 cm are nearly always fatal.',
      rows, xmax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'share of worlds that fell (0.3 m/s, 20 s)', xshort: 'share of worlds that fell', valW: 120 }, tipc));
  });
}

// -------------------------------------------------------------------------------- 4. navigation per sensor
const NAV_ROWS = [
  ['nav_blind', 'blind', 'straight to the goal'], ['nav_oracle', 'perfect height map', 'upper bound'], ['nav_tof18_head', 'ToF 18 x 14, head', 'pitch -25 deg'], ['nav_cam32_head', 'depth camera 32 x 24, head', 'pitch -25 deg'],
  ['nav_tof8_head', 'ToF 8 x 8, head', 'pitch -25 deg'], ['nav_tof18_low', 'ToF 18 x 14, low mount', 'at the base, pitch -10 deg'], ['nav_tof18_cones', 'two ToF 18 x 14 on the body cones', 'rigid with the body'],
  ['nav_blind_pp', 'blind, with pushes and pulls', ''], ['nav_oracle_pp', 'perfect map, with pushes and pulls', ''], ['nav_tof18_head_pp', 'ToF 18 x 14 head, with pushes and pulls', ''],
];
function chartNav(host, d) {
  shell(host, (host, tipc) => {
    const W = Math.max(280, host.clientWidth), narrow = W < 600, lw = narrow ? 0 : Math.min(260, Math.round(W * 0.38));
    const labelH = narrow ? 18 : 0, barH = 15, rowH = labelH + barH + 12, top = 6, x0 = narrow ? 4 : lw + 10, x1 = W - 10, xs = lin(0, 1, x0, x1);
    const H = top + NAV_ROWS.length * rowH + 40 + (narrow ? 0 : 0);
    host.append(keyLegend([['background:var(--good)', 'reached the goal'], ['background:var(--warn)', 'stuck'], ['background:var(--bad)', 'fell']]));
    const svg = h('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-labelledby': 'nv-t nv-d' }, h('title', { id: 'nv-t' }, 'Navigation outcome per hazard source'),
      h('desc', { id: 'nv-d' }, 'Share of 528 worlds that reached a goal behind an 8 cm block, got stuck, or fell, for each source of obstacle information given to the planner; the balance controller is the same in every row.'));
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => svg.append(h('line', { x1: xs(t), x2: xs(t), y1: top, y2: H - 30, class: 'grid' }), h('text', { x: xs(t), y: H - 14, class: 'tick', 'text-anchor': 'middle' }, t === 0 || t === 1 ? t : f2(t))));
    svg.append(h('text', { x: (x0 + x1) / 2, y: H - 1, class: 'axl', 'text-anchor': 'middle' }, 'share of worlds'));
    NAV_ROWS.forEach(([key, label, small], i) => {
      const r = d.nav[key]; if (!r) return;
      const y = top + i * rowH, by = y + labelH;
      if (i === 7) svg.append(h('line', { x1: x0, x2: x1, y1: y - 5, y2: y - 5, class: 'ref' }));
      if (narrow) svg.append(h('text', { x: x0, y: y + 12, class: 'val' }, label), small ? h('text', { x: x0 + 6 + label.length * 7.2, y: y + 12, class: 'tick' }, small) : null);
      else svg.append(h('text', { x: lw, y: by + 11 - (small ? 5 : 0), class: 'val', 'text-anchor': 'end' }, label), small ? h('text', { x: lw, y: by + 19, class: 'tick', 'text-anchor': 'end' }, small) : null);
      const segs = [['success', C.good, 'reached the goal'], ['stuck', C.warn, 'stuck'], ['fall', C.bad, 'fell']];
      let x = 0;
      const g = h('g');
      segs.forEach(([k, col, name]) => { const v = r[k] || 0; if (v > 0.002) g.append(h('rect', { x: xs(x), y: by, width: Math.max(1, xs(x + v) - xs(x)), height: barH, fill: col, opacity: 0.92 })); x += v; });
      g.append(h('text', { x: xs(Math.min(r.success, 0.999)) + (r.success > 0.3 ? -5 : 5), y: by + barH - 3, class: 'val', style: `font-size:11px;${r.success > 0.3 ? 'fill:#04120f' : ''}`, 'text-anchor': r.success > 0.3 ? 'end' : 'start' }, f2(r.success)));
      g.append(h('rect', { x: x0, y: by - 2, width: x1 - x0, height: barH + 4, fill: 'transparent' }));
      svg.append(focusable(g, tipc, `${label}${small ? ', ' + small : ''}: reached ${f2(r.success)}, stuck ${f2(r.stuck)}, fell ${f2(r.fall)}, rolled onto a block ${f2(r.over_block)}`,
        `<b>${label}</b>${small ? '<br>' + small : ''}<br>reached the goal <b>${f2(r.success)}</b><br>stuck ${f2(r.stuck)}<br>fell <span class="bad">${f2(r.fall)}</span><br>rolled onto a block ${f2(r.over_block)}`));
    });
    host.append(svg);
  });
}

// -------------------------------------------------------------------------------- 5. unseen designs
const HO = [
  ['g18_ret_rs025_s1', 'LQR + residual 0.25', 'then-recommended, arrestors, seed 1', C.hybrid], ['g12_dr175_s1', 'LQR + residual 0.5', 'no arrestors, seed 1', C.hybrid],
  ['g16_pure_ret100_s1', 'Pure learning', 'arrestors, seed 1', C.learned], ['g21_ret_rs025_tau3_s0', 'LQR + residual 0.25, 12 designs', 'trained with tau3, seed 0', C.hybrid], ['g21_ret_rs025_tau3_s1', 'LQR + residual 0.25, 12 designs', 'trained with tau3, seed 1', C.hybrid],
];
function chartDesigns(host, d) {
  shell(host, (host, tipc) => {
    host.append(keyLegend([['background:var(--hybrid)', 'LQR + learned residual'], ['background:var(--learned)', 'pure learning'], ['background:var(--text-2);opacity:.95', 'solid: 1.5x robot (13.5 cm ball)'], ['background:var(--text-2);opacity:.4', 'faded: tenth-torque robot']]));
    const rows = HO.map(([run, label, small, fill]) => {
      const r = d.designs_ho[run]; if (!r) return null;
      const mk = (des, op, name) => ({ v: r.u[des], fill, op, text: `${f2(r.u[des])}  (${f2(r.d[des])} with pushes)`, aria: `${label}, ${small}, ${name}: goal success ${f2(r.u[des])} undisturbed, ${f2(r.d[des])} with pushes and pulls`,
        tip: `<b>${label}</b><br>${small}<br>${name}<br>undisturbed <b>${f2(r.u[des])}</b><br>pushes + pulls <b>${f2(r.d[des])}</b>` });
      return { label, small, bars: [mk('s1.5', 0.95, '1.5x robot, unseen'), mk('tau3', 0.4, 'tenth-torque robot, ' + (run.includes('tau3') ? 'trained on' : 'unseen'))] };
    }).filter(Boolean);
    host.append(hbars(host, { id: 'ho', title: 'Goal success on robot designs the policy never trained on', desc: 'Undisturbed goal success on a robot scaled 1.5x and on a robot with one tenth of the wheel torque, for LQR-based and pure-learning policies; the value with pushes and pulls is given in brackets.',
      rows, xmax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'goal success, undisturbed (with pushes and pulls in brackets)', xshort: 'goal success (pushes in brackets)', valW: 150 }, tipc));
  });
}

// -------------------------------------------------------------------------------- boot
const CHARTS = { journey: chartJourney, dial: chartDial, size: chartSize, nav: chartNav, designs: chartDesigns };
async function boot() {
  const nodes = [...document.querySelectorAll('[data-chart]')];
  if (!nodes.length) return;
  let data = null;
  try { data = await get('data/study.json'); } catch (err) { nodes.forEach((el) => el.append(h('p', { class: 'callout bad' }, 'Could not load data/study.json: ' + err.message))); console.error(err); return; }
  nodes.forEach((el) => { const fn = CHARTS[el.dataset.chart]; if (fn) { try { fn(el, data); } catch (err) { el.append(h('p', { class: 'callout bad' }, 'Chart failed: ' + err.message)); console.error(err); } } });
}
boot();
