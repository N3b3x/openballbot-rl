// Small UI kit for the course widgets: theme colours, HiDPI canvases, an SVG line/scatter/bar chart, sliders, segmented
// buttons, read-outs and an animation clock. No dependencies. Colours come from the CSS tokens so everything follows the theme.
const NS = 'http://www.w3.org/2000/svg';

export function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'; }
export function palette() {
  const n = ['bg', 'bg-2', 'bg-3', 'line', 'line-2', 'text', 'text-2', 'text-3', 'model', 'learned', 'hybrid', 'warn', 'bad', 'good'];
  const o = {}; n.forEach((k) => { o[k.replace('-', '')] = css('--' + k); }); return o;
}
export function el(tag, attrs = {}, ...kids) {
  const svg = ['svg', 'g', 'path', 'line', 'circle', 'rect', 'text', 'polyline', 'polygon', 'title', 'desc', 'defs', 'marker', 'tspan', 'clipPath'].includes(tag);
  const e = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  kids.flat(9).forEach((c) => { if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c))); });
  return e;
}
export const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? 'n/a' : (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d));
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, t) => a + (b - a) * t;

/** HiDPI canvas filling `host` width; height = f(width). Calls draw(ctx, w, h) on resize and theme change. */
export function canvas(host, heightOf, draw, ctx0) {
  const c = el('canvas');
  host.append(c);
  const g = c.getContext('2d');
  const ttl = host.closest('figure') && host.closest('figure').querySelector('h3');
  c.setAttribute('role', 'img'); c.setAttribute('aria-label', `${ttl ? ttl.textContent : 'Figure'} (drawing; the same values are in the read-outs)`);
  const st = { c, g, w: 0, h: 0, draw };
  const size = () => {
    const w = Math.max(200, host.clientWidth);
    const h = Math.round(heightOf(w));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.height = h + 'px';
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    st.w = w; st.h = h;
    st.draw && st.draw(g, w, h);
  };
  new ResizeObserver(() => size()).observe(host);
  size();
  if (ctx0) ctx0.onTheme(() => st.draw && st.draw(g, st.w, st.h));
  st.redraw = () => st.draw && st.draw(g, st.w, st.h);
  return st;
}

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

/**
 * SVG chart. opts: { x:[lo,hi], y:[lo,hi], xlabel, ylabel, height, series:[{data:[[x,y],..], color:'--model', width, dash, label,
 * area, points}], bars:[{x, y, color, label}], hlines:[{y, color, label}], vlines:[{x, color, label}], dots:[{x,y,color,r,label}],
 * xfmt, yfmt, crosshair:true, desc }. Returns { update(partial) }.
 */
export function chart(host, opts) {
  const wrap = el('div', { style: { position: 'relative' } });
  host.append(wrap);
  const tip = el('div', { class: 'tip', hidden: true });
  wrap.append(tip);
  let o = { height: 220, xfmt: (v) => +v.toPrecision(4) + '', yfmt: (v) => +v.toPrecision(4) + '', ...opts };
  let svg = null, W = 0;
  const color = (c) => (c && c.startsWith('--') ? `var(${c})` : c || 'var(--text-2)');
  function render() {
    W = Math.max(240, wrap.clientWidth || host.clientWidth || 320);
    const H = o.height;
    const m = { l: o.ml ?? 54, r: o.mr ?? 12, t: o.title ? 24 : 10, b: o.xlabel ? 38 : 24 };
    const iw = W - m.l - m.r;
    const xt = o.xticks || ticks(o.x[0], o.x[1], Math.max(2, Math.floor(iw / 70)));
    // crowded text tick labels (category axes on a phone): rotate and make room instead of overlapping
    const xl = xt.map((v) => String(o.xfmt(v)));
    const maxLen = Math.min(15, Math.max(0, ...xl.map((t) => t.length)));
    const minGap = xt.length > 1 ? Math.min(...xt.slice(1).map((v, i) => Math.abs(v - xt[i]))) * (iw / ((o.x[1] - o.x[0]) || 1)) : iw;
    const rot = Math.max(0, ...xl.map((t) => t.length)) * 6.4 + 4 > minGap;
    if (rot) m.b += Math.min(46, maxLen * 4.2);
    const ih = H - m.t - m.b;
    const X = (v) => m.l + ((v - o.x[0]) / (o.x[1] - o.x[0])) * iw;
    const Y = (v) => m.t + ih - ((v - o.y[0]) / (o.y[1] - o.y[0])) * ih;
    const s = el('svg', { class: 'chart', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.desc || o.title || 'chart' });
    if (o.title) s.append(el('text', { class: 'ttl', x: m.l, y: 14, text: o.title }));
    const gg = el('g', { class: 'grid' });
    const yt = o.yticks || ticks(o.y[0], o.y[1], Math.max(2, Math.floor(ih / 40)));
    yt.forEach((v) => { gg.append(el('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) })); s.append(el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', text: o.yfmt(v) })); });
    xt.forEach((v, i) => {
      if (rot) { const tx = el('text', { transform: `translate(${X(v) + 3} ${m.t + ih + 12}) rotate(-40)`, 'text-anchor': 'end', text: xl[i].length > 15 ? xl[i].slice(0, 14) + '…' : xl[i] }); tx.append(el('title', { text: xl[i] })); s.append(tx); }
      else s.append(el('text', { x: X(v), y: m.t + ih + 15, 'text-anchor': 'middle', text: xl[i] }));
    });
    s.prepend(gg);
    if (o.y[0] < 0 && o.y[1] > 0) s.append(el('line', { class: 'zero', x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0) }));
    if (o.xlabel) s.append(el('text', { x: m.l + iw / 2, y: H - 4, 'text-anchor': 'middle', text: o.xlabel }));
    if (o.ylabel) s.append(el('text', { transform: `translate(11 ${m.t + ih / 2}) rotate(-90)`, 'text-anchor': 'middle', text: o.ylabel }));
    const clipId = 'c' + Math.random().toString(36).slice(2, 8);
    const defs = el('defs'); const cp = el('clipPath', { id: clipId }); cp.append(el('rect', { x: m.l, y: m.t - 2, width: iw, height: ih + 4 })); defs.append(cp); s.append(defs);
    const body = el('g', { 'clip-path': `url(#${clipId})` });
    (o.bands || []).forEach((b) => body.append(el('rect', { x: X(b.x0), width: Math.max(0, X(b.x1) - X(b.x0)), y: m.t, height: ih, style: { fill: color(b.color), opacity: b.opacity ?? 0.1 } })));
    (o.hbands || []).forEach((b) => body.append(el('rect', { x: m.l, width: iw, y: Y(b.y1), height: Math.max(0, Y(b.y0) - Y(b.y1)), style: { fill: color(b.color), opacity: b.opacity ?? 0.1 } })));
    (o.bars || []).forEach((b) => {
      const bw = b.w ?? (iw / ((o.x[1] - o.x[0]) || 1)) * 0.7;
      const y0 = Y(Math.max(o.y[0], 0)), y1 = Y(b.y);
      const r = el('rect', { x: X(b.x) - bw / 2 + (b.dx || 0), width: bw, y: Math.min(y0, y1), height: Math.abs(y0 - y1), rx: 3, style: { fill: color(b.color), opacity: b.opacity ?? 0.9 } });
      if (b.label) r.append(el('title', { text: b.label }));
      body.append(r);
    });
    (o.series || []).forEach((sr) => {
      if (!sr.data || !sr.data.length) return;
      const d = sr.data.filter((p) => Number.isFinite(p[1])).map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
      if (sr.area) body.append(el('path', { d: `${d}L${X(sr.data.at(-1)[0])},${Y(Math.max(o.y[0], 0))}L${X(sr.data[0][0])},${Y(Math.max(o.y[0], 0))}Z`, style: { fill: color(sr.color), opacity: 0.12 } }));
      body.append(el('path', { d, style: { fill: 'none', stroke: color(sr.color), strokeWidth: sr.width || 2, strokeDasharray: sr.dash || '', strokeLinejoin: 'round', strokeLinecap: 'round', opacity: sr.opacity ?? 1 } }));
    });
    (o.hlines || []).forEach((hl) => { body.append(el('line', { x1: m.l, x2: m.l + iw, y1: Y(hl.y), y2: Y(hl.y), style: { stroke: color(hl.color), strokeDasharray: hl.dash || '4 4', strokeWidth: 1.2 } })); if (hl.label) s.append(el('text', { x: m.l + iw - 4, y: Y(hl.y) - 4, 'text-anchor': 'end', text: hl.label, style: { fill: color(hl.color) } })); });
    (o.vlines || []).forEach((vl) => { body.append(el('line', { y1: m.t, y2: m.t + ih, x1: X(vl.x), x2: X(vl.x), style: { stroke: color(vl.color), strokeDasharray: vl.dash || '4 4', strokeWidth: 1.2 } })); if (vl.label) s.append(el('text', { x: X(vl.x) + 4, y: m.t + 12, text: vl.label, style: { fill: color(vl.color) } })); });
    s.append(body);
    const dotG = el('g');
    (o.dots || []).forEach((p) => {
      const c = el('circle', { cx: X(p.x), cy: Y(p.y), r: p.r || 4.5, style: { fill: p.hollow ? 'var(--bg-2)' : color(p.color), stroke: color(p.color), strokeWidth: 2 }, tabindex: p.label ? 0 : null, 'aria-label': p.label || null });
      if (p.label) {
        c.append(el('title', { text: p.label }));
        const show = () => { tip.hidden = false; tip.textContent = p.label; const x = X(p.x), y = Y(p.y); tip.style.left = Math.min(W - 200, Math.max(0, x - 60)) + 'px'; tip.style.top = Math.max(0, y - 46) + 'px'; };
        c.addEventListener('mouseenter', show); c.addEventListener('focus', show);
        c.addEventListener('mouseleave', () => { tip.hidden = true; }); c.addEventListener('blur', () => { tip.hidden = true; });
      }
      dotG.append(c);
      if (p.text) dotG.append(el('text', { x: X(p.x) + (p.tx ?? 7), y: Y(p.y) + (p.ty ?? 4), text: p.text, style: { fill: color(p.color) } }));
    });
    s.append(dotG);
    if (o.crosshair && (o.series || []).length) {
      const cl = el('line', { y1: m.t, y2: m.t + ih, style: { stroke: 'var(--text-3)', strokeWidth: 1, opacity: 0 } });
      s.append(cl);
      const ov = el('rect', { x: m.l, y: m.t, width: iw, height: ih, style: { fill: 'transparent' } });
      const move = (ev) => {
        const r = s.getBoundingClientRect();
        const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
        const xv = o.x[0] + ((px - m.l) / iw) * (o.x[1] - o.x[0]);
        cl.setAttribute('x1', px); cl.setAttribute('x2', px); cl.style.opacity = 0.6;
        const rows = o.series.filter((sr) => sr.label && sr.data && sr.data.length).map((sr) => {
          let best = sr.data[0];
          for (const p of sr.data) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
          return `<span style="color:${color(sr.color)}">${sr.label}</span> ${o.yfmt(best[1])}`;
        });
        tip.innerHTML = `${o.xlabel ? o.xlabel.split(' (')[0] : 'x'} ${o.xfmt(xv)}<br>${rows.join('<br>')}`;
        tip.hidden = false;
        tip.style.left = Math.min(W - 170, Math.max(0, px + 10)) + 'px'; tip.style.top = m.t + 'px';
      };
      ov.addEventListener('mousemove', move); ov.addEventListener('touchmove', move, { passive: true });
      ov.addEventListener('mouseleave', () => { tip.hidden = true; cl.style.opacity = 0; });
      s.append(ov);
    }
    if (o.after) o.after(s, { X, Y, m, iw, ih, W, H });
    if (svg) svg.replaceWith(s); else wrap.prepend(s);
    svg = s;
  }
  let lastW = 0;
  new ResizeObserver(() => { const w = wrap.clientWidth; if (Math.abs(w - lastW) > 2) { lastW = w; render(); } }).observe(wrap);
  render();
  return { update(p) { o = { ...o, ...p }; render(); }, get opts() { return o; } };
}

/** Labelled range slider. Returns the input element; calls onInput(value) on every change. */
export function slider(host, { label, min, max, step, value, fmt: f = (v) => v, onInput, id }) {
  const uid = id || 's' + Math.random().toString(36).slice(2, 8);
  const lab = el('label', { for: uid, text: label });
  const out = el('output', { for: uid, text: f(value) });
  const inp = el('input', { type: 'range', id: uid, min, max, step, value });
  inp.addEventListener('input', () => { const v = +inp.value; out.textContent = f(v); onInput && onInput(v); });
  inp.set = (v) => { inp.value = v; out.textContent = f(+inp.value); };
  host.append(el('div', { class: 'ctl' }, lab, out, inp));
  return inp;
}
/** Segmented toggle buttons. items: [[value, label], ...]. */
export function seg(host, items, value, onPick, label) {
  const g = el('div', { class: 'seg', role: 'group', 'aria-label': label || 'options' });
  const bs = items.map(([v, t]) => {
    const b = el('button', { type: 'button', 'aria-pressed': String(v === value), text: t });
    b.onclick = () => { bs.forEach((x) => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); onPick(v); };
    g.append(b); return b;
  });
  host.append(g);
  g.pick = (v) => { items.forEach(([iv], i) => bs[i].setAttribute('aria-pressed', String(iv === v))); };
  return g;
}
/** Read-out tiles. items: [[key, label, cls]]. Returns { set(key, text) }. */
export function readouts(host, items) {
  const box = el('div', { class: 'readouts', 'aria-live': 'polite' });
  const m = {};
  items.forEach(([k, lab, cls]) => { const b = el('b', { text: '...' }); m[k] = b; box.append(el('div', { class: 'ro ' + (cls || '') }, el('span', { text: lab }), b)); });
  host.append(box);
  return { set: (k, t) => { if (m[k]) m[k].textContent = t; }, box };
}
export function button(host, text, onClick, cls = '') {
  const b = el('button', { type: 'button', class: 'btn ' + cls, text });
  b.onclick = onClick; host.append(b); return b;
}
export function legend(host, items) {
  const l = el('div', { class: 'legend' });
  items.forEach(([c, t, dash]) => l.append(el('span', {}, el('i', { class: dash ? 'd' : '', style: { background: c.startsWith('--') ? `var(${c})` : c, color: c.startsWith('--') ? `var(${c})` : c } }), t)));
  host.append(l); return l;
}
/** requestAnimationFrame clock that pauses when the figure is off screen. step(dt) returns false to stop. */
export function clock(target, step) {
  let on = false, last = 0, raf = 0, visible = true;
  const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible && on && !raf) { last = 0; raf = requestAnimationFrame(f); } });
  io.observe(target);
  function f(t) {
    raf = 0;
    if (!on || !visible) return;
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 0.016;
    last = t;
    if (step(dt) === false) { on = false; return; }
    raf = requestAnimationFrame(f);
  }
  return {
    start() { if (!on) { on = true; last = 0; if (!raf) raf = requestAnimationFrame(f); } },
    stop() { on = false; },
    get running() { return on; },
  };
}

/** Draw the planar ballbot side view on a 2D canvas. x, theta in metres / rad; opts: { scale px per m, ox, oy (floor y),
 * l (COM height), color, ghost, push (N, arrow), comDot, label } */
export function drawBot(g, x, theta, o) {
  const pal = o.pal;
  const S = o.scale, r = (o.r || 0.09) * S;
  const cx = o.ox + x * S, cy = o.oy - r;
  const roll = -(x / (o.r || 0.09));
  g.save();
  g.globalAlpha = o.ghost ? 0.28 : 1;
  // ball
  const grd = g.createRadialGradient(cx - r * 0.35, cy - r * 0.35, r * 0.1, cx, cy, r);
  grd.addColorStop(0, pal.text2); grd.addColorStop(1, pal.line2);
  g.fillStyle = o.ghost ? pal.line2 : grd;
  g.beginPath(); g.arc(cx, cy, r, 0, 2 * Math.PI); g.fill();
  g.strokeStyle = pal.text3; g.lineWidth = 1.2; g.stroke();
  g.strokeStyle = pal.bg; g.lineWidth = 2;
  for (let k = 0; k < 3; k++) { const a = roll + (k * 2 * Math.PI) / 3; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * r * 0.92, cy + Math.sin(a) * r * 0.92); g.stroke(); }
  // body (rotates about the ball centre)
  g.translate(cx, cy); g.rotate(theta);
  const col = o.color || pal.model;
  const L = (o.l || 0.13) * S;
  const bodyTop = Math.max(0.46 * S, L * 2.2), bodyW = 0.22 * S;
  g.fillStyle = pal.bg3; g.strokeStyle = col; g.lineWidth = 2;
  roundRect(g, -bodyW / 2, -bodyTop, bodyW, bodyTop - r * 0.95, 8); g.fill(); g.stroke();
  // wheels touching the ball
  g.fillStyle = col;
  [-0.78, 0.78].forEach((a) => { g.save(); g.rotate(a); g.fillRect(-0.025 * S, -r - 0.022 * S, 0.05 * S, 0.02 * S); g.restore(); });
  // head with sensor
  g.fillStyle = col; roundRect(g, -bodyW * 0.32, -bodyTop - 0.035 * S, bodyW * 0.64, 0.04 * S, 4); g.fill();
  if (o.comDot !== false) {
    g.fillStyle = pal.warn; g.beginPath(); g.arc(0, -L, Math.max(4, 0.016 * S), 0, 2 * Math.PI); g.fill();
    g.strokeStyle = pal.bg; g.lineWidth = 1.5; g.stroke();
  }
  g.restore();
  if (o.push) {
    const ax = cx + Math.sin(theta) * (o.l || 0.13) * S, ay = cy - Math.cos(theta) * (o.l || 0.13) * S;
    const len = Math.sign(o.push) * Math.min(90, 18 + Math.abs(o.push) * 2.2);
    arrow(g, ax - len - Math.sign(o.push) * 12, ay, ax - Math.sign(o.push) * 12, ay, pal.bad, 3);
  }
}
export function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
export function arrow(g, x0, y0, x1, y1, color, w = 2) {
  const a = Math.atan2(y1 - y0, x1 - x0), hd = 6 + w * 2;
  g.strokeStyle = color; g.fillStyle = color; g.lineWidth = w; g.lineCap = 'round';
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - Math.cos(a) * hd * 0.6, y1 - Math.sin(a) * hd * 0.6); g.stroke();
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - hd * Math.cos(a - 0.45), y1 - hd * Math.sin(a - 0.45)); g.lineTo(x1 - hd * Math.cos(a + 0.45), y1 - hd * Math.sin(a + 0.45)); g.closePath(); g.fill();
}
export function floor(g, w, y, pal, x0 = 0, scale = 100) {
  g.strokeStyle = pal.line2; g.lineWidth = 1.5; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  g.fillStyle = pal.text3; g.font = '10px ' + css('--font-mono');
  const step = scale >= 300 ? 0.1 : 0.5;
  const startM = Math.ceil((-x0) / scale / step) * step;
  for (let m = startM; x0 + m * scale < w; m += step) {
    const px = x0 + m * scale;
    g.strokeStyle = pal.line; g.beginPath(); g.moveTo(px, y); g.lineTo(px, y + 5); g.stroke();
    if (Math.abs(m / step) % 2 < 1e-6 || step === 0.5) g.fillText(`${+m.toFixed(1)} m`, px + 2, y + 15);
  }
}
