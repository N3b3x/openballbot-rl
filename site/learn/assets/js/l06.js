// Lesson 6 figures: the simulated ToF depth sensor, the wavefront planner with the carrot follower, the contact-escape fallback, and outcomes.
import { el, canvas, chart, slider, seg, button, readouts, legend, clock, palette, drawBot, fmt, arrow } from './ui.js';

const DEG = Math.PI / 180;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ------------------------------------------------------------------------------------------------ ToF sensor
// Sensor of train_dev.py / sensor_view.py: 18 x 14 zones, 55 x 42 degree field of view, pitched down 25 degrees, 0.05 to 4 m range,
// range noise 0.005 d, 2 percent dropout that reads as maximum range, mounted 0.15 m above the base origin (0.24 m above the floor at rest).
const SEN = { w: 18, h: 14, fovH: 55 * DEG, fovV: 42 * DEG, rmin: 0.05, rmax: 4, noise: 0.005, drop: 0.02, up: 0.15, baseZ: 0.24 };

function mulberry(seed) { let a = seed >>> 0; return () => { a += 0x6d2b79f5; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

function scan(box, pitch, noisy, hs) {
  const rnd = mulberry(7);
  const zones = [];
  for (let r = 0; r < SEN.h; r++) for (let c = 0; c < SEN.w; c++) {
    const az = -SEN.fovH / 2 + ((c + 0.5) / SEN.w) * SEN.fovH;
    const el = pitch + SEN.fovV / 2 - ((r + 0.5) / SEN.h) * SEN.fovV;
    const d = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    let t = Infinity, kind = 'none';
    if (d[2] < -1e-6) { const tf = -hs / d[2]; if (tf > 0) { t = tf; kind = 'floor'; } }
    if (box.h > 0) {
      let lo = 0, hi = Infinity; const o = [0, 0, hs];
      const bmin = [box.x, -box.w / 2, 0], bmax = [box.x + box.d, box.w / 2, box.h];
      let ok = true;
      for (let k = 0; k < 3; k++) {
        if (Math.abs(d[k]) < 1e-9) { if (o[k] < bmin[k] || o[k] > bmax[k]) ok = false; } else {
          let t1 = (bmin[k] - o[k]) / d[k], t2 = (bmax[k] - o[k]) / d[k]; if (t1 > t2) [t1, t2] = [t2, t1];
          lo = Math.max(lo, t1); hi = Math.min(hi, t2);
        }
      }
      if (ok && lo <= hi && lo > 0 && lo < t) { t = lo; kind = 'block'; }
    }
    let range = t;
    if (noisy && Number.isFinite(range)) range += gauss(rnd) * SEN.noise * range;
    const drop = noisy && rnd() < SEN.drop;
    if (!Number.isFinite(range) || range > SEN.rmax || range < SEN.rmin || drop) { zones.push({ r, c, range: SEN.rmax, hit: null, kind: 'none', d }); continue; }
    const p = [d[0] * range, d[1] * range, hs + d[2] * range];
    zones.push({ r, c, range, hit: p, kind, d });
  }
  return zones;
}

export function tof(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const hs = SEN.baseZ + SEN.up;
  const S = { x: 0.7, h: 0.08, w: 0.4, pitch: -25, noisy: true };
  const side = el('div'); const dep = el('div');
  stage.append(el('div', { class: 'split', style: { padding: '0 12px', alignItems: 'center' } }, side, dep));
  let zones = [];
  const cs = canvas(side, (w) => Math.min(260, Math.max(180, w * 0.55)), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const sc = Math.min((w - 30) / 2.2, (h - 30) / 0.75), fy = h - 22, ox = 36;
    g.strokeStyle = pal.line2; g.beginPath(); g.moveTo(0, fy); g.lineTo(w, fy); g.stroke();
    g.fillStyle = pal.text3; g.font = '10px ' + getComputedStyle(document.body).fontFamily;
    for (let m = 0; m <= 2; m += 0.5) { g.fillText(`${m} m`, ox + m * sc - 8, fy + 14); g.beginPath(); g.moveTo(ox + m * sc, fy); g.lineTo(ox + m * sc, fy + 4); g.stroke(); }
    drawBot(g, 0, 0, { pal, scale: sc, ox, oy: fy, l: 0.13, color: pal.model });
    // block
    if (S.h > 0) { g.fillStyle = pal.warn; g.globalAlpha = 0.85; g.fillRect(ox + S.x * sc, fy - S.h * sc, 0.3 * sc, S.h * sc); g.globalAlpha = 1; }
    // central column rays
    const sx = ox, sy = fy - hs * sc;
    g.fillStyle = pal.hybrid; g.beginPath(); g.arc(sx, sy, 3.5, 0, 6.3); g.fill();
    zones.filter((z) => z.c === 8 || z.c === 9).forEach((z) => {
      if (z.c === 9) return;
      const t = z.hit ? z.range : 2.4;
      g.strokeStyle = z.kind === 'block' ? pal.warn : pal.hybrid; g.globalAlpha = z.kind === 'block' ? 0.9 : 0.3; g.lineWidth = 1;
      g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx + z.d[0] / Math.hypot(z.d[0], z.d[1]) * Math.hypot(z.d[0], z.d[1]) * t * sc, sy - z.d[2] * t * sc); g.stroke();
      g.globalAlpha = 1;
      if (z.hit) { g.fillStyle = z.kind === 'block' ? pal.warn : pal.text2; g.beginPath(); g.arc(sx + z.hit[0] * sc, fy - z.hit[2] * sc, 2.6, 0, 6.3); g.fill(); }
    });
    g.fillStyle = pal.text3; g.fillText('side view, centre column of zones', 8, 14);
  }, ctx);
  const cd = canvas(dep, (w) => Math.min(300, w * (SEN.h / SEN.w) + 18), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const cw = Math.min(26, (w - 4) / SEN.w), chh = Math.min(cw, (h - 18) / SEN.h);
    const ox = (w - cw * SEN.w) / 2;
    const hex = (c) => { const m = /^#([0-9a-f]{6})$/i.exec(c.trim()); return m ? [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)] : [128, 128, 128]; };
    const A = hex(pal.hybrid), Bc = hex(pal.bg3);
    zones.forEach((z) => {
      const u = clamp(z.range / 2.5, 0, 1);
      g.fillStyle = `rgb(${A.map((a, i) => Math.round(a * (1 - u) + Bc[i] * u)).join(',')})`;
      g.fillRect(ox + z.c * cw + 0.5, 16 + z.r * chh + 0.5, cw - 1, chh - 1);
      if (z.hit && z.hit[2] > 0.03) { g.strokeStyle = pal.warn; g.lineWidth = 2; g.strokeRect(ox + z.c * cw + 1.5, 16 + z.r * chh + 1.5, cw - 3, chh - 3); }
    });
    g.fillStyle = pal.text3; g.font = '10px ' + getComputedStyle(document.body).fontFamily;
    g.fillText('what the sensor reads (near = teal)', 4, 11);
  }, ctx);
  cd.c.setAttribute('role', 'img'); cd.c.setAttribute('aria-label', 'Depth image of 18 by 14 zones; zones that hit something taller than 3 cm are outlined.');
  cs.c.setAttribute('role', 'img'); cs.c.setAttribute('aria-label', 'Side view of the robot with the sensor beams hitting the floor and a block.');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const upd = () => { zones = scan({ x: S.x, d: 0.3, w: S.w, h: S.h }, S.pitch * DEG, S.noisy, hs); cs.redraw(); cd.redraw(); readout(); };
  slider(ctl, { label: 'Block distance', min: 0.3, max: 2.5, step: 0.05, value: S.x, fmt: (v) => `${v.toFixed(2)} m`, onInput: (v) => { S.x = v; upd(); } });
  slider(ctl, { label: 'Block height', min: 0, max: 0.15, step: 0.005, value: S.h, fmt: (v) => `${(v * 100).toFixed(1)} cm`, onInput: (v) => { S.h = v; upd(); } });
  slider(ctl, { label: 'Block width', min: 0.05, max: 1, step: 0.05, value: S.w, fmt: (v) => `${v.toFixed(2)} m`, onInput: (v) => { S.w = v; upd(); } });
  slider(ctl, { label: 'Sensor pitch', min: -45, max: 0, step: 1, value: S.pitch, fmt: (v) => `${v} deg`, onInput: (v) => { S.pitch = v; upd(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  seg(bt, [[true, 'Noise and dropout on'], [false, 'Perfect ranges']], true, (v) => { S.noisy = v; upd(); }, 'noise');
  const ro = readouts(body, [['n', 'zones above 3 cm', 'warn'], ['near', 'nearest obstacle zone', 'hybrid'], ['floor', 'floor zones seen', ''], ['gap', 'closest floor row', '']]);
  const msg = el('p', { class: 'status' }); body.append(msg);
  function readout() {
    const ob = zones.filter((z) => z.hit && z.hit[2] > 0.03);
    ro.set('n', `${ob.length} of ${zones.length}`);
    ro.set('near', ob.length ? `${Math.min(...ob.map((z) => z.hit[0])).toFixed(2)} m ahead` : 'none');
    const fl = zones.filter((z) => z.kind === 'floor');
    ro.set('floor', `${fl.length}`);
    ro.set('gap', fl.length ? `${Math.min(...fl.map((z) => z.hit[0])).toFixed(2)} m ahead` : 'none');
    msg.className = 'status ' + (ob.length ? 'warn' : '');
    msg.textContent = S.h > 0 && !ob.length ? `A ${(S.h * 100).toFixed(1)} cm block is ${S.h <= 0.03 ? 'below the 3 cm planner threshold' : 'out of the field of view or past the beams'}: nothing is marked.` : ob.length ? 'Zones above 3 cm become obstacle cells in the planner (the study uses 3 cm).' : '';
  }
  upd();
}

// ------------------------------------------------------------------------------------------------ planner
const GW = 30, GH = 18, CS = 0.1;      // 3.0 x 1.8 m, 10 cm cells
const NB = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const INF = 1e9;

function inflate(occ, k) {
  if (k <= 0) return occ.slice();
  const out = new Uint8Array(GW * GH);
  for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
    let o = 0;
    for (let dy = -k; dy <= k && !o; dy++) for (let dx = -k; dx <= k; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && xx < GW && yy >= 0 && yy < GH && occ[yy * GW + xx]) { o = 1; break; } }
    out[y * GW + x] = o;
  }
  return out;
}
function wavefront(infl, goal) {
  const d = new Float32Array(GW * GH).fill(INF);
  const q = [goal[1] * GW + goal[0]]; d[q[0]] = 0;
  for (let i = 0; i < q.length; i++) {
    const c = q[i], cx = c % GW, cy = (c / GW) | 0;
    for (const [dx, dy] of NB) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= GW || y >= GH) continue;
      const n = y * GW + x;
      if (infl[n] || d[n] < INF) continue;
      d[n] = d[c] + 1; q.push(n);
    }
  }
  return d;
}

export function planner(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const S = { infl: 2, mode: 'oracle', smooth: true, tool: 'draw', field: true, carrot: 3 };
  let truth = new Uint8Array(GW * GH), known = new Uint8Array(GW * GH);
  let start = [2, 9], goal = [27, 9];
  const preset = (n) => {
    truth = new Uint8Array(GW * GH);
    const set = (x, y) => { truth[y * GW + x] = 1; };
    if (n === 'wall') { for (let y = 0; y < GH; y++) if (y < 5 || y > 11) { set(14, y); set(15, y); } }
    if (n === 'block') { for (let y = 7; y <= 11; y++) for (let x = 13; x <= 15; x++) set(x, y); }
    if (n === 'maze') { for (let y = 0; y < 13; y++) { set(9, y); } for (let y = 5; y < GH; y++) { set(18, y); } for (let y = 0; y < 12; y++) set(24, y); }
    reset();
  };
  let sim = null, running = false, cv = null, ro = null, cur = [GW >> 1, GH >> 1];
  function reset() {
    known = S.mode === 'oracle' ? truth.slice() : new Uint8Array(GW * GH);
    sim = { p: [(start[0] + 0.5) * CS, (start[1] + 0.5) * CS], h: 0, v: 0, t: 0, path: [], head: [], blocked: 0, back: 0, fails: 0, bumps: 0, done: '', field: null, dirty: true, len: 0, last: 0 };
    running = false; replan(); cv && cv.redraw(); stats();
  }
  const cellOf = (p) => [clamp(Math.floor(p[0] / CS), 0, GW - 1), clamp(Math.floor(p[1] / CS), 0, GH - 1)];
  function replan() {
    const infl = inflate(known, S.infl);
    infl[goal[1] * GW + goal[0]] = 0;
    sim.infl = infl; sim.field = wavefront(infl, goal); sim.dirty = false;
  }
  function target() {
    const f = sim.field, p = sim.p;
    let [cx, cy] = cellOf(p);
    if (f[cy * GW + cx] >= INF) {      // inside the inflated margin: start from the best nearby cell
      let best = null, bv = INF;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) { const x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= GW || y >= GH) continue; const v = f[y * GW + x]; if (v < bv) { bv = v; best = [x, y]; } }
      if (!best) return null; [cx, cy] = best;
    }
    const n = S.smooth ? S.carrot : 1;
    for (let i = 0; i < n; i++) {
      let bv = f[cy * GW + cx], b = null;
      for (const [dx, dy] of NB) { const x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= GW || y >= GH) continue; const v = f[y * GW + x]; if (v < bv) { bv = v; b = [x, y]; } }
      if (!b) break; [cx, cy] = b;
    }
    return [(cx + 0.5) * CS, (cy + 0.5) * CS];
  }
  const wrapA = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  function reveal() {
    if (S.mode !== 'sensor') return;
    const [px, py] = sim.p; let ch = false;
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      if (!truth[y * GW + x] || known[y * GW + x]) continue;
      const dx = (x + 0.5) * CS - px, dy = (y + 0.5) * CS - py, d = Math.hypot(dx, dy);
      if (d > 1.2 || d < 0.05) continue;
      if (Math.abs(wrapA(Math.atan2(dy, dx) - sim.h)) < 30 * DEG) { known[y * GW + x] = 1; ch = true; }
    }
    if (ch) sim.dirty = true;
  }
  function tick(dt) {
    if (sim.done) return;
    sim.t += dt;
    const goalD = Math.hypot((goal[0] + 0.5) * CS - sim.p[0], (goal[1] + 0.5) * CS - sim.p[1]);
    if (goalD < 0.2) { sim.done = 'reached'; return; }
    if (sim.t > 90) { sim.done = 'timeout'; return; }
    reveal();
    if (sim.dirty || ((sim.t * 2) | 0) !== sim.last) { sim.last = (sim.t * 2) | 0; replan(); }
    if (sim.back > 0) { sim.back -= dt; const nx = sim.p[0] - Math.cos(sim.h) * 0.15 * dt, ny = sim.p[1] - Math.sin(sim.h) * 0.15 * dt; sim.p = [clamp(nx, 0.05, GW * CS - 0.05), clamp(ny, 0.05, GH * CS - 0.05)]; sim.path.push(sim.p.slice()); if (sim.back <= 0) { replan(); } return; }
    const tg = target();
    if (!tg) { sim.done = 'no path'; return; }
    const near = goalD < 0.35;
    const to = near ? [(goal[0] + 0.5) * CS - sim.p[0], (goal[1] + 0.5) * CS - sim.p[1]] : [tg[0] - sim.p[0], tg[1] - sim.p[1]];
    const ht = Math.atan2(to[1], to[0]);
    const h0 = sim.h;
    if (S.smooth) { const wl = 2.0 * dt; sim.h = wrapA(sim.h + clamp(wrapA(ht - sim.h), -wl, wl)); } else sim.h = ht;
    const err = Math.abs(wrapA(ht - sim.h));
    const vmax = S.smooth ? Math.min(0.3, 1.5 * goalD) * Math.max(0.25, Math.cos(Math.min(err, 1.4))) : 0.3;
    sim.v = S.smooth ? sim.v + clamp(vmax - sim.v, -0.6 * dt, 0.6 * dt) : vmax;
    sim.head.push(wrapA(sim.h - h0) / dt);
    const nx = sim.p[0] + Math.cos(sim.h) * sim.v * dt, ny = sim.p[1] + Math.sin(sim.h) * sim.v * dt;
    const rad = 0.08, hit = [[nx + rad, ny], [nx - rad, ny], [nx, ny + rad], [nx, ny - rad], [nx, ny]].some(([x, y]) => { const c = cellOf([x, y]); return x < 0 || y < 0 || x > GW * CS || y > GH * CS || truth[c[1] * GW + c[0]]; });
    if (hit) {
      sim.blocked += dt; sim.v = 0;
      if (sim.blocked >= 1) {   // stall detected: mark a 0.3 m wide segment 0.2 m ahead as an obstacle, back up, re-plan (S109)
        sim.blocked = 0; sim.bumps++; sim.fails++;
        for (let l = -0.15; l <= 0.151; l += 0.05) { const px = sim.p[0] + Math.cos(sim.h) * 0.2 - Math.sin(sim.h) * l, py = sim.p[1] + Math.sin(sim.h) * 0.2 + Math.cos(sim.h) * l; const c = cellOf([px, py]); known[c[1] * GW + c[0]] = 1; }
        sim.back = 1; sim.dirty = true;
        if (sim.fails > 6) sim.done = 'gave up';
      }
    } else { sim.blocked = 0; sim.len += Math.hypot(nx - sim.p[0], ny - sim.p[1]); sim.p = [nx, ny]; sim.path.push([nx, ny]); }
  }
  // drawing
  preset('block');
  cv = canvas(stage, (w) => w * (GH / GW) + 8, (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const c = (w - 8) / GW, ox = 4, oy = 4;
    const f = sim.field;
    let mx = 1; if (S.field && f) for (let i = 0; i < f.length; i++) if (f[i] < INF) mx = Math.max(mx, f[i]);
    g.globalAlpha = 1;
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      const i = y * GW + x, X = ox + x * c, Y = oy + (GH - 1 - y) * c;
      g.fillStyle = pal.bg3; g.fillRect(X, Y, c - 0.5, c - 0.5);
      if (S.field && f && f[i] < INF) { g.fillStyle = pal.hybrid; g.globalAlpha = 0.07 + 0.38 * (1 - f[i] / mx); g.fillRect(X, Y, c - 0.5, c - 0.5); g.globalAlpha = 1; }
      if (sim.infl && sim.infl[i] && !known[i]) { g.fillStyle = pal.warn; g.globalAlpha = 0.22; g.fillRect(X, Y, c - 0.5, c - 0.5); g.globalAlpha = 1; }
      else if (sim.infl && sim.infl[i]) { g.fillStyle = pal.warn; g.globalAlpha = 0.22; g.fillRect(X, Y, c - 0.5, c - 0.5); g.globalAlpha = 1; }
      if (truth[i]) { g.fillStyle = known[i] ? pal.text2 : pal.line2; g.fillRect(X, Y, c - 0.5, c - 0.5); if (!known[i] && S.mode !== 'oracle') { g.strokeStyle = pal.warn; g.setLineDash([3, 3]); g.strokeRect(X + 1, Y + 1, c - 2.5, c - 2.5); g.setLineDash([]); } }
      else if (known[i]) { g.strokeStyle = pal.bad; g.strokeRect(X + 1, Y + 1, c - 2.5, c - 2.5); }
    }
    const P = (p) => [ox + (p[0] / CS) * c, oy + (GH - p[1] / CS) * c];
    g.strokeStyle = pal.hybrid; g.lineWidth = 2.4; g.lineJoin = 'round'; g.beginPath();
    sim.path.forEach((p, i) => { const q = P(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.stroke();
    const gp = P([(goal[0] + 0.5) * CS, (goal[1] + 0.5) * CS]);
    g.strokeStyle = pal.good; g.lineWidth = 2; g.beginPath(); g.arc(gp[0], gp[1], 0.2 / CS * c, 0, 6.3); g.stroke(); g.fillStyle = pal.good; g.beginPath(); g.arc(gp[0], gp[1], 3.5, 0, 6.3); g.fill();
    const sp = P([(start[0] + 0.5) * CS, (start[1] + 0.5) * CS]); g.fillStyle = pal.text3; g.beginPath(); g.arc(sp[0], sp[1], 3.5, 0, 6.3); g.fill();
    const rp = P(sim.p); g.fillStyle = pal.model; g.beginPath(); g.arc(rp[0], rp[1], 0.11 / CS * c, 0, 6.3); g.fill(); g.strokeStyle = pal.bg; g.lineWidth = 2; g.stroke();
    arrow(g, rp[0], rp[1], rp[0] + Math.cos(sim.h) * 0.2 / CS * c, rp[1] - Math.sin(sim.h) * 0.2 / CS * c, pal.bg, 2);
    if (S.mode === 'sensor') { g.strokeStyle = pal.hybrid; g.globalAlpha = 0.5; g.lineWidth = 1; g.beginPath(); g.moveTo(rp[0], rp[1]); for (const s of [-1, 1]) g.lineTo(rp[0] + Math.cos(sim.h + s * 30 * DEG) * 1.2 / CS * c, rp[1] - Math.sin(sim.h + s * 30 * DEG) * 1.2 / CS * c); g.stroke(); g.globalAlpha = 1; }
    if (cur && cv && document.activeElement === cv.c) { g.strokeStyle = pal.text; g.lineWidth = 2; g.strokeRect(ox + cur[0] * c + 1, oy + (GH - 1 - cur[1]) * c + 1, c - 2.5, c - 2.5); }
  }, ctx);
  cv.c.tabIndex = 0;
  cv.c.setAttribute('role', 'application');
  cv.c.setAttribute('aria-label', 'Planner grid. Click or drag to draw or erase obstacles; with the keyboard use the arrow keys to move the cursor and space to toggle a cell.');
  const cellAt = (ev) => { const r = cv.c.getBoundingClientRect(), c = r.width / GW; return [clamp(Math.floor((ev.clientX - r.left) / c), 0, GW - 1), clamp(GH - 1 - Math.floor((ev.clientY - r.top - 4) / c), 0, GH - 1)]; };
  let paint = null;
  const apply = (cell) => {
    if (S.tool === 'start') { start = cell; truth[cell[1] * GW + cell[0]] = 0; reset(); return; }
    if (S.tool === 'goal') { goal = cell; truth[cell[1] * GW + cell[0]] = 0; reset(); return; }
    const i = cell[1] * GW + cell[0];
    if (paint === null) paint = S.tool === 'erase' ? 0 : truth[i] ? 0 : 1;
    if ((cell[0] === start[0] && cell[1] === start[1]) || (cell[0] === goal[0] && cell[1] === goal[1])) return;
    if (truth[i] !== paint) { truth[i] = paint; if (S.mode === 'oracle') known[i] = paint; sim.dirty = true; if (!running && !sim.path.length) { reset(); } else { replanNow(); } }
  };
  function replanNow() { replan(); cv.redraw(); stats(); }
  cv.c.classList.add('draw');
  cv.c.parentNode.append(el('p', { class: 'hint', style: { margin: '4px 0 0' }, text: 'On a touch screen, dragging inside the grid draws; scroll the page by dragging outside it.' }));
  cv.c.addEventListener('pointerdown', (e) => { cv.c.focus(); cv.c.setPointerCapture(e.pointerId); paint = null; apply(cellAt(e)); cur = cellAt(e); });
  cv.c.addEventListener('pointermove', (e) => { if (e.buttons) { apply(cellAt(e)); } });
  cv.c.addEventListener('pointerup', () => { paint = null; });
  cv.c.addEventListener('keydown', (e) => {
    const m = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (m) { cur = [clamp(cur[0] + m[0], 0, GW - 1), clamp(cur[1] + m[1], 0, GH - 1)]; e.preventDefault(); cv.redraw(); }
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); paint = null; apply(cur); paint = null; }
  });
  cv.c.addEventListener('focus', () => cv.redraw()); cv.c.addEventListener('blur', () => cv.redraw());

  const sg = el('div', { class: 'btns' }); body.append(sg);
  seg(sg, [['oracle', 'Perfect map'], ['sensor', 'ToF-like sensor'], ['blind', 'Blind + bump escape']], S.mode, (v) => { S.mode = v; reset(); }, 'what the planner knows');
  const sg2 = el('div', { class: 'btns' }); body.append(sg2);
  seg(sg2, [['draw', 'Draw walls'], ['erase', 'Erase'], ['start', 'Move start'], ['goal', 'Move goal']], S.tool, (v) => { S.tool = v; }, 'tool');
  seg(sg2, [[true, 'Smooth follower'], [false, 'Jerky follower']], true, (v) => { S.smooth = v; reset(); }, 'follower');
  const sg3 = el('div', { class: 'btns' }); body.append(sg3);
  seg(sg3, [['block', 'Block ahead'], ['wall', 'Wall with a gap'], ['maze', 'Zigzag walls'], ['clear', 'Clear']], '', (v) => preset(v), 'preset');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Obstacle inflation (cells of 10 cm; the study uses about 0.22 m)', min: 0, max: 4, step: 1, value: S.infl, fmt: (v) => `${v} (${(v * 10)} cm)`, onInput: (v) => { S.infl = v; reset(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  button(bt, 'Run', () => { if (sim.done) reset(); if (ctx.reduced) { for (let i = 0; i < 1800 && !sim.done; i++) tick(0.05); cv.redraw(); stats(); } else { running = true; clk.start(); } }, 'primary');
  button(bt, 'Reset', () => reset());
  const wb = el('label', { style: { display: 'inline-flex', gap: '.5em', alignItems: 'center', fontSize: '.88rem', color: 'var(--text-2)' } }, el('input', { type: 'checkbox', checked: true, onchange: (e) => { S.field = e.target.checked; cv.redraw(); } }), 'Show distance field');
  bt.append(wb);
  ro = readouts(body, [['st', 'state', ''], ['t', 'time', ''], ['len', 'path length', 'model'], ['bump', 'bumps', 'warn'], ['wob', 'heading-rate RMS', 'hybrid']]);
  function stats() {
    if (!ro) return;
    const rms = sim.head.length ? Math.sqrt(sim.head.reduce((s, v) => s + v * v, 0) / sim.head.length) : 0;
    ro.set('st', sim.done ? (sim.done === 'reached' ? 'reached the goal' : sim.done) : running ? 'driving' : 'ready');
    ro.set('t', `${sim.t.toFixed(1)} s`); ro.set('len', `${sim.len.toFixed(2)} m`); ro.set('bump', `${sim.bumps}`); ro.set('wob', `${rms.toFixed(2)} rad/s`);
  }
  legend(body, [['--hybrid', 'path'], ['--warn', 'inflated margin'], ['--text-2', 'known obstacle'], ['--line-2', 'hidden obstacle (dashed outline)'], ['--bad', 'bump mark']]);
  const clk = clock(stage, (dt) => { if (!running) return false; for (let i = 0; i < 4; i++) tick(0.05); cv.redraw(); stats(); if (sim.done) { running = false; return false; } return true; });
  stats();
}

// ------------------------------------------------------------------------------------------------ outcomes
export function outcomes(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const R = [
    { n: 'blind, straight', lab: 'Blind, straight at the goal (g26, S104)', r: 0.09, f: 0.91, s: 0.0 },
    { n: 'blind + escape', lab: 'Blind, with contact escape (g34, 40 s, S115)', r: 0.634, f: 0.261, s: 0.104 },
    { n: 'ToF + planner', lab: 'ToF 18 x 14 with the planner (g39, S121)', r: 0.866, f: 0.057, s: 0.077 },
    { n: 'perfect map', lab: 'Perfect map with the planner (g39, S121)', r: 0.953, f: 0.045, s: 0.001 },
  ];
  legend(stage, [['--good', 'reached the goal'], ['--bad', 'fell'], ['--warn', 'stuck']]);
  chart(stage, {
    x: [-0.6, R.length - 0.4], y: [0, 1], xticks: R.map((_, i) => i), xfmt: (v) => (R[v] ? R[v].n : ''), ylabel: 'fraction of trials', height: 250, ml: 40,
    bars: R.flatMap((r, i) => [{ x: i, dx: -18, w: 16, y: r.r, color: '--good', label: `${r.lab}: reached ${r.r}` }, { x: i, dx: 0, w: 16, y: r.f, color: '--bad', label: `${r.lab}: fell ${r.f}` }, { x: i, dx: 18, w: 16, y: r.s, color: '--warn', label: `${r.lab}: stuck ${r.s}` }]),
    desc: 'Outcomes of goal navigation behind a block for four ways of knowing the map.',
  });
  body.append(el('p', { class: 'hint', text: 'Goal 1.2 m away behind an 8 cm block, 528 simulated worlds (11 designs by 48), domain randomisation and noise on, no pushes. Different controller checkpoints and episode lengths are labelled in each bar; read the pattern, not the small differences.' }));
}
