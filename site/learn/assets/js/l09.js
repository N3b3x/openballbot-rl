// Lesson 9 figures: the goal point that swung around the ball (S118), the exposure artefact (S103), terrain tiles and the zigzag (S113), and seed spread.
import { el, canvas, chart, slider, seg, button, readouts, legend, palette, fmt, arrow, floor } from './ui.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ------------------------------------------------------------------------------------------------ S118: the measured point
export function swing(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const r = 0.09, off = ctx.data.facts.find((f) => f.what === 'ball body origin offset').values.offset_m;
  let s = 0.3;
  const cv = canvas(stage, (w) => Math.min(250, Math.max(170, w * 0.32)), (g, w, h) => {
    const pal = palette(); g.clearRect(0, 0, w, h);
    const sc = Math.min(w / 0.8, 520), fy = h - 30;
    floor(g, w, fy, pal, w / 2 - s * sc, sc);
    const phi = s / r, cx = w / 2, cy = fy - r * sc;
    // ball
    g.fillStyle = pal.bg3; g.strokeStyle = pal.text3; g.lineWidth = 1.5; g.beginPath(); g.arc(cx, cy, r * sc, 0, 6.3); g.fill(); g.stroke();
    // the ball body origin: starts 0.14 m above the centre, rotates with the ball
    const ox2 = off * Math.sin(phi), oz = off * Math.cos(phi);
    g.strokeStyle = pal.warn; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + ox2 * sc, cy - oz * sc); g.stroke(); g.setLineDash([]);
    g.strokeStyle = pal.bg2; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx - Math.sin(phi) * r * sc, cy + Math.cos(phi) * r * sc); g.strokeStyle = pal.line2; g.stroke();
    g.fillStyle = pal.hybrid; g.beginPath(); g.arc(cx, cy, 5, 0, 6.3); g.fill();
    g.fillStyle = pal.bad; g.beginPath(); g.arc(cx + ox2 * sc, cy - oz * sc, 5, 0, 6.3); g.fill();
    // goal marker held by the old logic: the origin sits on the goal, so the true centre is displaced
    g.fillStyle = pal.text2; g.fillText('teal: true ball centre', 8, 14); g.fillStyle = pal.bad; g.fillText('red: ball body origin (what the old goal logic held)', 8, 28);
  }, ctx);
  cv.c.setAttribute('role', 'img'); cv.c.setAttribute('aria-label', 'Side view of the ball rolling to the right. The red body origin point is 14 centimetres from the ball centre and moves around it as the ball rotates.');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Ball has rolled', min: 0, max: 1.6, step: 0.01, value: s, fmt: (v) => `${v.toFixed(2)} m (${((v / r) * 180 / Math.PI % 360).toFixed(0)} deg)`, onInput: (v) => { s = v; upd(); } });
  const ro = readouts(body, [['o', 'horizontal offset of the origin', 'bad'], ['d', 'if the origin is held on the goal, the centre is off by', 'warn']]);
  const ec = chart(body, { x: [0, 1.6], y: [-16, 16], xlabel: 'distance rolled (m)', ylabel: 'offset (cm)', height: 170, ml: 40, series: [{ data: Array.from({ length: 161 }, (_, i) => [i / 100, 100 * off * Math.sin(i / 100 / r)]), color: '--bad' }], desc: 'Horizontal offset of the ball body origin from the ball centre against distance rolled.' });
  const hold = [['origin logic (g34, three seeds)', 12.6, 11.4, 13.6], ['centre logic (g39)', 3.6]];
  body.append(el('p', { class: 'hint', text: 'Goal-test hold error, RMS, measured at the true ball centre (S118): old origin logic 12.6, 11.4 and 13.6 cm over three seeds; goal-centre logic 3.6 cm. The earlier 6 to 8 cm figures of S110 to S117 were measured at the swinging origin.' }));
  function upd() {
    const phi = s / r, o = off * Math.sin(phi);
    ro.set('o', `${(o * 100).toFixed(1)} cm`); ro.set('d', `up to ${(Math.abs(o) * 100).toFixed(1)} cm`);
    ec.update({ vlines: [{ x: s, color: '--text-3', dash: '2 3' }] }); cv.redraw();
  }
  upd();
}

// ------------------------------------------------------------------------------------------------ S103: exposure
export function exposure(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const F = ctx.data.facts;
  const first = F.find((f) => f.text.includes('0.15 to 0.17')).values;      // 15 s episodes: none vs ToF layer
  const re = F.find((f) => f.text.includes('0.38 / 0.33 / 0.35')).values;   // 40 s episodes: none / oracle / ToF
  let ep = 15;
  const sg = el('div', { class: 'btns' }); stage.append(sg);
  seg(sg, [[15, 'Episodes of 15 s (first result, S101 to S103)'], [40, 'Episodes of 40 s (the re-test)']], 15, (v) => { ep = v; draw(); }, 'episode length');
  legend(stage, [['--model', 'no speed layer'], ['--hybrid', 'speed layer (slows before features)']]);
  const ch = chart(stage, { x: [-0.8, 0.8], y: [0, 0.5], xticks: [0], xfmt: () => 'obstacle falls, velocity task 0.3 m/s', ylabel: 'fall fraction', height: 220, ml: 40, desc: 'Obstacle falls with and without the speed layer for two episode lengths.' });
  const note = el('p', { class: 'status' }); body.append(note);
  // toy: how many robots have met the obstacle by time T
  body.append(el('h3', { style: { padding: '0 18px', margin: '.8em 0 0', fontSize: '1rem' }, text: 'A toy that shows the mechanism' }));
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  const S = { T: 15, slow: 0.12, fast: 0.3, p: 0.38 };
  slider(ctl, { label: 'Episode length', min: 5, max: 60, step: 1, value: S.T, fmt: (v) => `${v} s`, onInput: (v) => { S.T = v; toy(); } });
  slider(ctl, { label: 'Slowed robot speed near features', min: 0.05, max: 0.3, step: 0.01, value: S.slow, fmt: (v) => `${v.toFixed(2)} m/s`, onInput: (v) => { S.slow = v; toy(); } });
  const tc = chart(body, { x: [5, 60], y: [0, 0.5], xlabel: 'episode length (s)', ylabel: 'fall fraction of all robots', height: 190, ml: 40, crosshair: true, desc: 'Toy: counted falls against episode length for a fast and a slowed robot with the same per-obstacle risk.' });
  legend(body, [['--model', 'fast (0.3 m/s)'], ['--hybrid', 'slowed']]);
  body.append(el('p', { class: 'hint', text: 'Toy model, not data: obstacles are 0.6 to 2.4 m away (uniform) and every robot that reaches one falls with the same probability 0.38 whatever its speed. The only difference is when it gets there.' }));
  function toy() {
    const fr = (v, T) => { let n = 0, tot = 0; for (let d = 0.6; d <= 2.4; d += 0.01) { tot++; if (d / v <= T) n++; } return n / tot; };
    const pts = (v) => Array.from({ length: 56 }, (_, i) => [5 + i, S.p * fr(v, 5 + i)]);
    tc.update({ series: [{ data: pts(S.fast), color: '--model', label: 'fast' }, { data: pts(S.slow), color: '--hybrid', label: 'slowed' }], vlines: [{ x: S.T, color: '--text-3', dash: '2 3' }] });
  }
  function draw() {
    const none = 0.38;
    const layer = ep === 15 ? first.tof : re.tof;
    const exits = ep === 15 ? [0.62, 0.54] : [0.62, 0.54];
    ch.update({ bars: [
      { x: 0, dx: -16, w: 28, y: none, color: '--model', label: `no layer: ${none}` }, { x: 0, dx: 16, w: 28, y: layer, color: '--hybrid', label: `speed layer: ${layer}` }] });
    note.className = 'status ' + (ep === 15 ? 'warn' : 'good');
    note.textContent = ep === 15 ? 'In 15 s episodes the speed layer seems to halve obstacle falls (0.38 to 0.15 to 0.17), but the slowed robots had not yet reached the obstacles.' : 'With 40 s every robot reaches its obstacle: the layer is worth 0.03 to 0.05 (0.38 against 0.33 with a perfect map and 0.35 with the ToF sensor). The earlier claim was withdrawn (S103).';
  }
  draw(); toy();
}

// ------------------------------------------------------------------------------------------------ S113: tiles, the zigzag and the renders
export function tiles(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  let flat = false;
  const sg = el('div', { class: 'btns' }); stage.append(sg);
  seg(sg, [[false, 'Standard terrain tiles'], [true, 'Fully flat map']], false, (v) => { flat = v; draw(); }, 'map');
  const D = [['zigzag (60 deg every 2 s)', 0.32, 0.051], ['goal test', 0.104, 0.093], ['curve (0.5 m circle)', 0.04, null]];
  const ch = chart(stage, { x: [-0.6, 2.6], y: [0, 0.45], xticks: [0, 1, 2], xfmt: (v) => (D[v] ? D[v][0] : ''), ylabel: 'fall fraction', height: 230, ml: 40, desc: 'Falls in three motion tests on terrain tiles and on a flat map.' });
  const note = el('p', { class: 'status' }); body.append(note);
  function draw() {
    ch.update({ bars: D.map((d, i) => ({ x: i, w: 46, y: flat ? (d[2] == null ? 0 : d[2]) : d[1], color: flat ? '--good' : '--bad', label: `${d[0]}: ${flat ? (d[2] == null ? 'not measured on the flat map' : d[2]) : d[1]}` })) });
    note.className = 'status ' + (flat ? 'good' : 'warn');
    note.textContent = flat ? 'On a fully flat map the zigzag falls 0.051: turning sharply is not the weakness. All bars are g32 s1 (S113). The curve was not repeated on the flat map: it stays on its start tile.' : 'On tiles the zigzag costs about a third of the robots, but only because its net forward travel (about 2 m) carries the robot onto neighbouring terrain. The curve stays on its start tile and falls 0.03 to 0.05.';
  }
  draw();
  // render clips: height profiles from the S113 note
  body.append(el('h3', { style: { padding: '0 18px', margin: '.8em 0 0', fontSize: '1rem' }, text: 'Why the 8 s showcase clips fell: the next feature' }));
  const prof = {
    'step up 3 cm': [[0, 0], [0.649, 0], [0.65, 0.03], [3, 0.03]],
    'step down, drop, step up': [[0, 0], [0.549, 0], [0.55, -0.026], [1.549, -0.026], [1.55, -0.051], [2.349, -0.051], [2.35, -0.001], [3, -0.001]],
    '11 deg ramp, then drop': [[0, 0], [0.55, 0], [1.45, 0.18], [1.451, -0.15], [3, -0.15]],
  };
  const keys = Object.keys(prof); let k = 0, dist = 1.2;
  const sg2 = el('div', { class: 'btns' }); body.append(sg2);
  seg(sg2, keys.map((n, i) => [i, n]), 0, (i) => { k = i; draw2(); }, 'tile feature');
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Distance travelled from the tile centre', min: 0, max: 3, step: 0.05, value: dist, fmt: (v) => `${v.toFixed(2)} m`, onInput: (v) => { dist = v; draw2(); } });
  const pc = chart(body, { x: [0, 3], y: [-0.2, 0.25], xlabel: 'distance along the clip path (m)', ylabel: 'ground height (m)', height: 180, ml: 44, desc: 'Ground height profile along a clip path for three tile types.' });
  body.append(el('p', { class: 'hint', text: 'Heights read from the map along the clip paths (S113 note): each tile has a flat pad at its centre; the first feature starts 0.55 to 0.65 m out and further features follow. The first explanation, "the robot drives off the map edge", was wrong: the map spans 10.5 m each way.' }));
  function draw2() { pc.update({ series: [{ data: prof[keys[k]], color: '--warn', width: 2.4 }], vlines: [{ x: dist, color: '--hybrid', dash: '', label: 'robot' }] }); }
  draw2();
}

// ------------------------------------------------------------------------------------------------ seeds
export function spread(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const gr = ctx.data.spread.groups;
  let a = 0, b = 2;
  const sa = el('div', { class: 'btns', style: { paddingBottom: 0 } }); stage.append(sa);
  const sb = el('div', { class: 'btns', style: { paddingTop: 0 } }); stage.append(sb);
  const lab = (t) => el('span', { style: { alignSelf: 'center', fontSize: '.85rem', color: 'var(--text-3)' }, text: t });
  sa.append(lab('Recipe A'));
  seg(sa, gr.map((g, i) => [i, g.name]), a, (i) => { a = i; draw(); }, 'recipe A');
  sb.append(lab('Recipe B'));
  seg(sb, gr.map((g, i) => [i, g.name]), b, (i) => { b = i; draw(); }, 'recipe B');
  const ch = chart(stage, { x: [-0.6, gr.length - 0.4], y: [0.7, 0.85], xticks: gr.map((_, i) => i), xfmt: (v) => (gr[v] ? gr[v].name : ''), ylabel: 'goal success, battery, DR 1, no pushes', height: 240, ml: 50, desc: 'Goal success for each training seed of three recipes.' });
  const ro = readouts(body, [['m', 'mean A / mean B', ''], ['d', 'difference of means', 'warn'], ['s', 'spread within A / B', ''], ['v', 'verdict', 'hybrid']]);
  const mean = (v) => v.reduce((x, y) => x + y, 0) / v.length;
  function draw() {
    const A = gr[a].values, B = gr[b].values;
    ch.update({
      dots: gr.flatMap((g, i) => g.values.map((v, j) => ({ x: i + (j - (g.values.length - 1) / 2) * 0.08, y: v, color: i === a || i === b ? '--hybrid' : '--text-3', r: 5, label: `${g.name}, seed ${j}: ${v}` }))),
      series: [{ data: [[a, Math.min(...A)], [a, Math.max(...A)]], color: '--hybrid', width: 3 }, { data: [[b, Math.min(...B)], [b, Math.max(...B)]], color: '--hybrid', width: 3 }],
    });
    const d = Math.abs(mean(A) - mean(B)), sa2 = Math.max(...A) - Math.min(...A), sb2 = Math.max(...B) - Math.min(...B);
    ro.set('m', `${mean(A).toFixed(3)} / ${mean(B).toFixed(3)}`); ro.set('d', d.toFixed(3)); ro.set('s', `${sa2.toFixed(2)} / ${sb2.toFixed(2)}`);
    ro.set('v', a === b ? 'same recipe' : d > Math.max(sa2, sb2) ? 'larger than the spread' : 'inside the seed spread: not a result');
  }
  draw();
}
