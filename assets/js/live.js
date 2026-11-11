// Live page: trained PPO policy (JavaScript forward pass) driving the ballbot simulated by MuJoCo WebAssembly (pinned 3.13.0).
// Physics wrapper and policy live in site/live/{sim.js,policy.js}; parity numbers come from site/live/parity.json and an in-browser re-run.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Policy } from '../../live/policy.js';
import { BallbotSim, loadModel } from '../../live/sim.js';

const MUJOCO_URL = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/mujoco.js';
const DT = 0.01;                       // control period (5 x 2 ms)
const VMAX = 1.5;                      // training command range (m/s)
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, n = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(n);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const OBS_INFO = {
  orientation: 'tilt and heading (rotation vector, rad)', angular_vel: 'rotation rate (rad/s)', vel: 'base velocity, world frame (m/s)',
  motor_state: 'see the note below', actions: 'its previous output', cmd: 'velocity command (m/s, m/s, rad/s)',
  heading: 'sin and cos of the heading', vint: 'running integral of the velocity error',
};

const S = { mj: null, sim: null, policy: null, spec: null, parityStatic: null, obs: null, paused: false, speed: 1, follow: true,
  cmd: [0, 0, 0], tour: false, tourT: 0, pushLeft: 0, pushVec: [0, 0], acc: 0, hist: null, banner: '', model: null };
const resetHist = () => { S.hist = { cx: [], cy: [], cw: [], vx: [], vy: [], wz: [], a0: [], a1: [], a2: [] }; };
resetHist(); window.__ballbot = S;   // exposed for debugging in the console

function badge(id, cls, text) { const el = $(id); el.className = 'lv-badge ' + cls; el.lastElementChild.textContent = text; }
function setLoad(title, msg, hide) { $('load-title').textContent = title; $('load-msg').textContent = msg; $('loading').hidden = !!hide; }
function banner(text, kind) { const b = $('banner'); b.hidden = !text; b.textContent = text || ''; b.className = 'lv-banner ' + (kind || ''); }

// ---------------------------------------------------------------------------------------------------------------- policy view
function buildObsTable() {
  const tb = $('obs-body');
  const order = ['orientation', 'angular_vel', 'vel', 'motor_state', 'actions', 'cmd', 'heading', 'vint'].filter((k) => S.spec.obs_keys.includes(k));
  tb.innerHTML = order.map((k) => `<tr><td>${k}</td><td class="num" id="o-${k}">&nbsp;</td><td>${OBS_INFO[k] || ''}${S.spec.normalised_keys.includes(k) ? ' <span class="lv-small">(standardised)</span>' : ''}</td></tr>`).join('');
  $('act-bars').innerHTML = [0, 1, 2].map((i) => `<div class="lv-bar"><span>wheel ${i}</span><div class="track"><div class="mid"></div><div class="fill" id="ab-${i}"></div></div><span id="av-${i}" style="text-align:right"></span></div>`).join('');
}
function updateObsView(obs, act) {
  for (const k of S.spec.obs_keys) { const el = $('o-' + k); if (el) el.textContent = obs[k].map((v) => fmt(v)).join(' '); }
  for (let i = 0; i < 3; i++) {
    const f = $('ab-' + i), a = act[i];
    f.style.left = (a >= 0 ? 50 : 50 + 50 * a) + '%'; f.style.width = Math.abs(a) * 50 + '%';
    $('av-' + i).textContent = fmt(a * S.spec.action.max_wheel_velocity, 1) + ' N m';
  }
}

// ---------------------------------------------------------------------------------------------------------------- command pad
const pad = $('pad'), pctx = pad.getContext('2d');
function drawPad() {
  const W = pad.width, c = W / 2, R = c - 14, k = R / VMAX;
  pctx.clearRect(0, 0, W, W);
  pctx.strokeStyle = css('--line-2'); pctx.fillStyle = css('--text-3'); pctx.lineWidth = 2; pctx.font = '500 20px ' + css('--font-mono');
  for (const r of [0.5, 1.0, 1.5]) { pctx.beginPath(); pctx.arc(c, c, r * k, 0, 7); pctx.stroke(); pctx.fillText(r + ' m/s', c + 6, c - r * k + 18); }
  pctx.beginPath(); pctx.moveTo(c, 10); pctx.lineTo(c, W - 10); pctx.moveTo(10, c); pctx.lineTo(W - 10, c); pctx.stroke();
  pctx.fillText('+y', c - 30, 30); pctx.fillText('+x', W - 44, c - 8);
  // actual velocity (violet) and command puck
  if (S.obs) { pctx.fillStyle = css('--learned'); pctx.beginPath(); pctx.arc(c + S.obs.vel[0] * k, c - S.obs.vel[1] * k, 9, 0, 7); pctx.fill(); }
  pctx.fillStyle = css('--text'); pctx.strokeStyle = css('--bg'); pctx.lineWidth = 4;
  pctx.beginPath(); pctx.arc(c + S.cmd[0] * k, c - S.cmd[1] * k, 14, 0, 7); pctx.stroke(); pctx.fill();
}
function setCmdFromPad(ev) {
  const r = pad.getBoundingClientRect(), nx = ((ev.clientX - r.left) / r.width) * 2 - 1, ny = ((ev.clientY - r.top) / r.height) * 2 - 1;
  let x = nx * (r.width / 2 / (r.width / 2 - 7)) * VMAX, y = -ny * (r.width / 2 / (r.width / 2 - 7)) * VMAX;
  const m = Math.hypot(x, y); if (m > VMAX) { x *= VMAX / m; y *= VMAX / m; }
  S.cmd[0] = Math.round(x * 100) / 100; S.cmd[1] = Math.round(y * 100) / 100; stopTour(); updateCmdReadout();
}
let dragging = false;
pad.addEventListener('pointerdown', (e) => { dragging = true; pad.setPointerCapture(e.pointerId); setCmdFromPad(e); });
pad.addEventListener('pointermove', (e) => { if (dragging) setCmdFromPad(e); });
pad.addEventListener('pointerup', () => { dragging = false; });
pad.addEventListener('keydown', (e) => {
  const step = 0.25, m = { ArrowUp: [0, step], ArrowDown: [0, -step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] }[e.key];
  if (m) { e.preventDefault(); stopTour(); S.cmd[0] = Math.max(-VMAX, Math.min(VMAX, S.cmd[0] + m[0])); S.cmd[1] = Math.max(-VMAX, Math.min(VMAX, S.cmd[1] + m[1])); updateCmdReadout(); }
  if (e.key === ' ' || e.key === '0') { e.preventDefault(); S.cmd[0] = S.cmd[1] = 0; updateCmdReadout(); }
});
function updateCmdReadout() {
  const sp = Math.hypot(S.cmd[0], S.cmd[1]);
  $('cmd-readout').innerHTML = `vx <b>${fmt(S.cmd[0])}</b> m/s<br>vy <b>${fmt(S.cmd[1])}</b> m/s<br>speed <b>${sp.toFixed(2)}</b> m/s<br>yaw <b>${fmt(S.cmd[2])}</b> rad/s`;
  $('wz').value = S.cmd[2]; $('wz-val').textContent = fmt(S.cmd[2]) + ' rad/s';
}
$('wz').addEventListener('input', (e) => { S.cmd[2] = +e.target.value; stopTour(); updateCmdReadout(); });
document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => { S.cmd = b.dataset.preset.split(',').map(Number); stopTour(); updateCmdReadout(); }));
document.querySelectorAll('[data-push]').forEach((b) => b.addEventListener('click', () => { S.pushVec = b.dataset.push.split(',').map(Number); S.pushLeft = 5; S.pushShow = 0.35; }));
const TOUR = [[0, 0, 0], [0, 0.6, 0], [0.6, 0, 0], [0, -0.6, 0.4], [-0.6, 0, 0.4], [0.4, 0.4, -0.6], [0, 0, 0]];
function stopTour() { if (S.tour) { S.tour = false; $('tour').setAttribute('aria-pressed', 'false'); } }
$('tour').addEventListener('click', () => { S.tour = !S.tour; S.tourT = 0; $('tour').setAttribute('aria-pressed', String(S.tour)); });
$('pause').addEventListener('click', () => setPaused(!S.paused));
function setPaused(p) { S.paused = p; $('pause').setAttribute('aria-pressed', String(p)); $('pause').textContent = p ? 'Resume' : 'Pause'; }
$('reset').addEventListener('click', doReset);
$('follow').addEventListener('click', () => { S.follow = !S.follow; $('follow').setAttribute('aria-pressed', String(S.follow)); });
$('speed').addEventListener('input', (e) => { S.speed = +e.target.value; $('speed-val').textContent = S.speed.toFixed(1) + 'x'; });
$('nofix').addEventListener('change', (e) => { if (S.sim) S.sim.fixFrames = !e.target.checked; doReset(); });
document.addEventListener('keydown', (e) => { if (e.key.toLowerCase() === 'p' && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) setPaused(!S.paused); });

function doReset() {
  if (!S.sim) return;
  S.sim.cmd = S.cmd.slice(); S.obs = S.sim.reset(); S.sim.push = [0, 0]; S.pushLeft = 0; S.acc = 0; resetHist(); banner(''); setPaused(false);
  if (S.trail) { S.trailN = 0; S.trail.geometry.setDrawRange(0, 0); }
}

// ---------------------------------------------------------------------------------------------------------------- stepping
function tick() {
  const sim = S.sim;
  if (S.tour) { S.tourT += DT; const i = Math.floor(S.tourT / 3) % TOUR.length; if (S.tourIdx !== i) { S.tourIdx = i; S.cmd = TOUR[i].slice(); updateCmdReadout(); } }
  sim.cmd = S.cmd.slice(); S.obs.cmd = S.cmd.slice();
  if (S.pushLeft > 0) { sim.push = S.pushVec; S.pushLeft--; } else sim.push = [0, 0];
  const a = S.policy.predict(S.obs);
  const r = sim.step(a); S.obs = r.obs; S.act = a;
  const h = S.hist, o = r.obs, push = (arr, v) => { arr.push(v); if (arr.length > 1000) arr.shift(); };
  push(h.cx, S.cmd[0]); push(h.cy, S.cmd[1]); push(h.cw, S.cmd[2]); push(h.vx, o.vel[0]); push(h.vy, o.vel[1]); push(h.wz, o.angular_vel[2]);
  push(h.a0, a[0]); push(h.a1, a[1]); push(h.a2, a[2]);
  const p = sim.basePos();
  if (r.failed) { setPaused(true); banner(`Fell over after ${(sim.stepCount * DT).toFixed(2)} s (${sim.tiltDeg > 70 ? 'tilt above 70 degrees' : 'left the ball'}). Press Reset to try again.`, 'bad'); }
  else if (Math.abs(p[0]) > 4.7 || Math.abs(p[1]) > 4.7) { setPaused(true); banner('Reached the edge of the 10 m x 10 m floor. Press Reset to start again from the centre.', 'warn'); }
}

S.tick = tick; S.doReset = doReset;

// ---------------------------------------------------------------------------------------------------------------- plots
function drawPlot(id, series, fixed) {
  const cv = $(id), dpr = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  let m = fixed || 0.5; if (!fixed) for (const s of series) for (const v of s.d) m = Math.max(m, Math.abs(v)); m = Math.ceil(m * 4) / 4;
  const L = 34, B = 16, T = 4, pw = w - L - 4, ph = h - B - T, y = (v) => T + ph * (0.5 - v / (2 * m));
  g.strokeStyle = css('--line'); g.fillStyle = css('--text-3'); g.font = '10px ' + css('--font-mono'); g.lineWidth = 1;
  for (const v of [-m, 0, m]) { g.beginPath(); g.moveTo(L, y(v)); g.lineTo(L + pw, y(v)); g.stroke(); g.fillText(v.toFixed(1), 2, y(v) + 3); }
  g.fillText('−10 s', L, h - 3); g.fillText('now', L + pw - 18, h - 3);
  for (const s of series) {
    g.strokeStyle = s.c; g.lineWidth = s.lw || 1.6; g.setLineDash(s.dash || []); g.beginPath();
    s.d.forEach((v, i) => { const x = L + pw * (1 - (s.d.length - 1 - i) / 1000); i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v)); });
    g.stroke();
  }
  g.setLineDash([]);
}
function drawPlots() {
  const h = S.hist, cmdc = css('--text-2'), act = css('--learned');
  drawPlot('p-vx', [{ d: h.cx, c: cmdc, dash: [5, 4] }, { d: h.vx, c: act }]);
  drawPlot('p-vy', [{ d: h.cy, c: cmdc, dash: [5, 4] }, { d: h.vy, c: act }]);
  drawPlot('p-wz', [{ d: h.cw, c: cmdc, dash: [5, 4] }, { d: h.wz, c: act }]);
  drawPlot('p-act', [{ d: h.a0, c: '#ef5350', lw: 1.2 }, { d: h.a1, c: '#43a047', lw: 1.2 }, { d: h.a2, c: '#42a5f5', lw: 1.2 }], 1);
}

// ---------------------------------------------------------------------------------------------------------------- 3D scene
let renderer, scene, camera, controls, geoms = [], arrows = {}, floorMat, dirLight;
function buildScene(mj, model) {
  const stage = $('stage');
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1)); renderer.shadowMap.enabled = true;
  stage.insertBefore(renderer.domElement, stage.firstChild);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.02, 60); camera.up.set(0, 0, 1); camera.position.set(0.9, -1.5, 0.9);
  controls = new OrbitControls(camera, renderer.domElement); controls.target.set(0, 0, 0.2); controls.enableDamping = !reduceMotion; controls.maxPolarAngle = Math.PI * 0.49; controls.minDistance = 0.4; controls.maxDistance = 8;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 0.9));
  dirLight = new THREE.DirectionalLight(0xffffff, 1.6); dirLight.position.set(2, -2, 4); dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(1024, 1024); Object.assign(dirLight.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 12 });
  scene.add(dirLight, dirLight.target);
  floorMat = new THREE.MeshStandardMaterial({ roughness: 0.9 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), floorMat); floor.receiveShadow = true; scene.add(floor);
  makeFloorTexture();
  const tex = new THREE.TextureLoader().load('live/model/textures/basketball_texture.png'); tex.colorSpace = THREE.SRGBColorSpace;
  const gt = model.geom_type, gs = model.geom_size, rgba = model.geom_rgba, gg = model.geom_group, did = model.geom_dataid;
  for (let i = 0; i < model.ngeom; i++) {
    const t = gt[i], sz = [gs[3 * i], gs[3 * i + 1], gs[3 * i + 2]]; let geo = null;
    if (gg[i] >= 3 || t === 1 /* hfield */ || t === 0) continue;
    if (t === 2) geo = new THREE.SphereGeometry(sz[0], 32, 24);
    else if (t === 3) { geo = new THREE.CapsuleGeometry(sz[0], 2 * sz[1], 6, 16); geo.rotateX(Math.PI / 2); }
    else if (t === 5) { geo = new THREE.CylinderGeometry(sz[0], sz[0], 2 * sz[1], 40); geo.rotateX(Math.PI / 2); }
    else if (t === 6) geo = new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]);
    else if (t === 7) {
      const id = did[i], va = model.mesh_vertadr[id], vn = model.mesh_vertnum[id], fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(model.mesh_vert.subarray(3 * va, 3 * (va + vn))), 3));
      geo.setIndex(Array.from(model.mesh_face.subarray(3 * fa, 3 * (fa + fn)))); geo.computeVertexNormals();
    } else continue;
    const isBall = i === S.sim.ballGeom, col = new THREE.Color(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]);
    const mat = new THREE.MeshStandardMaterial({ color: isBall ? 0xffffff : col, map: isBall ? tex : null, roughness: isBall ? 0.55 : 0.5, metalness: isBall ? 0 : 0.15,
      transparent: rgba[4 * i + 3] < 1, opacity: rgba[4 * i + 3] });
    const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.matrixAutoUpdate = false; m.userData.gi = i; scene.add(m); geoms.push(m);
  }
  arrows.cmd = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xffffff, 0.08, 0.05);
  arrows.vel = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xa78bfa, 0.08, 0.05);
  arrows.push = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xff6b6b, 0.1, 0.06);
  Object.values(arrows).forEach((a) => { a.visible = false; scene.add(a); });
  const N = 4000, tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * N), 3)); tg.setDrawRange(0, 0);
  S.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xa78bfa })); S.trail.frustumCulled = false; S.trailN = 0; scene.add(S.trail);
  resize(); new ResizeObserver(resize).observe(stage); applyTheme();
}
function makeFloorTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'), light = document.documentElement.dataset.theme === 'light';
  g.fillStyle = light ? '#dfe5ee' : '#323a48'; g.fillRect(0, 0, 64, 64); g.fillStyle = light ? '#9aa6b8' : '#1e2430'; g.fillRect(0, 0, 32, 32); g.fillRect(32, 32, 32, 32);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(20, 20); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;   // 0.5 m squares
  floorMat.map = t; floorMat.needsUpdate = true;
}
function applyTheme() { if (!renderer) return; scene.background = new THREE.Color(css('--bg-3')); makeFloorTexture(); }
document.addEventListener('themechange', () => { applyTheme(); drawPad(); drawPlots(); });
function resize() { const st = $('stage'), w = st.clientWidth, h = st.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }

const _m = new THREE.Matrix4();
function syncScene(dt) {
  const d = S.sim.d, xp = d.geom_xpos, xm = d.geom_xmat;
  for (const m of geoms) {
    const i = m.userData.gi, p = 3 * i, r = 9 * i;
    _m.set(xm[r], xm[r + 1], xm[r + 2], xp[p], xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1], xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2], 0, 0, 0, 1);
    m.matrix.copy(_m); m.matrixWorldNeedsUpdate = true;
  }
  const b = S.sim.basePos(), o = S.obs;
  if (S.follow) {
    const nt = new THREE.Vector3(b[0], b[1], 0.22), delta = nt.clone().sub(controls.target); controls.target.copy(nt); camera.position.add(delta);
  }
  dirLight.position.set(b[0] + 2, b[1] - 2, 4); dirLight.target.position.set(b[0], b[1], 0);
  const setArrow = (a, vx, vy, base, scale) => {
    const L = Math.hypot(vx, vy); a.visible = L > 0.03;
    if (a.visible) { a.position.set(base[0], base[1], 0.01); a.setDirection(new THREE.Vector3(vx / L, vy / L, 0)); a.setLength(Math.max(0.12, L * scale), 0.08, 0.05); }
  };
  setArrow(arrows.cmd, S.cmd[0], S.cmd[1], b, 0.7); if (o) setArrow(arrows.vel, o.vel[0], o.vel[1], b, 0.7);
  if (S.pushShow > 0) { S.pushShow -= dt; setArrow(arrows.push, S.pushVec[0], S.pushVec[1], [b[0], b[1]], 0.03); arrows.push.position.z = 0.35; } else arrows.push.visible = false;
  if (S.frameCount % 3 === 0 && !S.paused) {   // trail
    const arr = S.trail.geometry.attributes.position.array;
    if (S.trailN < 4000) { arr.set([b[0], b[1], 0.005], 3 * S.trailN++); S.trail.geometry.attributes.position.needsUpdate = true; S.trail.geometry.setDrawRange(0, S.trailN); }
  }
  controls.update();
}

// ---------------------------------------------------------------------------------------------------------------- main loop
let last = performance.now(), fpsAcc = 0, fpsN = 0, rtf = 0;
S.frameCount = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now; S.frameCount++;
  if (!S.sim || !renderer) { drawPad(); return; }
  if (!S.paused && !S.sim.failed) {
    S.acc += dt * S.speed; let n = 0; const t0 = performance.now();
    while (S.acc >= DT && n < 15 && !S.paused) { tick(); S.acc -= DT; n++; }
    if (S.acc > 0.2) S.acc = 0;
    if (n) rtf = 0.9 * rtf + 0.1 * (n * DT) / Math.max(1e-4, (performance.now() - t0) / 1000);
  }
  syncScene(dt); renderer.render(scene, camera);
  if (S.frameCount % 3 === 0) { drawPlots(); drawPad(); }
  if (S.frameCount % 6 === 0 && S.act) {
    updateObsView(S.obs, S.act);
    $('hud').innerHTML = `t <b>${(S.sim.stepCount * DT).toFixed(2)} s</b><br>speed <b>${Math.hypot(S.obs.vel[0], S.obs.vel[1]).toFixed(2)} m/s</b><br>tilt <b>${S.sim.tiltDeg.toFixed(1)}&deg;</b><br>physics <b>${rtf ? Math.min(999, rtf).toFixed(0) + 'x real time' : '...'}</b>`;
  }
}

// ---------------------------------------------------------------------------------------------------------------- parity
const mm = (v) => (v * 1000 >= 100 ? (v * 1000).toFixed(0) : (v * 1000).toPrecision(2));
function renderParityTable(rows, source) {
  $('par-body').innerHTML = rows.map((r) => `<tr><td>${r.name}<div class="lv-small">${r.cmd}</div></td><td class="v">${r.e1}</td><td class="v">${r.e4}</td><td class="v">${r.e8}</td><td class="v">${r.max}</td></tr>`).join('')
    + `<tr><td colspan="5" class="lv-small">Source: ${source}</td></tr>`;
}
const CASE_LABEL = { fwd_diag: 'Diagonal drive', yaw_fwd: 'Forward + turning', turn_back: 'Reverse + turning', hold: 'Hold still (3 s)' };
const cmdStr = (c) => `command (${c.map((v) => v.toFixed(1)).join(', ')})`;
function staticParity(p) {
  const rows = ['fwd_diag', 'yaw_fwd', 'turn_back', 'hold'].filter((k) => p.cases[k]).map((k) => {
    const e = p.cases[k].closed_loop.xy_err_m, rc = p.ref_cmds && p.ref_cmds[k];
    return { name: CASE_LABEL[k], cmd: rc ? cmdStr(rc) : '', e1: mm(e['1s']), e4: k === 'hold' ? '-' : mm(e['4s']), e8: k === 'hold' ? mm(e.final) + ' (at 3 s)' : mm(e['8s']), max: mm(e.max) };
  });
  renderParityTable(rows, `${p.mujoco_wasm} under Node.js, script site/live/parity_node.mjs (static numbers; press the button to repeat in this browser).`);
  const nf = p.cases.fwd_diag_nofix, os = p.cases.fwd_diag.one_step;
  setExtra([
    ['Network export', `JS forward pass vs Stable-Baselines3 on ${S.spec.verification.n_random_obs} random observations: max difference ${S.spec.verification.max_abs_err.toExponential(1)} (limit 1e-5)`],
    ['Policy on Python-recorded observations', `max action difference ${p.policy_max_abs_action_error.toExponential(1)} over all recorded steps`],
    ['Single control step', `from identical states and actions, median base-position difference ${(os.qpos_median * 1000).toExponential(0)} mm, median velocity difference ${os.qvel_median.toExponential(0)} m/s (fwd_diag, ${os.n} steps)`],
    ['Fix disabled', `Python falls at control step ${nf.closed_loop.ref_fell_at_step}; WebAssembly falls at step ${nf.closed_loop.fell_at_step}`],
  ]);
}
function setExtra(items) { $('par-extra').innerHTML = items.map(([a, b]) => `<tr><td style="white-space:nowrap;vertical-align:top">${a}</td><td>${b}</td></tr>`).join(''); }

async function runParity() {
  if (!S.mj) return;
  const btn = $('par-run'); btn.disabled = true; btn.textContent = 'Running...'; badge('b-parity', 'busy', 'Parity vs Python: running');
  try {
    const ref = await (await fetch('live/ref_web.json')).json();
    const files = S.files, { model } = S; const data = new S.mj.MjData(model);
    const sim = new BallbotSim(S.mj, model, data, { fixFrames: true }), nof = new BallbotSim(S.mj, model, data, { fixFrames: false });
    const pol = S.policy, rows = []; let polWorst = 0, nofix = null, worstAll = 0, nCases = 0;
    const refFull = S.parityStatic && S.parityStatic.cases;
    for (const c of ref.cases) {
      for (let t = 0; t < c.obs.length; t++) { const a = pol.predict(c.obs[t]); for (let i = 0; i < 3; i++) polWorst = Math.max(polWorst, Math.abs(a[i] - c.act[t][i])); }
      const useFix = !c.name.endsWith('nofix'), s = useFix ? sim : nof;
      s.cmd = c.cmd.slice(); let obs = s.reset(), fell = null; const err = [];
      for (let t = 0; t < c.steps; t++) {
        const r = s.step(pol.predict(obs)); obs = r.obs; const p = s.basePos();
        err.push(Math.hypot(p[0] - c.pos_xy[t][0], p[1] - c.pos_xy[t][1]));
        if (r.failed) { fell = t; break; }
      }
      if (!useFix) { nofix = { ref: c.fell_at_step, js: fell }; continue; }
      const at = (sec) => err[Math.min(err.length, Math.round(sec * 100)) - 1], mx = Math.max(...err); worstAll = Math.max(worstAll, mx); nCases++;
      rows.push({ name: CASE_LABEL[c.name] || c.name, cmd: cmdStr(c.cmd), e1: mm(at(1)), e4: c.steps >= 800 ? mm(at(4)) : '-', e8: c.steps >= 800 ? mm(at(8)) : mm(err[err.length - 1]) + ' (at 3 s)', max: mm(mx) });
    }
    data.delete();
    renderParityTable(rows, `computed just now in this browser (${S.mj_version}), against the Python reference in live/ref_web.json.`);
    const prev = [...$('par-extra').querySelectorAll('tr')].map((tr) => [tr.children[0].textContent, tr.children[1].innerHTML]).filter(([a]) => !/^(Policy on|Fix disabled)/.test(a));
    setExtra([...prev, ['Policy on Python-recorded observations (this browser)', `max action difference ${polWorst.toExponential(1)} over the first 150 steps of each run`],
      ['Fix disabled (this browser)', nofix ? `Python falls at control step ${nofix.ref}; WebAssembly falls at step ${nofix.js}` : 'n/a']]);
    badge('b-parity', worstAll < 0.05 ? 'ok' : 'warn', `Parity vs Python (this browser): max ${mm(worstAll)} mm over ${nCases} reference runs`);
  } catch (e) {
    console.error(e); badge('b-parity', 'bad', 'Parity vs Python: check failed to run');
  }
  btn.disabled = false; btn.textContent = 'Run the check in this browser';
}
$('par-run').addEventListener('click', runParity);

// ---------------------------------------------------------------------------------------------------------------- boot
async function boot() {
  drawPad(); updateCmdReadout(); requestAnimationFrame(frame);
  let spec, parity;
  try {
    [spec, parity] = await Promise.all([fetch('live/policy_robs_c_s0.json').then((r) => r.json()), fetch('live/parity.json').then((r) => r.json())]);
  } catch (e) { setLoad('Could not load the policy', String(e)); badge('b-engine', 'bad', 'Policy files missing'); return; }
  S.spec = spec; S.parityStatic = parity; S.policy = new Policy(spec); buildObsTable();
  try { const refc = await fetch('live/ref_web.json').then((r) => r.json()); parity.ref_cmds = Object.fromEntries(refc.cases.map((c) => [c.name, c.cmd])); } catch (e) { /* optional */ }
  staticParity(parity);
  badge('b-parity', 'warn', `Parity vs Python (Node reference): max ${mm(Math.max(...['fwd_diag', 'yaw_fwd', 'turn_back', 'hold'].map((k) => parity.cases[k].closed_loop.xy_err_m.max)))} mm over 4 runs`);
  let mj;
  try {
    setLoad('Loading the simulator', 'Fetching the MuJoCo WebAssembly build (about 10 MB, cached afterwards).');
    const mod = await import(MUJOCO_URL); mj = await mod.default();
  } catch (e) {
    console.error(e);
    setLoad('The physics engine could not be loaded', 'The page needs to download MuJoCo 3.13.0 (WebAssembly) from cdn.jsdelivr.net. You appear to be offline or the CDN is blocked. The network, the parity figures and the explanation below still work; the live simulation does not.');
    badge('b-engine', 'bad', 'MuJoCo WebAssembly: unavailable'); badge('b-fix', 'bad', 'Contact-frame fix: not running'); return;
  }
  S.mj = mj; S.mj_version = 'MuJoCo WebAssembly 3.13.0';
  try {
    setLoad('Loading the robot model', 'ballbot.xml, meshes and textures.');
    const get = async (p) => new Uint8Array(await (await fetch('live/model/' + p)).arrayBuffer());
    S.files = { 'ballbot.xml': await get('ballbot.xml'), 'meshes/cone.stl': await get('meshes/cone.stl'), 'textures/basketball_texture.png': await get('textures/basketball_texture.png') };
    const { model, data } = loadModel(mj, S.files); S.model = model;
    S.sim = new BallbotSim(mj, model, data); S.sim.cmd = S.cmd.slice(); S.obs = S.sim.reset();
    buildScene(mj, model);
    // self-check of the contact-frame hook: run a few control steps and require the heap layout to match data.contact
    let ok = S.sim.frameFix; for (let i = 0; i < 30 && ok; i++) { S.sim.step([0, 0, 0]); ok = S.sim.verify(); }
    S.obs = S.sim.reset();
    badge('b-engine', 'ok', 'MuJoCo WebAssembly 3.13.0 running');
    if (ok) badge('b-fix', 'ok', 'Contact-frame fix: active, memory layout verified');
    else { badge('b-fix', 'bad', 'Contact-frame fix: layout check FAILED, robot will not balance'); banner('The contact-frame fix could not be verified in this build of MuJoCo, so the physics is not faithful. Parity figures below are from the Node reference run.', 'warn'); S.sim.fixFrames = false; }
    $('loading').hidden = true;
    (window.requestIdleCallback || ((f) => setTimeout(f, 300)))(() => runParity());
  } catch (e) {
    console.error(e); setLoad('The model could not be loaded', String(e)); badge('b-engine', 'bad', 'MuJoCo WebAssembly: model failed to load');
  }
}
boot();
