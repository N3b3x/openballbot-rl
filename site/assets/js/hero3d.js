// Landing hero: a realistic rendering of the SIMULATED nominal ballbot (9 cm ball, about 4 kg; dimensions in robot3d.js), animated
// by the same planar model + LQR that the math page uses (ballbot-model.js), one planar simulation per horizontal axis.
// No physical robot exists. Page contract (unchanged): #hero-canvas, #hero-fallback, #hero-hud, #hero-play, #hero-toast, the
// html classes .webgl-ready / .no-webgl, an import map for 'three', and window.__hero3d.
// Lazy: nothing is downloaded or created until the hero is on screen. Quality adapts to frame rate and device.

const host = document.getElementById('hero-canvas');
const fallback = document.getElementById('hero-fallback');
const hud = document.getElementById('hero-hud');
const playBtn = document.getElementById('hero-play');
const toast = document.getElementById('hero-toast');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isLight = () => document.documentElement.dataset.theme === 'light';

function webglOk() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  } catch (e) { return false; }
}
function noWebgl() { document.documentElement.classList.add('no-webgl'); if (fallback) fallback.hidden = false; }

// minimal styles for the controls this script adds (the page CSS is owned elsewhere)
function injectStyles() {
  const s = document.createElement('style');
  s.textContent = `
  .hero3d-tools{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;pointer-events:auto}
  .hero3d-tools button{font:inherit;font-size:.82rem;cursor:pointer;display:inline-flex;align-items:center;gap:7px;padding:7px 13px;border-radius:999px;border:1px solid var(--line-2,#3a4a63);background:color-mix(in srgb,var(--bg-2,#10161f) 86%,transparent);color:var(--text,#e8eef7);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);transition:border-color .15s,background .15s}
  .hero3d-tools button:hover{border-color:var(--hybrid,#2ee6c4)}
  .hero3d-tools button:focus-visible{outline:2px solid var(--hybrid,#2ee6c4);outline-offset:2px}
  .hero3d-tools button[aria-pressed="true"]{border-color:var(--hybrid,#2ee6c4);background:color-mix(in srgb,var(--hybrid,#2ee6c4) 16%,var(--bg-2,#10161f))}
  .hero3d-tools .dot{width:8px;height:8px;border-radius:50%;background:var(--hybrid,#2ee6c4);box-shadow:0 0 0 3px color-mix(in srgb,var(--hybrid,#2ee6c4) 25%,transparent)}
  .hero3d-cap{flex-basis:100%;margin:0;font-size:.74rem;color:var(--text-3,#8393ab);max-width:60ch}
  .hero3d-wrap{display:flex;flex-direction:column;gap:6px;align-items:flex-end;text-align:right;margin-left:auto}
  .hero3d-wrap .hero3d-tools{justify-content:flex-end}
  .hero3d-floating{position:absolute;right:16px;bottom:12px;z-index:2}
  .hero3d-tools #hero-play{font-size:.82rem;padding:7px 13px;border-radius:999px;min-height:0;line-height:inherit}
  .hero3d-wrap .hero-note{max-width:46ch;margin:0 0 2px}
  @media (pointer:coarse){.hero3d-tools button{min-height:44px}}
  @media (max-width:900px){.hero3d-wrap{align-items:flex-start;text-align:left;margin-left:0}.hero3d-wrap .hero3d-tools{justify-content:flex-start}}
  @media (prefers-reduced-motion:reduce){.hero3d-tools button{transition:none}}`;
  document.head.appendChild(s);
}

async function init() {
  if (!host) return;
  if (!webglOk()) { noWebgl(); return; }
  // lazy: wait until the hero is (nearly) on screen before downloading three.js or touching the GPU
  await new Promise((res) => {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); res(); } }, { rootMargin: '200px' });
    io.observe(host);
  });
  const [THREE, { OrbitControls }, R3, Model] = await Promise.all([
    import('three'), import('three/addons/controls/OrbitControls.js'), import('./robot3d.js'), import('./ballbot-model.js'),
  ]);
  let geo = null;
  try { geo = await fetch(new URL('../../data/geometry_table.json', import.meta.url)).then((r) => (r.ok ? r.json() : null)); } catch (e) { /* lean law fallback below */ }

  const mobile = matchMedia('(pointer: coarse)').matches || Math.min(innerWidth, innerHeight) < 700;
  let level = mobile || (navigator.hardwareConcurrency || 8) <= 4 ? 1 : 2;      // 0 low, 1 medium, 2 high
  const DPR = [1, 1.5, 2];

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: level > 0, alpha: false, powerPreference: 'low-power', preserveDrawingBuffer: false });
  } catch (e) { noWebgl(); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, DPR[level]));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const cv = renderer.domElement;
  cv.setAttribute('role', 'img');
  cv.setAttribute('aria-label', 'Rendering of the simulated ballbot: a white tower with a time-of-flight sensor head, three omniwheel drives and an orange basketball-textured ball. Drag to orbit, click the robot to push it.');
  cv.style.touchAction = 'pan-y';

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 40);
  const stage = R3.createStage(THREE, renderer, scene, { quality: level });
  const bot = R3.createBallbot(THREE, { quality: level });
  scene.add(bot.root);

  const controls = new OrbitControls(camera, cv);
  controls.target.set(0, 0.25, 0);
  controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.minPolarAngle = 0.5; controls.maxPolarAngle = Math.PI / 2 - 0.04;
  controls.minDistance = 1.0; controls.maxDistance = 3.4;
  controls.rotateSpeed = 0.7;
  controls.autoRotateSpeed = 0.55;                 // slow idle drift (about 11 minutes per turn)
  controls.autoRotate = !reduced;

  // ---- theme sync
  function applyTheme() {
    const light = isLight();
    stage.setTheme(light, cssVar('--bg') || (light ? '#f6f8fb' : '#0a0e14'));
    bot.setTheme(light);
    dirty = true;
  }
  let dirty = true;
  document.addEventListener('themechange', applyTheme);

  // ---- sizing: shift the robot right on wide screens so the text sits left
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const portrait = h > w * 1.05;
    if (w > 900) camera.setViewOffset(w, h, -w * (w < 1200 ? 0.23 : 0.17), h * 0.1, w, h);
    else if (portrait) camera.clearViewOffset();
    else camera.setViewOffset(w, h, -w * 0.24, -h * 0.04, w, h);
    camera.fov = portrait ? 38 : 30;
    if (!resize.done) {
      resize.done = true;
      const d = portrait ? 1.4 : (w > 900 ? 2.1 : 2.5);
      camera.position.set(0.7, 0.42, 0.9).normalize().multiplyScalar(d).add(controls.target);
    }
    camera.updateProjectionMatrix();
    dirty = true;
  }
  new ResizeObserver(resize).observe(host);

  // ---- physics: one planar LQR simulation per horizontal axis (the model is planar; the two axes are decoupled here)
  const P = geo ? Model.nominalParams(geo) : null;
  const kappa = P ? Model.kappaLean(P) : 0.176;
  const mk = () => new Model.PlanarSim(P, { dtc: 0.01, sub: 6, aMax: 0.6, failDeg: 45 });
  let sx = P ? mk() : null, sz = P ? mk() : null;
  const DT = 0.01, AMP = [0.3, 0.2], OM = [1.2, 0.8], PH = [0, 1.3];
  let acc = 0, tSim = 0, playing = !reduced, visible = true, last = performance.now(), pokeUntil = 0, lastV = 0, aNow = 0;
  const state = { x: 0, z: 0, tx: 0, tz: 0 };

  function stepSim(dt) {
    if (!sx) {      // no model data: derived lean law theta = kappa a with a scripted acceleration (still honest about it)
      tSim += dt; const a = 0.4 * Math.sin(1.4 * tSim);
      state.x = -(0.4 / 1.96) * Math.sin(1.4 * tSim); state.z = 0; state.tx = kappa * a; state.tz = 0; aNow = a; return;
    }
    acc += dt;
    while (acc >= DT) {
      acc -= DT; tSim += DT;
      if (playing) {
        // the command is a gentle Lissajous velocity, pulled back toward the origin by a weak position loop
        sx.setCommand(AMP[0] * Math.sin(OM[0] * tSim + PH[0]) - 0.8 * sx.x);
        sz.setCommand(AMP[1] * Math.sin(OM[1] * tSim + PH[1]) - 0.8 * sz.x);
      } else { sx.setCommand(-0.8 * sx.x); sz.setCommand(-0.8 * sz.x); }
      sx.step(); sz.step();
      if (sx.fell || sz.fell) { sx = mk(); sz = mk(); }
      aNow = (sx.v - lastV) / DT; lastV = sx.v;
    }
    state.x = sx.x; state.z = sz.x; state.tx = sx.theta; state.tz = sz.theta;
  }
  const settled = () => !sx || (Math.abs(sx.s[1]) + Math.abs(sz.s[1]) + Math.abs(sx.v) + Math.abs(sz.v) < 2e-3 && Math.abs(sx.x) + Math.abs(sz.x) < 5e-3);

  let hudText = '';
  function pose(dt) {
    bot.setPose(state, dt);
    stage.followBall(bot);
    if (hud) {
      const s = `a = ${aNow.toFixed(2).padStart(5)} m/s²   lean = ${(state.tx * 180 / Math.PI).toFixed(1).padStart(5)}°   (kappa ${(kappa * 180 / Math.PI).toFixed(1)}° per m/s²)`;
      if (s !== hudText) { hudText = s; hud.textContent = s; }
    }
  }

  // ---- push: click the robot or press the button
  const arrowMat = new THREE.MeshBasicMaterial({ color: 0x2ee6c4, transparent: true, depthWrite: false, depthTest: false });
  const arrow = new THREE.Group();
  { const sh = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.12, 12), arrowMat); sh.position.y = -0.06;
    const hd = new THREE.Mesh(new THREE.ConeGeometry(0.017, 0.04, 16), arrowMat); hd.position.y = -0.005; arrow.add(sh, hd); }
  arrow.visible = false; arrow.renderOrder = 5; scene.add(arrow);
  let arrowT = 0;
  function showArrow(point, dir) {
    arrow.position.copy(point);
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0).negate(), dir.clone().normalize());   // arrow points along dir, tip at the point
    arrow.visible = true; arrowT = 0.7; arrowMat.opacity = 1;
  }
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let toastTimer;
  function say(msg) { if (!toast) return; toast.textContent = msg; toast.classList.add('on'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('on'), 3600); }
  function push(dir, height, point) {
    if (reduced && !playing) { say('Motion is reduced on this device. Press Play to enable the push.'); return; }
    if (!sx) { say('Model data did not load, so the push is unavailable.'); return; }
    const F = 10 + 6 * Math.min(1, height / 0.5);                               // newtons on the body, 0.12 s
    const d = dir.clone(); d.y = 0; d.normalize();
    sx.push(F * d.x, 0.12); sz.push(F * d.z, 0.12);
    bot.setStatus(1, 0xffb347); setTimeout(() => bot.setStatus(0.55, 0x2ee6c4), 900);
    showArrow(point || new THREE.Vector3(bot.rig.position.x, 0.3, bot.rig.position.z).addScaledVector(d, -0.14), d);
    say(`Pushed with ${F.toFixed(0)} N for 0.12 s. In the simulation the LQR leans into the push and drives the ball back under the body.`);
    pokeUntil = performance.now() + 3000; dirty = true;
  }
  let down = null;
  cv.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; controls.autoRotate = false; clearTimeout(resumeT); });
  cv.addEventListener('pointerup', (e) => {
    resumeDrift();
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
    down = null;
    const r = cv.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(bot.rig, true)[0];
    if (hit) push(ray.ray.direction, hit.point.y, hit.point);
  });
  let resumeT;
  function resumeDrift() { clearTimeout(resumeT); resumeT = setTimeout(() => { controls.autoRotate = playing; }, 6000); }

  // ---- injected controls (the page markup is owned elsewhere): push button, sensor toggle, honest caption
  injectStyles();
  const wrap = document.createElement('div'); wrap.className = 'hero3d-wrap';
  wrap.innerHTML = `<div class="hero3d-tools"><button type="button" id="hero-push"><span class="dot" aria-hidden="true"></span>Push the robot</button><button type="button" id="hero-sensor" aria-pressed="false">Show the sensor</button></div><p class="hero3d-cap">Rendering of the simulated nominal design (9 cm ball, about 4 kg). Simulation only.</p>`;
  const ui = document.querySelector('.hero-ui');
  const note = document.querySelector('.hero-note'); if (note) wrap.prepend(note);   // one tidy right-hand column: note, buttons, caption
  if (ui) ui.appendChild(wrap); else { wrap.classList.add('hero3d-floating'); host.parentElement.appendChild(wrap); }
  wrap.querySelector('#hero-push').addEventListener('click', () => {
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion); right.y = 0; right.normalize();
    push(right.multiplyScalar(Math.sin(performance.now() / 1000) > 0 ? 1 : -1), 0.3);
  });
  const sensorBtn = wrap.querySelector('#hero-sensor');
  sensorBtn.addEventListener('click', () => {
    const on = !bot.sensorVisible; bot.setSensorVisible(on); sensorBtn.setAttribute('aria-pressed', String(on));
    sensorBtn.textContent = on ? 'Hide the sensor' : 'Show the sensor';
    if (on) say('Head time-of-flight sensor: 55 x 42 degree field of view, pitched 25 degrees down, 8 x 8 zones binned from the 54 x 42 sensor.');
    dirty = true;
  });

  if (playBtn) {
    wrap.querySelector('.hero3d-tools').appendChild(playBtn);
    playBtn.hidden = false;                       // a pause control for everyone (moving content), not only reduced-motion users
    playBtn.textContent = playing ? 'Pause animation' : 'Play animation';
    playBtn.addEventListener('click', () => { playing = !playing; controls.autoRotate = playing; playBtn.textContent = playing ? 'Pause animation' : 'Play animation'; dirty = true; });
  }

  // ---- adaptive quality: step down when the frame rate is low (and never back up, to avoid flapping)
  function applyLevel(l) {
    level = l;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, DPR[l]));
    stage.setQuality(l);
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    dirty = true;
  }
  const perf = { n: 0, sum: 0, since: 0, fps: 60 };
  function watchPerf(dt, now) {
    if (dt > 0.1) return;                                   // tab switches and stalls are not render cost
    perf.n++; perf.sum += dt;
    if (perf.n >= 90) {
      perf.fps = perf.n / perf.sum;
      if (perf.fps < 38 && level > 0 && now - perf.since > 2500) { applyLevel(level - 1); perf.since = now; }
      perf.n = 0; perf.sum = 0;
    }
  }

  // ---- loop: renders only while visible; a settled scene renders on demand
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0 }).observe(host);
  let frozen = false;                                          // debugging and poster capture: stops the loop
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden || frozen) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const active = playing || now < pokeUntil || !settled();
    if (active || dirty) { stepSim(playing || now < pokeUntil || !settled() ? dt : 0); pose(dt); }
    if (arrow.visible) { arrowT -= dt; arrowMat.opacity = Math.max(0, arrowT / 0.7); if (arrowT <= 0) arrow.visible = false; }
    const pulse = 0.5 + 0.08 * Math.sin(now / 900);
    if (playing && now > pokeUntil) bot.setStatus(pulse);
    const moved = controls.update();
    if (active || dirty || moved || arrow.visible) { renderer.render(scene, camera); dirty = false; if (playing) watchPerf(dt, now); }
  }
  applyTheme(); resize(); stepSim(0); pose(0);
  renderer.render(scene, camera);
  document.documentElement.classList.add('webgl-ready');
  if (fallback) fallback.hidden = true;
  requestAnimationFrame(frame);
  window.__hero3d = {
    renderer, scene, camera, bot, stage, controls, THREE,
    poke: () => push(new THREE.Vector3(1, 0, 0), 0.3),
    setLevel: applyLevel, perf, freeze: (f = true) => { frozen = f; }, get level() { return level; },
  };
}

init().catch((e) => { console.warn('hero3d failed, showing the poster', e); noWebgl(); });
