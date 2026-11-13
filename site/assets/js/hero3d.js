// Landing hero: the project's CURRENT parametric robot, built at load time from data/robot_model.json. That file is written by
// scripts/study/export_robot_json.py from the Python model (ballbot_layout.py + layout_render.py), so the picture here follows the
// model: regenerate the JSON and this page shows the new design. Nothing about the robot's shape is hard-coded in this script.
// Motion: the same planar model + LQR that the math page uses (ballbot-model.js), one planar simulation per horizontal axis. The ball
// rolls without slip, the body leans about the ball centre, and each tagged wheel spins at the rate the ball's motion asks of it.
// No physical robot exists. Page contract (unchanged): #hero-canvas, #hero-fallback, #hero-hud, #hero-play, #hero-toast, the
// html classes .webgl-ready / .no-webgl, an import map for 'three', and window.__hero3d.
// Lazy: nothing is downloaded or created until the hero is on screen. If the JSON, three.js or WebGL is unavailable the static
// picture in #hero-fallback (rendered from the same model by the export script) stays in place.

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
// static fallback: keep the poster, drop the canvas box and the controls that need the live robot
function noWebgl() {
  document.documentElement.classList.add('no-webgl');
  if (fallback) fallback.hidden = false;
  if (host) host.hidden = true;
  if (hud) hud.hidden = true;
  const note = document.querySelector('.hero-note');
  if (note) note.textContent = 'Still picture of the simulated design. The interactive 3D view is unavailable in this browser.';
}

// minimal styles for the controls this script adds (the page CSS is owned elsewhere)
function injectStyles() {
  const s = document.createElement('style');
  s.textContent = `
  .hero3d-tools{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;pointer-events:auto}
  .hero3d-tools button{font:inherit;font-size:.82rem;cursor:pointer;display:inline-flex;align-items:center;gap:7px;padding:7px 13px;border-radius:999px;border:1px solid var(--line-2,#3a4a63);background:color-mix(in srgb,var(--bg-2,#10161f) 86%,transparent);color:var(--text,#e8eef7);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);transition:border-color .15s,background .15s}
  .hero3d-tools button:hover{border-color:var(--hybrid,#2ee6c4)}
  .hero3d-tools button:focus-visible{outline:2px solid var(--hybrid,#2ee6c4);outline-offset:2px}
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

// ---------------------------------------------------------------------------------------------------------------------
// The model file. Throws when it is missing or malformed, which leaves the static picture in place.
async function loadModel() {
  const r = await fetch(new URL('../../data/robot_model.json', import.meta.url), { cache: 'no-cache' });   // revalidate, so a regenerated model shows up
  if (!r.ok) throw new Error('robot_model.json: HTTP ' + r.status);
  const m = await r.json();
  if (!m || !m.meta || !Array.isArray(m.geoms) || !m.geoms.length || !(m.meta.ball_radius > 0)) throw new Error('robot_model.json: unexpected content');
  return m;
}

// Surface finish per colour name of layout_render.py (COL); unknown names get a neutral plastic.
const FINISH = {
  ball: [0.5, 0.05], seam: [0.6, 0.05], wheel: [0.42, 0.1], roller: [0.88, 0], hub: [0.5, 0.4], motor: [0.32, 0.75], gear: [0.5, 0.45], steel: [0.38, 0.8],
  dark: [0.45, 0.1], white: [0.3, 0.02], blue: [0.38, 0.08], arm: [0.36, 0.1], armj: [0.42, 0.15], screen: [0.22, 0], belt: [0.9, 0], pulley: [0.35, 0.7],
  pad: [0.6, 0], tray: [0.3, 0.75], red: [0.4, 0],
};

/**
 * Build the robot from the exported geoms. The file is in MuJoCo's convention (metres, z up, front +x, quaternion w x y z, capsule and
 * cylinder size = [radius, half-length] along local z). three.js is y up, so every part sits under a group rotated -90 degrees about x,
 * which maps (x, y, z) to (x, z, -y) and keeps the frame right-handed; inside those groups the numbers are used as exported.
 * Static geoms are baked and merged per material, so the whole robot is a few dozen draw calls.
 */
function buildRobot(THREE, merge, model, quality) {
  const { Group, Mesh, Matrix4, Quaternion, Vector3 } = THREE;
  const R = model.meta.ball_radius, c = new Vector3(0, 0, R);
  const detail = [0.6, 0.8, 1][quality];
  const seg = (r) => Math.max(6, Math.round((r < 0.006 ? 8 : r < 0.02 ? 12 : r < 0.05 ? 24 : r < 0.085 ? 36 : 64) * detail));

  const mats = new Map(), envMats = [], glow = [];
  function material(g) {
    const [r, gg, b, a = 1] = g.rgba, key = g.rgba.join(',');
    if (mats.has(key)) return mats.get(key);
    const [rough, metal] = FINISH[g.color] || [0.5, 0.05];
    const m = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(r, gg, b, THREE.SRGBColorSpace), roughness: rough, metalness: metal });
    if (a < 0.999) { m.transparent = true; m.opacity = a; m.depthWrite = false; }
    if (g.color === 'screen' && a >= 0.999) { m.emissive = m.color.clone(); m.emissiveIntensity = 0.3; glow.push(m); }
    mats.set(key, m); envMats.push(m);
    return m;
  }

  function shape(g) {
    const s = g.size;
    let geo;
    if (g.type === 'sphere') geo = new THREE.SphereGeometry(s[0], seg(s[0]), Math.max(6, Math.round(seg(s[0]) * 0.6)));
    else if (g.type === 'capsule') { geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], Math.max(3, Math.round(seg(s[0]) / 3)), seg(s[0])); geo.rotateX(Math.PI / 2); }
    else if (g.type === 'cylinder') { geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], seg(s[0])); geo.rotateX(Math.PI / 2); }
    else if (g.type === 'box') geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    else if (g.type === 'ellipsoid') { geo = new THREE.SphereGeometry(1, 32, 20); geo.scale(s[0], s[1], s[2]); }
    else if (g.type === 'mesh' && g.vertices && g.faces) {
      geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices, 3)); geo.setIndex(g.faces);
      geo.computeVertexNormals(); geo.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array((g.vertices.length / 3) * 2), 2));
    } else return null;
    return geo.index ? geo : null;
  }

  const mtx = new Matrix4(), q = new Quaternion(), p = new Vector3(), one = new Vector3(1, 1, 1);
  /** Bake a list of geoms (positions relative to `origin`) into one mesh per material and add them to `parent`. */
  function bake(list, origin, parent) {
    const byMat = new Map();
    for (const g of list) {
      const geo = shape(g); if (!geo) continue;
      q.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); p.fromArray(g.pos).sub(origin);
      geo.applyMatrix4(mtx.compose(p, q, one));
      const m = material(g); if (!byMat.has(m)) byMat.set(m, []); byMat.get(m).push(geo);
    }
    for (const [m, geos] of byMat) {
      const merged = geos.length > 1 ? merge(geos, false) : geos[0];
      if (!merged) continue;
      if (geos.length > 1) for (const g of geos) g.dispose();
      const mesh = new Mesh(merged, m); mesh.castShadow = !m.transparent; mesh.receiveShadow = !m.transparent; if (m.transparent) mesh.renderOrder = 2;
      parent.add(mesh);
    }
  }
  const zUp = (parent) => { const g = new Group(); g.rotation.x = -Math.PI / 2; parent.add(g); return g; };

  const root = new Group();
  const rig = new Group(); rig.position.y = R; root.add(rig);        // translates with the ball centre
  const ballSpin = new Group(); rig.add(ballSpin);
  const tilt = new Group(); rig.add(tilt);                           // body frame, pivots about the ball centre
  bake(model.geoms.filter((g) => g.part === 'ball'), c, zUp(ballSpin));
  const bodyZ = zUp(tilt);
  bake(model.geoms.filter((g) => g.part !== 'ball' && g.part !== 'wheel'), c, bodyZ);

  // wheels: one spinning group per tagged wheel; geoms tagged with an unknown wheel index stay fixed to the body
  const toThree = (a) => new Vector3(a[0], a[2], -a[1]);
  const wheels = [], loose = [];
  const frames = new Map((model.wheels || []).map((w) => [w.index, w]));
  const tagged = new Map();
  for (const g of model.geoms) {
    if (g.part !== 'wheel') continue;
    if (!frames.has(g.wheel)) { loose.push(g); continue; }
    if (!tagged.has(g.wheel)) tagged.set(g.wheel, []);
    tagged.get(g.wheel).push(g);
  }
  if (loose.length) bake(loose, c, bodyZ);
  for (const [idx, list] of tagged) {
    const w = frames.get(idx), hub = new Vector3().fromArray(w.hub);
    const spin = new Group(); spin.position.copy(hub).sub(c); bodyZ.add(spin);
    bake(list, hub, spin);
    const axis = new Vector3().fromArray(w.axis).normalize();          // z-up frame: the group lives under bodyZ
    const n = toThree(w.normal), t = toThree(w.tangent), a3 = toThree(w.axis);
    const denom = new Vector3().crossVectors(a3, n).dot(t) || 1;        // sign of the spin axis relative to the drive direction
    wheels.push({ spin, axis, n, t, gain: -(R / w.radius) / denom, angle: 0 });
  }

  // ---- pose update, ball roll, wheel spin
  const prev = { x: 0, z: 0, tx: 0, tz: 0, ok: false };
  const qBall = new Quaternion(), qd = new Quaternion(), ax = new Vector3(), wBody = new Vector3(), wBall = new Vector3(), wRel = new Vector3(), tmp = new Vector3(), qInv = new Quaternion();
  /** Pose in metres and radians: ball centre (x, z) on the floor, body lean about the ball centre toward +x (tiltX) and +z (tiltZ). */
  function setPose({ x = 0, z = 0, tiltX = 0, tiltZ = 0 }, dt = 0) {
    rig.position.x = x; rig.position.z = z;
    tilt.rotation.set(tiltZ, 0, -tiltX, 'XYZ');
    if (prev.ok && dt > 1e-5) {
      const vx = (x - prev.x) / dt, vz = (z - prev.z) / dt;
      wBall.set(vz / R, 0, -vx / R);                                    // rolling without slip on the floor
      const w = wBall.length();
      if (w > 1e-6) { ax.copy(wBall).divideScalar(w); qd.setFromAxisAngle(ax, w * dt); qBall.premultiply(qd).normalize(); ballSpin.quaternion.copy(qBall); }
      wBody.set((tiltZ - prev.tz) / dt, 0, -(tiltX - prev.tx) / dt);
      qInv.copy(tilt.quaternion).invert();
      wRel.copy(wBall).sub(wBody).applyQuaternion(qInv);                // ball rate relative to the body, in the body frame
      for (const wh of wheels) {
        // no slip along the drive tangent at the contact: wheel rate = -(R / rw) (w_rel x n) . t / ((axis x n) . t)
        wh.angle = (wh.angle + wh.gain * tmp.crossVectors(wRel, wh.n).dot(wh.t) * dt) % (2 * Math.PI);
        wh.spin.quaternion.setFromAxisAngle(wh.axis, wh.angle);
      }
    }
    prev.x = x; prev.z = z; prev.tx = tiltX; prev.tz = tiltZ; prev.ok = true;
  }
  const screenBase = glow.map((m) => m.color.clone());
  /** Face screen glow: level 0..1; an optional hex tints it (used as a short flash when the robot is pushed). */
  function setStatus(level, hex) {
    glow.forEach((m, i) => { m.emissiveIntensity = 0.18 + 0.3 * level; if (hex !== undefined) { if (hex === null) m.emissive.copy(screenBase[i]); else m.emissive.set(hex); } });
  }
  function setTheme(light) { for (const m of envMats) m.envMapIntensity = light ? 0.85 : 1.05; }
  function dispose() { root.traverse((o) => { if (o.geometry) o.geometry.dispose(); }); for (const m of envMats) m.dispose(); }
  setTheme(false);
  return { root, rig, tilt, ballSpin, wheels, setPose, setStatus, setTheme, dispose, meta: model.meta, drawCalls: () => { let k = 0; root.traverse((o) => { if (o.isMesh) k++; }); return k; } };
}

async function init() {
  if (!host) return;
  if (!webglOk()) { noWebgl(); return; }
  // lazy: wait until the hero is (nearly) on screen before downloading anything or touching the GPU
  await new Promise((res) => {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); res(); } }, { rootMargin: '200px' });
    io.observe(host);
  });
  const model = await loadModel();                 // first: without the model there is nothing to draw, so keep the poster and skip three.js
  const [THREE, { OrbitControls }, { mergeGeometries }, R3, Model] = await Promise.all([
    import('three'), import('three/addons/controls/OrbitControls.js'), import('three/addons/utils/BufferGeometryUtils.js'), import('./robot3d.js'), import('./ballbot-model.js'),
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

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 40);
  const stage = R3.createStage(THREE, renderer, scene, { quality: level });      // lights, floor and contact shadow only; the robot comes from the JSON
  const bot = buildRobot(THREE, mergeGeometries, model, level);
  scene.add(bot.root);
  const meta = model.meta, RB = meta.ball_radius, H = meta.height || 0.6, size = H / 0.6;
  stage.contact.scale.setScalar(RB / 0.09);

  host.appendChild(renderer.domElement);
  const cv = renderer.domElement;
  cv.setAttribute('role', 'img');
  cv.setAttribute('aria-label', `Animated 3D view of the simulated robot, drawn from the project's current design model: a box-shaped body with a face screen and two arms, carried by ${meta.wheel_count} double omni wheels on a single dark ball, with curved guard arms around the ball. Drag to orbit, click the robot to push it.`);
  cv.style.touchAction = 'pan-y';

  const controls = new OrbitControls(camera, cv);
  controls.target.set(0, H * 0.44, 0);
  controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.minPolarAngle = 0.5; controls.maxPolarAngle = Math.PI / 2 - 0.04;
  controls.minDistance = 1.0 * size; controls.maxDistance = 3.4 * size;
  controls.rotateSpeed = 0.7;
  controls.autoRotateSpeed = 0.45;                 // slow orbit: about 2 minutes per turn (2.0 is 30 s per turn)
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
    if (w > 900) camera.setViewOffset(w, h, -w * (w < 1200 ? 0.23 : 0.17), h * 0.06, w, h);
    else if (portrait) camera.clearViewOffset();
    else camera.setViewOffset(w, h, -w * 0.24, -h * 0.02, w, h);
    camera.fov = portrait ? 38 : 30;
    if (!resize.done) {
      resize.done = true;
      const d = (portrait ? 1.3 : (w > 900 ? 1.95 : 2.2)) * size;
      camera.position.set(0.82, 0.36, 0.72).normalize().multiplyScalar(d).add(controls.target);   // front quarter: the face (+x) is toward the viewer
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
  const state = { x: 0, z: 0, tiltX: 0, tiltZ: 0 };

  function stepSim(dt) {
    if (!sx) {      // no model data: derived lean law theta = kappa a with a scripted acceleration (still honest about it)
      tSim += dt; const a = 0.4 * Math.sin(1.4 * tSim);
      state.x = -(0.4 / 1.96) * Math.sin(1.4 * tSim); state.z = 0; state.tiltX = kappa * a; state.tiltZ = 0; aNow = a; return;
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
    state.x = sx.x; state.z = sz.x; state.tiltX = sx.theta; state.tiltZ = sz.theta;
  }
  const settled = () => !sx || (Math.abs(sx.s[1]) + Math.abs(sz.s[1]) + Math.abs(sx.v) + Math.abs(sz.v) < 2e-3 && Math.abs(sx.x) + Math.abs(sz.x) < 5e-3);

  let hudText = '';
  function pose(dt) {
    bot.setPose(state, dt);
    stage.followBall(bot);
    if (hud) {
      const s = `a = ${aNow.toFixed(2).padStart(5)} m/s²   lean = ${(state.tiltX * 180 / Math.PI).toFixed(1).padStart(5)}°   (kappa ${(kappa * 180 / Math.PI).toFixed(1)}° per m/s²)`;
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
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());   // arrow points along dir, tip at the point
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
    bot.setStatus(1, 0xffb347); setTimeout(() => bot.setStatus(0.55, null), 900);
    showArrow(point || new THREE.Vector3(bot.rig.position.x, H * 0.55, bot.rig.position.z).addScaledVector(d, -0.16), d);
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

  // ---- injected controls (the page markup is owned elsewhere): push button, play / pause, honest caption
  injectStyles();
  const wrap = document.createElement('div'); wrap.className = 'hero3d-wrap';
  const cm = (v) => (v * 100).toFixed(1).replace(/\.0$/, '');
  wrap.innerHTML = `<div class="hero3d-tools"><button type="button" id="hero-push"><span class="dot" aria-hidden="true"></span>Push the robot</button></div><p class="hero3d-cap"></p>`;
  wrap.querySelector('.hero3d-cap').textContent = `The current design, drawn from the project's generated model (ball radius ${cm(RB)} cm, ${meta.wheel_count} wheels, model of ${meta.generated || 'unknown date'}). Simulation only.`;
  const ui = document.querySelector('.hero-ui');
  const note = document.querySelector('.hero-note'); if (note) wrap.prepend(note);   // one tidy right-hand column: note, buttons, caption
  if (ui) ui.appendChild(wrap); else { wrap.classList.add('hero3d-floating'); host.parentElement.appendChild(wrap); }
  wrap.querySelector('#hero-push').addEventListener('click', () => {
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion); right.y = 0; right.normalize();
    push(right.multiplyScalar(Math.sin(performance.now() / 1000) > 0 ? 1 : -1), 0.3);
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

  // ---- loop: renders only while on screen and the tab is visible; a settled scene renders on demand
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0 }).observe(host);
  let frozen = false;                                          // debugging and poster capture: stops the loop
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden || frozen) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const active = playing || now < pokeUntil || !settled();
    if (active || dirty) { stepSim(active ? dt : 0); pose(dt); }
    if (arrow.visible) { arrowT -= dt; arrowMat.opacity = Math.max(0, arrowT / 0.7); if (arrowT <= 0) arrow.visible = false; }
    if (playing && now > pokeUntil) bot.setStatus(0.5 + 0.12 * Math.sin(now / 900));
    const moved = controls.update();
    if (active || dirty || moved || arrow.visible) { renderer.render(scene, camera); dirty = false; if (playing) watchPerf(dt, now); }
  }
  applyTheme(); resize(); stepSim(0); pose(0);
  renderer.render(scene, camera);
  document.documentElement.classList.add('webgl-ready');
  if (fallback) fallback.hidden = true;
  requestAnimationFrame(frame);
  window.__hero3d = {
    renderer, scene, camera, bot, stage, controls, THREE, model,
    poke: () => push(new THREE.Vector3(1, 0, 0), 0.3),
    setLevel: applyLevel, perf, freeze: (f = true) => { frozen = f; }, get level() { return level; },
  };
}

init().catch((e) => { console.warn('hero3d: the live robot is unavailable, showing the static picture', e); noWebgl(); });
