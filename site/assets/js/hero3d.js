// Procedural 3-D ballbot for the landing hero. ILLUSTRATION ONLY: the motion is a scripted sinusoid plus a damped
// oscillator for the poke; it is not a physics simulation. Dimensions come from ballbot_gym/models/ballbot.xml
// (ball radius 0.09 m, body cylinder radius 0.11 m / height 0.28 m, wheel contacts at zenith ~51 deg, azimuth 0/120/240).
// The idle lean follows the derived lean law theta = kappa * a, kappa = 0.176 rad per m/s^2 (paper_v2/main.tex).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const KAPPA = 0.176;          // rad per m/s^2 (10.1 deg)
const R_BALL = 0.09, R_BODY = 0.11, H_BODY = 0.28, ZENITH = 51 * Math.PI / 180;
const BODY_BOTTOM = 0.20;     // body cylinder bottom above ball centre (xml: frame +0.14, geom centre +0.2, half-height 0.14 -> 0.06+0.14)

const host = document.getElementById('hero-canvas');
const fallback = document.getElementById('hero-fallback');
const hud = document.getElementById('hero-hud');
const playBtn = document.getElementById('hero-play');
const toast = document.getElementById('hero-toast');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function webglOk() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  } catch (e) { return false; }
}

function init() {
  if (!host || !webglOk()) { document.documentElement.classList.add('no-webgl'); return; }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  } catch (e) { document.documentElement.classList.add('no-webgl'); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D illustration of a ballbot balancing on a basketball-textured ball. Drag to orbit, click the robot to poke it.');

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 40);
  camera.position.set(0.95, 0.62, 1.15);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.22, 0);
  controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minPolarAngle = 0.35; controls.maxPolarAngle = Math.PI / 2 - 0.03;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.7;
  renderer.domElement.style.touchAction = 'pan-y';

  // ---- lights
  scene.add(new THREE.HemisphereLight(0xdfeaff, 0x30384a, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(1.4, 2.4, 1.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.5, far: 6 });
  sun.shadow.radius = 5; sun.shadow.bias = -0.0004;
  scene.add(sun);

  // ---- floor (checker, like the simulator's floor_chk material)
  const checker = document.createElement('canvas'); checker.width = checker.height = 256;
  const floorMat = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0 });
  function paintChecker() {
    const g = checker.getContext('2d');
    const dark = cssVar('--theme-floor-a') || '#1b2433', light = cssVar('--theme-floor-b') || '#27344a';
    g.fillStyle = dark; g.fillRect(0, 0, 256, 256);
    g.fillStyle = light; g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
    if (floorMat.map) floorMat.map.needsUpdate = true;
  }
  const floorTex = new THREE.CanvasTexture(checker);
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.repeat.set(30, 30);
  floorTex.colorSpace = THREE.SRGBColorSpace; floorTex.anisotropy = 4;
  floorMat.map = floorTex;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), floorMat);
  floor.rotation.x = -Math.PI / 2; floor.position.y = 0; floor.receiveShadow = true;
  scene.add(floor);

  // ---- basketball texture
  const bt = document.createElement('canvas'); bt.width = 1024; bt.height = 512;
  {
    const g = bt.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, '#d9611c'); grad.addColorStop(0.5, '#ee7a2a'); grad.addColorStop(1, '#cf5718');
    g.fillStyle = grad; g.fillRect(0, 0, 1024, 512);
    let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;     // deterministic speckle
    for (let i = 0; i < 9000; i++) { g.fillStyle = `rgba(${rnd() < 0.5 ? '90,30,5' : '255,170,100'},${0.05 + rnd() * 0.12})`; g.fillRect(rnd() * 1024, rnd() * 512, 2, 2); }
    g.strokeStyle = '#1a0d06'; g.lineWidth = 9; g.lineCap = 'round';
    g.beginPath(); g.moveTo(0, 256); g.lineTo(1024, 256); g.stroke();                       // equator
    for (const u of [0, 512, 1024]) { g.beginPath(); g.moveTo(u, 0); g.lineTo(u, 512); g.stroke(); }   // meridians
    for (const c of [256, 768]) {                                                          // curved side channels
      for (const sgn of [-1, 1]) {
        g.beginPath();
        for (let v = 0; v <= 512; v += 8) { const tt = v / 512 * 2 - 1; const x = c + sgn * 110 * (1 - tt * tt); if (v === 0) g.moveTo(x, v); else g.lineTo(x, v); }
        g.stroke();
      }
    }
  }
  const ballTex = new THREE.CanvasTexture(bt); ballTex.colorSpace = THREE.SRGBColorSpace; ballTex.anisotropy = 4;
  const ballMat = new THREE.MeshStandardMaterial({ map: ballTex, bumpMap: ballTex, bumpScale: 0.6, roughness: 0.7 });
  const ball = new THREE.Mesh(new THREE.SphereGeometry(R_BALL, 64, 40), ballMat);
  ball.castShadow = true;

  // ---- body (tilts about ball centre)
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xdfe6f1, roughness: 0.45, metalness: 0.25 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x20262f, roughness: 0.5, metalness: 0.5 });
  const ringMat = new THREE.MeshStandardMaterial({ color: 0x2ee6c4, emissive: 0x2ee6c4, emissiveIntensity: 0.6, roughness: 0.3 });
  const tilt = new THREE.Group();
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(R_BODY, R_BODY, H_BODY, 56), bodyMat);
  tower.position.y = BODY_BOTTOM + H_BODY / 2; tower.castShadow = true; tilt.add(tower);
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(R_BODY * 0.96, R_BODY, 0.02, 56), darkMat);
  cap.position.y = BODY_BOTTOM + H_BODY + 0.01; cap.castShadow = true; tilt.add(cap);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(R_BODY + 0.001, 0.0045, 12, 72), ringMat);
  ring.rotation.x = Math.PI / 2; ring.position.y = BODY_BOTTOM + H_BODY * 0.62; tilt.add(ring);
  const eye = new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 12), darkMat);   // sensor window
  eye.position.set(R_BODY - 0.002, BODY_BOTTOM + H_BODY * 0.8, 0); eye.scale.set(0.6, 1, 1); tilt.add(eye);
  const skirt = new THREE.Mesh(new THREE.CylinderGeometry(R_BODY, 0.07, 0.03, 48), darkMat);
  skirt.position.y = BODY_BOTTOM - 0.015; skirt.castShadow = true; tilt.add(skirt);

  // three omniwheels at the real contact geometry: contact point p = R*(sin z cos a, cos z, sin z sin a) (y up)
  const wheelR = 0.025, wheelW = 0.02;
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x3a4250, roughness: 0.4, metalness: 0.6 });
  const rollerMat = new THREE.MeshStandardMaterial({ color: 0xc9d2e0, roughness: 0.35, metalness: 0.3 });
  [0, 120, 240].forEach((azDeg) => {
    const az = azDeg * Math.PI / 180;
    const n = new THREE.Vector3(Math.sin(ZENITH) * Math.cos(az), Math.cos(ZENITH), Math.sin(ZENITH) * Math.sin(az));  // outward normal
    const centre = n.clone().multiplyScalar(R_BALL + wheelR);
    const w = new THREE.Group();
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(wheelR, wheelR, wheelW, 28), wheelMat);
    w.add(hub);
    for (let k = 0; k < 10; k++) {                           // small rollers around the rim
      const th = k / 10 * Math.PI * 2;
      const rl = new THREE.Mesh(new THREE.CapsuleGeometry(0.0055, 0.012, 4, 8), rollerMat);
      rl.position.set(Math.cos(th) * wheelR, 0, Math.sin(th) * wheelR); rl.rotation.set(0, -th, Math.PI / 2); rl.rotateY(0);
      w.add(rl);
    }
    // wheel spin axis is tangent (azimuthal direction), cylinder axis is local y -> align y with tangent
    const tangent = new THREE.Vector3(-Math.sin(az), 0, Math.cos(az));
    w.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
    w.position.copy(centre);
    w.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    tilt.add(w);
    // strut from wheel assembly to the body skirt
    const top = new THREE.Vector3(Math.cos(az) * 0.07, BODY_BOTTOM - 0.02, Math.sin(az) * 0.07);
    const dir = top.clone().sub(centre); const len = dir.length();
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, len, 10), darkMat);
    strut.position.copy(centre).addScaledVector(dir, 0.5);
    strut.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    strut.castShadow = true; tilt.add(strut);
  });

  const rig = new THREE.Group();     // translates with the ball
  const ballSpin = new THREE.Group(); ballSpin.add(ball);
  rig.add(ballSpin); rig.add(tilt);
  scene.add(rig);
  rig.position.y = R_BALL;

  // contact shadow blob under the ball (cheap, keeps the contact grounded)
  const blob = new THREE.Mesh(new THREE.CircleGeometry(0.13, 32), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2; blob.position.y = 0.002; scene.add(blob);

  // ---- theme sync
  function applyTheme() {
    const light = document.documentElement.dataset.theme === 'light';
    const bg = new THREE.Color(cssVar('--bg') || '#0a0e14');
    scene.background = bg; scene.fog = new THREE.Fog(bg, 2.2, 6.0);
    document.documentElement.style.setProperty('--theme-floor-a', light ? '#dfe6f0' : '#18202d');
    document.documentElement.style.setProperty('--theme-floor-b', light ? '#f4f7fb' : '#263349');
    paintChecker(); floorTex.needsUpdate = true;
    bodyMat.color.set(light ? 0xf4f6fa : 0xdfe6f1);
    renderer.toneMappingExposure = light ? 0.95 : 1.05;
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
    if (w > 900) camera.setViewOffset(w, h, -w * 0.16, 0, w, h);
    else if (portrait) camera.clearViewOffset();
    else camera.setViewOffset(w, h, -w * 0.18, -h * 0.1, w, h);
    camera.fov = portrait ? 40 : 32;
    if (!resize.done) { resize.done = true; const d = portrait ? 2.0 : (w > 900 ? 1.75 : 2.25); camera.position.set(0.55, 0.5, 1).normalize().multiplyScalar(d).add(controls.target); }
    camera.updateProjectionMatrix();
    dirty = true;
  }
  new ResizeObserver(resize).observe(host);

  // ---- state: idle sinusoid + poke oscillator
  const A = 0.5, W = 1.5;                       // commanded acceleration amplitude (m/s^2) and angular frequency (rad/s)
  const poke = { l: new THREE.Vector2(), v: new THREE.Vector2(), p: new THREE.Vector2(), pv: new THREE.Vector2() };
  let playing = !reduced, t = 0, last = performance.now(), visible = true, pokeUntil = 0;
  const idle = (tt) => {
    const a = A * Math.sin(W * tt), a2 = 0.12 * Math.sin(0.7 * W * tt + 1.2);
    return { a, lx: KAPPA * a, ly: KAPPA * a2,
             px: -(A / (W * W)) * Math.sin(W * tt), pz: -(0.12 / ((0.7 * W) ** 2)) * Math.sin(0.7 * W * tt + 1.2) };
  };
  let ballRoll = new THREE.Vector2();

  function pose(tt, dt) {
    const s = playing ? idle(tt) : idle(0);
    // poke: lean oscillator (underdamped, zeta ~0.35, w0 ~ 7 rad/s: stands in for the closed loop recovering)
    const w0 = 7, z = 0.35;
    for (const k of ['x', 'y']) {
      const acc = -w0 * w0 * poke.l[k] - 2 * z * w0 * poke.v[k];
      poke.v[k] += acc * dt; poke.l[k] += poke.v[k] * dt;
      // ball shifts under the leaning body, then returns
      const acc2 = -((w0 * 0.55) ** 2) * poke.p[k] - 2 * 0.6 * w0 * 0.55 * poke.pv[k] + 0.9 * poke.l[k];
      poke.pv[k] += acc2 * dt; poke.p[k] += poke.pv[k] * dt;
    }
    const lx = s.lx + poke.l.x, lz = s.ly + poke.l.y;
    const px = s.px + poke.p.x * 0.35, pz = s.pz + poke.p.y * 0.35;
    rig.position.x = px; rig.position.z = pz;
    blob.position.x = px; blob.position.z = pz;
    tilt.rotation.set(lz, 0, -lx, 'XYZ');       // lean toward +x = rotate about -z in a y-up frame; +z lean = about +x
    ballSpin.rotation.set(pz / R_BALL, 0, -px / R_BALL, 'XYZ');
    if (hud) {
      const aTot = s.a + (poke.l.x / KAPPA);
      hud.textContent = `a = ${aTot.toFixed(2).padStart(5)} m/s²   lean = ${(lx * 180 / Math.PI).toFixed(1).padStart(5)}°   (kappa ${(KAPPA * 180 / Math.PI).toFixed(1)}° per m/s²)`;
    }
  }

  // ---- poke interaction (click without dragging)
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; controls.autoRotate = false; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
    down = null;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(rig, true)[0];
    if (!hit) return;
    if (reduced && !playing) { say('Motion is reduced on this device. Press Play to enable the poke illustration.'); return; }
    const d = ray.ray.direction.clone(); d.y = 0; d.normalize();                  // push along the ray, horizontally
    const mag = 1.0 + 0.8 * Math.min(1, hit.point.y / 0.5);
    poke.v.x += d.x * mag; poke.v.y += d.z * mag;                                 // lean x = +x, lean y = +z
    say('Poked. This wobble-and-recover is an illustration of the idea, not a physics simulation.');
    pokeUntil = performance.now() + 2500; dirty = true;
  });
  let toastTimer;
  function say(msg) { if (!toast) return; toast.textContent = msg; toast.classList.add('on'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('on'), 3200); }

  if (playBtn) {
    playBtn.hidden = !reduced;
    playBtn.addEventListener('click', () => { playing = !playing; controls.autoRotate = playing; playBtn.textContent = playing ? 'Pause animation' : 'Play animation'; dirty = true; });
    if (reduced) playBtn.textContent = 'Play animation';
  }

  // ---- loop (renders only when visible; static scene renders on demand)
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0 }).observe(host);
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const poking = now < pokeUntil || poke.l.lengthSq() > 1e-6 || poke.v.lengthSq() > 1e-6;
    if (playing) t += dt;
    if (playing || poking || dirty) { pose(t, dt); }
    const moved = controls.update();
    if (playing || poking || dirty || moved) { renderer.render(scene, camera); dirty = false; }
  }
  applyTheme(); resize(); pose(0, 0);
  renderer.render(scene, camera);
  document.documentElement.classList.add('webgl-ready');
  if (fallback) fallback.hidden = true;
  requestAnimationFrame(frame);
  window.__hero3d = { renderer, scene, poke: () => { poke.v.x += 1.4; pokeUntil = performance.now() + 2500; } };
}

try { init(); } catch (e) { console.warn('hero3d failed, showing poster', e); document.documentElement.classList.add('no-webgl'); if (fallback) fallback.hidden = false; }
