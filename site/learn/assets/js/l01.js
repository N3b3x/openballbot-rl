// Lesson 1 figures: the 3D robot you can push, and the fall race (ballbot against the same body on a fixed pivot).
import { paramsFrom, linearize, eig, zeroNMP, pivotPole, PlanarBot, dlqr, planeTorqueMax } from './physics.js';
import { el, canvas, chart, slider, button, readouts, legend, clock, palette, drawBot, floor, fmt, seg } from './ui.js';

const DEG = Math.PI / 180;
const unstable = (P) => Math.max(...eig(linearize(P).A).map((z) => z[0]));

// ------------------------------------------------------------------------------------------- 3D robot
export async function robot3d(root, ctx) {
  const stage = root.querySelector('.stage');
  const body = root.querySelector('.body');
  const P = paramsFrom(ctx.data.model);
  const { A, B } = linearize(P);
  const K = dlqr(A, B, ctx.data.lqr.q, ctx.data.lqr.r, 0.01);
  const tauMax = planeTorqueMax(ctx.data.model.n, 30);
  const test = document.createElement('canvas');
  if (!(test.getContext('webgl2') || test.getContext('webgl'))) {
    stage.append(el('p', { class: 'status warn', text: 'WebGL is not available in this browser, so the 3D view is off. The 2D figure below shows the same physics.' }));
    return;
  }
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  // The shared, realistic robot model of the main site (read-only reuse). If it cannot be loaded, the figure falls back to a simple
  // pendulum drawn here from the same planar-model dimensions.
  let R3 = null;
  try { R3 = await import('../../../assets/js/robot3d.js'); } catch (e) { R3 = null; }
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  stage.append(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(32, 1, 0.02, 30);
  cam.position.set(0.95, 0.62, 1.25);
  const controls = new OrbitControls(cam, renderer.domElement);
  renderer.domElement.setAttribute('role', 'img'); renderer.domElement.setAttribute('aria-label', '3D view of the robot; drag to orbit. Tilt, position and torque are in the read-outs below.');
  renderer.domElement.style.touchAction = 'pan-y'; // horizontal drag orbits; vertical drag still scrolls the page
  controls.target.set(0, 0.22, 0);
  controls.enableDamping = !ctx.reduced;
  controls.minDistance = 0.6; controls.maxDistance = 4; controls.maxPolarAngle = Math.PI * 0.49;
  const comMesh = new THREE.Mesh(new THREE.SphereGeometry(0.016, 24, 16), new THREE.MeshStandardMaterial({ color: 0xffb454, emissive: 0x884400, depthTest: false }));
  comMesh.renderOrder = 5;
  const arrowObj = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-0.35, P.l, 0), 0.25, 0xff6b6b, 0.07, 0.045);
  arrowObj.visible = false; scene.add(arrowObj);
  let place3 = null, theme3 = null;
  if (R3) {
    const st3 = R3.createStage(THREE, renderer, scene, { quality: 1 });
    const rb = R3.createBallbot(THREE, { quality: 1 });
    scene.add(rb.root);
    rb.tilt.add(comMesh); comMesh.position.set(0, P.l, 0);
    place3 = (x, th, dt) => { rb.setPose({ x, z: 0, tiltX: th, tiltZ: 0 }, dt); st3.followBall(rb); };
    theme3 = (pal) => { const light = document.documentElement.dataset.theme === 'light'; st3.setTheme(light, pal.bg2); rb.setTheme(light); };
  } else {
    scene.add(new THREE.HemisphereLight(0xdfeaff, 0x303848, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1.2, 2.4, 1.1); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.1, far: 6 });
    scene.add(sun);
    const floorMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
    const fl = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), floorMat);
    fl.rotation.x = -Math.PI / 2; fl.receiveShadow = true; scene.add(fl);
    let grid = null;
    const tc = document.createElement('canvas'); tc.width = 256; tc.height = 128;
    const tg = tc.getContext('2d');
    const ballTex = new THREE.CanvasTexture(tc); ballTex.colorSpace = THREE.SRGBColorSpace;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(P.r, 48, 32), new THREE.MeshStandardMaterial({ map: ballTex, roughness: 0.55 }));
    ball.castShadow = true; scene.add(ball);
    const bodyG = new THREE.Group(); scene.add(bodyG);
    const towerMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.15, transparent: true, opacity: 0.55 });
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.28, 48), towerMat);
    tower.position.y = 0.32; tower.castShadow = true; bodyG.add(tower);
    const deckMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.3 });
    const deck = new THREE.Mesh(new THREE.CylinderGeometry(0.115, 0.1, 0.03, 48), deckMat);
    deck.position.y = 0.165; bodyG.add(deck);
    bodyG.add(comMesh); comMesh.position.y = P.l;
    const wheelMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.2 });
    const zen = (ctx.data.model.params.zenith_deg || 51) * DEG;
    for (let k = 0; k < 3; k++) {
      const az = (k * 2 * Math.PI) / 3 + 0.3;
      const u = new THREE.Vector3(Math.sin(zen) * Math.cos(az), Math.cos(zen), Math.sin(zen) * Math.sin(az));
      const w = new THREE.Mesh(new THREE.TorusGeometry(0.022, 0.009, 12, 28), wheelMat);
      w.position.copy(u.clone().multiplyScalar(P.r + 0.026));
      const tang = new THREE.Vector3(-Math.sin(az), 0, Math.cos(az));
      w.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3().crossVectors(u, tang).normalize());
      bodyG.add(w);
    }
    place3 = (x, th) => { ball.position.set(x, P.r, 0); ball.rotation.z = -x / P.r; bodyG.position.set(x, P.r, 0); bodyG.rotation.z = -th; };
    theme3 = (pal) => {
      scene.background = new THREE.Color(pal.bg2); floorMat.color = new THREE.Color(pal.bg3);
      if (grid) scene.remove(grid);
      grid = new THREE.GridHelper(8, 80, new THREE.Color(pal.line2), new THREE.Color(pal.line)); grid.position.y = 0.0005; scene.add(grid);
      towerMat.color = new THREE.Color(pal.model); deckMat.color = new THREE.Color(pal.text3); wheelMat.color = new THREE.Color(pal.hybrid);
      tg.fillStyle = pal.text2; tg.fillRect(0, 0, 256, 128); tg.fillStyle = pal.line2;
      for (let i = 0; i < 8; i += 2) tg.fillRect(i * 32, 0, 32, 128);
      tg.fillStyle = pal.warn; tg.fillRect(0, 60, 256, 8); ballTex.needsUpdate = true;
    };
  }
  function themeColours() { theme3(palette()); render(); }
  const bot = new PlanarBot(P, { K: null, tauMax });
  let mode = 'off', pushT = 0, pushF = 0, idle = 0;
  const ro = readouts(body, [['tilt', 'body tilt', 'warn'], ['x', 'ball position', 'model'], ['tau', 'motor torque (plane)', 'hybrid'], ['st', 'state', '']]);
  const btns = el('div', { class: 'btns' });
  body.append(btns);
  seg(btns, [['off', 'Motors off'], ['lqr', 'Balance on (LQR)']], 'off', (v) => { mode = v; bot.K = v === 'lqr' ? K : null; if (v === 'lqr' && bot.fallen) reset(); status(); run.start(); }, 'controller');
  button(btns, 'Push forward', () => push(1), 'primary');
  button(btns, 'Push back', () => push(-1));
  button(btns, 'Reset', () => reset());
  const st = el('p', { class: 'status' }); body.append(st);
  function status() {
    st.className = 'status ' + (bot.fallen ? 'bad' : mode === 'lqr' ? 'good' : '');
    st.textContent = bot.fallen ? 'Fallen: with the motors off nothing stops the tilt from growing. Reset and try Balance on.'
      : mode === 'lqr' ? 'Balancing: the LQR rolls the ball under the body. Push it.' : (Math.abs(bot.theta) > 0.002 ? 'Falling...' : 'Standing exactly upright: an unstable balance point. Push it.');
  }
  function push(dir) { if (bot.fallen) reset(); pushT = 0.1; pushF = 12 * dir; run.start(); }
  function reset() { bot.reset(mode === 'off' ? 0.0 : 0); pushT = 0; idle = 0; place(); status(); render(); }
  function place(dt = 0) {
    const x = bot.x, th = bot.theta;
    place3(x, th, dt);
    arrowObj.visible = pushT > 0;
    if (pushT > 0) {
      const dir = new THREE.Vector3(Math.sign(pushF), 0, 0);
      const cx = x + Math.sin(th) * P.l, cy = P.r + Math.cos(th) * P.l;
      arrowObj.setDirection(dir); arrowObj.position.set(cx - Math.sign(pushF) * 0.33, cy, 0);
    }
    ro.set('tilt', `${fmt(th / DEG, 1)} deg`); ro.set('x', `${fmt(x * 100, 1)} cm`); ro.set('tau', `${fmt(bot.tau, 1)} N m`);
    ro.set('st', bot.fallen ? 'fallen' : mode === 'lqr' ? 'balancing' : 'free');
  }
  function render() { renderer.render(scene, cam); }
  function size() {
    const w = Math.max(240, stage.clientWidth), h = Math.round(w < 560 ? Math.min(380, w * 1.0) : Math.min(460, Math.max(280, w * 0.6)));
    renderer.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix();
    const k = w < 560 ? 0.8 : 1; cam.position.set(bot.x + 0.95 * k, 0.55 * k, 1.25 * k); controls.target.set(bot.x, 0.2, 0); controls.update(); render();
  }
  new ResizeObserver(size).observe(stage);
  controls.addEventListener('change', render);
  let acc = 0;
  const run = clock(stage, (dt) => {
    acc += dt;
    while (acc >= bot.dt) {
      acc -= bot.dt;
      const F = pushT > 0 ? pushF : 0;
      if (pushT > 0) pushT -= bot.dt;
      bot.tick(0, F);
    }
    place(dt); status();
    if (controls.enableDamping) controls.update();
    // keep the robot in view
    const dx = (bot.x - controls.target.x) * 0.05; controls.target.x += dx; cam.position.x += dx;
    render();
    const calm = bot.fallen || (Math.abs(bot.theta) < 1e-4 && Math.abs(bot.s[3]) < 1e-4 && pushT <= 0 && Math.abs(bot.vball) < 1e-3);
    idle = calm ? idle + dt : 0;
    return idle < 1.5;
  });
  ctx.onTheme(themeColours);
  themeColours(); size(); place(); status();
}


// ------------------------------------------------------------------------------------------- fall race
export function fall(root, ctx) {
  const stage = root.querySelector('.stage');
  const body = root.querySelector('.body');
  const P0 = paramsFrom(ctx.data.model);
  const sweep = ctx.data.sweep.groups.find((g) => g.vary === 'body_com_height_m');
  let l = P0.l, th0 = 2 * DEG, t = 0, run = false, bot, piv;
  const P = () => ({ ...P0, l });
  const pivAcc = (Pp, th) => (Pp.m_a * 9.81 * Pp.l * Math.sin(th)) / (Pp.I_a + Pp.m_a * Pp.l * Pp.l);
  const trace = { a: [], b: [] };
  function reset() {
    bot = new PlanarBot(P(), { K: null, theta0: th0 });
    piv = { th: th0, w: 0, fallen: false };
    t = 0; trace.a = [[0, th0 / DEG]]; trace.b = [[0, th0 / DEG]];
  }
  function stepSim(dt) {
    const n = Math.max(1, Math.round(dt / 0.01));
    for (let i = 0; i < n; i++) {
      bot.tick(0, 0);
      if (!piv.fallen) {
        const Pp = P(); const h = 0.001;
        for (let k = 0; k < 10; k++) { piv.w += pivAcc(Pp, piv.th) * h; piv.th += piv.w * h; }
        if (piv.th > 75 * DEG) { piv.th = 75 * DEG; piv.fallen = true; }
      }
      t += 0.01;
      trace.a.push([t, Math.min(75, bot.theta / DEG)]); trace.b.push([t, piv.th / DEG]);
    }
  }
  reset();
  const cv = canvas(stage, (w) => Math.min(300, Math.max(200, w * 0.4)), (g, w, h) => {
    const pal = palette();
    g.clearRect(0, 0, w, h);
    const S = Math.min(w / 2.4, h / 0.75) * 0.9, fy = h - 28;
    const fam = getComputedStyle(document.body).fontFamily;
    g.fillStyle = pal.text2; g.font = '600 12px ' + fam;
    g.fillText('Ballbot (ball free)', 10, 16);
    g.fillText('Body on a fixed pivot', w / 2 + 10, 16);
    for (const cx of [w * 0.25, w * 0.75]) { g.strokeStyle = pal.line2; g.beginPath(); g.moveTo(cx - w * 0.22, fy); g.lineTo(cx + w * 0.22, fy); g.stroke(); }
    drawBot(g, 0 * bot.x, bot.theta, { pal, scale: S, ox: w * 0.25, oy: fy, l, color: pal.model });
    drawBot(g, 0, piv.th, { pal, scale: S, ox: w * 0.75, oy: fy, l, color: pal.learned });
    g.fillStyle = pal.text3; g.font = '11px ' + fam;
    g.fillText(`t = ${t.toFixed(2)} s`, 10, 32);
  }, ctx);
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Side view: a ballbot with its motors off and the same body on a fixed pivot, released from the same small tilt. A chart below plots both tilts.');
  legend(body, [['--model', 'ballbot tilt'], ['--learned', 'fixed-pivot tilt']]);
  const tc = chart(body, { x: [0, 1.5], y: [0, 75], xlabel: 'time (s)', ylabel: 'tilt (deg)', height: 190, crosshair: true, series: [], desc: 'Tilt against time for both bodies.' });
  const ro = readouts(body, [['p', 'ballbot unstable pole', 'model'], ['pv', 'fixed-pivot pole', 'learned'], ['dbl', 'tilt doubles every', 'warn'], ['z', 'NMP zero (lesson 2)', '']]);
  const ctl = el('div', { class: 'ctls' }); body.append(ctl);
  slider(ctl, { label: 'Body centre of mass above ball centre', min: 0.06, max: 0.3, step: 0.005, value: l, fmt: (v) => `${(v * 100).toFixed(1)} cm`, onInput: (v) => { l = v; reset(); upd(); } });
  slider(ctl, { label: 'Initial tilt', min: 0.5, max: 5, step: 0.5, value: 2, fmt: (v) => `${v} deg`, onInput: (v) => { th0 = v * DEG; reset(); upd(); } });
  const bt = el('div', { class: 'btns' }); body.append(bt);
  const rel = button(bt, 'Release both', () => {
    reset();
    if (ctx.reduced) { while (t < 1.5 && !(bot.fallen && piv.fallen)) stepSim(0.01); upd(); } else { run = true; clk.start(); }
  }, 'primary');
  button(bt, 'Reset', () => { run = false; clk.stop(); reset(); upd(); });
  body.append(el('p', { class: 'hint', text: 'Lines: the planar model as the centre of mass moves, body inertia held at its nominal value. Dots: the S119 design-sweep variants, where moving the ballast also changes that inertia.' }));
  legend(body, [['--model', 'ballbot pole p'], ['--learned', 'fixed-pivot pole'], ['--warn', 'zero z', true]]);
  const ls = []; for (let v = 0.06; v <= 0.3001; v += 0.005) ls.push(v);
  const curve = (f) => ls.map((v) => [v * 100, f({ ...P0, l: v })]);
  const pc = chart(body, {
    x: [6, 30], y: [3, 9], xlabel: 'COM height above ball centre (cm)', ylabel: '1/s', height: 200,
    series: [{ data: curve(unstable), color: '--model', label: 'p' }, { data: curve(pivotPole), color: '--learned', label: 'pivot' }, { data: curve(zeroNMP), color: '--warn', dash: '5 4', label: 'z' }],
    dots: sweep.rows.flatMap((r) => [{ x: r.phys.l * 100, y: r.phys.p, color: '--model', label: `${r.nominal ? 'nominal' : r.name}: l ${(r.phys.l * 100).toFixed(1)} cm, p ${r.phys.p.toFixed(2)} 1/s` }, { x: r.phys.l * 100, y: r.phys.z, color: '--warn', hollow: true, label: `${r.nominal ? 'nominal' : r.name}: z ${r.phys.z.toFixed(2)} 1/s` }]),
    desc: 'Unstable pole, fixed-pivot pole and non-minimum-phase zero against centre-of-mass height.',
  });
  function upd() {
    const Pp = P(), p = unstable(Pp);
    ro.set('p', `${p.toFixed(2)} 1/s`); ro.set('pv', `${pivotPole(Pp).toFixed(2)} 1/s`); ro.set('dbl', `${(Math.LN2 / p).toFixed(3)} s`); ro.set('z', `${zeroNMP(Pp).toFixed(2)} 1/s`);
    tc.update({ series: [{ data: trace.a, color: '--model', label: 'ballbot' }, { data: trace.b, color: '--learned', label: 'pivot' }] });
    pc.update({ vlines: [{ x: l * 100, color: '--text-3', label: `${(l * 100).toFixed(1)} cm` }] });
    cv.redraw();
  }
  const clk = clock(stage, (dt) => {
    if (!run) return false;
    stepSim(dt); cv.redraw();
    tc.update({ series: [{ data: trace.a, color: '--model', label: 'ballbot' }, { data: trace.b, color: '--learned', label: 'pivot' }] });
    if (t > 1.5 || (bot.fallen && piv.fallen)) { run = false; return false; }
  });
  reset(); upd();
}
