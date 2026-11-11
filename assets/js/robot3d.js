// Reusable 3-D ballbot for the site (landing hero, course pages). RENDERING OF THE SIMULATED NOMINAL DESIGN ONLY: no physical
// robot exists. Everything is procedural three.js geometry (lathes, rounded boxes, tubes, instanced rollers), no mesh download.
//
// Real dimensions (metres, from ballbot_gym/models/ballbot.xml and the nominal variant files; y is up here, the simulator's z):
//   ball radius 0.09 (basketball surface), tower cylinder radius 0.11, bottom 0.18 and top 0.46 above the ball centre
//   (ball body origin 0.02 above the base origin, ball geom 0.14 below it, so the base origin is 0.12 above the ball centre;
//   tower centre +0.2 above the base, half height 0.14),
//   three omniwheels of radius 0.025 touching the ball 51 deg from its top, 120 deg apart (the xml puts wheel 0 at 102.7 deg
//   azimuth in this frame), three passive ball-transfer arrestors 8 mm radius below the equator opposite the wheels
//   (ballbot_gym/models/variants/*nom.xml), two side ToF modules on stalks at +-0.17 m, 0.06 m above the ball centre (cam_0/1
//   bodies), head ToF 0.15 m above the base origin (0.27 above the ball centre) pitched -25 deg, 55 x 42 deg field of view (experiments/warp_ppo/train.py).
// Detail beyond those dimensions (motor frames, panel lines, fasteners) is an artist's plausible choice and is not measured.
//
// Usage:
//   import * as THREE from 'three';
//   import { createBallbot, createStage } from './robot3d.js';
//   const stage = createStage(THREE, renderer, scene, { quality: 2 });   // env map, lights, floor, contact shadow
//   const bot = createBallbot(THREE, { quality: 2 });                   // quality 0 low, 1 medium, 2 high
//   scene.add(bot.root);
//   bot.setPose({ x, z, tiltX, tiltZ }, dt);  // metres, radians (lean toward +x / +z); ball roll and wheel spin follow
//   stage.followBall(bot);  bot.setSensorVisible(true);
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const DEG = Math.PI / 180;

/** Real nominal dimensions (metres, degrees). Exported so a proportion check can compare against ballbot.xml. */
export const DIMS = {
  ballR: 0.09, towerR: 0.11, towerBottom: 0.18, towerTop: 0.46,
  wheelR: 0.025, zetaDeg: 51, wheelAz0Deg: 102.7,
  arrestorR: 0.008, arrestorAbove: -0.0818, arrestorHoriz: 0.0685,
  headUp: 0.27, headPitchDeg: -25, fovDeg: [55, 42],
  sideSensor: { lateral: 0.17, up: 0.06, pitchDeg: -25 },
};

export function createBallbot(THREE, opts = {}) {
  const q = opts.quality ?? 2;
  const SEG = [20, 28, 40][q];
  const { Vector3: V3, Matrix4, Quaternion, Group, Mesh, Color } = THREE;
  const Y = new V3(0, 1, 0), ONE = new V3(1, 1, 1);
  const { ballR: RB, towerR: RT, wheelR: RW } = DIMS;

  // ---------------------------------------------------------------- materials (PBR)
  const std = (o) => new THREE.MeshStandardMaterial(o);
  const phys = (o) => new THREE.MeshPhysicalMaterial(o);
  const M = {
    shell: phys({ color: 0xe8ebef, roughness: 0.42, metalness: 0, clearcoat: 0.45, clearcoatRoughness: 0.35 }),
    graphite: std({ color: 0x2c323b, roughness: 0.5, metalness: 0.08 }),
    anod: std({ color: 0x5b6470, roughness: 0.4, metalness: 0.85 }),
    alu: std({ color: 0xc9ced6, roughness: 0.27, metalness: 1 }),
    motor: std({ color: 0x8d949e, roughness: 0.34, metalness: 0.95 }),
    steel: std({ color: 0xa9aeb6, roughness: 0.3, metalness: 1 }),
    rubber: std({ color: 0x121316, roughness: 0.85, metalness: 0 }),
    roller: std({ color: 0x1f2227, roughness: 0.62, metalness: 0.1 }),
    glass: phys({ color: 0x04070a, roughness: 0.05, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.03, envMapIntensity: 1.6 }),
    red: std({ color: 0xb82229, roughness: 0.4, metalness: 0 }),
    label: std({ color: 0x28b8a0, roughness: 0.5, metalness: 0 }),
    ring: std({ color: 0x071d19, emissive: 0x2ee6c4, emissiveIntensity: 0.85, roughness: 0.3 }),
  };
  const envMats = Object.values(M);

  // ---------------------------------------------------------------- basketball textures (equirectangular, procedural)
  function ballTextures() {
    const W = [512, 1024, 2048][q], H = W / 2, S = W / 2048;
    const mk = () => { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; };
    const cc = mk(), cb = mk(), g = cc.getContext('2d'), b = cb.getContext('2d');
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const grad = g.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#e4681a'); grad.addColorStop(0.5, '#f58a2f'); grad.addColorStop(1, '#da6015');
    g.fillStyle = grad; g.fillRect(0, 0, W, H);
    b.fillStyle = '#b8b8b8'; b.fillRect(0, 0, W, H);
    // pebble grain: jittered rows, spacing in longitude widened toward the poles so dimples stay round on the sphere
    const pitch = 0.0032 / RB, rows = Math.round(Math.PI / pitch);
    for (let r = 0; r < rows; r++) {
      const lat = (r + 0.5) / rows * Math.PI - Math.PI / 2, ring = Math.max(6, Math.round(2 * Math.PI * Math.cos(lat) / pitch));
      const y = (r + 0.5) / rows * H, rx = (W / ring) * 0.36, ry = (H / rows) * 0.36;
      for (let i = 0; i < ring; i++) {
        const x = (i + 0.5 + (r % 2) * 0.5 + (rnd() - 0.5) * 0.3) / ring * W, yy = y + (rnd() - 0.5) * ry * 0.5;
        b.fillStyle = 'rgba(30,30,30,0.7)'; b.beginPath(); b.ellipse(x, yy, rx, ry, 0, 0, 7); b.fill();
        g.fillStyle = `rgba(120,40,6,${0.08 + rnd() * 0.08})`; g.beginPath(); g.ellipse(x, yy, rx * 0.9, ry * 0.9, 0, 0, 7); g.fill();
      }
    }
    // seams: equator, vertical great circle, two curved side channels (pole to pole)
    const lines = (ctx, w, col) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
      for (const u of [0, W / 2, W]) { ctx.beginPath(); ctx.moveTo(u, 0); ctx.lineTo(u, H); ctx.stroke(); }
      for (const c of [W * 0.25, W * 0.75]) for (const sg of [-1, 1]) {
        ctx.beginPath();
        for (let v = 0; v <= H; v += 4) { const t = v / H * 2 - 1, x = c + sg * W * 0.107 * (1 - t * t); v ? ctx.lineTo(x, v) : ctx.moveTo(x, v); }
        ctx.stroke();
      }
    };
    lines(g, 11 * S + 2, '#15100d'); lines(b, 11 * S + 2, '#202020');
    // soft overall mottling so the surface does not look printed
    for (let i = 0; i < 260; i++) { g.fillStyle = `rgba(${rnd() < 0.5 ? '70,20,0' : '255,160,90'},${0.025 + rnd() * 0.035})`; g.beginPath(); g.arc(rnd() * W, rnd() * H, (20 + rnd() * 70) * S, 0, 7); g.fill(); }
    const map = new THREE.CanvasTexture(cc); map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = [2, 4, 8][q];
    const bump = new THREE.CanvasTexture(cb); bump.anisotropy = [2, 4, 8][q];
    return { map, bump };
  }
  const bt = ballTextures();
  M.ball = std({ map: bt.map, bumpMap: bt.bump, bumpScale: [0.8, 1.4, 2.0][q], roughness: 0.68, metalness: 0 });
  envMats.push(M.ball);

  // ---------------------------------------------------------------- geometry helpers
  const mat = (px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0) =>
    new Matrix4().compose(new V3(px, py, pz), new Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), ONE);
  const basisM = (x, y, z, p) => new Matrix4().makeBasis(x, y, z).setPosition(p);
  const lathe = (pts, seg = SEG) => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), seg);

  class Bucket {
    constructor() { this.m = new Map(); }
    add(key, geo, m) { const g = geo.index ? geo.toNonIndexed() : geo.clone(); if (m) g.applyMatrix4(m); (this.m.get(key) || this.m.set(key, []).get(key)).push(g); }
    rod(key, a, b, r, seg = 10) {
      const d = b.clone().sub(a), len = d.length();
      this.add(key, new THREE.CylinderGeometry(r, r, len, seg, 1), new Matrix4().compose(a.clone().add(b).multiplyScalar(0.5), new Quaternion().setFromUnitVectors(Y, d.normalize()), ONE));
    }
    tube(key, pts, r, seg = 40, rad = 6) { this.add(key, new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), seg, r, rad, false)); }
    build(parent, shadow = true) {
      for (const [k, list] of this.m) {
        const mesh = new Mesh(mergeGeometries(list, false), M[k]);
        mesh.castShadow = shadow; mesh.receiveShadow = true; parent.add(mesh);
      }
    }
  }
  // radial fastener (hex head) on a vertical cylinder of radius r at azimuth phi, height y
  const screwGeo = new THREE.CylinderGeometry(0.0021, 0.0021, 0.0017, 6);
  function radialScrew(bk, phi, y, r, key = 'steel') {
    const rad = new V3(Math.cos(phi), 0, Math.sin(phi));
    bk.add(key, screwGeo, basisM(new V3(0, 1, 0), rad, new V3(Math.sin(phi), 0, -Math.cos(phi)), new V3(rad.x * r, y, rad.z * r)));
  }
  // cylindrical panel (open arc) at azimuth phi (0 = +x), span radians
  const arc = (phi, span, y0, y1, r, seg = 16) =>
    new THREE.CylinderGeometry(r, r, y1 - y0, seg, 1, true, Math.PI / 2 - phi - span / 2, span).translate(0, (y0 + y1) / 2, 0);

  // ---------------------------------------------------------------- scene graph
  const root = new Group(); root.name = 'ballbot';
  const rig = new Group(); root.add(rig);                    // translates with the ball centre
  const ballSpin = new Group(); rig.add(ballSpin);
  const tilt = new Group(); rig.add(tilt);                   // body frame, pivots about the ball centre
  rig.position.y = RB;

  const ball = new Mesh(new THREE.SphereGeometry(RB, [48, 72, 96][q], [32, 48, 64][q]), M.ball);
  ball.castShadow = true; ball.receiveShadow = true; ballSpin.add(ball);

  const B = new Bucket();
  const TOWER_DY = DIMS.towerBottom - 0.20;
  const tower = new Group(); tower.position.y = TOWER_DY; tilt.add(tower);

  // ---- tower shell -------------------------------------------------------------------------------------------------
  const yB = 0.20;                                          // tower parts are authored with the bottom at 0.20 and shifted by TOWER_DY
  const TB = new Bucket();
  B.add('anod', lathe([[0.000, 0.146], [0.050, 0.146], [0.060, 0.150], [0.085, 0.164], [0.100, 0.174], [0.104, 0.178], [0.104, DIMS.towerBottom]]));   // chassis hub
  TB.add('alu', lathe([[0.100, yB - 0.001], [0.1115, yB - 0.001], [0.1125, yB + 0.001], [0.1125, yB + 0.005], [0.1105, yB + 0.0066], [0.100, yB + 0.0066]]));   // base flange
  TB.add('graphite', lathe([[0.1085, 0.2066], [0.1100, 0.2085], [0.1100, 0.3165], [0.1090, 0.3185], [0.1060, 0.3185]]));                                  // battery bay band
  TB.add('shell', lathe([[0.1060, 0.3205], [0.1090, 0.3205], [0.1100, 0.3225], [0.1100, 0.4040], [0.1098, 0.4300], [0.1060, 0.4500], [0.1010, 0.4590], [0.0950, 0.4620]]));
  TB.add('alu', lathe([[0.0000, 0.4800], [0.0880, 0.4800], [0.0950, 0.4775], [0.0975, 0.4720], [0.0975, 0.4590], [0.0000, 0.4590]]));              // lid
  TB.add('graphite', lathe([[0.0952, 0.4605], [0.0990, 0.4605], [0.0990, 0.4660], [0.0952, 0.4660]]));                                                   // lid seal
  for (const y of [0.3195, 0.4045]) TB.add('graphite', new THREE.CylinderGeometry(0.1103, 0.1103, 0.0016, SEG, 1, true).translate(0, y, 0));            // panel gaps
  for (const ph of [30, 150, 270]) TB.add('graphite', arc(ph * DEG, 0.9 * DEG, 0.3225, 0.4035, RT + 0.0004, 3));                                       // vertical seams
  TB.add('graphite', arc(180 * DEG, 84 * DEG, 0.226, 0.309, RT + 0.0012, 20));                                                                          // battery cover
  TB.add('label', arc(180 * DEG, 56 * DEG, 0.262, 0.270, RT + 0.0019, 16));
  for (let i = -3; i <= 3; i++) TB.add('rubber', arc(180 * DEG + i * 6.5 * DEG, 2.2 * DEG, 0.279, 0.302, RT + 0.0019, 2));                              // vent slits
  const nScrew = 12;
  for (let i = 0; i < nScrew; i++) {
    radialScrew(TB, i / nScrew * Math.PI * 2, 0.2122, RT - 0.0001);
    radialScrew(TB, (i + 0.5) / nScrew * Math.PI * 2, 0.3095, RT + 0.0004);
  }
  for (let i = 0; i < 8; i++) radialScrew(TB, i / 8 * Math.PI * 2 + 0.2, 0.4560, 0.1058, 'steel');
  // lid: vent slots, e-stop, antenna
  for (let i = 0; i < 8; i++) { const a = i / 8 * Math.PI * 2 + Math.PI / 8; TB.add('rubber', new RoundedBoxGeometry(0.016, 0.0018, 0.0036, 1, 0.0008), mat(Math.cos(a) * 0.062, 0.4802, Math.sin(a) * 0.062, 0, -a, 0)); }
  TB.add('graphite', new THREE.CylinderGeometry(0.0125, 0.0135, 0.004, 24), mat(-0.045, 0.482, 0.04));
  TB.add('red', new THREE.CylinderGeometry(0.0085, 0.0095, 0.0085, 24).translate(0, 0.0045, 0), mat(-0.045, 0.484, 0.04));
  TB.add('anod', new THREE.CylinderGeometry(0.0034, 0.004, 0.034, 12), mat(0.04, 0.497, -0.045));
  TB.add('graphite', new THREE.SphereGeometry(0.0036, 12, 8), mat(0.04, 0.514, -0.045));
  TB.add('alu', new THREE.CylinderGeometry(0.0075, 0.0085, 0.004, 16), mat(0.04, 0.482, -0.045));
  // rear cable gland and a hanging cable bundle
  radialScrew(TB, Math.PI, 0.345, RT + 0.002, 'anod');
  TB.add('anod', new THREE.CylinderGeometry(0.0062, 0.0062, 0.008, 16), basisM(new V3(0, 1, 0), new V3(-1, 0, 0), new V3(0, 0, 1), new V3(-RT - 0.002, 0.345, 0)));
  TB.tube('rubber', [new V3(-RT - 0.008, 0.345, 0), new V3(-RT - 0.02, 0.32, 0.002), new V3(-RT - 0.016, 0.255, -0.004), new V3(-0.113, 0.205, 0), new V3(-0.092, 0.186, 0.002)], 0.0022, 40, 6);

  // ---- head ToF sensor (VL53L9CX-like multizone module behind a window) ------------------------------------------------
  const head = new Group(); head.position.set(RT - 0.002, DIMS.headUp - TOWER_DY, 0); head.rotation.z = DIMS.headPitchDeg * DEG; tower.add(head);
  const HB = new Bucket();
  HB.add('graphite', new RoundedBoxGeometry(0.030, 0.042, 0.066, 3, 0.006), mat(0.004, 0, 0));
  HB.add('anod', new RoundedBoxGeometry(0.004, 0.047, 0.071, 3, 0.0015), mat(0.0195, 0, 0));        // bezel plate
  HB.add('glass', new RoundedBoxGeometry(0.0025, 0.032, 0.056, 2, 0.001), mat(0.0222, 0, 0));       // window
  const lensG = new THREE.CylinderGeometry(0.0085, 0.0085, 0.0014, 32), lensS = new THREE.CylinderGeometry(0.0055, 0.0055, 0.0014, 28);
  HB.add('alu', new THREE.TorusGeometry(0.0098, 0.0011, 8, 36), mat(0.0238, 0, -0.012, 0, Math.PI / 2, 0));
  HB.add('alu', new THREE.TorusGeometry(0.0066, 0.0009, 8, 30), mat(0.0238, 0, 0.0155, 0, Math.PI / 2, 0));
  HB.add('glass', lensG, mat(0.0242, 0, -0.012, 0, 0, Math.PI / 2));
  HB.add('glass', lensS, mat(0.0242, 0, 0.0155, 0, 0, Math.PI / 2));
  for (const [dy, dz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) HB.add('steel', new THREE.CylinderGeometry(0.0016, 0.0016, 0.0012, 6), mat(0.0221 + 0.0012, dy * 0.0205, dz * 0.0325, 0, 0, Math.PI / 2));
  HB.build(head);
  const ringMesh = new Mesh(new THREE.TorusGeometry(0.0118, 0.0008, 8, 40), M.ring); ringMesh.rotation.y = Math.PI / 2; ringMesh.position.set(0.0236, 0, -0.012); head.add(ringMesh);
  // body status band: a lit groove around the shoulder
  const band = new Mesh(new THREE.TorusGeometry(0.1092, 0.0017, 8, 96), M.ring); band.rotation.x = Math.PI / 2; band.position.y = 0.4385; tower.add(band);

  // ---- three omniwheel assemblies + motors, three arrestors -------------------------------------------------------------
  const zeta = DIMS.zetaDeg * DEG, rContact = RB + RW;
  const wheels = [], arrestors = [];
  const rollerGeo = lathe([[0.0, -0.0083], [0.0035, -0.0083], [0.0058, -0.0065], [0.0065, -0.003], [0.0065, 0.003], [0.0058, 0.0065], [0.0035, 0.0083], [0.0, 0.0083]], 12);
  const NR = [10, 12, 14][q];
  for (let k = 0; k < 3; k++) {
    const phi = (DIMS.wheelAz0Deg + 120 * k) * DEG;
    const n = new V3(Math.sin(zeta) * Math.cos(phi), Math.cos(zeta), Math.sin(zeta) * Math.sin(phi));        // outward contact normal
    const a = new V3(Math.cos(zeta) * Math.cos(phi), -Math.sin(zeta), Math.cos(zeta) * Math.sin(phi));        // wheel axis: meridian tangent
    const xd = new V3().crossVectors(a, n);                                                                  // drive direction (azimuthal)
    const c = n.clone().multiplyScalar(rContact);
    const frame = new Group(); frame.matrixAutoUpdate = false; frame.matrix.copy(basisM(xd, a, n, c)); frame.matrixWorldNeedsUpdate = true; tilt.add(frame);
    const spin = new Group(); frame.add(spin);
    // spinning part: hub plates, boss, bolts, instanced barrel rollers (two staggered rows)
    const WB = new Bucket();
    WB.add('alu', new THREE.CylinderGeometry(0.0165, 0.0165, 0.004, 28));
    for (const s of [-1, 1]) {
      WB.add('alu', new THREE.CylinderGeometry(0.0172, 0.0172, 0.0025, 28), mat(0, s * 0.0158, 0));
      WB.add('anod', new THREE.CylinderGeometry(0.0062, 0.0062, 0.0022, 20), mat(0, s * 0.0176, 0));
      WB.add('steel', new THREE.TorusGeometry(0.0125, 0.0007, 6, 24), mat(0, s * 0.0173, 0, Math.PI / 2, 0, 0));
      for (let i = 0; i < 6; i++) { const t = i / 6 * Math.PI * 2; WB.add('steel', new THREE.CylinderGeometry(0.0013, 0.0013, 0.0014, 6), mat(Math.cos(t) * 0.0125, s * 0.0178, Math.sin(t) * 0.0125)); }
    }
    WB.add('anod', new THREE.CylinderGeometry(0.0045, 0.0045, 0.037, 14));
    WB.build(spin);
    const rollers = new THREE.InstancedMesh(rollerGeo, M.roller, NR * 2); rollers.castShadow = true;
    const rc = RW - 0.0065, tmp = new Matrix4(), mR = new Matrix4();
    for (let row = 0; row < 2; row++) for (let i = 0; i < NR; i++) {
      const t = (i + row * 0.5) / NR * Math.PI * 2, rh = new V3(Math.cos(t), 0, Math.sin(t)), tg = new V3(-Math.sin(t), 0, Math.cos(t));
      mR.makeBasis(rh, tg, new V3(0, -1, 0)).setPosition(rh.x * rc, (row ? 1 : -1) * 0.0085, rh.z * rc);
      rollers.setMatrixAt(row * NR + i, tmp.copy(mR));
    }
    spin.add(rollers);
    // static part in the wheel frame (local y = axis): motor, gearbox, C-bracket
    const SB = new Bucket(), Lm = (y) => mat(0, y, 0);
    SB.add('anod', new THREE.CylinderGeometry(0.0228, 0.0228, 0.004, 28), Lm(-0.0215));
    SB.add('alu', new THREE.CylinderGeometry(0.0205, 0.0205, 0.024, 28), Lm(-0.0355));
    SB.add('motor', new THREE.CylinderGeometry(0.0196, 0.0196, 0.045, 28), Lm(-0.0705));
    SB.add('steel', new THREE.CylinderGeometry(0.0199, 0.0199, 0.0014, 28), Lm(-0.0486));
    SB.add('steel', new THREE.CylinderGeometry(0.0199, 0.0199, 0.0014, 28), Lm(-0.0922));
    SB.add('graphite', new THREE.CylinderGeometry(0.0188, 0.0188, 0.004, 28), Lm(-0.0950));
    SB.add('anod', new THREE.CylinderGeometry(0.0235, 0.0235, 0.007, 28), Lm(-0.0975));            // mount boss against the chassis
    SB.add('rubber', new THREE.CylinderGeometry(0.0055, 0.0055, 0.008, 12), mat(0.0205, -0.082, 0, 0, 0, Math.PI / 2));   // cable boot
    for (let i = 0; i < 4; i++) { const t = i / 4 * Math.PI * 2 + Math.PI / 4; SB.add('steel', new THREE.CylinderGeometry(0.0016, 0.0016, 0.0016, 6), mat(Math.cos(t) * 0.0185, -0.0236, Math.sin(t) * 0.0185)); }
    // C-bracket: plate on the motor face, bridge over the outside of the wheel, tab carrying the outboard bearing
    SB.add('anod', new RoundedBoxGeometry(0.013, 0.050, 0.0035, 2, 0.001), mat(0, 0.0015, RW + 0.0072));
    SB.add('anod', new RoundedBoxGeometry(0.013, 0.0035, 0.0345, 2, 0.001), mat(0, 0.0235, RW + 0.0072 - 0.0155 + 0.0030 - 0.0025, 0));
    SB.add('anod', new RoundedBoxGeometry(0.013, 0.0035, 0.0345, 2, 0.001), mat(0, -0.0235, RW + 0.0072 - 0.0155 + 0.0030 - 0.0025, 0));
    SB.add('steel', new THREE.CylinderGeometry(0.0042, 0.0042, 0.003, 14), Lm(0.0224));
    SB.build(frame);
    wheels.push({ spin, n, a, xd, c, angle: 0 });

    // arrestor: ball-transfer unit opposite the wheel, on a slim strut curving around the ball from the base flange
    const psi = phi + Math.PI, ca = Math.cos(psi), sa = Math.sin(psi);
    const dH = DIMS.arrestorHoriz, dV = DIMS.arrestorAbove, dl = Math.hypot(dH, dV), uh = dH / dl, uv = dV / dl;
    const ac = new V3(ca * (RB + DIMS.arrestorR) * uh, (RB + DIMS.arrestorR) * uv, sa * (RB + DIMS.arrestorR) * uh);
    const adir = new V3(ca * uh, uv, sa * uh);
    const aq = new Quaternion().setFromUnitVectors(Y, adir);
    B.add('steel', new THREE.SphereGeometry(DIMS.arrestorR, 20, 14), new Matrix4().compose(ac, aq, ONE));
    B.add('anod', new THREE.CylinderGeometry(0.0108, 0.0098, 0.0085, 20), new Matrix4().compose(ac.clone().addScaledVector(adir, 0.0072), aq, ONE));
    const P = (r, y) => new V3(ca * r, y, sa * r);
    B.tube('anod', [P(0.098, DIMS.towerBottom + 0.001), P(0.1135, 0.14), P(0.1125, 0.04), P(0.099, -0.045), ac.clone().addScaledVector(adir, 0.0135)], 0.0037, 36, 8);
    B.add('alu', new THREE.CylinderGeometry(0.0062, 0.0062, 0.003, 14), mat(ca * 0.0985, DIMS.towerBottom - 0.0035, sa * 0.0985));
    arrestors.push(ac);

    // motor cable from the boot up into the chassis
    const boot = c.clone().addScaledVector(a, -0.082).addScaledVector(xd, 0.0205).addScaledVector(n, 0);
    B.tube('rubber', [boot, boot.clone().addScaledVector(xd, 0.014).addScaledVector(n, 0.012), new V3(Math.cos(phi + 0.28) * 0.071, 0.158, Math.sin(phi + 0.28) * 0.071), new V3(Math.cos(phi + 0.3) * 0.088, 0.172, Math.sin(phi + 0.3) * 0.088)], 0.0018, 24, 6);
  }

  // ---- side ToF modules on stalks (cam_0 / cam_1 bodies in ballbot.xml) --------------------------------------------------
  const sides = [];
  for (const s of [1, -1]) {
    const { lateral, up, pitchDeg } = DIMS.sideSensor, tip = new V3(s * lateral, up, 0.01);
    B.rod('anod', new V3(s * 0.06, 0.15, 0.01), tip, 0.0075, 12);
    B.add('alu', new THREE.SphereGeometry(0.0095, 14, 10), mat(s * 0.06, 0.15, 0.01));
    const sg = new Group(); sg.position.copy(tip); sg.rotation.set(0, s > 0 ? 0 : Math.PI, pitchDeg * DEG); tilt.add(sg);
    const SB = new Bucket();
    SB.add('graphite', new RoundedBoxGeometry(0.018, 0.026, 0.034, 2, 0.004), mat(0.004, 0, 0));
    SB.add('glass', new RoundedBoxGeometry(0.0018, 0.019, 0.026, 2, 0.0008), mat(0.0136, 0, 0));
    SB.add('alu', new THREE.TorusGeometry(0.006, 0.0007, 8, 24), mat(0.0148, 0, -0.005, 0, Math.PI / 2, 0));
    SB.add('glass', new THREE.CylinderGeometry(0.0055, 0.0055, 0.001, 20), mat(0.0152, 0, -0.005, 0, 0, Math.PI / 2));
    SB.add('glass', new THREE.CylinderGeometry(0.0036, 0.0036, 0.001, 18), mat(0.0152, 0, 0.0075, 0, 0, Math.PI / 2));
    SB.build(sg);
    sides.push(sg);
  }

  B.build(tilt); TB.build(tower);

  // ---------------------------------------------------------------- sensor field of view (head ToF) + 8 x 8 floor hits
  const viz = new Group(); viz.visible = false; head.add(viz);
  const fovL = 0.85, hx = Math.tan(DIMS.fovDeg[0] / 2 * DEG) * fovL, hy = Math.tan(DIMS.fovDeg[1] / 2 * DEG) * fovL;
  const apex = new V3(0.0245, 0, 0);
  const far = [[hy, -hx], [hy, hx], [-hy, hx], [-hy, -hx]].map(([yy, zz]) => new V3(apex.x + fovL, yy, zz));
  const pos = [], idx = [];
  const addTri = (p0, p1, p2) => pos.push(...p0.toArray(), ...p1.toArray(), ...p2.toArray());
  for (let i = 0; i < 4; i++) addTri(apex, far[i], far[(i + 1) % 4]);
  addTri(far[0], far[1], far[2]); addTri(far[0], far[2], far[3]);
  const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const fovMat = new THREE.MeshBasicMaterial({ color: 0x2ee6c4, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false });
  viz.add(new Mesh(fg, fovMat));
  const lp = []; for (let i = 0; i < 4; i++) lp.push(apex, far[i], far[i], far[(i + 1) % 4]);
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x2ee6c4, transparent: true, opacity: 0.55 });
  viz.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lp), edgeMat));
  const dots = new THREE.InstancedMesh(new THREE.CircleGeometry(0.0085, 10).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x2ee6c4, transparent: true, opacity: 0.9, depthWrite: false }), 64);
  dots.frustumCulled = false; dots.visible = false; dots.count = 0; root.add(dots);
  const _o = new V3(), _d = new V3(), _m = new Matrix4(), _p = new Matrix4(), _s = new V3(1, 1, 1), _qq = new Quaternion();
  function updateDots() {
    head.updateWorldMatrix(true, false);
    _o.copy(apex).applyMatrix4(head.matrixWorld);
    let n = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      const yy = Math.tan((i + 0.5 - 4) / 8 * DIMS.fovDeg[1] * DEG), zz = Math.tan((j + 0.5 - 4) / 8 * DIMS.fovDeg[0] * DEG);
      _d.set(1, -yy, zz).transformDirection(head.matrixWorld);
      if (_d.y > -0.02) continue;
      const t = -_o.y / _d.y; if (t > 3.8) continue;
      _p.compose(new V3(_o.x + _d.x * t, 0.003, _o.z + _d.z * t), _qq, _s.set(1, 1, 1));
      dots.setMatrixAt(n++, _p);
    }
    dots.count = n; dots.instanceMatrix.needsUpdate = true;
  }

  // ---------------------------------------------------------------- pose update, ball roll, wheel spin
  const prev = { x: 0, z: 0, tx: 0, tz: 0, ok: false };
  const qBall = new Quaternion(), qd = new Quaternion(), ax = new V3(), wBody = new V3(), wBall = new V3(), wRel = new V3(), tmpV = new V3();
  const qInv = new Quaternion();
  let sensorOn = false, statusT = 0;

  /** Pose in metres and radians: ball centre (x,z) on the floor, body lean about the ball centre toward +x (tiltX) and +z (tiltZ). */
  function setPose({ x = 0, z = 0, tiltX = 0, tiltZ = 0 }, dt = 0) {
    rig.position.x = x; rig.position.z = z;
    tilt.rotation.set(tiltZ, 0, -tiltX, 'XYZ');
    if (prev.ok && dt > 1e-5) {
      const vx = (x - prev.x) / dt, vz = (z - prev.z) / dt;
      // rolling without slip on the floor: omega = (vz, 0, -vx) / R
      wBall.set(vz / RB, 0, -vx / RB);
      const w = wBall.length();
      if (w > 1e-6) { ax.copy(wBall).divideScalar(w); qd.setFromAxisAngle(ax, w * dt); qBall.premultiply(qd); ballSpin.quaternion.copy(qBall); }
      wBody.set((tiltZ - prev.tz) / dt, 0, -(tiltX - prev.tx) / dt);
      qInv.copy(tilt.quaternion).invert();
      wRel.copy(wBall).sub(wBody).applyQuaternion(qInv);                          // ball rate relative to the body, body frame
      for (const wh of wheels) {
        // no-slip along the drive direction: wheel rate = -(omega_rel x n) . d * R / r_w
        const rate = -tmpV.crossVectors(wRel, wh.n).dot(wh.xd) * RB / RW;
        wh.angle += rate * dt; wh.spin.quaternion.setFromAxisAngle(Y, wh.angle);
      }
    }
    prev.x = x; prev.z = z; prev.tx = tiltX; prev.tz = tiltZ; prev.ok = true;
    if (sensorOn) { tilt.updateWorldMatrix(true, true); updateDots(); }
  }
  function setSensorVisible(on) { sensorOn = !!on; viz.visible = sensorOn; dots.visible = sensorOn; if (sensorOn) { rig.updateWorldMatrix(true, true); updateDots(); } }
  /** Status ring glow: level 0..1 (default 0.55). hex optional colour. */
  function setStatus(level, hex) { M.ring.emissiveIntensity = 0.25 + 1.3 * level; if (hex !== undefined) M.ring.emissive.set(hex); }
  function setTheme(light) {
    M.shell.color.set(light ? 0xf3f5f8 : 0xe3e7ec);
    const k = light ? 0.8 : 1;
    for (const m of envMats) m.envMapIntensity = (m === M.glass ? 1.6 : 1) * k;
    fovMat.opacity = light ? 0.16 : 0.1; edgeMat.opacity = light ? 0.8 : 0.55;
    const c = light ? 0x0f9b86 : 0x2ee6c4; fovMat.color.set(c); edgeMat.color.set(c); dots.material.color.set(c);
  }
  function dispose() {
    root.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    for (const m of envMats) { m.map?.dispose?.(); m.bumpMap?.dispose?.(); m.dispose(); }
  }
  setTheme(false);

  // geometry numbers for the proportion check against ballbot.xml (all in metres relative to the ball centre, y up)
  const report = () => ({
    ballR: RB, towerR: RT, towerBottom: DIMS.towerBottom, towerTop: DIMS.towerTop, overallHeight: DIMS.towerTop + RB, overallHeightWithAntenna: 0.4976 + RB,
    wheelCentres: wheels.map((w) => w.c.toArray()), wheelR: RW,
    wheelZenithDeg: wheels.map((w) => Math.acos(w.n.y) / DEG), arrestorCentres: arrestors.map((v) => v.toArray()),
    head: { up: DIMS.headUp, pitchDeg: DIMS.headPitchDeg }, sideSensorLateral: DIMS.sideSensor.lateral,
  });

  return { root, rig, tilt, ball, ballSpin, head, wheels, sides, materials: M, setPose, setSensorVisible, setStatus, setTheme, dispose, report, dims: DIMS,
           get sensorVisible() { return sensorOn; } };
}

// =====================================================================================================================
// Stage: procedural image-based lighting, key light with soft shadows, clean floor, contact shadow
// =====================================================================================================================
export function createStage(THREE, renderer, scene, opts = {}) {
  let quality = opts.quality ?? 2;

  // ---- tiny procedural environment: gradient dome + soft boxes, prefiltered once with PMREM (no HDR download)
  const envScene = new THREE.Scene();
  const dome = new THREE.SphereGeometry(20, 48, 24), cols = [];
  const p = dome.attributes.position, top = new THREE.Color(0xb9c9e2), hor = new THREE.Color(0x6f7b8c), bot = new THREE.Color(0x323b49), c = new THREE.Color();
  for (let i = 0; i < p.count; i++) { const t = p.getY(i) / 20; c.copy(t > 0 ? hor.clone().lerp(top, Math.pow(t, 0.7)) : hor.clone().lerp(bot, Math.pow(-t, 0.6))); cols.push(c.r, c.g, c.b); }
  dome.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  envScene.add(new THREE.Mesh(dome, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  const softbox = (w, h, pos, intensity, tint = 0xffffff) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(tint).multiplyScalar(intensity), side: THREE.DoubleSide }));
    m.position.set(...pos); m.lookAt(0, 0.3, 0); envScene.add(m);
  };
  softbox(7, 5, [-6, 9, 7], 6.5, 0xfff3e6);          // key, same side as the shadow-casting light
  softbox(2.5, 9, [9, 4, 1], 3.2, 0xd7e6ff);         // tall strip fill from the right
  softbox(8, 2.2, [-3, 5, -10], 4.2, 0xdaf5ee);      // back rim
  softbox(10, 10, [0, 14, 0], 1.6);                  // overhead
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envRT = pmrem.fromScene(envScene, 0.035);
  scene.environment = envRT.texture; pmrem.dispose();
  envScene.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });

  // ---- key light and soft shadows
  const sun = new THREE.DirectionalLight(0xfff4e8, 2.6);
  sun.position.set(-1.1, 2.6, 1.5); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -0.9, right: 0.9, top: 0.9, bottom: -0.9, near: 0.5, far: 6 });
  sun.shadow.bias = -0.0003; sun.shadow.normalBias = 0.004; sun.shadow.radius = 4;
  scene.add(sun); scene.add(sun.target);
  const rim = new THREE.DirectionalLight(0xcfe6ff, 0.7); rim.position.set(1.6, 1.4, -2.2); scene.add(rim);

  // ---- floor: clean matte surface with a faint 0.25 m tile pattern like the simulator floor, fading into the background
  const fc = document.createElement('canvas'); fc.width = fc.height = 512;
  const floorTex = new THREE.CanvasTexture(fc); floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.colorSpace = THREE.SRGBColorSpace; floorTex.anisotropy = 8;
  const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.52, metalness: 0, envMapIntensity: 0.55 });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(7, 72), floorMat); floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  floorTex.repeat.set(14, 14);
  const paintFloor = (a, b, line) => {
    const g = fc.getContext('2d'); g.fillStyle = a; g.fillRect(0, 0, 512, 512); g.fillStyle = b; g.fillRect(0, 0, 256, 256); g.fillRect(256, 256, 256, 256);
    g.strokeStyle = line; g.lineWidth = 3; g.strokeRect(0, 0, 512, 512); g.beginPath(); g.moveTo(256, 0); g.lineTo(256, 512); g.moveTo(0, 256); g.lineTo(512, 256); g.stroke();
    floorTex.needsUpdate = true;
  };

  // ---- contact shadow: soft dark core under the ball so it never looks like it floats
  const cs = document.createElement('canvas'); cs.width = cs.height = 128;
  { const g = cs.getContext('2d'), gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, 'rgba(0,0,0,0.85)'); gr.addColorStop(0.22, 'rgba(0,0,0,0.55)'); gr.addColorStop(0.5, 'rgba(0,0,0,0.16)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128); }
  const contact = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.42), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(cs), transparent: true, depthWrite: false, opacity: 0.9 }));
  contact.rotation.x = -Math.PI / 2; contact.position.y = 0.0015; contact.renderOrder = 1; scene.add(contact);

  function setQuality(level) {
    quality = level;
    const size = [0, 1024, 2048][level];
    sun.castShadow = level > 0;
    if (level > 0 && sun.shadow.mapSize.x !== size) { sun.shadow.mapSize.set(size, size); sun.shadow.map?.dispose(); sun.shadow.map = null; }
    rim.visible = level > 0;
  }
  function setTheme(light, bgColor) {
    const bg = new THREE.Color(bgColor);
    scene.background = bg; scene.fog = new THREE.Fog(bg, 2.4, 6.4);
    if (light) paintFloor('#dde3ec', '#e9eef5', 'rgba(120,135,160,0.35)'); else paintFloor('#141b27', '#1a2331', 'rgba(150,170,200,0.12)');
    sun.intensity = light ? 2.3 : 2.6; contact.material.opacity = light ? 0.75 : 0.95;
    renderer.toneMappingExposure = light ? 1.0 : 1.08;
  }
  /** Keep contact shadow, shadow camera and light under the moving ball. */
  function followBall(bot) {
    const { x, z } = bot.rig.position;
    contact.position.x = x; contact.position.z = z; sun.target.position.set(x, 0.2, z); sun.position.set(x - 1.1, 2.6, z + 1.5);
    sun.target.updateMatrixWorld();
  }
  /** Poster mode: transparent background and a shadow-only floor so a render can be saved with alpha. */
  function posterMode(on) {
    floor.material = on ? new THREE.ShadowMaterial({ opacity: 0.28 }) : floorMat; scene.fog = on ? null : scene.fog; if (on) scene.background = null;
  }
  setQuality(quality);
  return { sun, rim, floor, contact, setQuality, setTheme, followBall, posterMode, envTexture: envRT.texture,
           dispose() { envRT.dispose(); floorTex.dispose(); } };
}
