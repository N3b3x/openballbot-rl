// Lesson 10 figures: the deployment stack (clickable), the safety supervisor and command gate, and the parity results.
import { el, chart, slider, seg, button, readouts, legend, palette } from './ui.js';

// ------------------------------------------------------------------------------------------------ stack diagram
const NODES = [
  { id: 'bt', x: 10, y: 10, w: 150, h: 50, t: ['behaviour tree', 'mission'], c: '--text-3', d: 'Mission logic. A behaviour-tree "stop" should send zero /cmd_vel (or let it time out): the robot then brakes and holds. Do not call disable while it is balancing unless someone is holding it.', f: 'external (not in this repo)' },
  { id: 'nav', x: 10, y: 90, w: 150, h: 60, t: ['Nav2', 'planner + controller'], c: '--model', d: 'Plans a path on the map and publishes /cmd_vel at 10 to 20 Hz in base_link. The ballbot is holonomic, so use a holonomic controller (MPPI with the Omni model, or DWB with max_vel_y above 0) and keep its limits at or below the motion layer\'s. Lessons 5 and 6 are the ideas behind it.', f: 'topic: /cmd_vel (geometry_msgs/Twist)' },
  { id: 'gate', x: 200, y: 10, w: 140, h: 50, t: ['command gate', 'timeout, stop, hold'], c: '--hybrid', d: 'Keeps the latest command. After cmd_timeout (0.5 s) without /cmd_vel the desired velocity becomes zero and, with hold_on_timeout, the robot holds the point where it will come to rest: current position + v|v| / (2 a_brk), using the trained goal-hold behaviour.', f: 'core/safety.py CommandGate' },
  { id: 'mref', x: 200, y: 90, w: 140, h: 50, t: ['motion layer', 'slew, turn, brake, hold'], c: '--hybrid', d: 'The layer of lesson 5: speed slew, turn-rate limit, braking curve and position hold, at 100 Hz. Settings come from the bundle (a_acc 0.4, a_brk 0.5, a_lat 0.35).', f: 'core/motion_ref.py (port of experiments/warp_ppo/motion_ref.py)' },
  { id: 'obs', x: 200, y: 170, w: 140, h: 50, t: ['observation', '25 + history 8 + design'], c: '--text-2', d: 'Builds the 25-number observation from the state estimate, appends the last 8 (observation, action) pairs and the 6-number design vector: the 255-number actor input. Reproduces the training quirks exactly, including the "motor" entries 9 to 11, which are base velocities divided by torque and not wheel rates.', f: 'core/observation.py' },
  { id: 'pi', x: 390, y: 170, w: 130, h: 50, t: ['PPO actor', '255-512-512-512-3'], c: '--learned', d: 'The learned residual: three hidden layers of 512 tanh units, about 0.66 M multiply-adds per tick. Exported as numpy weights, ONNX and TorchScript; the normalisation is inside every file. A numpy tick costs about 0.7 ms for the whole loop on the development PC.', f: 'core/policy.py, config/bundle/<run>/actor.*' },
  { id: 'lqr', x: 390, y: 90, w: 130, h: 50, t: ['LQR base', 'per-design gains K'], c: '--model', d: 'The lesson 3 controller with its command governor and velocity integral, one law per horizontal plane, mapped through the wheel matrix W to three wheel torques.', f: 'core/lqr.py (port of lqr_gpu.py GpuLQR.act)' },
  { id: 'auth', x: 390, y: 10, w: 130, h: 50, t: ['authority', '0.15 to 0.3 adaptive'], c: '--learned', d: 'Adaptive residual authority of lesson 4: 0.15 while cruising, 0.3 on command changes, near-zero commands and tilt, with the 0.05 m/s hold dead band.', f: 'core/authority.py' },
  { id: 'mix', x: 575, y: 90, w: 130, h: 50, t: ['mix', 'clip(u + rs a)'], c: '--hybrid', d: 'The applied action is clip(lqr + rs a, -1, 1), converted to a wheel torque of applied x tau, clamped to the actuator limit (30 N m for the nominal design).', f: 'core/controller.py' },
  { id: 'safe', x: 575, y: 170, w: 130, h: 50, t: ['safety', 'HOLD / ACTIVE / FAULT'], c: '--bad', d: 'Zero torque until /ballbot/enable and the robot is upright (tilt under 3 degrees) with fresh data. Tilt above 30 degrees, a non-finite state or torque, or stale sensor data (over 50 ms) latch FAULT. Try it in the next figure.', f: 'core/safety.py SafetySupervisor' },
  { id: 'mcu', x: 750, y: 115, w: 120, h: 60, t: ['motor driver', '/wheel_torque_cmd'], c: '--text-3', d: 'Three torques in newton metres, published every tick (zeros unless ACTIVE). The hardware driver must map its motor directions to the wheel axes in the bundle; check with the robot on a stand. There is no hardware yet: this part is unverified.', f: 'topic: /wheel_torque_cmd (Float64MultiArray)' },
  { id: 'imu', x: 10, y: 240, w: 150, h: 50, t: ['IMU + encoders', '/imu, /joint_states'], c: '--text-3', d: 'Sensors, sampled at the same instant just before the tick; the filter assumes dt = 10 ms. Encoder calibration and IMU mounting are unverified on hardware.', f: 'sensor_msgs/Imu, JointState' },
  { id: 'est', x: 200, y: 240, w: 140, h: 50, t: ['estimator', 'attitude + Kalman'], c: '--hybrid', d: 'The lesson 7 estimator: attitude filter, gyro bias from standing periods, encoder ball rate, Kalman filter. The controller runs its own copy, with no transport delay inside the balance loop; estimator_node runs identical code to publish /odom and the TF odom to base_link for Nav2.', f: 'core/estimator.py, nodes/estimator_node.py' },
];
const EDGES = [['bt', 'nav'], ['nav', 'gate'], ['gate', 'mref'], ['mref', 'obs'], ['obs', 'pi'], ['mref', 'lqr'], ['auth', 'mix'], ['lqr', 'mix'], ['pi', 'mix'], ['mix', 'safe'], ['safe', 'mcu'], ['imu', 'est'], ['est', 'obs']];

export function stack(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const W = 880, H = 305;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': 'Deployment stack. Each box is a button; select one to read about it.', style: { width: '100%', height: 'auto', padding: '8px 12px', minWidth: '560px' } });
  svg.append(el('defs', {}, el('marker', { id: 'arw', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto' }, el('path', { d: 'M0,0L8,4L0,8z', style: { fill: 'var(--text-3)' } }))));
  const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
  const edgeEls = EDGES.map(([a, b]) => {
    const A = byId[a], B = byId[b];
    let x1 = A.x + A.w, y1 = A.y + A.h / 2, x2 = B.x, y2 = B.y + B.h / 2;
    if (B.x <= A.x + 1 && Math.abs(B.x - A.x) < A.w) { x1 = A.x + A.w / 2; x2 = B.x + B.w / 2; if (B.y > A.y) { y1 = A.y + A.h; y2 = B.y; } else { y1 = A.y; y2 = B.y + B.h; } }
    else if (A.x > B.x) { x1 = A.x; x2 = B.x + B.w; }
    else if (B.x < A.x + A.w) { x1 = A.x + A.w / 2; x2 = B.x + B.w / 2; if (B.y > A.y) { y1 = A.y + A.h; y2 = B.y; } else { y1 = A.y; y2 = B.y + B.h; } }
    const p = el('path', { d: `M${x1},${y1} L${x2},${y2}`, style: { stroke: 'var(--line-2)', strokeWidth: 1.5, fill: 'none' }, 'marker-end': 'url(#arw)' });
    svg.append(p); return [a, b, p];
  });
  const boxes = {};
  NODES.forEach((n) => {
    const g = el('g', { tabindex: 0, role: 'button', 'aria-label': `${n.t.join(', ')}`, style: { cursor: 'pointer', outline: 'none' } });
    const r = el('rect', { x: n.x, y: n.y, width: n.w, height: n.h, rx: 9, style: { fill: 'var(--bg-3)', stroke: `var(${n.c})`, strokeWidth: 1.6 } });
    g.append(r, el('text', { x: n.x + n.w / 2, y: n.y + n.h / 2 - 3, 'text-anchor': 'middle', style: { fill: 'var(--text)', font: '600 12px var(--font-sans)' }, text: n.t[0] }), el('text', { x: n.x + n.w / 2, y: n.y + n.h / 2 + 13, 'text-anchor': 'middle', style: { fill: 'var(--text-3)', font: '10px var(--font-mono)' }, text: n.t[1] }));
    const pick = () => select(n.id);
    g.addEventListener('click', pick); g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    g.addEventListener('focus', () => { r.style.strokeWidth = 3; }); g.addEventListener('blur', () => { r.style.strokeWidth = sel === n.id ? 3 : 1.6; });
    svg.append(g); boxes[n.id] = r;
  });
  ['controller_node: 100 Hz, torque held for 10 ms', 'Nav2 publishes /cmd_vel at 10 to 20 Hz', 'the estimator also feeds the LQR, the authority rule and the motion layer'].forEach((t, i) => svg.append(el('text', { x: 380, y: 252 + i * 16, style: { fill: 'var(--text-3)', font: '10px var(--font-mono)' }, text: t })));
  stage.append(el('div', { style: { overflowX: 'auto', padding: '0 4px' } }, svg));
  const card = el('div', { class: 'callout', style: { margin: '12px 18px 16px', maxWidth: 'none' } });
  body.append(card);
  let sel = 'mref';
  function select(id) {
    sel = id; const n = byId[id];
    Object.entries(boxes).forEach(([k, r]) => { r.style.strokeWidth = k === id ? 3 : 1.6; r.style.fill = k === id ? 'color-mix(in srgb,var(--hybrid) 12%,var(--bg-3))' : 'var(--bg-3)'; });
    edgeEls.forEach(([a, b, p]) => { p.style.stroke = a === id || b === id ? 'var(--hybrid)' : 'var(--line-2)'; p.style.strokeWidth = a === id || b === id ? 2.4 : 1.5; });
    card.replaceChildren(el('h3', { style: { margin: '0 0 .3em', fontSize: '1.05rem' }, text: n.t.join(': ') }), el('p', { text: n.d }), el('p', { class: 'mono', style: { fontSize: '.8rem', color: 'var(--text-3)' }, text: n.f }));
  }
  select('mref');
}

// ------------------------------------------------------------------------------------------------ safety supervisor and command gate
export function safety(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const cfg = { cutoff: 30, engage: 3, stateTimeout: 0.05, cmdTimeout: 0.5, abrk: ctx.data.motion_ref.a_brk };
  const S = { mode: 'HOLD', reason: 'startup', enableReq: false, tilt: 1, age: 0.01, tau: 12, cmdAge: 0, v: 0.5, log: [] };
  const push = (t) => { S.log.unshift(t); S.log = S.log.slice(0, 7); };
  function update() {
    const fresh = S.age <= cfg.stateTimeout;
    const prev = S.mode;
    if (S.mode === 'ACTIVE') {
      if (!Number.isFinite(S.tilt)) fault('non-finite state');
      else if (S.tilt > cfg.cutoff) fault(`tilt ${S.tilt.toFixed(1)} deg > cutoff ${cfg.cutoff}`);
      else if (!fresh) fault('state estimate stale');
      else if (!S.enableReq) { S.mode = 'HOLD'; S.reason = 'disabled'; }
    } else if (S.mode === 'HOLD' && S.enableReq) {
      if (fresh && S.tilt < cfg.engage) { S.mode = 'ACTIVE'; S.reason = 'enabled (controller reset)'; }
      else S.reason = `enable pending: needs a fresh state and tilt below ${cfg.engage} deg`;
    }
    if (S.mode !== prev) push(`${prev} to ${S.mode}: ${S.reason}`);
    render();
  }
  function fault(why) { S.mode = 'FAULT'; S.reason = why; S.enableReq = false; }
  const colour = { HOLD: 'var(--warn)', ACTIVE: 'var(--good)', FAULT: 'var(--bad)' };
  const row = el('div', { style: { display: 'flex', gap: '10px', padding: '0 18px', flexWrap: 'wrap' } });
  const pills = ['HOLD', 'ACTIVE', 'FAULT'].map((m) => { const p = el('div', { style: { flex: '1 1 120px', textAlign: 'center', padding: '14px 8px', borderRadius: '12px', border: '2px solid var(--line)', font: '600 1rem var(--font-mono)', color: 'var(--text-3)', background: 'var(--bg-3)' }, text: m }); row.append(p); return p; });
  stage.append(row);
  const ctl = el('div', { class: 'ctls' }); stage.append(ctl);
  slider(ctl, { label: 'Body tilt (from the estimator)', min: 0, max: 40, step: 0.5, value: S.tilt, fmt: (v) => `${v.toFixed(1)} deg`, onInput: (v) => { S.tilt = v; update(); } });
  slider(ctl, { label: 'Age of the newest IMU + encoder sample', min: 0, max: 120, step: 1, value: 10, fmt: (v) => `${v} ms`, onInput: (v) => { S.age = v / 1000; update(); } });
  const bt = el('div', { class: 'btns' }); stage.append(bt);
  button(bt, 'Enable  (/ballbot/enable true)', () => { S.enableReq = true; update(); }, 'primary');
  button(bt, 'Disable', () => { S.enableReq = false; update(); });
  button(bt, 'Clear fault', () => { if (S.mode === 'FAULT') { S.mode = 'HOLD'; S.reason = 'fault cleared'; push('FAULT to HOLD: fault cleared'); } update(); });
  const ro = readouts(body, [['m', 'mode', ''], ['t', 'wheel torque output', 'hybrid'], ['why', 'reason', '']]);
  const lg = el('pre', { class: 'mono', style: { margin: '0 18px 8px', fontSize: '.78rem', color: 'var(--text-3)', whiteSpace: 'pre-wrap' } }); body.append(lg);
  body.append(el('h3', { style: { padding: '0 18px', margin: '.6em 0 0', fontSize: '1rem' }, text: 'The command gate' }));
  const c2 = el('div', { class: 'ctls' }); body.append(c2);
  slider(c2, { label: 'Time since the last /cmd_vel', min: 0, max: 1.5, step: 0.05, value: 0, fmt: (v) => `${v.toFixed(2)} s`, onInput: (v) => { S.cmdAge = v; render(); } });
  slider(c2, { label: 'Speed when the commands stop', min: 0.1, max: 0.9, step: 0.05, value: S.v, fmt: (v) => `${v.toFixed(2)} m/s`, onInput: (v) => { S.v = v; render(); } });
  const ro2 = readouts(body, [['g', 'desired velocity', 'model'], ['b', 'braking distance v|v| / (2 a_brk)', 'warn'], ['h', 'then', '']]);
  function render() {
    pills.forEach((p, i) => { const m = ['HOLD', 'ACTIVE', 'FAULT'][i], on = m === S.mode; p.style.borderColor = on ? colour[m] : 'var(--line)'; p.style.color = on ? colour[m] : 'var(--text-3)'; p.style.background = on ? `color-mix(in srgb,${colour[m]} 14%,var(--bg-3))` : 'var(--bg-3)'; });
    ro.set('m', S.mode); ro.set('t', S.mode === 'ACTIVE' ? 'controller output (up to 30 N m)' : 'zero'); ro.set('why', S.reason);
    lg.textContent = S.log.length ? S.log.join('\n') : 'No transitions yet. Press Enable.';
    const timed = S.cmdAge > cfg.cmdTimeout;
    ro2.set('g', timed ? 'zero (timeout)' : `${S.v.toFixed(2)} m/s`);
    ro2.set('b', `${((S.v * S.v) / (2 * cfg.abrk) * 100).toFixed(0)} cm`);
    ro2.set('h', timed ? (S.mode === 'ACTIVE' ? 'brakes and holds that point (hold_on_timeout)' : 'robot is not active: zero torque') : 'follows /cmd_vel');
  }
  update();
}

// ------------------------------------------------------------------------------------------------ parity
export function parity(root, ctx) {
  const stage = root.querySelector('.stage'), body = root.querySelector('.body');
  const P = ctx.data.parity;
  let which = 'g39', stat = 0;
  const sg = el('div', { class: 'btns' }); stage.append(sg);
  seg(sg, [['g39', 'g39 (recommended)'], ['g34', 'g34 (earlier)']], 'g39', (v) => { which = v; draw(); }, 'bundle');
  seg(sg, [[0, 'Worst tick'], [1, 'Median tick']], 0, (v) => { stat = v; draw(); }, 'statistic');
  const ch = chart(stage, { x: [-12, -1], y: [-0.5, P.rows.length - 0.5], height: 340, ml: 176, xlabel: 'difference core minus simulator, power of ten (log scale)', desc: 'Parity differences on a log scale.', xticks: [-12, -10, -8, -6, -4, -2], xfmt: (v) => `1e${v}`, yticks: P.rows.map((_, i) => i), yfmt: (v) => (P.rows[v] ? P.rows[v].name : '') });
  const ro = readouts(body, [['n', 'control ticks compared', 'model'], ['f', 'float formats', '']]);
  body.append(el('p', { class: 'hint', text: 'Max over ticks and median, absolute difference between the ROS 2 core (float64) and the simulator (float32 on the GPU), 33 worlds over all 11 designs, domain randomisation and pushes on, noise and latency off, about half the episodes in goal mode. A zero median prints at the left edge.' }));
  // the chart helper has no horizontal bars: draw each value as a thick line with a dot
  function draw() {
    const pts = P.rows.map((r, i) => { const v = r[which][stat]; return { x: v > 0 ? Math.log10(v) : -12, y: P.rows.length - 1 - i, name: r.name, v }; });
    ch.update({
      yfmt: (v) => { const r = P.rows[P.rows.length - 1 - v]; return r ? r.name.replace('observation, goal-mode ticks only', 'observation (goal ticks)') : ''; },
      bars: [], series: pts.map((p) => ({ data: [[-12, p.y], [p.x, p.y]], color: stat ? '--model' : '--hybrid', width: 5 })),
      dots: pts.map((p) => ({ x: p.x, y: p.y, color: stat ? '--model' : '--hybrid', r: 5, label: `${p.name}: ${p.v}` })),
    });
    ro.set('n', which === 'g39' ? P.ticks[0] : P.ticks[1]); ro.set('f', 'simulator float32, core float64');
  }
  draw();
}
