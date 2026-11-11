// Architecture page: inlines the static SVG diagrams (so the theme's CSS variables apply), swaps each diagram for its
// phone-width variant on narrow screens, and shows a detail panel for the block the reader clicks or focuses.
// Facts come from the repo code and the study log; every panel carries its sources. Simulation only.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const src = (t) => `<span class="src">${t}</span>`;
const kv = (pairs) => `<div class="kv">${pairs.map(([k, v]) => `<div><small>${k}</small>${v}</div>`).join('')}</div>`;
const eq = (t) => `<pre class="eq">${t}</pre>`;
const COL = { model: 'var(--model)', learned: 'var(--learned)', hybrid: 'var(--hybrid)', warn: 'var(--warn)', neutral: 'var(--text-2)', good: 'var(--good)', bad: 'var(--bad)' };
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduced) document.body.classList.add('reduced');
const LOG = 'tracking_study_log.md';

/* ------------------------------------------------------------------ diagram loading */
const narrow = matchMedia('(max-width: 700px)');
const cache = new Map();
async function fetchSVG(url) {
  if (!cache.has(url)) cache.set(url, fetch(url, { cache: 'no-cache' }).then((r) => (r.ok ? r.text() : Promise.reject(r.status))).then((t) => {
    const svg = new DOMParser().parseFromString(t, 'image/svg+xml').documentElement;
    if (svg.nodeName !== 'svg') throw new Error('not svg');
    return svg;
  }));
  return cache.get(url);
}
async function mount(host) {
  const url = narrow.matches && host.dataset.svgM ? host.dataset.svgM : host.dataset.svg;
  if (host.dataset.mounted === url) return;
  try {
    const svg = document.importNode(await fetchSVG(url), true);
    const img = host.querySelector('img');
    if (img && img.alt && !svg.getAttribute('aria-label')) svg.setAttribute('aria-label', img.alt);
    host.replaceChildren(svg);
    host.dataset.mounted = url;
    const key = host.dataset.detail;
    if (key && DETAILS[key]) bind(svg, $(`#${key}-detail`), DETAILS[key], host.dataset.first);
  } catch (e) { /* keep the <img> fallback */ }
}
function mountAll() { $$('[data-svg]').forEach(mount); }

/* ------------------------------------------------------------------ detail panels */
function show(panel, d) {
  if (!panel) return;
  panel.style.setProperty('--c', COL[d.c] || 'var(--hybrid)');
  panel.innerHTML = `<h4>${d.title}</h4>${d.html}<div>${(d.src || []).map(src).join('')}</div>`;
}
function bind(root, panel, dict, first) {
  const els = $$('[data-d]', root);
  const pick = (el) => {
    const d = dict[el.dataset.d]; if (!d) return;
    els.forEach((x) => x.classList.toggle('sel', x === el));
    show(panel, d);
  };
  els.forEach((el) => {
    el.addEventListener('click', () => pick(el));
    el.addEventListener('focus', () => pick(el));
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(el); } });
  });
  const f = els.find((x) => x.dataset.d === first) || els[0];
  if (f) pick(f);
}

const DETAILS = {
  stack: {
    goal: { c: 'good', title: 'Goal', html: `<p>The task is either a planar velocity command or a goal point to reach. In the goal task the point is about 1.1 m away across a terrain tile, and the robot must get there without falling. The goal enters as a direction and distance in the observation and, when a map is used, as the zero of the planner's distance field.</p>${kv([['goal task', 'point about 1.1 m away'], ['velocity task', 'hold 0.3 m/s across tiles'], ['success', 'reach without a fall']])}`, src: ['train_dev.py: goal_feat, cross-focus 1.1 to 1.3 m', `${LOG} S79 to S91`] },
    nav: { c: 'model', title: 'Navigation layer', html: `<p>Turns an obstacle map into a smooth velocity command. A 64 x 64 grid of 5 cm cells marks cells more than 3 cm above the start ground as obstacles, inflates them by 0.22 m, and propagates a distance-to-goal field from the goal over free cells (8-connected) every 0.2 s. The follower descends the field 0.3 m to a carrot point, turns the commanded heading toward it at most 2 rad/s, and sets speed = min(0.3 m/s, 1.5 x goal distance, a turn-dependent cap), changing at most 0.6 m/s², all times the design's Froude factor.</p><p>The balance controller never sees the map: it receives only this command. Derivation and a live toy on the <a href="math.html#s9">physics page, section 9</a>.</p>${kv([['grid', '64 x 64, 5 cm'], ['margin', '0.22 m'], ['carrot', '0.3 m'], ['turn rate', '&lt;= 2 rad/s'], ['speed', '&lt;= 0.3 m/s x sqrt(s)']])}`, src: ['nav.py: WavePlanner.field, command', `${LOG} S89, S90`] },
    lqr: { c: 'model', title: 'LQR with governor and integral', html: `<p>Per vertical plane a 4-state linear quadratic regulator designed on that robot design's derived planar model, discretised at the 10 ms control period. A reference governor limits the acceleration of the internal velocity reference to 0.5 m/s² (about 5 degrees of lean), and a slow integral of the velocity error (gain 1, clipped at ±2) removes steady offset from contact creep. A proportional yaw loop (gain 5) handles spin. The three torques are mapped to the wheels by the omniwheel transmission and normalised by the torque limit.</p>${eq('s = (0, theta, phidot - phidot_ref, thetadot)      tau = -K s\nQ = diag(1e-6, 50, 1, 1),  R = 1,  K from the discrete Riccati equation at dt = 0.01 s\nv_ref += dv * min(1, 0.5 dt / |dv|)        xi = clip(xi + (v_ref - v) dt, ±2)\nphidot_ref = (v_ref + xi) / r              tau_z = -5 (omega_cmd - omega_z)')}<p>Full derivation on the <a href="math.html#s4">physics page, section 4</a>.</p>`, src: ['lqr_gpu.py: design_tables, GpuLQR.act', 'paper_v2/main.tex: LQR section'] },
    policy: { c: 'learned', title: 'Residual PPO policy', html: `<p>A two-layer MLP (256 tanh units each) trained with PPO on 4096 parallel worlds. Its input is 25 proprioceptive and command values (orientation, angular and linear velocity, wheel rates, last action, command, heading, velocity integral, goal) plus a 6-value design vector when several robot designs are in the pool. Its three outputs are wheel residuals in [-1, 1], multiplied by the authority 0.25 before being added to the LQR action. The output layer starts small, so training begins close to the pure LQR.</p>${kv([['input', '25 + 6 values'], ['hidden', '256 x 256 tanh'], ['output', '3 residuals'], ['authority', '0.25']])}`, src: ['train_dev.py: OBS_DIM, Net, obs()', `${LOG} S82, S83`] },
    mix: { c: 'hybrid', title: 'Sum and clip', html: `<p>The applied action is the LQR action plus the authority times the policy action, clipped to the wheel range. Authority 0.25 is the measured knee between tracking accuracy and robustness: tracking 0.99 to 1.01 of the command, goal success 0.71 to 0.76 undisturbed and 0.42 to 0.44 with pushes and pulls over three seeds; higher authority loses tracking, lower authority loses robustness.</p>${eq('a = clip( a_LQR + 0.25 a_pi , -1, 1 )')}<p>Why this trade-off exists and the full authority table: <a href="math.html#s5">physics page, section 5</a>.</p>`, src: ['train_dev.py: step(), rs = residual authority', `${LOG} S82, S83, S84`] },
    torque: { c: 'warn', title: 'Wheel torques', html: `<p>Each normalised wheel command is multiplied by that design's torque limit (30 N m on the nominal robot) and clipped. Domain randomisation scales the limit by a per-episode factor and, with some probability, applies the previous command one step late. Torque saturation is not what makes the robot fall: under 1 percent of wheels were at the limit in the last second before a fall (S79).</p>`, src: ['train_dev.py: tau_w, drc torque, latency', `${LOG} S79`] },
    robot: { c: 'neutral', title: 'The simulated robot', html: `<p>A MuJoCo-Warp float32 model stepped on the GPU: a body on a ball driven by three omniwheels with anisotropic friction, fitted with three passive ball arrestors (8 mm rollers at 100 degrees polar angle, touching the ball only at the loaded seat). Decisions every 10 ms (100 Hz). A fall is a body tilt beyond 70 degrees or the body more than 30 degrees off the ball. The arrestors were the first change that improved every robustness measure at once; why they leave the model unchanged is on the <a href="math.html#s8">physics page, section 8</a>.</p>${kv([['engine', 'MuJoCo-Warp, float32'], ['rate', '100 Hz'], ['arrestors', '3 x 8 mm at 100 deg'], ['fall', 'tilt &gt; 70 deg or off-ball &gt; 30 deg']])}`, src: ['retainer.py', 'train_dev.py: fell', `${LOG} S79, S80`] },
  },
  sense: {
    tof: { c: 'warn', title: 'Time-of-flight sensor', html: `<p>A simulated multizone ToF sensor returns one range per zone: 18 x 14 zones over a 55 x 42 degree field of view, with range-proportional noise of 0.5 percent and 2 percent dropout. The reference mount is a gimbal-stabilised head 15 cm above the base origin, pitched 25 degrees down. Where it looks matters more than its resolution: a 32 x 24 depth camera at the same mount was no better, pitching to -40 degrees left a third of worlds stuck, an 8 x 8 sensor or a low mount lost 7 to 12 points, and two body-cone sensors were worst (0.58 success).</p>`, src: [`${LOG} S89 (sensor table)`] },
    hits: { c: 'model', title: 'Hit points', html: `<p>Each range and its known ray direction, together with the sensor pose, gives a 3-D point in the world frame. No learning: pure geometry. On hardware this is the step that needs the sensor's calibration and the gimbal angle.</p>`, src: [`${LOG} S89 continued`] },
    mem: { c: 'model', title: 'Memory', html: `<p>Hits more than 3 cm above the ground under the ball are kept in a per-world memory of 600 points. Low hits are the floor and are dropped. The memory lets the planner remember blocks that have left the field of view.</p>`, src: [`${LOG} S89 continued`, 'nav.py: occupancy_points'] },
    occ: { c: 'model', title: 'Occupancy grid', html: `<p>The memory is rasterised onto the 64 x 64 grid of 5 cm cells and inflated by 0.22 m (robot half width plus margin) with a max-pool. A wider margin does not help: 0.30 or 0.38 m closes most of the 0.7 m gaps between blocks and worlds get stuck instead (S89).</p>${kv([['cells', '64 x 64 x 5 cm'], ['inflation', '0.22 m']])}`, src: ['nav.py: field (max_pool2d)', `${LOG} S89`] },
    field: { c: 'hybrid', title: 'Distance field and follower', html: `<p>From the goal cell outward, each free cell takes one plus the smallest of its eight neighbours, repeated 110 times; inflated cells stay infinite. Replanned every 0.2 s. The carrot follower reads it (see the controller stack). With a real sensor the planner also aims the sensor gimbal at its target heading: letting the sensor follow the slowly turning command made it see side blocks late (success 0.72, 0.16 of worlds rolled onto a block); aiming it at the target gave 0.78 and 0.03 (S90).</p>`, src: ['nav.py: field, command', `${LOG} S89, S90`] },
  },
  train: {
    worlds: { c: 'warn', title: '4096 worlds on one GPU', html: `<p>Training runs 4096 simulated robots in parallel on MuJoCo-Warp in float32, each with its own design, randomised parameters, surface, terrain tile and disturbance schedule. Decisions at 100 Hz; 200 million steps in total for the recommended controller (twice as many gave no gain, S84). Everything here is simulation.</p>`, src: ['train_dev.py: GpuBallbot', 'BRIEF fact sheet', `${LOG} S84`] },
    designs: { c: 'model', title: '11 robot designs', html: `<p>Variants of the nominal robot: geometric scale, ball mass, ballast, torque limit and rotor inertia. Each design carries its own derived planar model and therefore its own LQR gain. The policy receives a 6-value design vector (log scale factor, log total-mass ratio, log torque ratio, log ball-mass ratio, log rotor-inertia ratio, log body-mass ratio) relative to the nominal design. Commands and pushes are Froude-scaled per design (<a href="math.html#s6">physics page, section 6</a>).</p>`, src: ['train_dev.py: dvec_tab, f_tab, push_tab', 'ballbot_gym/models/variants/'] },
    dr: { c: 'warn', title: 'Domain randomisation at scale 1.75', html: `<p>Each world draws body and ball mass, wheel-ball friction, torque limit, wheel damping, action latency and observation noise from ranges widened by a factor 1.75 over the base config. Evaluation repeats every test at scales 1, 1.75 and 2.5. A late-found engine error (stale subtree mass under mass randomisation) invalidated earlier randomised results; everything cited here is from the corrected engine (S69, S70).</p>`, src: ['train_dev.py: set_dr_scale, drc', `${LOG} S69, S70`] },
    push: { c: 'warn', title: 'Pushes and pulls', html: `<p>Pushes: a horizontal force of 5 to 20 N on the base for 50 ms in a random direction, every 2 to 4 s, scaled with the design's weight. Pulls: a sustained 3 to 12 N for 0.5 to 2 s, every 4 to 10 s. In the fall analysis before the arrestors, pulls were active in 58 percent of disturbed falls and pushes in 3 percent (S79).</p>`, src: ['train_dev.py: push_force, pull', `${LOG} S79`] },
    terrain: { c: 'warn', title: 'Terrain tiles', html: `<p>Ten terrain kinds: flat, ramp up, ramp down, rough, step down, small drop, stairs, step up, large drop and bump, each at a range of sizes, plus (for navigation training) a blocks tile with unclimbable 8 cm blocks. Surfaces vary in friction (about 0.4 to 1.2) and softness. The remaining falls of the recommended controller concentrate on discrete obstacles above the traction limit (S83 addendum, S88).</p>`, src: ['tiles.py', `${LOG} S83, S88, S89`] },
    task: { c: 'good', title: 'Tasks and termination', html: `<p>A world either tracks a velocity command (speed up to 1.2 x sqrt(s) m/s, direction and yaw rate drawn at random, zero with probability 0.1) or drives to a goal point. The reward pays for tracking accuracy and smoothness and charges for action size, action rate, body offset from the ball, and a fall. A fall (tilt over 70 degrees or body more than 30 degrees off the ball) ends the episode with a penalty of 2.</p>`, src: ['train_dev.py: step(), reward terms', 'paper_v2/main.tex: MDP'] },
    obs: { c: 'learned', title: 'What the policy sees', html: `<p>25 values every 10 ms: body orientation as a rotation vector (3), angular velocity (3), linear velocity (3), wheel rates (3), the previous applied action (3), the command (3), heading as sine and cosine (2), the integrated velocity error (2), and the goal vector and flag (3). With several designs in the pool, a 6-value design vector is appended. Observation noise is added under randomisation. These are all quantities a real robot's IMU, encoders and a goal input could provide.</p>${eq('o = [ rotvec(3), omega(3), v(3), wheels(3), a_prev(3), cmd(3), sin/cos yaw(2), v_int(2), goal(3) ]  (+ design(6))')}`, src: ['train_dev.py: OBS_DIM = 25, obs()'] },
    priv: { c: 'learned', title: 'What only the critic sees', html: `<p>The value network is asymmetric: in addition to the policy's input it receives privileged quantities that exist only in simulation: the floor friction and softness, the mass and torque scales drawn for the episode, the wheel damping, a local height scan ahead, and the design vector. This helps the critic judge returns without giving the policy information a robot would not have. The policy itself never sees the map, the friction, the masses or the pushes.</p>`, src: ['train_dev.py: priv_obs, --asym'] },
    actor: { c: 'learned', title: 'Actor', html: `<p>MLP with two hidden layers of 256 tanh units; three outputs are the means of a Gaussian over wheel residuals, scaled by the authority 0.25. At evaluation the mean is used. The output layer is initialised small so the controller starts as the LQR.</p>`, src: ['train_dev.py: Net, Policy'] },
    critic: { c: 'learned', title: 'Critic', html: `<p>Same shape as the actor, one output: the expected return. Trained on the privileged input (asymmetric actor-critic). Discarded after training.</p>`, src: ['train_dev.py: Net(use_priv=True)'] },
  },
  evalp: {
    goal: { c: 'good', title: 'Goal task', html: `<p>Reach a point about 1.1 m away across a terrain tile, 11 designs, 176 worlds per terrain kind. Scored by success, falls and median time. Recommended controller: 0.71 to 0.76 success undisturbed and 0.42 to 0.44 with pushes and pulls over three seeds (S83, S84).</p>`, src: [`${LOG} S83, S84`] },
    vel: { c: 'model', title: 'Velocity task', html: `<p>Hold 0.3 m/s (Froude-scaled) across terrain tiles for 20 s, or 8 s in the standard battery, 640 worlds per kind in the 20 s by-size test. Scored by falls and by whether the feature was crossed, per obstacle kind and size. Longer episodes were needed after 8 s episodes made slow robots look safe because they never reached the obstacle (S86 withdrawn, S87).</p>`, src: [`${LOG} S87, S88`] },
    flat: { c: 'model', title: 'Flat tracking', html: `<p>Commands of 0.3, 0.6 and 0.9 m/s on the nominal robot with randomisation and noise off, 64 worlds. Scored by achieved speed over commanded speed and the steady-state error. After a map-edge artefact was found (robots driving off the height field were counted as falls), the test uses interior tiles only (S92).</p>`, src: [`${LOG} S83, S92`] },
    cond: { c: 'warn', title: 'Conditions every test is crossed with', html: `<p>Randomisation scale 1, 1.75 or 2.5; undisturbed or with pushes and pulls; surfaces on; three training seeds. One JSON file per setting, with every setting in the file name (a lesson from results that were silently overwritten).</p>`, src: [`${LOG} S82 to S84`, 'BRIEF lessons 2'] },
    held: { c: 'neutral', title: 'Held-out sets', html: `<p>Terrain: tile seed 1, while training used seed 0 (the arrestor gain holds there: +0.15 undisturbed, +0.12 disturbed, S84). Robot designs never trained on: the 1.5x robot (13.5 cm ball), where LQR-based controllers keep 0.61 to 0.71 goal success and pure learning collapses to 0.09 to 0.15; a lower-torque design (tau3); and a tenth-torque design that is motor limited for every controller, at most 0.30 (S85-R). Held-out designs cost more than held-out terrain.</p>`, src: [`${LOG} S83 addendum, S84, S85-R`] },
  },
};

/* ------------------------------------------------------------------ boot */
mountAll();
narrow.addEventListener('change', mountAll);
let rz = 0;
window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(mountAll, 200); });

(function spy() {
  const links = $$('.subnav a');
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-30% 0px -60% 0px' });
  links.forEach((a) => { const t = $(a.getAttribute('href')); t && io.observe(t); });
})();
