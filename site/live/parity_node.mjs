// Parity test: JS policy + MuJoCo WASM (3.13.0) closed loop vs the Python reference rollouts (site/live/ref_rollout.json).
// Usage:  npm i @mujoco/mujoco@3.13.0 (anywhere) ; MUJOCO_JS=/abs/path/node_modules/@mujoco/mujoco/mujoco.js node site/live/parity_node.mjs
// Writes site/live/parity.json (numbers shown on the live page).
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { Policy } from './policy.js';
import { BallbotSim, loadModel } from './sim.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const models = path.resolve(here, '../../ballbot_gym/models');
const loadMujoco = (await import(process.env.MUJOCO_JS ? pathToFileURL(process.env.MUJOCO_JS).href : '@mujoco/mujoco')).default;
const mj = await loadMujoco();
const files = {
  'ballbot.xml': fs.readFileSync(path.join(models, 'ballbot.xml')),
  'meshes/cone.stl': fs.readFileSync(path.join(models, 'meshes/cone.stl')),
  'textures/basketball_texture.png': fs.readFileSync(path.join(models, 'textures/basketball_texture.png')),
};
const spec = JSON.parse(fs.readFileSync(path.join(here, 'policy_robs_c_s0.json')));
const ref = JSON.parse(fs.readFileSync(path.join(here, 'ref_rollout.json')));
const pol = new Policy(spec);
const maxabs = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const out = { mujoco_wasm: '@mujoco/mujoco 3.13.0', python_mujoco: '3.13.0', cases: {} };

// 1) policy parity on the recorded observations
let worstPol = 0;
for (const c of ref.cases) for (let t = 0; t < c.steps; t++) worstPol = Math.max(worstPol, maxabs(Array.from(pol.predict(c.obs[t])), c.act[t]));
out.policy_max_abs_action_error = worstPol;
console.log('policy: max |a_js - a_python| over all recorded steps =', worstPol.toExponential(2));

for (const c of ref.cases) {
  const { model, data } = loadModel(mj, files);
  const sim = new BallbotSim(mj, model, data, { fixFrames: !c.name.endsWith('nofix') });
  sim.cmd = c.cmd;
  const res = {};
  // 2) initial state agreement
  let obs = sim.reset();
  res.init_qpos_err = maxabs(Array.from(data.qpos), c.qpos0); res.init_obs_err = Math.max(...Object.keys(c.obs0).map(k => maxabs(obs[k], c.obs0[k])));
  res.frame_fix_layout_verified = sim.verify(); res.frame_fix_applied = sim.frameFix && sim.fixFrames;
  // 3) teacher-forced single control step: reference state(t) + reference action(t) -> compare state(t+1)
  const eq = [], ev = [], ep = [];
  for (let t = 0; t < Math.min(c.steps, 300); t++) {
    const q = t === 0 ? c.qpos0 : c.qpos[t - 1], v = t === 0 ? c.qvel0 : c.qvel[t - 1];
    data.qpos.set(q); data.qvel.set(v); mj.mj_forward(model, data);
    sim.step(c.act[t]);
    eq.push(maxabs(Array.from(data.qpos), c.qpos[t])); ev.push(maxabs(Array.from(data.qvel), c.qvel[t]));
  }
  const med = a => a.slice().sort((x, y) => x - y)[a.length >> 1];
  res.one_step = { n: eq.length, qpos_max: Math.max(...eq), qpos_median: med(eq), qvel_max: Math.max(...ev), qvel_median: med(ev) };
  // 4) free-running closed loop with the JS policy
  obs = sim.reset(); sim.cmd = c.cmd; obs = sim.getObs([0, 0, 0]); sim.vint = [0, 0]; obs = sim.reset();
  const err = [], traj = []; let fell = null, layoutOk = true;
  for (let t = 0; t < c.steps; t++) {
    const a = pol.predict(obs);
    const r = sim.step(a); obs = r.obs;
    const p = sim.basePos();
    err.push(Math.hypot(p[0] - c.pos[t][0], p[1] - c.pos[t][1]));
    traj.push([p[0], p[1]]);
    if (sim.frameFix && !sim.verify()) layoutOk = false;
    if (r.failed && fell === null) { fell = t; break; }
  }
  const at = s => err[Math.min(err.length - 1, Math.round(s * 100) - 1)];
  res.closed_loop = { layout_verified_every_step: layoutOk, steps_run: err.length, fell_at_step: fell, ref_fell_at_step: c.fell_at_step,
    xy_err_m: { '1s': at(1), '2s': at(2), '3s': at(3), '4s': at(4), '5s': at(5), '6s': at(6), '8s': at(8), final: err[err.length - 1], max: Math.max(...err) },
    final_js_xy: traj[traj.length - 1], final_ref_xy: c.pos[c.pos.length - 1].slice(0, 2),
    path_len_ref_m: c.pos.reduce((s, p, i) => s + (i ? Math.hypot(p[0] - c.pos[i - 1][0], p[1] - c.pos[i - 1][1]) : 0), 0) };
  out.cases[c.name] = res;
  console.log(c.name, JSON.stringify(res, (k, v) => typeof v === 'number' ? +v.toPrecision(3) : v));
  data.delete(); model.delete();
}
fs.writeFileSync(path.join(here, 'parity.json'), JSON.stringify(out, null, 1));
