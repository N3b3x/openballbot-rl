// Closed-loop simulation of the ballbot with the official MuJoCo WebAssembly build (@mujoco/mujoco 3.13.0, the same MuJoCo
// version as the Python training environment). Mirrors ballbot_gym/envs/ballbot_env.py (observation, action scaling, 5 sub-steps
// of 2 ms with implicitfast) and ballbot_gym/utils/contact_frame.py (ContactFrameStepper) for flat ground and no domain
// randomisation (no latency, no sensor noise, nominal masses and friction).
//
// Contact-frame fix: the JS bindings return `data.contact` BY COPY (writes are lost), so the frames are rewritten through the
// WebAssembly heap: `data.arena` is a live view of the arena inside mjData, where the mjContact array lives. The struct layout
// is calibrated against `data.contact` at start-up and can be re-verified (verify()); if it does not match, frameFix is false.
export const NU = 3;
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class BallbotSim {
  constructor(mj, model, data, opts = {}) {
    this.mj = mj; this.m = model; this.d = data;
    this.maxVel = opts.maxWheelVelocity ?? 30.0; this.repeat = opts.actionRepeat ?? 5;
    this.vclip = opts.velClip ?? 8.0; this.fixFrames = opts.fixFrames ?? true;
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST.value;           // env setting "integrator: implicitfast"
    for (let i = 0; i < NU; i++) { model.actuator_ctrlrange[2 * i] = -this.maxVel; model.actuator_ctrlrange[2 * i + 1] = this.maxVel; }
    const adr = v => (typeof v === 'number' ? v : v[0]);
    this.wheelDof = [0, 1, 2].map(k => adr(model.jnt('wheel_joint_' + k).dofadr));
    for (const dof of this.wheelDof) model.dof_damping[dof] = opts.wheelDamping ?? 0.1;   // env setting "wheel_damping: 0.1"
    // NOTE (faithful quirk): BallbotEnv._get_obs reads qvel[joint.id] (joint index 1..3), NOT qvel[dofadr] (6..8), so the policy's
    // 'motor_state' is really [base vy, base vz, base wx]/30. The policy was trained on exactly this signal, so it is reproduced.
    this.motorIdx = [0, 1, 2].map(k => model.jnt('wheel_joint_' + k).id);
    this.baseId = model.body('base').id; this.ballId = model.body('ball').id; this.ballGeom = model.geom('the_ball').id;
    this.isWheel = new Uint8Array(model.ngeom);
    for (let k = 0; k < 3; k++) this.isWheel[model.geom('wheel_mesh_' + k).id] = 1;
    this.cmd = [0, 0, 0]; this.vint = [0, 0]; this.lastAct = [0, 0, 0];
    this.layout = null; this.frameFix = false; this.stepCount = 0; this.failed = false; this.push = [0, 0];
    this.tiltDeg = 0; this.offBallDeg = 0;
  }

  // ---- contact-frame hook (see header) ------------------------------------------------------------------------
  _arena() {
    const ar = this.d.arena;
    return { f64: new Float64Array(ar.buffer, ar.byteOffset, ar.byteLength >> 3), i32: new Int32Array(ar.buffer, ar.byteOffset, ar.byteLength >> 2) };
  }
  calibrate() {                       // call right after a forward pass that has >= 2 contacts
    const d = this.d, cs = d.contact; let ok = false;
    if (cs.size() >= 2) {
      const { f64, i32 } = this._arena(), c0 = cs.get(0), c1 = cs.get(1);
      const p0 = Array.from(c0.pos), p1 = Array.from(c1.pos);
      let base = -1, stride = 0;
      for (let i = 1; i < f64.length - 4 && base < 0; i++) if (f64[i] === p0[0] && f64[i + 1] === p0[1] && f64[i + 2] === p0[2]) base = i - 1;  // base = index of mjContact.dist
      if (base >= 0) {
        for (let i = base + 8; i < base + 400; i++) if (f64[i + 1] === p1[0] && f64[i + 2] === p1[1] && f64[i + 3] === p1[2]) { stride = i - base; break; }
        let geomI = -1;                                                      // int32 index of (geom1, geom2), found by matching two contacts
        for (let j = 2 * (base + 13); j < 2 * (base + stride) - 1 && geomI < 0; j++) {
          if (i32[j] === c0.geom1 && i32[j + 1] === c0.geom2 && i32[j + 2 * stride] === c1.geom1 && i32[j + 2 * stride + 1] === c1.geom2) geomI = j;
        }
        ok = stride > 0 && geomI >= 0 && Array.from(c0.frame).every((v, j) => v === f64[base + 4 + j]) && Array.from(c1.frame).every((v, j) => v === f64[base + stride + 4 + j]);
        if (ok) this.layout = { base, stride, geomI };
      }
      c0.delete(); c1.delete();
    }
    cs.delete();
    this.frameFix = ok; return ok;
  }
  verify() {                          // exact comparison of arena contents with data.contact for every active contact
    if (!this.layout) return false;
    const d = this.d, cs = d.contact, n = cs.size(), { f64, i32 } = this._arena(), { base, stride, geomI } = this.layout;
    let ok = n === d.ncon;
    for (let i = 0; i < n && ok; i++) {
      const c = cs.get(i);
      ok = c.geom1 === i32[geomI + 2 * stride * i] && c.geom2 === i32[geomI + 2 * stride * i + 1] &&
           Array.from(c.frame).every((v, j) => v === f64[base + i * stride + 4 + j]);
      c.delete();
    }
    cs.delete(); return ok;
  }
  alignContacts() {                   // JS port of ContactFrameStepper._align_contacts
    const d = this.d, n = d.ncon; if (!n || !this.layout) return;
    const { f64, i32 } = this._arena(), { base, stride, geomI } = this.layout, xmat = d.geom_xmat, bg = this.ballGeom, isW = this.isWheel;
    for (let i = 0; i < n; i++) {
      const g1 = i32[geomI + 2 * stride * i], g2 = i32[geomI + 2 * stride * i + 1];
      if (!((g1 === bg && isW[g2]) || (g2 === bg && isW[g1]))) continue;
      const w = isW[g2] ? g2 : g1, o = base + i * stride + 4;               // frame rows: normal, tangent1, tangent2
      const ax = xmat[9 * w + 2], ay = xmat[9 * w + 5], az = xmat[9 * w + 8];   // capsule axis (local z) in the world frame
      const nx = f64[o], ny = f64[o + 1], nz = f64[o + 2];
      const k = ax * nx + ay * ny + az * nz;
      let tx = ax - k * nx, ty = ay - k * ny, tz = az - k * nz;
      const inv = 1 / (Math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12);
      tx *= inv; ty *= inv; tz *= inv;
      f64[o + 3] = tx; f64[o + 4] = ty; f64[o + 5] = tz;
      f64[o + 6] = ny * tz - nz * ty; f64[o + 7] = nz * tx - nx * tz; f64[o + 8] = nx * ty - ny * tx;
    }
  }

  // ---- stepping ---------------------------------------------------------------------------------------------
  forward() {                         // ContactFrameStepper.forward
    const mj = this.mj, m = this.m, d = this.d;
    mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_camlight(m, d); mj.mj_crb(m, d); mj.mj_factorM(m, d);
    mj.mj_collision(m, d);
    if (this.fixFrames && this.frameFix) this.alignContacts();
    mj.mj_makeConstraint(m, d); mj.mj_island(m, d); mj.mj_transmission(m, d);
    mj.mj_fwdVelocity(m, d); mj.mj_fwdActuation(m, d); mj.mj_fwdAcceleration(m, d); mj.mj_fwdConstraint(m, d);
  }
  subStep() { this.forward(); this.mj.mj_implicit(this.m, this.d); }

  reset(init = null) {
    const mj = this.mj, d = this.d;
    mj.mj_resetData(this.m, d);
    if (init) { d.qpos.set(init.qpos0); d.qvel.set(init.qvel0); }
    else { d.qpos[2] += 0.01; d.qpos[12] += 0.01; }                          // env: robot placed 1 cm above the (flat) terrain
    mj.mj_forward(this.m, d);
    this.stepCount = 0; this.failed = false; this.vint = [0, 0]; this.lastAct = [0, 0, 0];
    if (!this.layout) this.calibrate();
    return this.getObs([0, 0, 0]);
  }

  step(action) {                      // action in [-1,1]^3 (already clipped by the policy)
    const d = this.d, a = Array.from(action).map(v => clip(v, -1, 1));
    for (let i = 0; i < NU; i++) d.ctrl[i] = -clip(a[i] * this.maxVel, -this.maxVel, this.maxVel);
    const b = 6 * this.baseId;
    for (let r = 0; r < this.repeat; r++) {
      d.xfrc_applied[b] = this.push[0]; d.xfrc_applied[b + 1] = this.push[1];
      this.subStep();
    }
    d.xfrc_applied[b] = 0; d.xfrc_applied[b + 1] = 0;
    this.stepCount++;
    const obs = this.getObs(a);
    const q = d.xquat, o = 4 * this.baseId, qx = q[o + 1], qy = q[o + 2];
    this.tiltDeg = Math.acos(clip(1 - 2 * (qx * qx + qy * qy), -1, 1)) * 180 / Math.PI;
    const gx = d.geom_xpos, bg = 3 * this.ballGeom, xp = d.xpos, bp = 3 * this.baseId;
    const rx = xp[bp] - gx[bg], ry = xp[bp + 1] - gx[bg + 1], rz = xp[bp + 2] - gx[bg + 2];
    this.offBallDeg = Math.atan2(Math.hypot(rx, ry), Math.max(rz, 1e-6)) * 180 / Math.PI;
    this.failed = this.tiltDeg > 70 || this.offBallDeg > 30;                 // env: max_allowed_tilt 70, max_ball_offset_deg 30
    return { obs, failed: this.failed };
  }

  getObs(lastAct) {                   // mirrors BallbotEnv._get_obs (xquat/cvel come from the last forward pass, qvel from after the step)
    const d = this.d, ob = 4 * this.baseId, q = d.xquat;
    const w = q[ob], x = q[ob + 1], y = q[ob + 2], z = q[ob + 3];
    const vn = Math.hypot(x, y, z), ang = 2 * Math.atan2(vn, w), sc = vn > 1e-12 ? ang / vn : 2 / w;   // numpy-quaternion as_rotation_vector
    const cv = d.cvel, c = 6 * this.baseId, L = this.vclip;
    const vel = [clip(cv[c + 3], -L, L), clip(cv[c + 4], -L, L), clip(cv[c + 5], -L, L)];
    const angv = [clip(cv[c], -L, L), clip(cv[c + 1], -L, L), clip(cv[c + 2], -L, L)];
    const motor = this.motorIdx.map(i => clip(d.qvel[i] / this.maxVel, -2, 2));
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    this.vint = [0, 1].map(i => clip(this.vint[i] + (this.cmd[i] - vel[i]) * 0.01, -1, 1));
    this.lastAct = Array.from(lastAct);
    return { orientation: [x * sc, y * sc, z * sc], angular_vel: angv, vel, motor_state: motor, actions: this.lastAct,
             cmd: this.cmd.slice(), heading: [Math.sin(yaw), Math.cos(yaw)], vint: this.vint.slice() };
  }
  basePos() { const p = this.d.xpos, i = 3 * this.baseId; return [p[i], p[i + 1], p[i + 2]]; }
  ballPos() { const p = this.d.xpos, i = 3 * this.ballId; return [p[i], p[i + 1], p[i + 2]]; }
}

// Load model + assets into the WASM virtual file system. files: {relativePath: Uint8Array}
// (ballbot.xml, meshes/cone.stl, textures/basketball_texture.png)
export function loadModel(mj, files) {
  const dirs = new Set(['/w']);
  try { mj.FS.mkdir('/w'); } catch (e) { /* exists after a previous load */ }
  for (const [p, bytes] of Object.entries(files)) {
    const parts = p.split('/'); let cur = '/w';
    for (const s of parts.slice(0, -1)) { cur += '/' + s; if (!dirs.has(cur)) { try { mj.FS.mkdir(cur); } catch (e) { /* exists */ } dirs.add(cur); } }
    mj.FS.writeFile('/w/' + p, bytes);
  }
  const model = mj.MjModel.from_xml_path('/w/ballbot.xml');
  return { model, data: new mj.MjData(model) };
}
