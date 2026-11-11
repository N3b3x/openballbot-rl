"""CHECK 1, MuJoCo side: single wheel on a fixed hinge, constant torque; accel and terminal speed. Also reads I_axis from dz_nom."""
import json, sys
import numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
m0 = mujoco.MjModel.from_xml_path(str(ROOT/'ballbot_gym/models/variants/dz_nom.xml'))
d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
M = np.zeros((m0.nv, m0.nv)); mujoco.mj_fullM(m0, d0, M)
a = m0.joint('wheel_joint_0').dofadr[0]
I_axis = M[a, a] - m0.dof_armature[a]          # wheel inertia about its hinge (armature removed)
print('dz_nom wheel_0: dof_armature', m0.dof_armature[a], 'dof_damping', m0.dof_damping[a], 'M_diag', M[a, a], '=> I_axis', I_axis)
json.dump(dict(I_axis=float(I_axis)), open(Path(__file__).with_name('check_armature_I.json'), 'w'))
XML = """<mujoco><option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/><worldbody><body name="w"><joint name="j" type="hinge" axis="1 0 0" armature="{arm}" damping="{damp}"/>
<inertial pos="0 0 0" mass="0.0893" diaginertia="{I} {I2} {I2}"/></body></worldbody><actuator><motor joint="j" gear="1"/></actuator></mujoco>"""
tau = 0.1
for arm, damp in [(0, 0), (0.005, 0), (0, 0.1), (0.005, 0.1)]:
    for integ in ['implicitfast']:
        m = mujoco.MjModel.from_xml_string(XML.format(arm=arm, damp=damp, I=I_axis, I2=I_axis*2))
        d = mujoco.MjData(m); d.ctrl[0] = tau
        v = []
        for i in range(600):
            mujoco.mj_step(m, d); v.append(d.qvel[0])
        v = np.array(v)
        acc0 = v[0] / 0.002
        print(f'MJ arm={arm} damp={damp}: accel(first step)={acc0:.3f} rad/s2 -> I_eff={tau/acc0:.3e}; v(1.2s)={v[-1]:.4f} rad/s (tau/damp={tau/damp if damp else float("inf"):.4f})')
