"""CHECK 1, PhysX side: single wheel body on a hinge under a fixed base, constant actuation force, no contacts, gravity 0.
Uses exactly the joint-attribute lines of build_usd.py (PhysxJointAPI armature + USD angular drive damping * pi/180) unless
variant says otherwise.  Run:  C:/Users/Owner/isaac_venv/Scripts/python.exe check_armature_px.py   (CPU PhysX, headless)
"""
import json, os, sys
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np
from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd, omni.physx, omni.timeline
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils
sys.path.insert(0, str(Path(__file__).parent))
import build_usd
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.physx.tensors")
import importlib
T = importlib.import_module("omni.physics.tensors")

I_AXIS = json.load(open(Path(__file__).with_name('check_armature_I.json')))['I_axis']
TAU = 0.1
# (label, armature attr, drive damping attr [USD units, per degree], extra)
VARIANTS = [
    ("arm0 damp0",               0.0,   0.0),
    ("arm0.005 damp0",           0.005, 0.0),
    ("arm0 damp0.1*pi/180",      0.0,   0.1 * np.pi / 180),
    ("arm0.005 damp0.1*pi/180",  0.005, 0.1 * np.pi / 180),     # == build_usd.py as written
    ("arm0.005 damp0.1 raw",     0.005, 0.1),                    # unscaled value (old bug case)
    ("arm0.005 damp0.1*180/pi",  0.005, 0.1 * 180 / np.pi),
]
if os.environ.get("PX_SCAN"):
    VARIANTS = [("arm0.005 d%.3g ang0" % d, 0.005, d * np.pi / 180) for d in (0.01, 0.03, 0.1, 0.3, 1.0)] +                [("arm0.005 d0.1 angdefault", 0.005, 0.1 * np.pi / 180)] +                [("arm%.3g d0.1 ang0" % a, a, 0.1 * np.pi / 180) for a in (0.0005, 0.001, 0.002, 0.01)]
ANG0 = bool(os.environ.get("PX_SCAN"))
ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.Xform.Define(stage, "/World")
sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); sc.CreateGravityMagnitudeAttr(0.0)
ps = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
ps.CreateTimeStepsPerSecondAttr(500); ps.CreateSolverTypeAttr("TGS"); ps.CreateEnableGPUDynamicsAttr(False); ps.CreateBroadphaseTypeAttr("MBP")
for i, (lab, arm, ddamp) in enumerate(VARIANTS):
    root = f"/World/t{i}"
    UsdGeom.Xform.Define(stage, root)
    b = UsdGeom.Xform.Define(stage, root + "/base"); b.AddTranslateOp().Set(Gf.Vec3d(i * 2.0, 0, 1))
    build_usd._rigid(b.GetPrim()); build_usd._mass(b.GetPrim(), 1.0, (0, 0, 0), (0.1, 0.1, 0.1))
    UsdPhysics.ArticulationRootAPI.Apply(b.GetPrim())
    a = PhysxSchema.PhysxArticulationAPI.Apply(b.GetPrim())
    a.CreateSolverPositionIterationCountAttr(32); a.CreateSolverVelocityIterationCountAttr(4)
    a.CreateSleepThresholdAttr(0.0); a.CreateStabilizationThresholdAttr(0.0)
    fj = UsdPhysics.FixedJoint.Define(stage, root + "/anchor"); fj.CreateBody1Rel().SetTargets([b.GetPath()])
    fj.CreateLocalPos0Attr(Gf.Vec3f(i * 2.0, 0, 1))
    w = UsdGeom.Xform.Define(stage, root + "/wheel"); w.AddTranslateOp().Set(Gf.Vec3d(i * 2.0, 0, 1))
    build_usd._rigid(w.GetPrim()); build_usd._mass(w.GetPrim(), 0.0893, (0, 0, 0), (I_AXIS, 2 * I_AXIS, 2 * I_AXIS))
    if ANG0 and "angdefault" not in lab:
        rbx = PhysxSchema.PhysxRigidBodyAPI(w.GetPrim()); rbx.CreateAngularDampingAttr(0.0); rbx.CreateLinearDampingAttr(0.0)
    j = UsdPhysics.RevoluteJoint.Define(stage, root + "/wheel_joint_0")
    j.CreateBody0Rel().SetTargets([b.GetPath()]); j.CreateBody1Rel().SetTargets([w.GetPath()]); j.CreateAxisAttr("X")
    jp = j.GetPrim()
    pj = PhysxSchema.PhysxJointAPI.Apply(jp)
    pj.CreateArmatureAttr(arm)                          # same as build_usd.py
    pj.CreateJointFrictionAttr(0.0)
    drv = UsdPhysics.DriveAPI.Apply(jp, "angular")
    drv.CreateTypeAttr("force"); drv.CreateStiffnessAttr(0.0); drv.CreateDampingAttr(float(ddamp)); drv.CreateMaxForceAttr(1e6)
tl = omni.timeline.get_timeline_interface(); tl.play(); app.update(); tl.pause()
sid = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
sv = T.create_simulation_view("numpy", sid)
sim = omni.physx.get_physx_simulation_interface()
arts = [sv.create_articulation_view(f"/World/t{i}/base") for i in range(len(VARIANTS))]
idx = np.array([0], dtype=np.int32)
print("PX dof counts", [a.max_dofs for a in arts])
for (lab, arm, dd), a in zip(VARIANTS, arts):
    try: print(f"PX view reports [{lab}]: armature={a.get_dof_armatures()}  damping={a.get_dof_dampings()}  stiffness={a.get_dof_stiffnesses()}  friction={a.get_dof_friction_coefficients() if hasattr(a,'get_dof_friction_coefficients') else 'n/a'}")
    except Exception as e: print("view query failed", lab, repr(e)[:150])
for a in arts:
    a.set_dof_actuation_forces(np.full((1, 1), TAU, np.float32), idx)
vs = [[] for _ in arts]; t = 0.0
for step in range(600):
    sim.simulate(0.002, t); sim.fetch_results(); t += 0.002
    for k, a in enumerate(arts): vs[k].append(float(a.get_dof_velocities()[0, 0]))
for (lab, arm, dd), v in zip(VARIANTS, vs):
    v = np.array(v); acc = v[0] / 0.002
    print(f"PX [{lab}]: accel(first step)={acc:.3f} rad/s2 -> I_eff={TAU/acc:.3e}; v(0.2s)={v[99]:.4f}; v(1.2s)={v[-1]:.4f} rad/s (tau/damp_rad={TAU/(dd*180/np.pi) if dd else float('inf'):.4f})")
app.close()
