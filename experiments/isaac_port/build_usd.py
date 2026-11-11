"""Build the ballbot as a USD/PhysX articulation from the exported MuJoCo data (export_model.export).

mode "roller": each omni-wheel contact is a passive roller sphere on a free revolute joint (axis = grip direction), the
only way to get anisotropic wheel/ball friction with PhysX's isotropic friction model.
mode "naive": capsule collider on the wheel with one isotropic friction coefficient, i.e. what a plain MJCF import gives
(MuJoCo <pair friction="0.001 1.0"> has no PhysX equivalent and is dropped / replaced by one material).

Must run inside a started SimulationApp (needs pxr + PhysxSchema).
"""
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


def quat_from_to(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    w = 1.0 + a @ b
    if w < 1e-9:
        p = np.cross(a, [1, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1, 0])
        p /= np.linalg.norm(p)
        return Gf.Quatf(0.0, *map(float, p))
    q = np.array([w, *v])
    q /= np.linalg.norm(q)
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def qmat(q):  # (w,x,y,z) -> R
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) / 2
    q = np.zeros(4)
    q[1 + i] = s
    q[0] = (R[k, j] - R[j, k]) / (4 * s)
    q[1 + j] = (R[j, i] + R[i, j]) / (4 * s)
    q[1 + k] = (R[k, i] + R[i, k]) / (4 * s)
    return q


def _mass(prim, mass, com, diag, quat=(1, 0, 0, 0)):
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diag)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(*map(float, quat)))


def _rigid(prim, report=False):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if report:
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)


def build(stage, root, M, mode="roller", fric=1.0, mu_grip=1.0, roller_cap_r=0.012, roller_mass=0.004, n_rollers=12, roller_friction=0.0, slide_range=0.04, element='roller', slide_k=20.0, slide_d=0.2,
          solver_pos=32, solver_vel=4, with_ground=True, ground_mu=1.0):
    """M = export_model.export(design). World placement at the reset pose, yaw 0. Returns dict of prim paths."""
    UsdGeom.Xform.Define(stage, root)
    paths = {}
    # ---- materials
    def material(path, mu):
        mat = UsdShade.Material.Define(stage, path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(mu)
        pm.CreateDynamicFrictionAttr(mu)
        pm.CreateRestitutionAttr(0.0)
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px.CreateFrictionCombineModeAttr("max")
        px.CreateRestitutionCombineModeAttr("min")
        return mat

    mu_wheel = mu_grip * fric
    mat_ball = material(root + "/Looks/ball_mat", mu_wheel)
    mat_roll = material(root + "/Looks/roller_mat", mu_wheel)
    mat_misc = material(root + "/Looks/misc_mat", 0.5)

    def collider(prim, mat, contact_offset=0.002, rest_offset=0.0):
        UsdPhysics.CollisionAPI.Apply(prim)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(contact_offset)
        pc.CreateRestOffsetAttr(rest_offset)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # ---- ball (separate rigid body, centre at its geometric centre)
    bc = np.array(M["ball_centre_world"])
    # MuJoCo ball body COM = geom centre (ipos -0.14 equals geom offset), so frame at centre
    ball = UsdGeom.Sphere.Define(stage, root + "/ball")
    ball.CreateRadiusAttr(M["ball"]["radius"])
    ball.AddTranslateOp().Set(Gf.Vec3d(*map(float, bc)))
    _rigid(ball.GetPrim(), report=True)
    _mass(ball.GetPrim(), M["ball"]["mass"], (0, 0, 0), M["ball"]["inertia"])
    collider(ball.GetPrim(), mat_ball)
    rb = PhysxSchema.PhysxRigidBodyAPI(ball.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(solver_pos)
    rb.CreateSolverVelocityIterationCountAttr(solver_vel)
    rb.CreateLinearDampingAttr(0.0)
    rb.CreateAngularDampingAttr(0.0)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    paths["ball"] = root + "/ball"

    # ---- robot articulation root = base
    base_pos = np.array(M["base"]["pos_settled"])
    robot = UsdGeom.Xform.Define(stage, root + "/robot")
    base = UsdGeom.Xform.Define(stage, root + "/robot/base")
    base.AddTranslateOp().Set(Gf.Vec3d(*map(float, base_pos)))
    bp = base.GetPrim()
    _rigid(bp)
    UsdPhysics.ArticulationRootAPI.Apply(bp)
    art = PhysxSchema.PhysxArticulationAPI.Apply(bp)
    art.CreateSolverPositionIterationCountAttr(solver_pos)
    art.CreateSolverVelocityIterationCountAttr(solver_vel)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSleepThresholdAttr(0.0)
    art.CreateStabilizationThresholdAttr(0.0)
    # base mass properties are given at the base frame: COM = ipos
    _mass(bp, M["base"]["mass"], M["base"]["ipos"], M["base"]["inertia"], M["base"]["iquat"])
    tw = UsdGeom.Cylinder.Define(stage, root + "/robot/base/tower")
    tw.CreateRadiusAttr(M["tower"]["r"])
    tw.CreateHeightAttr(2 * M["tower"]["hl"])
    tw.CreateAxisAttr("Z")
    tw.AddTranslateOp().Set(Gf.Vec3d(*map(float, M["tower"]["pos"])))
    collider(tw.GetPrim(), mat_misc)
    paths["base"] = root + "/robot/base"

    wheels = M["wheels"]
    for k, W in enumerate(wheels):
        wpos = base_pos + np.array(W["body_pos"])
        Rw = qmat(W["body_quat"])
        wx = UsdGeom.Xform.Define(stage, f"{root}/robot/wheel_{k}")
        wx.AddTranslateOp().Set(Gf.Vec3d(*map(float, wpos)))
        qw = mat_to_quat(Rw)
        wx.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*map(float, qw)))
        wp = wx.GetPrim()
        _rigid(wp, report=(mode == 'naive'))
        _mass(wp, W["mass"], W["ipos"], W["inertia"], W["iquat"])
        # joint wheel <-> base (hinge about axis_world through anchor_world)
        ax = np.array(W["axis_world"])
        anchor = np.array(W["anchor_world"])
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/robot/wheel_joint_{k}")
        j.CreateBody0Rel().SetTargets([bp.GetPath()])
        j.CreateBody1Rel().SetTargets([wp.GetPath()])
        j.CreateAxisAttr("X")
        qa = quat_from_to([1, 0, 0], ax)                       # local x -> hinge axis (world; base and wheel_k frames differ)
        Rq = qmat([qa.GetReal(), *qa.GetImaginary()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor - base_pos)))
        j.CreateLocalRot0Attr(qa)
        j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, Rw.T @ (anchor - wpos))))
        qa1 = mat_to_quat(Rw.T @ Rq)
        j.CreateLocalRot1Attr(Gf.Quatf(*map(float, qa1)))
        jp = j.GetPrim()
        pj = PhysxSchema.PhysxJointAPI.Apply(jp)
        pj.CreateArmatureAttr(M["armature"])
        pj.CreateJointFrictionAttr(0.0)
        drv = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(0.0)
        drv.CreateDampingAttr(M["damping"] * np.pi / 180.0)   # USD angular drive damping is per degree/s
        drv.CreateMaxForceAttr(1e6)
        paths[f"wheel_joint_{k}"] = f"{root}/robot/wheel_joint_{k}"
        paths[f"wheel_{k}"] = f"{root}/robot/wheel_{k}"

        n = np.array(W["frame_normal"])        # ball centre -> wheel side
        t_grip = np.array(W["frame_t2"])
        if mode == "naive":
            cap = UsdGeom.Capsule.Define(stage, f"{root}/robot/wheel_{k}/cap")
            cap.CreateRadiusAttr(W["cap_r"])
            cap.CreateHeightAttr(2 * W["cap_hl"])
            cap.CreateAxisAttr("Z")
            # MuJoCo geom pose in the wheel body frame
            cap_pos = np.array([-0.018, -0.08, -0.053])
            cap.AddTranslateOp().Set(Gf.Vec3d(*map(float, cap_pos)))
            # capsule orientation: euler -45 9 0 (xyz intrinsic, degrees)
            e = np.radians([-45.0, 9.0, 0.0])
            cx, cy = np.cos(e[0]), np.cos(e[1])
            sx, sy = np.sin(e[0]), np.sin(e[1])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            qc = mat_to_quat(Rx @ Ry)
            cap.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*map(float, qc)))
            collider(cap.GetPrim(), mat_roll)
        else:
            # ring of passive barrel rollers (capsules whose long axis is the circumferential tangent), outer radius chosen so the
            # settled geometry equals MuJoCo's settled geometry (cap radius minus soft-contact penetration)
            r_out = W["cap_r"] - W["penetration"]
            rho = roller_cap_r
            Rr = r_out - rho
            d0 = np.array(W["contact_world"]) - anchor
            s_ax = float(d0 @ ax / np.linalg.norm(ax))
            axn = ax / np.linalg.norm(ax)
            e1 = d0 - (d0 @ axn) * axn
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axn, e1)
            half = np.pi * Rr / n_rollers
            for i in range(n_rollers):
                ph = 2 * np.pi * i / n_rollers
                rad = np.cos(ph) * e1 + np.sin(ph) * e2
                tan = -np.sin(ph) * e1 + np.cos(ph) * e2
                cen = anchor + s_ax * axn + Rr * rad
                if element == "slider":
                    Rn = np.stack([axn, tan, np.cross(axn, tan)], axis=1)       # x = axle (slide axis), y = tangent
                    cap_axis = "Y"
                else:
                    Rn = np.stack([tan, rad, -axn], axis=1)                     # x = tangent (roller spin axis)
                    cap_axis = "X"
                rx = UsdGeom.Xform.Define(stage, f"{root}/robot/rl_{k}_{i}")
                rx.AddTranslateOp().Set(Gf.Vec3d(*map(float, cen)))
                rx.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*map(float, mat_to_quat(Rn))))
                rp = rx.GetPrim()
                _rigid(rp, report=True)
                ii = max(0.4 * roller_mass * rho ** 2, 2e-7)
                _mass(rp, roller_mass, (0, 0, 0), (ii, ii, ii))
                cap = UsdGeom.Capsule.Define(stage, f"{root}/robot/rl_{k}_{i}/shape")
                cap.CreateRadiusAttr(rho)
                cap.CreateHeightAttr(2 * half)
                cap.CreateAxisAttr(cap_axis)
                collider(cap.GetPrim(), mat_roll)
                if element == "slider":
                    jr = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/robot/roller_joint_{k}_{i}")
                else:
                    jr = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/robot/roller_joint_{k}_{i}")
                jr.CreateBody0Rel().SetTargets([wp.GetPath()])
                jr.CreateBody1Rel().SetTargets([rp.GetPath()])
                jr.CreateAxisAttr("X")
                if element == "slider":
                    jr.CreateLowerLimitAttr(-slide_range)
                    jr.CreateUpperLimitAttr(slide_range)
                jr.CreateLocalPos0Attr(Gf.Vec3f(*map(float, Rw.T @ (cen - wpos))))
                jr.CreateLocalRot0Attr(Gf.Quatf(*map(float, mat_to_quat(Rw.T @ Rn))))
                jr.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                jr.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
                PhysxSchema.PhysxJointAPI.Apply(jr.GetPrim()).CreateJointFrictionAttr(roller_friction)
                drv2 = UsdPhysics.DriveAPI.Apply(jr.GetPrim(), "linear" if element == "slider" else "angular")
                drv2.CreateTypeAttr("force")
                drv2.CreateStiffnessAttr(slide_k if element == "slider" else 0.0)
                drv2.CreateDampingAttr(slide_d if element == "slider" else 0.0)
                drv2.CreateMaxForceAttr(1e6)
                paths[f"roller_{k}_{i}"] = f"{root}/robot/rl_{k}_{i}"

    if with_ground:
        g = UsdGeom.Cube.Define(stage, root + "/ground")
        g.CreateSizeAttr(1.0)
        g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
        g.AddScaleOp().Set(Gf.Vec3f(40, 40, 1))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        gm = material(root + "/Looks/ground_mat", ground_mu)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(gm, UsdShade.Tokens.weakerThanDescendants, "physics")
        paths["ground"] = root + "/ground"
    return paths
