"""Task 2: run the official MJCF -> USD route (mujoco-usd-converter 0.6.1, shipped with Isaac Sim 6.2) on dz_nom and list what survives.

Step 1 (system python):  python import_check.py prep     writes import_test/dz_nom_plane.xml (hfield replaced by a plane, absolute asset dirs)
Step 2 (isaac venv):     mujoco_usd_converter import_test/dz_nom_plane.xml import_test/usd --no-layer-structure
Step 3 (isaac venv):     python import_check.py inspect  dumps masses, joints, colliders, friction attributes of the produced USD
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
T = HERE / "import_test"

if sys.argv[1] == "prep":
    src = (ROOT / "ballbot_gym/models/variants/dz_nom.xml").read_text()
    mesh = str((ROOT / "ballbot_gym/models/meshes").resolve()).replace("\\", "/")
    tex = str((ROOT / "ballbot_gym/models/textures").resolve()).replace("\\", "/")
    src = src.replace('meshdir="../meshes"', f'meshdir="{mesh}"').replace('texturedir="../textures"', f'texturedir="{tex}"')
    src = re.sub(r"<hfield[^>]*/>", "", src)
    src = re.sub(r'<geom name="terrain"[^>]*/>', '<geom name="terrain" type="plane" size="5 5 0.1" material="floor_chk"/>', src)
    T.mkdir(exist_ok=True)
    (T / "dz_nom_plane.xml").write_text(src)
    print("wrote", T / "dz_nom_plane.xml")
else:
    from pxr import Usd, UsdGeom, UsdPhysics
    f = next((T / "usd").glob("*.usd*"))
    stage = Usd.Stage.Open(str(f))
    out = dict(file=str(f), bodies=[], joints=[], colliders=[], materials=[], other=[])
    for prim in stage.Traverse():
        tn = prim.GetTypeName()
        attrs = {a.GetName(): (str(a.Get()) if a.Get() is not None else None) for a in prim.GetAttributes()
                 if any(k in a.GetName() for k in ("mass", "Inertia", "inertia", "centerOfMass", "axis", "localPos", "friction", "Friction", "radius", "height", "limit", "armature", "damping", "stiffness", "restitution", "principal"))}
        apis = [str(s) for s in prim.GetAppliedSchemas()]
        if "PhysicsRigidBodyAPI" in apis or "PhysicsMassAPI" in apis:
            out["bodies"].append(dict(path=str(prim.GetPath()), type=tn, apis=apis, attrs=attrs))
        elif "Joint" in tn:
            out["joints"].append(dict(path=str(prim.GetPath()), type=tn, attrs=attrs))
        elif "PhysicsCollisionAPI" in apis:
            out["colliders"].append(dict(path=str(prim.GetPath()), type=tn, apis=apis, attrs=attrs))
        elif "PhysicsMaterialAPI" in apis or tn == "Material":
            out["materials"].append(dict(path=str(prim.GetPath()), apis=apis, attrs=attrs))
        elif attrs:
            out["other"].append(dict(path=str(prim.GetPath()), type=tn, attrs=attrs))
    json.dump(out, open(T / "inspect.json", "w"), indent=1)
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})
