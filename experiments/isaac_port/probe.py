"""Smoke test: start Isaac Sim headless, report versions, GPU, and PhysX schema attributes relevant to friction/solver."""
import sys, time
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_STARTED_S", time.time() - t0, flush=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
import omni.kit.app
print("kit", omni.kit.app.get_app().get_build_version(), flush=True)
from pxr import Usd, PhysxSchema, UsdPhysics
reg = Usd.SchemaRegistry()
for api in ("PhysxMaterialAPI", "PhysxSceneAPI", "PhysxRigidBodyAPI", "PhysxJointAPI", "PhysxArticulationAPI", "PhysxCollisionAPI"):
    d = reg.FindAppliedAPIPrimDefinition(api)
    print(api, sorted(d.GetPropertyNames()) if d else None, flush=True)
app.close()
