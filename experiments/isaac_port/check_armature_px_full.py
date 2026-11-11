"""CHECK 1b: read back armature/damping PhysX holds for the wheel DOFs (and roller DOFs) of the full emulated robot."""
import numpy as np
from common import DesignCfg
from backend_px import PXBackend
be = PXBackend(DesignCfg("nom"))
a = be.art
arm = a.get_dof_armatures()[0]; dmp = a.get_dof_dampings()[0]
print("FULL wheel dofs armature", arm[be.wdof], "damping", dmp[be.wdof])
print("FULL roller dofs armature unique", np.unique(arm[be.rdof]), "damping unique", np.unique(dmp[be.rdof]))
