"""Controller bundle: the JSON file written by experiments/warp_ppo/export_policy.py plus the actor weights it points to.

The bundle is the single source of every number the deployment needs (observation layout and scales, history length, action scaling,
LQR gains per design, adaptive-authority and motion-layer parameters, design vectors, kinematic offsets, control rate). Nothing in the
core hard-codes a value that the bundle carries.
"""
import json
from pathlib import Path

import numpy as np

BUNDLE_FORMAT = "ballbot_controller_bundle"


class Bundle:
    def __init__(self, data, root):
        if data.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a controller bundle (format={data.get('format')!r})")
        self.data, self.root = data, Path(root)

    # ---- sections (plain dicts; see export_policy.py for the schema)
    def __getitem__(self, k):
        return self.data[k]

    @property
    def dt(self):
        return float(self.data["control"]["dt"])

    @property
    def design_names(self):
        return list(self.data["designs"]["table"].keys())

    def design(self, name=None):
        """Per-design constants: tau (N m, actuator limit), f (Froude speed factor), dvec (6), lqr (K, W, r_ball), kinematics."""
        name = name or self.data["designs"]["nominal"]
        if name not in self.data["designs"]["table"]:
            raise KeyError(f"design {name!r} not in bundle (have {self.design_names})")
        return self.data["designs"]["table"][name]

    @property
    def track_point(self):
        """Goal point the policy was trained with: 'ball_centre' (goal_centre runs, g39 and later) or 'ball_body_origin' (older runs).
        Bundles exported before the goal_point section existed fall back to the estimator setting, then to ball_body_origin."""
        gp = self.data.get("goal_point") or {}
        return gp.get("track_point") or self.data.get("estimator", {}).get("track_point") or "ball_body_origin"

    @property
    def goal_centre(self):
        return self.track_point == "ball_centre"

    def path(self, key):
        """Absolute path of a file referenced in policy.files."""
        return self.root / self.data["policy"]["files"][key]

    def weights(self):
        """Actor weights and input normalisation as numpy arrays (from the .npz written by the exporter)."""
        z = np.load(self.path("weights"))
        return {k: z[k] for k in z.files}


def load_bundle(path):
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return Bundle(data, path.parent)
