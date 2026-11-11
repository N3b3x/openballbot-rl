"""pytest setup: the core is importable without installing the package and without ROS."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from parity_replay import DEFAULT_RUN, RUNS, bundle_path, fixture_path, full_rollout_path  # noqa: E402

REPO = HERE.parents[2]


@pytest.fixture(scope="session")
def bundle():
    """Bundle of the recommended run (DEFAULT_RUN)."""
    from ballbot_control.core import load_bundle
    return load_bundle(bundle_path(DEFAULT_RUN))


@pytest.fixture(scope="session")
def fixture_rollout():
    return dict(np.load(fixture_path(DEFAULT_RUN)))


@pytest.fixture(scope="session", params=RUNS)
def run_case(request):
    """(run, bundle, fixture rollout) for every exported run."""
    from ballbot_control.core import load_bundle
    return request.param, load_bundle(bundle_path(request.param)), dict(np.load(fixture_path(request.param)))
