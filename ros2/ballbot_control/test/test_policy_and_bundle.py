"""Exported actor and bundle consistency."""
import numpy as np
import pytest

from ballbot_control.core.policy import MlpPolicy, make_policy


def test_bundle_dims(bundle):
    p, o = bundle["policy"], bundle["observation"]
    assert p["input_dim"] == o["obs_dim"] + o["hist_k"] * (o["obs_dim"] + o["act_dim"]) + 6 == 255
    assert o["hist_k"] == 8 and o["obs_dim"] == 25
    assert bundle["control"]["rate_hz"] == 100.0
    assert sum(e["size"] for e in o["layout"]) == 25
    assert bundle["action"]["rs_adapt"]["lo"] == 0.15 and bundle["action"]["rs_adapt"]["hi"] == 0.3 and bundle["action"]["rs_adapt"]["v_db"] == 0.05


def test_numpy_policy_matches_sim_actions(fixture_rollout, bundle):
    """Recorded noisy-phase observations (as in training: obs noise, latency on) -> recorded GPU actions."""
    pol = make_policy(bundle, "numpy")
    y = pol(fixture_rollout["x_noisy"])
    assert np.abs(y - fixture_rollout["act_noisy"]).max() < 1e-5


def test_onnx_policy_matches_numpy(fixture_rollout, bundle):
    pytest.importorskip("onnxruntime")
    x = fixture_rollout["x_noisy"]
    y_onnx = make_policy(bundle, "onnx")(x)
    y_np = make_policy(bundle, "numpy")(x)
    assert np.abs(y_onnx - y_np).max() < 1e-5
    assert np.abs(make_policy(bundle, "onnx")(x[0]) - y_np[0]).max() < 1e-5      # single-sample call


def test_torchscript_matches_numpy(fixture_rollout, bundle):
    torch = pytest.importorskip("torch")
    ts = torch.jit.load(str(bundle.path("torchscript")))
    x = fixture_rollout["x_noisy"]
    with torch.no_grad():
        y = ts(torch.from_numpy(x)).numpy()
    assert np.abs(y - make_policy(bundle)(x)).max() < 1e-5


def test_export_verification_recorded(bundle):
    v = bundle["verification"]
    for res in v.values():
        for k in ("torchscript", "numpy_weights_float64", "recorded_sim_actions_vs_reference"):
            assert res[k] < 1e-5
        if res.get("onnx_onnxruntime") is not None:
            assert res["onnx_onnxruntime"] < 1e-5


def test_lqr_gains_recomputable(bundle):
    """K in the bundle = discrete LQR on the planar model json (lqr_gpu.design_tables), so a new design needs only its A, B."""
    scipy = pytest.importorskip("scipy")
    import json
    import scipy.linalg as la
    from scipy.signal import cont2discrete
    from conftest import REPO
    L = bundle["lqr"]
    for n in bundle.design_names:
        M = json.load(open(REPO / "ballbot_gym" / "models" / "variants" / f"dz_{n}.json"))
        A, B, dt = np.array(M["A"]), np.array(M["B"]), L["dt"]
        Q, r = np.diag(L["q"]), L["r"]
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), dt)
        X = la.solve_discrete_are(Ad, Bd, Q * dt, np.array([[r * dt]]))
        K = np.linalg.solve(np.array([[r * dt]]) + Bd.T @ X @ Bd, Bd.T @ X @ Ad).ravel()
        np.testing.assert_allclose(bundle.design(n)["lqr"]["K"], K, rtol=1e-5, atol=1e-6)


def test_policy_input_dim_guard(bundle):
    w = bundle.weights()
    p = MlpPolicy(w)
    with pytest.raises(ValueError):
        p(np.zeros(254))
