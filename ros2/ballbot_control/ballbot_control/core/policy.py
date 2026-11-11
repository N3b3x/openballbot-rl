"""Learned residual actor (numpy forward pass, no torch needed on the robot).

The actor is the MLP of train_dev.Net (tanh hidden layers, linear output). Its input is the 255-D actor vector built by
ObservationBuilder; normalisation ((x - mean) / sqrt(var + 1e-8), clipped to +-clip_obs) and the output clip to [-1, 1] are part of
the policy here, exactly as in the evaluation scripts. An ONNX backend is available when onnxruntime is installed; both backends take
the raw (un-normalised) actor vector.
"""
import numpy as np


class MlpPolicy:
    def __init__(self, weights, clip_obs=10.0, act_clip=1.0):
        self.mean = np.asarray(weights["mean"], dtype=np.float64)
        self.std = np.asarray(weights["std"], dtype=np.float64)
        n = int(weights["n_layers"])
        self.W = [np.asarray(weights[f"W{i}"], dtype=np.float64) for i in range(n)]
        self.b = [np.asarray(weights[f"b{i}"], dtype=np.float64) for i in range(n)]
        self.clip_obs, self.act_clip = float(clip_obs), float(act_clip)
        self.input_dim = self.W[0].shape[1]

    def __call__(self, x):
        """x (input_dim,) or (B, input_dim) raw actor vector -> action in [-1, 1]."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"actor input has {x.shape[-1]} columns, expected {self.input_dim}")
        h = np.clip((x - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return np.clip(h, -self.act_clip, self.act_clip)


class OnnxPolicy:
    """Same contract as MlpPolicy, backed by onnxruntime (normalisation and clip are inside the ONNX graph)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.input_dim = self.sess.get_inputs()[0].shape[-1]

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        y = self.sess.run(None, {self.inp: x[None] if single else x})[0].astype(np.float64)
        return y[0] if single else y


def make_policy(bundle, backend="numpy"):
    p = bundle["policy"]
    if backend == "onnx":
        return OnnxPolicy(bundle.path("onnx"))
    if backend != "numpy":
        raise ValueError(f"unknown policy backend {backend!r}")
    return MlpPolicy(bundle.weights(), clip_obs=p["clip_obs"], act_clip=p["action_clip"])
