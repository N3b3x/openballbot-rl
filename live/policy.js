// Forward pass of the exported PPO policy (see scripts/site/export_policy.py). Pure JS, no dependencies.
// Reproduces SB3 `policy.predict(normalise(obs), deterministic=True)`: observation normalisation (VecNormalize on five keys),
// tanh MLP 18 -> 256 -> 256 -> 3, clipped to the action bounds.
export class Policy {
  constructor(spec) {
    this.spec = spec;
    this.keys = spec.obs_keys;
    this.layers = spec.layers.map(L => ({ W: L.W.map(r => Float64Array.from(r)), b: Float64Array.from(L.b) }));
    this.nin = this.keys.reduce((s, k) => s + spec.obs_sizes[k], 0);
    this.low = spec.action.low; this.high = spec.action.high;
    this.h = this.layers.map(L => new Float64Array(L.b.length));
    this.x = new Float64Array(this.nin);
  }
  // obs: {key: array-like}. Returns {action: Float64Array(3), x: normalised input (for inspection)}
  predict(obs) {
    const s = this.spec, x = this.x; let o = 0;
    for (const k of this.keys) {
      const v = obs[k], nm = s.norm[k];
      for (let i = 0; i < s.obs_sizes[k]; i++) {
        let z = v[i];
        if (nm) z = Math.min(s.clip_obs, Math.max(-s.clip_obs, (z - nm.mean[i]) / Math.sqrt(nm.var[i] + s.norm_eps)));
        x[o++] = z;
      }
    }
    let inp = x;
    for (let l = 0; l < this.layers.length; l++) {
      const { W, b } = this.layers[l], out = this.h[l], last = l === this.layers.length - 1;
      for (let j = 0; j < b.length; j++) {
        let acc = b[j]; const w = W[j];
        for (let i = 0; i < inp.length; i++) acc += w[i] * inp[i];
        out[j] = last ? acc : Math.tanh(acc);
      }
      inp = out;
    }
    const a = new Float64Array(this.low.length);
    for (let i = 0; i < a.length; i++) a[i] = Math.min(this.high[i], Math.max(this.low[i], inp[i]));
    return a;
  }
}
