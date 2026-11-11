"""Patch train/evaluate scripts to pass ToF / height_scan config and to evaluate book heights."""
for p, pairs in {
    "scripts/study/train_ppo.py": [(
        'env_config = {k: cfg[k] for k in ("camera", "env", "logging")}\n    fns = [',
        'env_config = {k: cfg[k] for k in ("camera", "env", "logging", "tof", "height_scan") if k in cfg}\n    fns = [')],
    "scripts/study/evaluate.py": [
        ('env_config = {k: cfg[k] for k in ("camera", "env", "logging")}',
         'env_config = {k: cfg[k] for k in ("camera", "env", "logging", "tof", "height_scan") if k in cfg}'),
        ('        if base.terrain_type == "perlin":\n            tcfg["amplitude"] = float(amp)\n            tcfg.pop("amplitude_range", None)\n',
         '        if base.terrain_type == "perlin":\n            tcfg["amplitude"] = float(amp)\n            tcfg.pop("amplitude_range", None)\n'
         '        elif base.terrain_type == "book":  # the "amplitude" column holds the book height [m]\n'
         '            tcfg["step_height_m"] = float(amp)\n            tcfg.pop("ranges", None)\n')],
}.items():
    s = open(p, encoding="utf-8").read()
    for o, n in pairs:
        assert s.count(o) == 1, (p, o[:50])
        s = s.replace(o, n, 1)
    open(p, "w", encoding="utf-8").write(s)
print("ok")
