"""Fall probe: 30 episodes of random training-distribution commands (deterministic policy, nominal robot).
Prints mean episode length, number of falls and the command active at each fall.
  python scripts/study/eval_falls.py <run> <config>"""
import sys, yaml, numpy as np, torch
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)); _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2]))
from policy_io import load_controller
from ballbot_rl.training.utils import make_ballbot_env
import ballbot_gym
run, conf = sys.argv[1], sys.argv[2]
cfg=yaml.safe_load(open(f"configs/env/{conf}.yaml")); cfg["camera"]["enabled"]=False
DR = len(sys.argv) > 3 and sys.argv[3] == "dr"   # keep randomisation + pushes from the config (robustness probe)
if DR:
    cfg["env"]["dr"] = yaml.safe_load(open("configs/env/rob_a.yaml"))["env"]["dr"]
else:
    cfg["env"].pop("dr", None)
env=make_ballbot_env(terrain_config=cfg["terrain"],reward_config=cfg["reward"],env_config={k:cfg[k] for k in ("camera","env","logging")},seed=0,eval_env=True)()
model,norm=load_controller(run, env)
b=env.unwrapped; falls=[]; lens=[]
for ep in range(30):
    b.cmd_fixed=None; obs,_=env.reset(seed=1000+ep)
    for t in range(800):
        with torch.no_grad(): a,_=model.predict(norm(obs),deterministic=True)
        obs,r,te,tr,info=env.step(a)
        if te: falls.append((t, np.round(b.cmd,2), round(info["off_ball_deg"],1))); break
        if tr: break
    lens.append(t+1)
print("mean len", np.mean(lens), "falls", len(falls), "/30")
for f in falls: print(" ", f)
