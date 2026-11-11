"""Print one line whenever a run crosses a new --every decisions boundary; exit when all summaries exist."""
import argparse, glob, time
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--every", type=int, default=100000)
a = ap.parse_args()
last = {}
while True:
    done = 0
    for n in a.runs:
        d = Path("outputs/study/runs") / n
        if (d / "summary.json").exists():
            done += 1
        f = d / "progress.csv"
        if f.exists():
            try:
                r = pd.read_csv(f).iloc[-1]
                ts = int(r["time/total_timesteps"])
                b = ts // a.every
                if last.get(n) != b:
                    last[n] = b
                    print(f"{n} ts={ts} len={r['rollout/ep_len_mean']:.0f} rew={r['rollout/ep_rew_mean']:.1f} fps={int(r['time/fps'])}", flush=True)
            except Exception:
                pass
    if done == len(a.runs):
        print("ALL_DONE", flush=True)
        break
    time.sleep(30)
