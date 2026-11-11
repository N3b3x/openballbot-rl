"""GPU thermal / memory guard for long training sessions.
Logs temperature, power, clocks, utilisation and dedicated/shared GPU memory every INTERVAL seconds to outputs/study/warp_ppo/gpu_guard.csv and
throttles the training jobs when needed by suspending whole jobs (resumed automatically):
  - temperature >= HOT for 2 samples in a row: suspend the most recently started job; resume the oldest suspended job when temperature <= COOL
  - dedicated GPU memory >= MEM_GB (WDDM starts paging to system RAM above that and everything slows down by 5 to 10x): suspend one job
  - hardware or software thermal slowdown flagged by the driver counts as HOT
Only processes whose command line contains one of PATTERNS are touched. Run:  python gpu_guard.py [--hot 83] [--cool 74] [--mem 15.2] [--interval 20]
"""
import argparse, csv, subprocess, time
from pathlib import Path
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--hot", type=float, default=83.0); ap.add_argument("--cool", type=float, default=74.0)
ap.add_argument("--mem", type=float, default=15.2); ap.add_argument("--interval", type=float, default=20.0)
a = ap.parse_args()
PATTERNS = ("train_dev.py --name", "train.py --name", "distill.py --teacher")
OUT = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo" / "gpu_guard.csv"
Q = "temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap"


def sample():
    r = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
    r = [x.strip() for x in r]
    return dict(temp=float(r[0]), power=float(r[1]), sm_mhz=float(r[2]), util=float(r[3]), mem_gb=float(r[4]) / 1024,
                hw_therm=r[5].startswith("Active"), sw_therm=r[6].startswith("Active"), pcap=r[7].startswith("Active"))


def workers():
    out = []
    for p in psutil.process_iter(["pid", "cmdline", "create_time", "status"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if any(k in cl for k in PATTERNS) and "run_queue" not in cl and "gpu_guard" not in cl:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return sorted(out, key=lambda p: p.info["create_time"])


def jobs():
    """Group launcher shim + worker (parent/child with the same command line) into one job."""
    ws = workers(); ids = {p.pid for p in ws}; res = []
    for p in ws:
        if p.ppid() in ids:
            continue
        res.append([p] + [c for c in ws if c.ppid() == p.pid])
    return res


def name_of(job):
    cl = " ".join(job[0].info["cmdline"] or [])
    return cl.split("--name")[1].split()[0] if "--name" in cl else cl[:40]


suspended = []      # list of jobs we suspended, oldest first
hot_streak = 0
OUT.parent.mkdir(parents=True, exist_ok=True)
new = not OUT.exists()
f = open(OUT, "a", newline=""); w = csv.writer(f)
if new:
    w.writerow(["time", "temp_c", "power_w", "sm_mhz", "util", "mem_gb", "hw_thermal", "sw_thermal", "power_cap", "running_jobs", "suspended_jobs", "event"]); f.flush()
print("GUARD start", flush=True)
while True:
    try:
        s = sample()
    except Exception as e:
        time.sleep(a.interval); continue
    event = ""
    hot = s["temp"] >= a.hot or s["hw_therm"] or s["sw_therm"]
    hot_streak = hot_streak + 1 if hot else 0
    susp_pids = {q.pid for j in suspended for q in j}
    try:
        live = [j for j in jobs() if j[0].pid not in susp_pids]
    except psutil.Error:
        time.sleep(2); continue
    if hot_streak >= 2 and len(live) > 0:
        j = live[-1]
        for p in j:
            try: p.suspend()
            except psutil.Error: pass
        suspended.append(j); event = f"GUARD hot {s['temp']:.0f}C: suspended {name_of(j)}"; hot_streak = 0
    elif s["mem_gb"] >= a.mem and len(live) > 1:
        j = live[-1]
        for p in j:
            try: p.suspend()
            except psutil.Error: pass
        suspended.append(j); event = f"GUARD memory {s['mem_gb']:.1f} GB: suspended {name_of(j)}"
    elif suspended and s["temp"] <= a.cool and not s["hw_therm"] and not s["sw_therm"] and s["mem_gb"] < a.mem - 2.0:
        j = suspended.pop(0)
        for p in j:
            try: p.resume()
            except psutil.Error: pass
        event = f"GUARD cool {s['temp']:.0f}C: resumed {name_of(j)}"
    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), s["temp"], s["power"], s["sm_mhz"], s["util"], round(s["mem_gb"], 2), int(s["hw_therm"]), int(s["sw_therm"]),
                int(s["pcap"]), len(live), len(suspended), event]); f.flush()
    if event:
        print(event, flush=True)
    time.sleep(a.interval)
