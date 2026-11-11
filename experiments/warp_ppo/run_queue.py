"""Run a queue of training commands on the one GPU with a concurrency limit, a memory-aware launch rule and simple dependencies.
One command per line in the queue file (lines starting with # are ignored). The job name is the value of --name; stdout goes to <out>/<name>.out.
A line may start with  after:NAME[,NAME2]  to wait until those runs have printed their final 'done' line.
  python run_queue.py queue.txt --par 3 --mem-launch 10.0
A new job is launched only if fewer than --par jobs of this queue are running AND the dedicated GPU memory in use is below --mem-launch GB
(one 4096-world job needs about 3 GB, one 8192-world 512x3 job about 4.5 GB; above ~15 GB Windows pages GPU memory into system RAM and everything slows 10x).
Measured on the RTX 5080: aggregate ~150k env-steps/s with 2 to 3 concurrent jobs."""
import argparse, re, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("queue"); ap.add_argument("--par", type=int, default=2); ap.add_argument("--mem-launch", type=float, default=10.0)
ap.add_argument("--py", default=str(Path(__file__).resolve().parents[1] / "mjx_port" / ".venv" / "Scripts" / "python.exe"))
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
out = HERE.parents[1] / "outputs" / "study" / "warp_ppo"


def mem_gb():
    try:
        return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()) / 1024
    except Exception:
        return 0.0


def finished(name):
    f = out / f"{name}.out"
    try:
        return any(l.startswith("done ") for l in open(f, errors="ignore").read().splitlines()[-5:])
    except OSError:
        return False


jobs = []
for l in open(a.queue):
    l = l.strip()
    if not l or l.startswith("#"):
        continue
    deps = []
    m = re.match(r"after:(\S+)\s+(.*)", l)
    if m:
        deps, l = m.group(1).split(","), m.group(2)
    jobs.append((deps, l))
running = []
last_start = 0.0
while jobs or running:
    running = [(p, n) for p, n in running if p.poll() is None]
    for j in list(jobs):
        deps, cmd = j
        if len(running) >= a.par or time.time() - last_start < 90:
            break
        if not all(finished(d) for d in deps) or mem_gb() >= a.mem_launch:
            continue
        name = re.search(r"--name\s+(\S+)", cmd).group(1)
        p = subprocess.Popen([a.py, *cmd.split()], stdout=open(out / f"{name}.out", "w"), stderr=subprocess.STDOUT, cwd=HERE)
        running.append((p, name)); jobs.remove(j); last_start = time.time()
        print("started", name, time.strftime("%H:%M:%S"), flush=True)
    time.sleep(20)
print("queue done", flush=True)
