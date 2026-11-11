"""Render the side-by-side clips (needs outputs/study/isaac_port/rec/*.npz from record.py for both engines) and write
videos/clips.json with data-derived outcome text.   python render_all.py [clip names]"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from common import ROOT

OUT = ROOT / "outputs/study/isaac_port"
NOTE = "Simulation only. No hardware claims."
JOBS = [
    # name, left, right, design, title, scenario, extra args, description
    ("nom_settle", "mj_nom_settle", "px_nom_settle", "nom", "LQR hold, robot dropped 2 cm onto the ball", "settle", [],
     "Robot and ball dropped 2 cm from the settled pose with the LQR holding zero command. Compares the ball and wheel contact transient and the contact forces."),
    ("nom_pulse", "mj_nom_pulse", "px_nom_pulse", "nom", "Open-loop torque pulse, 0.5 Nm on wheels 0 and 2 for 0.1 s (0.3x speed)", "torque_step", ["--slow"],
     "No feedback: 0.5 Nm on wheels 0 and 2 for 0.1 s after 0.1 s of rest. The unstable plant is only comparable for the first 0.3 s; ball and body response is the check."),
    ("nom_track04", "mj_nom_track04", "px_nom_track04", "nom", "LQR, 0.4 scaled speed", "lqr_0.4", [],
     "LQR (nominal gains) tracking 0.4 of the scaled speed along +x for 5 s."),
    ("nom_track08", "mj_nom_track08", "px_nom_track08", "nom", "LQR, 0.8 scaled speed", "lqr_0.8", [],
     "LQR tracking 0.8 of the scaled speed along +x for 5 s."),
    ("nom_push", "mj_nom_push", "px_nom_push", "nom", "LQR, 1x body-weight 50 ms push at 1.5 s", "push_1x", [],
     "One horizontal 50 ms push of 1x total weight (35 degrees from +x) while driving at 0.4 scaled speed."),
    ("zeta66_push", "mj_zeta66_push", "px_zeta66_push", "zeta66", "Contact zenith 66 deg, same 1x push", "push_1x_zeta66", [],
     "Design variant with the wheels at 66 degrees zenith (more upright contact) under the same push."),
    ("zeta41_push", "mj_zeta41_push", "px_zeta41_push", "zeta41", "Contact zenith 41 deg, same 1x push", "push_1x_zeta41", [],
     "Design variant with the wheels at 41 degrees zenith under the same push (known to fall more often in MuJoCo)."),
    ("zeta41_track08", "mj_zeta41_track08", "px_zeta41_track08", "zeta41", "Contact zenith 41 deg, 0.8 scaled speed", "lqr_0.8_zeta41", [],
     "Zenith 41 degree design at 0.8 scaled speed; MuJoCo with the same LQR loses balance."),
    ("naive_track04", "mj_nom_track04", "pxnaive_nom_track04", "nom", "PhysX naive import (isotropic capsule wheels), 0.4 scaled speed", "naive_import_lqr_0.4", [],
     "What a plain MJCF to PhysX conversion gives: capsule wheels with one isotropic friction coefficient (the anisotropic <pair friction> cannot be represented). Failure case."),
]


def summarize(rec):
    fell = int(rec["fell_at"])
    tail = slice(max(0, len(rec["t"]) - 150), None)
    return dict(fell=fell >= 0, fell_t=float(rec["t"][fell]) if fell >= 0 else None, final_speed=float(rec["speed"][tail].mean()),
                cmd=float(rec["cmd"][-1]), max_tilt=float(rec["tilt"].max()), t_end=float(rec["t"][-1]),
                mean_fn=float(np.mean(rec["contacts"][:, :, 6][rec["contacts"][:, :, 8] > 0.5])) if (rec["contacts"][:, :, 8] > 0.5).any() else float("nan"))


def main():
    names = sys.argv[1:]
    clips_f = OUT / "videos" / "clips.json"
    clips = {c["id"].replace("isaac_", "", 1): c for c in json.load(open(clips_f))} if clips_f.exists() else {}
    for name, l, r, design, title, scen, extra, desc in JOBS:
        if names and name not in names:
            continue
        if not ((OUT / "rec" / f"{l}.npz").exists() and (OUT / "rec" / f"{r}.npz").exists()):
            print("skip", name)
            continue
        cmd = [sys.executable, str(Path(__file__).parent / "render.py"), "--left", l, "--right", r, "--design", design, "--title", title, "--out", name] + extra
        print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip(), flush=True)
        mp4 = OUT / "videos" / f"{name}.mp4"
        poster = OUT / "videos" / f"{name}.jpg"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", "2.0", "-i", str(mp4), "-frames:v", "1", "-q:v", "4", str(poster)], capture_output=True)
        A, B = summarize(np.load(OUT / "rec" / f"{l}.npz")), summarize(np.load(OUT / "rec" / f"{r}.npz"))
        engine_r = "PhysX 5 (Isaac Sim 6.2), naive import" if "naive" in r else "PhysX 5 (Isaac Sim 6.2), roller-ring omni wheel emulation"
        outcome = []
        for lab, S_ in (("MuJoCo", A), ("PhysX", B)):
            if "naive" in r and lab == "PhysX":
                lab = "PhysX naive"
            outcome.append(f"{lab}: " + (f"fell at {S_['fell_t']:.2f} s" if S_["fell"] else f"stays upright, mean speed {S_['final_speed']:.2f} m/s (cmd {S_['cmd']:.2f}), max tilt {S_['max_tilt']:.1f} deg") + f", mean wheel-ball normal force {S_['mean_fn']:.1f} N")
        clips[name] = dict(id=f"isaac_{name}", title=title, description=desc, engines=["MuJoCo 3.13 CPU (ContactFrameStepper)", engine_r],
                           scenario=scen, parameters=dict(design=f"dz_{design}", dt_s=0.002, control_hz=100, controller="LQR (design gains)" if scen not in ("torque_step",) else "open loop"),
                           outcome="; ".join(outcome), fell=dict(mujoco=A["fell"], physx=B["fell"]),
                           file=f"outputs/study/isaac_port/videos/{name}.mp4", poster=f"outputs/study/isaac_port/videos/{name}.jpg",
                           size_mb=round(mp4.stat().st_size / 1e6, 2), note=NOTE)
    json.dump(list(clips.values()), open(clips_f, "w"), indent=1)


if __name__ == "__main__":
    main()
