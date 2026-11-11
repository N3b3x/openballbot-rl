"""Re-encode the curated preview videos for the website and write site/data/videos.json + site/data/traj/*.json.

Source: outputs/study/preview/*.mp4 and *_traj.npz (written by scripts/study/render_control.py / render_book.py).
Output: site/videos/<id>.mp4 (H.264, +faststart, CRF 26, max 720p), site/videos/<id>.jpg (poster),
        site/data/traj/<id>.json (plan / real path at 10 Hz, path error at 10 Hz), site/data/videos.json.
Run:    python scripts/site/build_videos.py [--force]
Texts below were written from the video HUD (controller name, command, fall banner) and the *_traj.npz files; the
controller is the on-screen title of each video, the outcome the fall banner / path error stored in the npz.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREV = ROOT / "outputs/study/preview"
SITE = ROOT / "site"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FORCE = "--force" in sys.argv

# family: model (LQR), learned (PPO / neural), hybrid (residual on LQR).  scenario ids drive the filter chips.
SCEN = {"flat": "Flat tracking", "square": "Square course", "push": "Pushes", "perlin": "Rough floor (Perlin)",
        "climb": "Step climb", "book": "Book / perception", "history": "Study history"}
CTRL = {
    "lqr": ("LQR", "model"),
    "res_c": ("res_c (residual on LQR)", "hybrid"),
    "robs_c": ("robs_c + integral", "learned"),
    "robs_b": ("robs_b + integral", "learned"),
    "trk_g": ("trk_g (PPO)", "learned"),
    "trk_h": ("trk_h (PPO, smooth)", "learned"),
    "trk_f": ("trk_f (PPO)", "learned"),
    "book": ("PPO, book study", "learned"),
    "early": ("early PPO", "learned"),
}
SQUARE_NOTE = "square course of 0.5 m/s legs with 1 rad/s turns, 26 s plan"
# id, source stem, controller key, scenario, title, what-you-see, outcome text builder hint, sources
V = [
    ("best_tour", "best_tour", "robs_c", "flat", "Tour on a flat floor",
     "robs_c with the outer velocity integral follows a 24.5 s sequence of velocity commands (forward, sideways, turning, stop) on a checkerboard; green is the planned path, red the actual one.",
     "robs_c_s0@ki=1.0, tour sequence, flat floor"),
    ("trk_f_final", "trk_f_final", "trk_f", "history", "trk_f: scratch policy with a 1 rad/s spin cap",
     "The first policy that really tracks speed (ratio 0.98 at 0.5 m/s) runs the standard command sequence, ending with a +2 rad/s spin command above its trained limit.",
     "trk_f_s0, full command sequence"),
    ("trk_g_final", "trk_g_final", "trk_g", "history", "trk_g: refined tracking policy",
     "trk_g (trk_f continued, action-rate 0.147 to 0.013) runs the standard command sequence including a 1.0 m/s leg.",
     "trk_g_s0, full command sequence"),
    ("lean_c_final", "lean_c_final", "early", "history", "lean_c: the lean-reward policy",
     "An early policy rewarded for leaning into the command; it balances but moves slowly (ratio about 0.3). Grey floor, no path overlay.",
     "lean_c_s0, full command sequence"),
    ("ballc_freed_final", "ballc_freed_final", "early", "history", "ballc_freed: first honest on-ball balance",
     "After adding the off-ball termination the policy balances on the ball but barely follows commands (ratio 0.21 to 0.33).",
     "ballc_freed_s0, full command sequence"),
    ("ctrl_free_c_final", "ctrl_free_c_final", "early", "history", "ctrl_free_c: the exploit (void result)",
     "With free limits and no off-ball check the policy leaves the ball and stands on its wheels; the tilt check never fires, so the reward looked good. Kept as a warning.",
     "ctrl_free_c_s0; result declared void in the study log"),
    ("lqr_square", "lqr_square", "lqr", "square", "LQR on the square course",
     f"The model-based LQR drives a {SQUARE_NOTE}; it tracks the 0.5 m/s legs closely.",
     "lqr (no training), square sequence"),
    ("robs_b_ki1_square", "robs_b_ki1_square", "robs_b", "square", "robs_b + integral on the square course",
     f"A domain-randomised PPO policy with an outer velocity integral on the {SQUARE_NOTE}.",
     "robs_b_s0@ki=1.0, square sequence"),
    ("trk_g_square_guide0", "trk_g_square_guide0", "trk_g", "square", "trk_g on the square, velocity tracking only",
     f"Pure velocity commands along the {SQUARE_NOTE}; position drift is not corrected.",
     "trk_g_s0, square sequence, guidance 0"),
    ("trk_g_square_guide1.5", "trk_g_square_guide1.5", "trk_g", "square", "trk_g on the square, position guidance 1.5",
     f"The same course with a position-feedback gain of 1.5 1/s toward the planned path ({SQUARE_NOTE}).",
     "trk_g_s0, square sequence, guidance 1.5"),
    ("trk_h_square_guide0", "trk_h_square_guide0", "trk_h", "square", "trk_h on the square, velocity tracking only",
     f"The smoother trk_h completes the full course ({SQUARE_NOTE}) without falling but drifts off the planned path.",
     "trk_h_s0, square sequence, guidance 0"),
    ("trk_h_square_guide1.5", "trk_h_square_guide1.5", "trk_h", "square", "trk_h on the square, position guidance 1.5",
     "Same policy with position guidance; it survives but the path error grows, so position feedback does not fix drift (open problem in the study log).",
     "trk_h_s0, square sequence, guidance 1.5"),
    ("best_push", "best_push", "robs_c", "push", "Pushes during a straight drive (robs_c)",
     "The robot drives forward at 0.5 m/s while an on-screen overlay announces pushes in random directions.",
     "robs_c_s0@ki=1.0, push sequence, nominal physics"),
    ("best_push_recover", "best_push_recover", "robs_b", "push", "Pushes: recovery then a fall (robs_b)",
     "robs_b with the integral follows a 0.5 m/s command while being pushed in random directions; the clip ends with a fall.",
     "robs_b_s0@ki=1.0, push sequence, nominal physics"),
    ("cand_push_robs_b", "cand_push_robs_b", "robs_b", "push", "Pushes during a straight drive (robs_b)",
     "A second push run of robs_b on a straight 0.5 m/s drive with random-direction pushes; the clip ends with a fall.",
     "robs_b_s0@ki=1.0, push sequence, nominal physics"),
    ("cand_perlin8_res_c", "cand_perlin8_res_c", "res_c", "perlin", "res_c on 8 cm Perlin terrain",
     "The residual-on-LQR controller completes the 24.5 s tour on an uneven floor (Perlin amplitude 8 cm, no terrain training).",
     "res_c_s0, tour sequence, Perlin amplitude 0.08 m"),
    ("cand_perlin12_res_c", "cand_perlin12_res_c", "res_c", "perlin", "res_c on 12 cm Perlin terrain",
     "The same controller on a rougher floor (amplitude 12 cm); no fall, path error peaks at 1.2 m.",
     "res_c_s0, tour sequence, Perlin amplitude 0.12 m"),
    ("cand_perlin8_lqr", "cand_perlin8_lqr", "lqr", "perlin", "LQR on 8 cm Perlin terrain",
     "Plain LQR on the same rough floor; it also finishes the tour at 0.5 m/s.",
     "lqr, tour sequence, Perlin amplitude 0.08 m"),
    ("best_perlin8_tour", "best_perlin8_tour", "robs_c", "perlin", "robs_c on 8 cm Perlin terrain (fails)",
     "robs_c with the integral on the 8 cm floor tracks for a while and then leaves the ball while turning at 0.8 rad/s.",
     "robs_c_s0@ki=1.0, tour sequence, Perlin amplitude 0.08 m"),
    ("best_perlin12_tour", "best_perlin12_tour", "robs_c", "perlin", "robs_c on 12 cm Perlin terrain (fails early)",
     "On the 12 cm floor robs_c falls within about 3 s (the terrain evaluation of robs_c covers 4 and 8 cm only).",
     "robs_c_s0@ki=1.0, tour sequence, Perlin amplitude 0.12 m"),
    ("best_perlin8_nav_trim", "best_perlin8_nav_trim", "robs_c", "perlin", "Waypoint navigation on 8 cm terrain (first 24 s)",
     "robs_c is commanded to navigate a list of waypoints at 0.5 m/s on the 8 cm Perlin floor; the clip is the first 24 s of a 60 s run.",
     "robs_c_s0@ki=1.0, nav sequence, Perlin amplitude 0.08 m"),
    ("climb_2cm_robs_c", "climb_2cm_robs_c", "robs_c", "climb", "Climbing a 2 cm step (robs_c)",
     "Side view: robs_c approaches a 2 cm raised floor at 0.7 m/s, crosses it and stops on top.",
     "robs_c_s0@ki=1.0, climb sequence, step 0.02 m"),
    ("climb_3cm_robs_c", "climb_3cm_robs_c", "robs_c", "climb", "A 3 cm step (robs_c fails)",
     "Same approach at 3 cm: the robot leaves the ball at the step. The study attributes this limit to wheel-ball traction (friction-scaling tests, 4 trials per cell).",
     "robs_c_s0@ki=1.0, climb sequence, step 0.03 m"),
    ("book_proprio_0.02", "book_proprio_0.02", "book", "book", "Book study: proprioceptive policy, 2 cm",
     "Side view of the book-climbing study with a proprioception-only policy and a 2 cm step.",
     "book_proprio policy, render_book.py, height 0.02 m"),
    ("book_scan_0.02", "book_scan_0.02", "book", "book", "Book study: height-scan policy, 2 cm",
     "The same task for a policy that also sees a height scan of the ground ahead (teacher observation).",
     "book_scan policy, render_book.py, height 0.02 m"),
    ("book_tof_0.02", "book_tof_0.02", "book", "book", "Book study: ToF policy, 2 cm",
     "The same task for a policy that sees the simulated 8x8 time-of-flight sensor.",
     "book_tof policy, render_book.py, height 0.02 m"),
]
BOOK_OUT = ("The video alone does not show a success or a fall banner. The matching evaluation (outputs/study/results/book2_*.csv, "
            "rough-floor amplitude sweep) reports 0 percent goal success for every perception variant at amplitude 1 cm and above.")
NAV_TRIM = 24.1


def sh(args):
    return subprocess.run(args, capture_output=True, text=True)


def duration(path):
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", sh([FF, "-i", str(path)]).stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def r(a, n=3):
    return np.round(np.asarray(a, float), n).tolist()


def main():
    (SITE / "videos").mkdir(parents=True, exist_ok=True)
    (SITE / "data/traj").mkdir(parents=True, exist_ok=True)
    out = []
    for vid, stem, ck, scen, title, see, desc in V:
        src = PREV / f"{stem}.mp4"
        dst = SITE / "videos" / f"{vid}.mp4"
        poster = SITE / "videos" / f"{vid}.jpg"
        if FORCE or not dst.exists():
            sh([FF, "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale='min(1280,iw)':-2:flags=lanczos,format=yuv420p",
                "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-threads", "2", "-movflags", "+faststart", "-an", str(dst)])
        dur = duration(dst)
        if FORCE or not poster.exists():
            sh([FF, "-y", "-loglevel", "error", "-ss", f"{min(dur * 0.3, 8):.2f}", "-i", str(dst), "-frames:v", "1",
                "-vf", "scale=640:-2", "-q:v", "4", str(poster)])
        entry = dict(id=vid, title=title, see=see, controller=CTRL[ck][0], family=CTRL[ck][1], scenario=scen,
                     scenario_label=SCEN[scen], duration_s=round(dur, 1), setup=desc,
                     video=f"videos/{vid}.mp4", poster=f"videos/{vid}.jpg",
                     size_kb=dst.stat().st_size // 1024,
                     sources=[f"outputs/study/preview/{stem}.mp4"])
        npz = PREV / f"{stem}_traj.npz"
        if stem == "best_perlin8_nav_trim":
            npz = PREV / "best_perlin8_nav_traj.npz"
        if npz.exists():
            z = np.load(npz)
            pe = z["path_err"]
            real = z["real"]
            plan = z["plan"]
            fell = bool(z["fell"])
            nav = "nav" in stem
            # time bases (verified against video HUD clock): path_err at 100 Hz, real at 10 Hz, plan at 100 Hz
            # (the nav plan is a waypoint polyline, not time indexed)
            if stem == "best_perlin8_nav_trim":
                pe = pe[: int(NAV_TRIM * 100)]
                real = real[: int(NAV_TRIM * 10) + 1]
                fell = False
            t_end = len(pe) / 100.0
            traj = dict(provenance=[f"outputs/study/preview/{npz.name}"], guide=float(z["guide"]), fell=fell, t_end=round(t_end, 2),
                        dt_real=0.1, dt_plan=0.1 if not nav else None, plan_time_indexed=not nav,
                        plan=r(plan[::10] if not nav else plan[::10], 3), real=r(real, 3), path_err=r(pe[::10], 3),
                        path_err_mean=round(float(pe.mean()), 3), path_err_max=round(float(pe.max()), 3),
                        path_err_final=round(float(pe[-1]), 3))
            (SITE / "data/traj" / f"{vid}.json").write_text(json.dumps(traj, separators=(",", ":")), encoding="utf-8")
            entry["traj"] = f"data/traj/{vid}.json"
            entry["sources"].append(f"outputs/study/preview/{npz.name}")
            entry["fell"] = fell
            entry["fall_t_s"] = round(t_end, 1) if fell else None
            if stem == "best_perlin8_nav_trim":
                entry["outcome"] = (f"No fall in the first {NAV_TRIM:.0f} s shown (the full 60 s run also ends without a fall); mean path error over this clip "
                                    f"{pe.mean():.2f} m, max {pe.max():.2f} m.")
            elif fell:
                outcome = (f"Falls at t = {t_end:.1f} s (path error {pe[-1]:.2f} m when it ends); mean path error before the fall {pe.mean():.2f} m.")
                entry["outcome"] = outcome
            else:
                entry["outcome"] = (f"No fall over {t_end:.1f} s; mean path error {pe.mean():.2f} m, max {pe.max():.2f} m, final {pe[-1]:.2f} m.")
        else:
            entry["traj"] = None
            entry["fell"] = None
            if vid == "ctrl_free_c_final":
                entry["outcome"] = "Result declared void in the study log: the policy leaves the ball within 0.2 to 0.6 s and balances on its wheels."
            elif vid.startswith("book_"):
                entry["outcome"] = BOOK_OUT
            else:
                entry["outcome"] = ("Rendered before the video fix described in the study log (the 8 s episode limit silently reset the robot to the start while the commanded path "
                                    "kept going), so treat it as an illustration of behaviour; no trajectory file was stored.")
        out.append(entry)
    meta = dict(provenance=["scripts/site/build_videos.py", "outputs/study/preview/*.mp4", "outputs/study/preview/*_traj.npz"],
                notes=["Videos are MuJoCo renders of the simulated robot; no hardware.",
                       "Trajectory time bases: path_err 100 Hz (stored here at 10 Hz), real path 10 Hz, planned path 100 Hz (stored at 10 Hz); video clock equals simulation time.",
                       "Controller and command names were read from the on-screen HUD of each video.",
                       "best_perlin8_nav_trim is the first 24.1 s of best_perlin8_nav (60 s); its plan is a waypoint polyline and not time indexed.",
                       "Push schedules (time and force) are not stored with the push videos."],
                scenarios=SCEN, controllers={k: dict(label=v[0], family=v[1]) for k, v in CTRL.items()}, videos=out)
    (SITE / "data/videos.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tot = sum(p.stat().st_size for p in (SITE / "videos").glob("*.mp4")) / 1e6
    print(f"{len(out)} videos, {tot:.1f} MB")


if __name__ == "__main__":
    main()
