"""Make the short animated previews shown in the README gallery (docs/assets/clips/).

GitHub READMEs cannot play repository mp4 files inline, so each preview is a small looping animated WebP (lossy,
about 0.3 to 0.6 MB at 560 px) cut from a simulation clip. A GIF fallback (two-pass ffmpeg palettegen / paletteuse) is
available with --format gif, at roughly six times the size. The full clips are on the website's videos page.

Usage (from the repository root, with the study venv that has imageio-ffmpeg):
    experiments/mjx_port/.venv/Scripts/python.exe scripts/docs/make_previews.py            # all previews (WebP)
    experiments/mjx_port/.venv/Scripts/python.exe scripts/docs/make_previews.py c13_push_pull_hold
    ... make_previews.py --format gif --width 480                                          # GIF instead

Never add c11_goal_bump_J02 (withdrawn clip).
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "clips")
VID = os.path.join(ROOT, "outputs", "study", "warp_ppo", "videos")
SITE = os.path.join(ROOT, "site", "videos")

# name: (source dir, start s, duration s of source, playback speed-up)
CLIPS = {
    "c12_nav_with_layer":  (VID, 0.0, 11.5, 2.0),    # g34, ToF navigation through the motion layer
    "c8_nav_tof18_smooth": (SITE, 0.0, 12.4, 2.0),   # g18, ToF navigation with the smooth follower
    "c13_push_pull_hold":  (VID, 0.0, 12.0, 2.0),    # g34, goal hold under pushes and pulls
    "c11_goal_hold_nom":   (VID, 0.0, 10.0, 2.0),    # g34, drive 1 m, brake, hold
    "c11_step15_nom":      (VID, 2.0, 4.5, 1.0),     # g34, 1.5 cm step at 0.3 m/s
    "c11_ramp_zhigh":      (VID, 0.0, 4.3, 1.0),     # g34, 11 degree ramp at 0.6 m/s, zhigh design
}
WITHDRAWN = {"c11_goal_bump_J02"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make(name, fmt="webp", width=560, fps=12, colors=192):
    if name in WITHDRAWN:
        raise SystemExit(f"{name} is withdrawn; do not publish it")
    src_dir, t0, dur, speed = CLIPS[name]
    src = os.path.join(src_dir, name + ".mp4")
    if not os.path.exists(src):
        raise SystemExit(f"missing source clip {src}")
    os.makedirs(OUT, exist_ok=True)
    ff = ffmpeg_exe()
    base = f"setpts=PTS/{speed},fps={fps},scale={width}:-2:flags=lanczos"
    cut = ["-ss", f"{t0}", "-t", f"{dur}", "-i", src]
    out = os.path.join(OUT, f"{name}.{fmt}")
    if fmt == "gif":
        with tempfile.TemporaryDirectory() as td:
            pal = os.path.join(td, "pal.png")
            subprocess.run([ff, "-v", "error", "-y", *cut, "-vf", f"{base},palettegen=max_colors={colors}:stats_mode=diff",
                            pal], check=True)
            subprocess.run([ff, "-v", "error", "-y", *cut, "-i", pal, "-lavfi",
                            f"{base}[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0", out],
                           check=True)
    elif fmt == "webp":
        subprocess.run([ff, "-v", "error", "-y", *cut, "-vf", base, "-c:v", "libwebp_anim", "-lossless", "0",
                        "-quality", "78", "-compression_level", "6", "-loop", "0", out], check=True)
    else:
        raise SystemExit(f"unknown format {fmt}")
    print(f"{out}  {os.path.getsize(out) / 1e6:.2f} MB  ({dur / speed:.1f} s at {speed:g}x)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="clips to make (default: all)")
    ap.add_argument("--format", default="webp", choices=["gif", "webp"])
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=192, help="GIF palette size")
    a = ap.parse_args()
    for n in a.names or list(CLIPS):
        make(n, a.format, a.width, a.fps, a.colors)


if __name__ == "__main__":
    sys.exit(main())
