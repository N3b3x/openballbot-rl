"""Re-encode the cinematic renders (outputs/study/cine/cine_*.mp4) for the site and add them to site/data/videos.json.
Each entry's text comes from the render's sidecar json. Run: python scripts/site/add_cinematic.py"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
CINE, SITE = ROOT / "outputs/study/cine", ROOT / "site"
FF = imageio_ffmpeg.get_ffmpeg_exe()
vj = SITE / "data/videos.json"
D = json.load(open(vj))
D["videos"] = [v for v in D["videos"] if not v["id"].startswith("cine_")]
for k, lab in (("compare", "Side by side"),):
    D["scenarios"].setdefault(k, lab)
for mp4 in sorted(CINE.glob("cine_*.mp4")):
    meta = json.load(open(mp4.with_suffix(".json")))
    vid = mp4.stem
    out = SITE / "videos" / f"{vid}.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "27",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)
    jpg = mp4.with_suffix(".jpg")
    if jpg.exists():
        (SITE / "videos" / f"{vid}.jpg").write_bytes(jpg.read_bytes())
    cmp_ = "controllers" in meta
    scen = "compare" if cmp_ else meta["scenario"]
    st = meta["stats"]
    fell = (st["a"]["fell"] or st["b"]["fell"]) if cmp_ else st["fell"]
    D["videos"].append({
        "id": vid, "title": meta["title"] + " (cinematic)", "see": meta["description"] + " " + meta.get("terrain", ""),
        "controller": " vs ".join(meta["controllers"]) if cmp_ else meta["controller"],
        "family": "learned" if cmp_ else meta["controller_family"],
        "scenario": scen, "scenario_label": D["scenarios"].get(scen, scen), "duration_s": round(meta["duration_s"], 1),
        "setup": meta["command"], "video": f"videos/{vid}.mp4", "poster": f"videos/{vid}.jpg", "size_kb": out.stat().st_size // 1024,
        "sources": [f"outputs/study/cine/{mp4.name}"], "traj": None, "fell": bool(fell),
        "fall_t_s": None if cmp_ else st["fall_t_s"], "outcome": meta["outcome"] + " Simulation only.",
    })
json.dump(D, open(vj, "w"), indent=1)
print(len(D["videos"]), "videos")
