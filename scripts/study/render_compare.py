"""Side-by-side comparison renderer (two controllers, same scenario, synchronised). Used by render_cinematic.py --compare.

  python scripts/study/render_cinematic.py --compare perlin8_tour_lqr perlin8_tour_res_c
  python scripts/study/render_cinematic.py --compare square_lqr square_robs_b
"""
import json
import time as _time

import numpy as np


def run(names, args, R):
    from PIL import Image, ImageDraw
    JA, JB = dict(R.JOBS[names[0]]), dict(R.JOBS[names[1]])
    assert JA["scn"] == JB["scn"], "compare needs the same scenario"
    W, H, FPS = R.W, R.H, R.FPS
    VW, VH = 640, 400
    t0 = _time.time()
    xml = R.make_xml()
    envs, Ls, Ms = [], [], []
    for J in (JA, JB):
        env, m, L = R.simulate(J, xml, args.sim_limit)
        envs.append(env), Ms.append(m), Ls.append(L)
        print(f"[compare] {J['run']}: {len(L['pos']) / 100:.1f} s fell={L['fell']} ({_time.time() - t0:.0f} s)", flush=True)
    terrs = [R.Terrain(m) for m in Ms]
    for m, t in zip(Ms, terrs):
        R.paint_floor(m, t)
    Js = [JA, JB]
    fams = [R.family(J["run"]) for J in Js]
    Ss = [R.stats_of(J, L) for J, L in zip(Js, Ls)]
    pre = [R.precompute_stats(L) for L in Ls]
    looks = [R.cam_track(J, L, t) for J, L, t in zip(Js, Ls, terrs)]
    stages = [R.Stage(m, t, J, L, w=VW, h=VH) for m, t, J, L in zip(Ms, terrs, Js, Ls)]
    n_all = max(len(L["pos"]) for L in Ls)
    n_sim = int(np.ceil(n_all / 100.0 * FPS))
    n_res = int(4.0 * FPS)
    total = n_sim + n_res
    T_end = max(L["n_main"] for L in Ls) / 100.0
    mm = R.MiniMap(terrs[0], Ls[0], size=250, extra=[Ls[1]["pos"][:, :2]])
    err_max = max(25.0 * np.ceil(max(L["err"][:L["n_main"]].max() for L in Ls) * 100 / 25.0), 50.0)
    stem = f"cine_compare_{JA['scn']}_{JA['run'].split('@')[0].rsplit('_s', 1)[0]}_vs_{JB['run'].split('@')[0].rsplit('_s', 1)[0]}"
    ff = None if args.no_video else R.open_ffmpeg(R.OUT / f"{stem}.mp4")
    (R.OUT / "_dbg").mkdir(exist_ok=True)
    dump = {int(float(x) * FPS) for x in args.dump.split(",") if x}
    fell_any = [L for L in Ls if L["fell"]]
    poster_k = int((min(L["fell_step"] for L in fell_any) / 100.0 + 0.8) * FPS) if fell_any else int(0.7 * n_sim)
    names_ = [R.ctrl_label(J["run"]) for J in Js]

    def mini_frame(i):
        im = mm.base.copy()
        dr = ImageDraw.Draw(im, "RGBA")
        for L, fam in zip(Ls, fams):
            n = min(i, L["n_main"] - 1)
            pts = [mm.px(p) for p in L["pos"][0:n + 1:5, :2]]
            if len(pts) > 1:
                dr.line(pts, fill=R.C[fam] + (255,), width=5, joint="curve")
            ii = min(i, L["fell_step"] if L["fell"] else len(L["pos"]) - 1)
            x, y = mm.px(L["pos"][ii, :2])
            if L["fell"] and i >= L["fell_step"]:
                dr.line([x - 12, y - 12, x + 12, y + 12], fill=R.C["bad"] + (255,), width=5)
                dr.line([x - 12, y + 12, x + 12, y - 12], fill=R.C["bad"] + (255,), width=5)
            else:
                dr.ellipse([x - 11, y - 11, x + 11, y + 11], fill=R.C[fam] + (255,), outline=(255, 255, 255, 255), width=2)
        return im.resize((mm.size, mm.size), Image.LANCZOS)

    for k in range(total):
        t_vid = k / FPS
        t_after = max(0.0, (k - n_sim) / FPS)
        idxs = [min(int(round(k / FPS * 100)), len(L["pos"]) - 1) if k < n_sim else len(L["pos"]) - 1 for L in Ls]
        canvas = Image.new("RGB", (W, H), (8, 11, 17))
        cams = []
        for v in range(2):
            cam = R.cam_at(Js[v], looks[v][idxs[v]], t_vid, t_after)
            cam.dist *= 1.12
            raw = stages[v].render(idxs[v], cam)
            canvas.paste(Image.fromarray(raw), (v * VW, 0))
            cams.append(cam)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov, "RGBA")
        dr.rectangle([VW - 1, 0, VW, VH], fill=(8, 11, 17, 255))
        iA = idxs
        # per-view header, status chip and live error
        for v in range(2):
            L, fam, J = Ls[v], fams[v], Js[v]
            i = idxs[v]
            x0 = v * VW
            R.card(dr, [x0 + 14, 14, x0 + VW - 14, 60], r=12)
            dr.rounded_rectangle([x0 + 14, 14, x0 + 20, 60], radius=3, fill=R.C[fam] + (255,))
            w = R.pill(dr, x0 + 32, 25, names_[v][0], R.font(15, "bold"), R.C[fam], h=26)
            R.label(dr, (x0 + 32 + w + 10, 38), names_[v][1], R.font(13), R.C["dim"] + (255,), "lm")
            fell = L["fell"] and i >= L["fell_step"]
            pushed = (not fell) and L["pushF"][i] > 0
            txt, col = ((f"FELL {L['fell_step'] / 100:.1f} s", R.C["bad"]) if fell else
                        (f"PUSHED {L['pushF'][i]:.0f} N", R.C["warn"]) if pushed else ("BALANCING", R.C["good"]))
            f = R.font(14, "bold")
            tw = dr.textlength(txt, font=f)
            bx = x0 + VW - 14 - 14 - tw - 30
            dr.rounded_rectangle([bx, 24, bx + tw + 30, 50], radius=13, fill=(10, 14, 22, 215), outline=col + (255,), width=2)
            dr.ellipse([bx + 9, 33, bx + 19, 43], fill=col + (255,))
            R.label(dr, (bx + 25, 37), txt, f, col + (255,), "lm")
            R.card(dr, [x0 + 14, VH - 46, x0 + 214, VH - 14], r=10)
            R.label(dr, (x0 + 26, VH - 30), f"position error {L['err'][i] * 100:4.0f} cm", R.font(14, "mono"), R.C["ink"] + (255,), "lm")
            if fell and (k / FPS - L["fell_step"] / 100.0) <= 1.5 and k < n_sim:
                dr.rounded_rectangle([x0 + 90, 78, x0 + VW - 90, 124], radius=14, fill=(150, 20, 28, 235), outline=(255, 160, 160, 255), width=2)
                R.label(dr, (x0 + VW / 2, 94), f"FELL at t = {L['fell_step'] / 100:.1f} s", R.font(20, "bold"), (255, 255, 255, 255), "mm")
                R.label(dr, (x0 + VW / 2, 114), f"{L['why']}; run ends", R.font(13), (255, 225, 225, 255), "mm")
            if J.get("step") is None and L["pushF"][i] > 0 and not fell:
                pass
        # bottom: minimap
        R.card(dr, [16, 414, 288, 700])
        R.label(dr, (30, 424), "TOP VIEW   white = plan, lines = trails", R.font(11, "bold"), R.C["dim"] + (255,))
        ov.alpha_composite(mini_frame(max(idxs)), (27, 444))
        dr = ImageDraw.Draw(ov, "RGBA")
                # bottom: error plot
        px0, px1, py0, py1 = 346, 900, 456, 640
        R.card(dr, [300, 414, 920, 700])
        R.label(dr, (318, 424), "POSITION ERROR vs TIME (cm)", R.font(11, "bold"), R.C["dim"] + (255,))
        for gv in np.arange(0, err_max + 1, 25.0 if err_max <= 100 else 50.0):
            y = py1 - (py1 - py0) * gv / err_max
            dr.line([(px0, y), (px1, y)], fill=(255, 255, 255, 28))
            R.label(dr, (px0 - 6, y), f"{gv:.0f}", R.font(11, "mono"), R.C["dim"] + (255,), "rm")
        for gt in np.arange(0, T_end + 0.1, 5.0):
            x = px0 + (px1 - px0) * gt / T_end
            dr.line([(x, py1), (x, py1 + 5)], fill=(255, 255, 255, 80))
            R.label(dr, (x, py1 + 8), f"{gt:.0f} s", R.font(11, "mono"), R.C["dim"] + (255,), "mt")
        for v in range(2):
            L, fam = Ls[v], fams[v]
            n = min(idxs[v], L["n_main"] - 1)
            pts = [(px0 + (px1 - px0) * (j / 100.0) / T_end, py1 - (py1 - py0) * min(L["err"][j] * 100, err_max) / err_max)
                   for j in range(0, n + 1, 4)]
            if len(pts) > 1:
                dr.line(pts, fill=R.C[fam] + (255,), width=3, joint="curve")
            if L["fell"] and idxs[v] >= L["fell_step"]:
                x, y = px0 + (px1 - px0) * L["fell_step"] / 100.0 / T_end, py1 - (py1 - py0) * min(L["err"][L["fell_step"]] * 100, err_max) / err_max
                dr.line([x - 7, y - 7, x + 7, y + 7], fill=R.C["bad"] + (255,), width=3)
                dr.line([x - 7, y + 7, x + 7, y - 7], fill=R.C["bad"] + (255,), width=3)
        xc = px0 + (px1 - px0) * min(max(idxs) / 100.0, T_end) / T_end
        dr.line([(xc, py0), (xc, py1)], fill=(255, 255, 255, 130), width=1)
        R.label(dr, (px0 + 8, 674), "tracking: distance to the time-indexed planned point", R.font(11), R.C["dim"] + (255,))
        # bottom: stats table
        R.card(dr, [932, 414, 1264, 700])
        R.label(dr, (950, 424), "LIVE", R.font(11, "bold"), R.C["dim"] + (255,))
        for v in range(2):
            R.label(dr, (1130 + v * 108, 440), names_[v][0].split(" ")[0], R.font(14, "bold"), R.C[fams[v]] + (255,), "ra")
        rows = [("time", lambda v, i, L: f"{min(i, L['n_main']) / 100:5.1f} s"),
                ("pos error", lambda v, i, L: f"{L['err'][i] * 100:4.0f} cm"),
                ("mean error", lambda v, i, L: f"{pre[v][1]['mean_err'][i] * 100:4.0f} cm"),
                ("speed err", lambda v, i, L: f"{np.linalg.norm(pre[v][0][i] - L['cmd'][i][:2]):4.2f} m/s"),
                ("lean", lambda v, i, L: f"{L['tilt'][i]:4.1f} deg"),
                ("off-ball", lambda v, i, L: f"{L['off'][i]:4.1f} deg"),
                ("travelled", lambda v, i, L: f"{pre[v][1]['dist'][i]:5.2f} m")]
        for r_, (lab, fn) in enumerate(rows):
            y = 482 + r_ * 30
            R.label(dr, (950, y), lab, R.font(13), R.C["dim"] + (255,), "lm")
            for v in range(2):
                R.label(dr, (1130 + v * 108, y), fn(v, idxs[v], Ls[v]), R.font(13, "mono"), R.C["ink"] + (255,), "rm")
            dr.line([(950, y + 15), (1248, y + 15)], fill=(255, 255, 255, 22))
        R.label(dr, (950, 694 - 14), "SIMULATION ONLY", R.font(10, "bold"), R.C["dim"] + (200,), "lm")
        # time bar
        R.card(dr, [16, 706, 1264, 718], r=6)
        fx = 16 + 1248 * min(1.0, (min(max(idxs), max(L["n_main"] for L in Ls)) / 100.0) / T_end)
        dr.rounded_rectangle([16, 706, fx, 718], radius=6, fill=(255, 255, 255, 90))
        # intro and result overlays
        if t_vid < 3.6:
            a = R.smooth(t_vid / 0.5) * (1 - R.smooth((t_vid - 2.7) / 0.9))
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ld = ImageDraw.Draw(lay, "RGBA")
            ld.rectangle([0, 150, W, 330], fill=(6, 9, 14, 190))
            R.label(ld, (640, 195), f"{JA['title']}: {names_[0][0]} vs {names_[1][0]}", R.font(40, "bold"), (255, 255, 255, 255), "mm")
            R.label(ld, (640, 245), f"same scenario, same terrain, synchronised   |   {JA['terrain']}", R.font(20), (200, 210, 224, 255), "mm")
            R.label(ld, (640, 285), "MuJoCo simulation, no hardware, DR: no", R.font(16), (200, 210, 224, 255), "mm")
            ld.rectangle([0, 150, W // 2, 154], fill=R.C[fams[0]] + (255,))
            ld.rectangle([W // 2, 150, W, 154], fill=R.C[fams[1]] + (255,))
            lay.putalpha(lay.getchannel("A").point(lambda x: int(x * a)))
            ov.alpha_composite(lay)
        if k >= n_sim:
            a = R.smooth(t_after / 0.5)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ld = ImageDraw.Draw(lay, "RGBA")
            ld.rectangle([0, 0, W, H], fill=(5, 8, 12, 175))
            bx0, by0, bx1, by1 = 250, 110, 1030, 610
            ld.rounded_rectangle([bx0, by0, bx1, by1], radius=22, fill=(10, 14, 22, 252), outline=(255, 255, 255, 40), width=1)
            R.label(ld, (bx0 + 36, by0 + 26), "RESULT", R.font(13, "bold"), R.C["dim"] + (255,))
            R.label(ld, (bx0 + 36, by0 + 48), f"{JA['title']}", R.font(26, "bold"), R.C["ink"] + (255,))
            for v in range(2):
                cx = bx0 + 420 + v * 170
                R.pill(ld, cx - 60, by0 + 96, names_[v][0].split(" ")[0], R.font(15, "bold"), R.C[fams[v]], h=28)
                s = Ss[v]
                out = f"FELL {s['fall_t_s']:.1f} s" if s["fell"] else "No fall"
                R.pill(ld, cx - 60, by0 + 132, out, R.font(14, "bold"), R.C["bad"] if s["fell"] else R.C["good"], h=26)
            rows = [("Falls", lambda s: "1" if s["fell"] else "0"), ("Mean position error", lambda s: f"{s['mean_err_cm']:.0f} cm"),
                    ("Final position error", lambda s: f"{s['final_err_cm']:.0f} cm"), ("Max position error", lambda s: f"{s['max_err_cm']:.0f} cm"),
                    ("Max off-ball angle", lambda s: f"{s['max_off_ball_deg']:.1f} deg"),
                    ("Mean speed error", lambda s: f"{s['mean_speed_err_mps']:.2f} m/s"),
                    ("Distance travelled", lambda s: f"{s['dist_travelled_m']:.1f} m"), ("Run length", lambda s: f"{s['duration_s']:.1f} s")]
            for r_, (lab, fn) in enumerate(rows):
                y = by0 + 196 + r_ * 36
                R.label(ld, (bx0 + 36, y), lab, R.font(17), R.C["dim"] + (255,), "lm")
                for v in range(2):
                    R.label(ld, (bx0 + 420 + v * 170 + 50, y), fn(Ss[v]), R.font(18, "monob"), R.C["ink"] + (255,), "rm")
                ld.line([(bx0 + 36, y + 18), (bx1 - 32, y + 18)], fill=(255, 255, 255, 25))
            R.label(ld, (bx0 + 36, by1 - 20), "Simulation only. Nominal physics. A fall ends that run; the other run continues.", R.font(12), R.C["dim"] + (255,), "lm")
            lay.putalpha(lay.getchannel("A").point(lambda x: int(x * a)))
            ov.alpha_composite(lay)
        img = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")
        if k in dump:
            img.save(R.OUT / "_dbg" / f"{stem}_{t_vid:05.1f}s.png")
        if k == poster_k:
            img.save(R.OUT / f"{stem}.jpg", quality=90)
        if ff is not None:
            ff.stdin.write(np.asarray(img).tobytes())
        if k % 150 == 0:
            print(f"  frame {k}/{total} ({_time.time() - t0:.0f} s)", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    for s_ in stages:
        s_.close()
    side = dict(id=stem, title=f"{JA['title']}: {names_[0][0]} vs {names_[1][0]}", scenario=JA["scn"], terrain=JA["terrain"],
                controllers=[JA["run"], JB["run"]], controller_families=fams,
                description="Two controllers on the same scenario and terrain, simulated separately and replayed in sync.",
                outcome="; ".join(f"{names_[v][0]}: " + (f"fell at {Ss[v]['fall_t_s']} s" if Ss[v]["fell"] else "no fall") +
                                  f", mean error {Ss[v]['mean_err_cm']} cm, max {Ss[v]['max_err_cm']} cm" for v in range(2)),
                stats=dict(zip(["a", "b"], Ss)), duration_s=round(total / FPS, 1), sim_only=True, domain_randomisation=False,
                command=f"python scripts/study/render_cinematic.py --compare {names[0]} {names[1]}")
    (R.OUT / f"{stem}.json").write_text(json.dumps(side, indent=1))
    print(f"[compare] done {stem} {total / FPS:.1f} s in {_time.time() - t0:.0f} s", flush=True)
