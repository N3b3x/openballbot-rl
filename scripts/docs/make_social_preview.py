"""Render the repository social preview image docs/assets/social-preview.png (1280 x 640, well under 1 MB).

GitHub shows this image when the repository is shared; the owner uploads it manually in the repository settings
(Settings, Social preview). It carries the title, the one-line pipeline, the simulation-only note and one headline
number (S121), drawn large enough to read at thumbnail size.

Usage (needs Pillow only):
    python scripts/docs/make_social_preview.py
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "social-preview.png")
W, H, S = 1280, 640, 2  # drawn at 2x and downsampled for smooth edges

BG, PANEL, STROKE = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED = "#f0f6fc", "#9da7b3"
BLUE, GREEN, PURPLE, ORANGE, TEAL = "#58a6ff", "#3fb950", "#bc8cff", "#f0883e", "#39c5cf"

FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
SANS = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
SANS_REG = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]


def font(names, size):
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size * S)
    print("warning: no TrueType font found, using the default bitmap font", file=sys.stderr)
    return ImageFont.load_default()


def main():
    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im)

    def rr(box, fill, outline=None, r=18, w=2):
        d.rounded_rectangle([v * S for v in box], radius=r * S, fill=fill, outline=outline, width=w * S)

    def text(xy, s, f, fill, anchor="la"):
        d.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor)

    # simulation-only badge
    f_badge = font(SANS, 22)
    rr((64, 52, 372, 100), "#2a1a0c", ORANGE, r=24)
    text((218, 77), "SIMULATION ONLY", f_badge, ORANGE, "mm")

    # title and subtitle
    text((64, 118), "openballbot-rl", font(SANS, 88), TEXT)
    text((64, 232), "An end-to-end ballbot pipeline, from robot spec to ROS 2", font(SANS_REG, 33), MUTED)
    text((64, 278), "GPU-trained LQR + PPO residual controller, ToF navigation, 11 designs", font(SANS_REG, 27), MUTED)

    # pipeline strip
    steps = [("Robot spec", BLUE), ("Simulate", GREEN), ("Train", PURPLE), ("Evaluate", ORANGE), ("Navigate", TEAL),
             ("ROS 2", BLUE)]
    n, gap = len(steps), 22
    x0, x1, y0, y1 = 64, 1216, 340, 416
    bw = (x1 - x0 - gap * (n - 1)) / n
    f_step = font(SANS, 25)
    for i, (label, col) in enumerate(steps):
        bx = x0 + i * (bw + gap)
        rr((bx, y0, bx + bw, y1), PANEL, col, r=14, w=3)
        text((bx + bw / 2, (y0 + y1) / 2), label, f_step, TEXT, "mm")
        if i < n - 1:
            ax = bx + bw + gap / 2
            ay = (y0 + y1) / 2
            d.polygon([((ax - 6) * S, (ay - 9) * S), ((ax + 6) * S, ay * S), ((ax - 6) * S, (ay + 9) * S)], fill=MUTED)

    # headline numbers (S121, S118)
    cards = [("0.87", "goal reached behind 8 cm blocks with a simulated ToF (S121)", GREEN),
             ("3.6 cm", "goal hold error on flat ground, RMS (S118)", BLUE),
             ("1.3e-3 N m", "ROS 2 core vs simulator, max torque difference", PURPLE)]
    cw, cg, cy0, cy1 = 368, 24, 458, 586
    f_big, f_small = font(SANS, 46), font(SANS_REG, 21)
    for i, (big, small, col) in enumerate(cards):
        cx = 64 + i * (cw + cg)
        rr((cx, cy0, cx + cw, cy1), PANEL, STROKE, r=14)
        d.rectangle([cx * S, (cy0 + 14) * S, (cx + 6) * S, (cy1 - 14) * S], fill=col)
        text((cx + 26, cy0 + 14), big, f_big, col)
        # wrap the caption to two lines
        words, lines, cur = small.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if d.textlength(t, font=f_small) / S > cw - 44:
                lines.append(cur)
                cur = w_
            else:
                cur = t
        lines.append(cur)
        for j, ln in enumerate(lines[:2]):
            text((cx + 26, cy0 + 68 + j * 27), ln, f_small, MUTED)

    text((64, 604), "github.com/N3b3x/openballbot-rl  |  Apache-2.0  |  fork of Salehi's OpenBallBot-RL", font(SANS_REG, 20), MUTED)

    im = im.resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    im.save(OUT, optimize=True)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
