import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(16, 6.5)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 6.5)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="k", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.2)
def arr(x0, y0, x1, y1, t=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3))
    if t: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.17, t, ha="center", fontsize=8, color="#444")
box(0.2, 4.2, 2.6, 1.4, "Goal / user command\n(where to go, how fast)", "#f7f7f7")
box(3.6, 4.2, 5.0, 1.4, "Navigation layer\nwavefront planner + smooth follower\n0.3 m carrot, 2 rad/s, 0.6 m/s^2", "#deebf7")
box(3.6, 2.0, 5.0, 1.4, "Terrain map\nToF 18x14 on the head, aimed by the planner\nhits > 3 cm kept in memory", "#e5f5e0")
box(9.6, 4.2, 2.8, 1.4, "LQR\nper-design gains\nvelocity governor", "#fee6ce")
box(9.6, 2.0, 2.8, 1.4, "Learned residual\nPPO, authority 0.25\n11 designs, DR, pushes", "#fee6ce")
box(13.3, 3.1, 2.5, 1.4, "Wheel torques\n3 omniwheels", "#f7f7f7")
box(3.6, 0.2, 12.2, 1.1, "Robot: 9 cm ball + Rezero-style arrestors (keep the body on the ball).   Simulation only: MuJoCo-Warp, 4096 worlds, 100 Hz", "#fdf0d5")
arr(2.8, 4.9, 3.6, 4.9); arr(8.6, 4.9, 9.6, 4.9, "smooth velocity cmd"); arr(6.1, 3.4, 6.1, 4.2); arr(8.6, 2.7, 9.6, 2.7, "same cmd"); arr(12.4, 4.9, 13.3, 4.1, "a_LQR"); arr(12.4, 2.7, 13.3, 3.5, "+ 0.25 a_policy")
ax.text(8.0, 6.2, "Controller stack (recommended configuration)", fontsize=13, ha="center", weight="bold")
fig.savefig("outputs/study/warp_ppo/review/4_architecture.png", dpi=130); print("ok")
