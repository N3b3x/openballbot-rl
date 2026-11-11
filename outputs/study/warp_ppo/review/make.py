import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
out = "outputs/study/warp_ppo/review/"
steps = [("Pure learning", 0.26), ("LQR + residual 0.5", 0.29), ("Soft LQR", 0.33), ("+ fall penalty /\npush curriculum", 0.27),
         ("+ depth sensor", 0.28), ("+ ball arrestors\n(cage)", 0.47), ("Cage, authority 0.25\n(recommended)", 0.43), ("Cage + pure\nlearning", 0.50)]
und = [0.44, 0.59, 0.63, 0.57, 0.59, 0.74, 0.74, 0.72]
fig, ax = plt.subplots(figsize=(12, 5.2)); x = np.arange(len(steps))
ax.bar(x - 0.2, und, 0.4, color="#9ecae1", label="undisturbed"); ax.bar(x + 0.2, [s[1] for s in steps], 0.4, color="#e6550d", label="pushes + pulls")
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5); ax.set_ylabel("goal success\n(1.1 m goal, 11 robot designs, DR, terrain tiles)"); ax.set_ylim(0, 1)
ax.axvline(4.5, color="k", ls="--", lw=1); ax.text(4.6, 0.95, "mechanical fix:\nball arrestors", fontsize=9, va="top")
ax.set_title("Journey: what moved robustness (simulation, corrected engine, seed means)"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(out + "1_journey.png", dpi=150); plt.close(fig)

pts = [("authority 0.15", 1.01, 0.35), ("authority 0.25 (recommended)", 1.00, 0.43), ("authority 0.35", 0.95, 0.45), ("authority 0.5", 0.85, 0.47), ("soft LQR", 0.72, 0.51), ("pure learning", 0.74, 0.50)]
fig, ax = plt.subplots(figsize=(7.5, 5))
for n, tr, rb in pts:
    ax.scatter(tr, rb, s=140, color="#e6550d" if "recommended" in n else "#3182bd", zorder=3); ax.annotate(n, (tr, rb), xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.scatter(0.98, 0.29, s=140, color="grey", marker="x", zorder=3); ax.annotate("no cage, authority 0.5", (0.98, 0.29), xytext=(6, -14), textcoords="offset points", fontsize=9, color="grey")
ax.set_xlabel("speed tracking: achieved / commanded speed, flat ground (1.0 = exact)"); ax.set_ylabel("robustness: goal success with pushes + pulls")
ax.set_xlim(0.65, 1.06); ax.set_ylim(0.2, 0.6); ax.grid(alpha=0.3); ax.set_title("The dial: how much the learned part may override the LQR (all with cage)")
fig.tight_layout(); fig.savefig(out + "2_tradeoff.png", dpi=150); plt.close(fig)

kinds = ["flat", "ramp up", "ramp down", "rough", "step down", "small drop", "stairs", "step up", "large drop", "bump"]
small = [0.10, 0.15, 0.28, 0.09, 0.12, 0.23, 0.27, 0.40, 0.61, 0.59]; large = [0.10, 0.12, 0.29, 0.27, 0.21, 0.34, 0.70, 0.95, 0.77, 0.90]
fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(kinds))
ax.bar(x - 0.2, small, 0.4, color="#74c476", label="smaller half of the feature range"); ax.bar(x + 0.2, large, 0.4, color="#cb181d", label="larger half")
ax.axhline(0.1, color="k", ls=":", lw=1); ax.text(-0.4, 0.12, "flat-ground base rate", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(kinds, rotation=20, ha="right"); ax.set_ylabel("fall rate (20 s at 0.3 m/s)"); ax.set_ylim(0, 1)
ax.set_title("Where it still falls: obstacle SIZE, not speed (traction limit about 2.6 cm for a 9 cm ball)"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(out + "3_falls_by_size.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(16, 6.5)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 6.5)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="k", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.2)
def arr(x0, y0, x1, y1, t=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3))
    if t: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, t, ha="center", fontsize=8.5, color="#444")
box(0.2, 3.6, 2.2, 1.1, "Goal / user command\n(where to go, how fast)", "#f7f7f7")
box(3.0, 3.6, 2.6, 1.1, "Navigation layer\nwavefront planner + smooth follower\n(0.3 m carrot, 2 rad/s, 0.6 m/s^2)", "#deebf7")
box(3.0, 1.9, 2.6, 1.1, "Terrain map\nToF 18x14 on the head, aimed by the planner\nhits > 3 cm kept in memory", "#e5f5e0")
box(6.3, 3.6, 2.0, 1.1, "LQR\n(per-design gains,\nvelocity governor)", "#fee6ce")
box(6.3, 1.9, 2.0, 1.1, "Learned residual\nPPO, authority 0.25\n11 designs, DR, pushes", "#fee6ce")
box(8.9, 2.75, 1.9, 1.1, "Wheel torques\n(3 omniwheels)", "#f7f7f7")
box(6.3, 0.3, 4.5, 1.0, "Robot: 9 cm ball + Rezero-style arrestors (keep the body on the ball)\nsimulation only: MuJoCo-Warp, 4096 worlds, 100 Hz", "#fdf0d5")
arr(2.4, 4.15, 3.0, 4.15); arr(5.6, 4.15, 6.3, 4.15, "smooth velocity cmd"); arr(4.3, 3.0, 4.3, 3.6); arr(5.6, 2.45, 6.3, 2.45, "same cmd"); arr(8.3, 4.15, 8.9, 3.5, "a_LQR"); arr(8.3, 2.45, 8.9, 3.0, "+ 0.25 a_policy")
ax.text(5.5, 5.0, "Controller stack (recommended configuration)", fontsize=13, ha="center", weight="bold")
fig.savefig(out + "4_architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)

rows = [("blind", 0.09, 0.91), ("perfect map", 0.91, 0.09), ("ToF 18x14 head (jerky planner)", 0.85, 0.11), ("ToF 18x14 head, smooth + aimed", 0.78, 0.16), ("body-cone sensors", 0.58, 0.35)]
fig, ax = plt.subplots(figsize=(8.5, 4)); y = np.arange(len(rows))
ax.barh(y - 0.2, [r[1] for r in rows], 0.4, color="#3182bd", label="reached goal"); ax.barh(y + 0.2, [r[2] for r in rows], 0.4, color="#cb181d", label="fell")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_title("Obstacle course: goal behind an 8 cm block (528 worlds, undisturbed)", fontsize=10.5); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(out + "5_navigation.png", dpi=150); plt.close(fig)
print("ok")
