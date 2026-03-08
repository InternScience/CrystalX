import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Data (from excerpt)
bins_atoms = ["1–20", "21–40", "41–60", "61–90", ">90"]
x = np.arange(len(bins_atoms))

shelxt   = np.array([33.62, 31.30, 16.30, 13.48, 0.00])
crystalx = np.array([93.53, 93.67, 85.16, 85.96, 72.73])
gain = crystalx - shelxt

# ---- Colors ----
C_CRYSTALX = "#2A9D8F"  # teal
C_SHELXT   = "#8D6E63"  # brown
C_GRID     = "#BBBBBB"  # light gray
C_REF      = "#666666"  # dark gray

# Fonts
plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
})

# Figure
fig = plt.figure(figsize=(14.2, 8.3))
gs = GridSpec(2, 1, height_ratios=[4.9, 3.2], hspace=0.14)

# ---- Top: accuracy ----
ax = fig.add_subplot(gs[0, 0])

ax.plot(x, crystalx, color=C_CRYSTALX, marker='o', linewidth=4.0, markersize=10, label="CrystalX")
ax.plot(x, shelxt,   color=C_SHELXT,   marker='o', linewidth=4.0, markersize=10, label="SHELXT")

# Dumbbell connectors + subtle gap shading (更克制一点)
for xi, y0, y1 in zip(x, shelxt, crystalx):
    ax.vlines(xi, y0, y1, linewidth=3.4, alpha=0.20, color=C_CRYSTALX)
ax.fill_between(x, shelxt, crystalx, alpha=0.06, color=C_CRYSTALX)

# Emphasize critical points
ax.plot(x[-1], shelxt[-1], marker='X', markersize=17, linestyle='None', color=C_SHELXT)
ax.plot(
    x[-1], crystalx[-1],
    marker='D', markersize=13, linestyle='None',
    markerfacecolor="white", markeredgecolor=C_CRYSTALX, markeredgewidth=3
)

ax.set_ylim(0, 112)  # 给顶部 +pp 留一点空间
ax.set_xlim(-0.5, x[-1] + 0.75)
ax.set_ylabel("Structural integrity %", fontsize=25)
ax.grid(True, axis='y', linewidth=1.0, alpha=0.25, color=C_GRID)

# Hide top x ticks + labels completely (和 SNR 图一致)
ax.set_xticks(x)
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Direct labels at the right (保留即可；legend 去掉更干净)
ax.text(x[-1] + 0.18, crystalx[-1], "CrystalX", va='center', fontsize=22, color=C_CRYSTALX)
ax.text(x[-1] + 0.18, max(shelxt[-2], 1), "SHELXT", va='center', fontsize=22, color=C_SHELXT)

# +pp labels（稍微下移一点，避免贴边）
for xi, g, y1 in zip(x, gain, crystalx):
    ax.text(
        xi, min(110, y1 + 5.0), f"+{g:.1f} pp",
        ha='center', va='bottom', fontsize=20, color=C_REF
    )

# Baseline collapse callout（加白底、缩一点，避免抢戏）
ax.annotate(
    "Baseline collapse:\n33.62% → 0.00%",
    xy=(x[-1], shelxt[-1]),
    xytext=(x[-2] + 0.05, 44),
    arrowprops=dict(arrowstyle="->", linewidth=2.0, color=C_REF),
    ha="left", va="center", fontsize=22, color=C_REF,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)
)

# ✅ 去掉 legend（右侧已有直接标注）
# leg = ax.legend(loc="lower left", framealpha=0.92)
# leg.get_frame().set_edgecolor(C_GRID)

# ---- Bottom: gain ----
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.bar(x, gain, color=C_CRYSTALX, alpha=0.90)

ax2.set_xticks(x)
ax2.set_xticklabels(bins_atoms)
ax2.tick_params(axis='x', pad=10)

ax2.set_ylabel("Gain (pp)", fontsize=25)
ax2.set_xlabel("Structural complexity (non-H atoms)", labelpad=10, fontsize=26)
ax2.grid(True, axis='y', linewidth=1.0, alpha=0.25, color=C_GRID)

# Truncated y-axis
ymin, ymax = 55, 76
ax2.set_ylim(ymin, ymax)
# ax2.text(0.01, 0.97, f"Y-axis truncated ({ymin}–{ymax} pp)",
#          transform=ax2.transAxes, fontsize=22, va='top', color=C_REF)

ax2.axhline(60, linewidth=1.8, linestyle='--', color=C_REF, alpha=0.9)
# ✅ 修正：与数据一致（第一桶是 59.9）
ax2.text(0.01, 0.97, "≥59.9 pp in every bin",
         transform=ax2.transAxes, fontsize=24, va='top', color=C_REF)

# Bar labels
for xi, g in zip(x, gain):
    ax2.text(xi, g + 0.50, f"{g:.1f}",
             ha='center', va='bottom', fontsize=20, color=C_REF)

# Axis break marks (left y-axis)
d = 0.015
ax2.plot((-d, +d), (0.04, 0.08), transform=ax2.transAxes,
         color=C_REF, clip_on=False, linewidth=1.8)
ax2.plot((-d, +d), (0.11, 0.15), transform=ax2.transAxes,
         color=C_REF, clip_on=False, linewidth=1.8)

# Shift overall down a bit
fig.subplots_adjust(top=0.87, bottom=0.12, left=0.11, right=0.96)

plt.savefig("figure1a_complexity.png", dpi=300, bbox_inches="tight")
plt.show()