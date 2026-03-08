import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Data (SNR bins)
bins_snr = ["0–5", "5–15", "15–30", "30–50", ">50"]
x = np.arange(len(bins_snr))

shelxt = np.array([14.81, 26.94, 31.37, 38.63, 43.64])
model  = np.array([85.19, 89.42, 93.07, 95.70, 95.90])
gain = model - shelxt

# ---- Colors (keep your palette) ----
C_CRYSTALX = "#2A9D8F"  # teal
C_SHELXT   = "#8D6E63"  # brown
C_GRID     = "#BBBBBB"  # light gray
C_REF      = "#666666"  # dark gray

# Fonts (keep your style)
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
})

# Figure
fig = plt.figure(figsize=(14.2, 8.4))
gs = GridSpec(2, 1, height_ratios=[4.9, 3.2], hspace=0.14)

# ---- Top: accuracy ----
ax = fig.add_subplot(gs[0, 0])

# 用 plot（去掉 errorbar 视觉噪声；风格不变）
ax.plot(
    x, model,
    color=C_CRYSTALX, marker='o', linewidth=4.0, markersize=10,
    label="CrystalX"
)
ax.plot(
    x, shelxt,
    color=C_SHELXT, marker='o', linewidth=4.0, markersize=10,
    label="SHELXT"
)

# Dumbbell connectors + subtle gap shading
for xi, y0, y1 in zip(x, shelxt, model):
    ax.vlines(xi, y0, y1, linewidth=3.4, alpha=0.22, color=C_CRYSTALX)
ax.fill_between(x, shelxt, model, alpha=0.08, color=C_CRYSTALX)

# Emphasize low-SNR point
ax.plot(x[0], shelxt[0], marker='X', markersize=17, linestyle='None', color=C_SHELXT)
ax.plot(
    x[0], model[0],
    marker='D', markersize=13, linestyle='None',
    markerfacecolor="white", markeredgecolor=C_CRYSTALX, markeredgewidth=3
)

ax.set_ylim(0, 110)
ax.set_xlim(-0.5, x[-1] + 0.75)
ax.set_ylabel("Structural integrity %",fontsize=25)
ax.grid(True, axis='y', linewidth=1.0, alpha=0.25, color=C_GRID)

# ✅ 关键修正：彻底隐藏上图的 x 轴刻度与标签（防止和下图抢空间）
ax.set_xticks(x)
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Direct labels at the right
ax.text(x[-1] + 0.18, model[-1], "CrystalX",  va='center', fontsize=22, color=C_CRYSTALX)
ax.text(x[-1] + 0.18, shelxt[-1], "SHELXT", va='center', fontsize=22, color=C_SHELXT)

# +pp labels
for xi, g, y1 in zip(x, gain, model):
    ax.text(
        xi, min(108, y1 + 6.0), f"+{g:.1f} pp",
        ha='center', va='bottom', fontsize=20, color=C_REF
    )

# ✅ 关键修正：把 noise sensitivity 缩小、加白底、放到不挡线的位置
ax.annotate(
    "SHELXT noise sensitivity:\n43.64% → 14.81%",
    xy=(x[0], shelxt[0]),
    xytext=(x[1] + 0.10, 46),
    arrowprops=dict(arrowstyle="->", linewidth=2.2, color=C_REF),
    ha="left", va="center", fontsize=22, color=C_REF,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)
)



# ---- Bottom: gain ----
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.bar(x, gain, color=C_CRYSTALX, alpha=0.90)

ax2.set_xticks(x)
ax2.set_xticklabels(bins_snr)
ax2.tick_params(axis='x', pad=14)  # 稍微再往下，保证不碰中间分隔线

ax2.set_ylabel("Gain (pp)", fontsize=25)
ax2.set_xlabel("Signal-to-noise ratio (SNR)", labelpad=10, fontsize=26)
ax2.grid(True, axis='y', linewidth=1.0, alpha=0.25, color=C_GRID)

# Truncated y-axis
ymin, ymax = 50, 75
ax2.set_ylim(ymin, ymax)

# ✅ 关键修正：把截断说明移到右上角并加白底，避免压到第一根柱子
ax2.text(
    0.99, 0.95, f"≥50 pp in every bin",
    transform=ax2.transAxes, fontsize=24, va='top', ha='right', color=C_REF,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)
)

ax2.axhline(60, linewidth=1.8, linestyle='--', color=C_REF, alpha=0.9)

# Bar labels
for xi, g in zip(x, gain):
    ax2.text(xi, g + 0.35, f"{g:.1f}",
             ha='center', va='bottom', fontsize=20, color=C_REF)

# Axis break marks (left y-axis)
d = 0.015
ax2.plot((-d, +d), (0.04, 0.08), transform=ax2.transAxes,
         color=C_REF, clip_on=False, linewidth=1.8)
ax2.plot((-d, +d), (0.11, 0.15), transform=ax2.transAxes,
         color=C_REF, clip_on=False, linewidth=1.8)

# Shift overall down a bit (keep your style)
fig.subplots_adjust(top=0.87, bottom=0.12, left=0.11, right=0.96)

plt.savefig("figure1b_snr.png", dpi=300, bbox_inches="tight")
plt.show()