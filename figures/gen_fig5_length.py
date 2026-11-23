"""Figure 5. The length channel: released records that fail the strict audit once the number of codes
the record shows is also known (K = 5, m = 3). Without the length, every bar is zero."""
import km2fig as k
from matplotlib.patches import Patch

plt = k.setup()
D = k.data()["length"]
ROWS = [("eicu_L1", "eICU-CRD, category"), ("eicu_L2", "eICU-CRD, block"), ("mimic_L2", "MIMIC-IV, block"),
        ("mimic_L3", "MIMIC-IV, chapter")]
FL = dict(facecolor=k.REFUSAL, edgecolor="white", linewidth=0)
SL = dict(facecolor=k.SUPPRESSION, edgecolor="white", linewidth=0, hatch="///")
fig, axes = plt.subplots(1, 2, figsize=(k.W, 3.6), sharey=True)
for ax, measure, name, letter in ((axes[0], "violators", "Violators", "a"), (axes[1], "unique", "Unique records", "b")):
    for i, (key, _) in enumerate(ROWS):
        ax.barh(i - 0.19, D[key]["floor"][measure], height=0.30, **FL)
        ax.barh(i + 0.19, D[key]["slocL"][measure], height=0.30, **SL)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=k.st.GRID, lw=0.8)
    ax.set_xlabel("Released records, %")
    k.panel(ax, letter, name)
axes[0].set_yticks(range(len(ROWS)), [r[1] for r in ROWS])
axes[0].set_ylim(len(ROWS) - 0.45, -0.55)
k.legend(fig, [Patch(label="Floor", **FL), Patch(label="S-loc", **SL)])
k.layout(fig, w_pad=1.6)
k.save(fig, "fig5-length")
