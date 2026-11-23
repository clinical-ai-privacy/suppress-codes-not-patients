"""Figure 3. The floor against its matched random control: the same number of training patients
removed at random, the same coding (K = 5, m = 3). Whiskers: Nadeau-Bengio 95 % interval."""
import km2fig as k

plt = k.setup()
D = k.data()["control"]
LEVELS = ["category", "block", "chapter"]            # one row per level, the same row in both panels
PANELS = [("eICU-CRD, death", ["eicu_death_L1", "eicu_death_L2", "eicu_death_L3"]),
          ("MIMIC-IV, readmission", [None, "mimic_readmit30_L2", "mimic_readmit30_L3"])]
fig, axes = plt.subplots(1, 2, figsize=(k.W, 3.3), sharex=True, sharey=True)
for ax, (name, rows), letter in zip(axes, PANELS, "ab"):
    for i, key in enumerate(rows):
        if key is None:
            continue
        for part, dy, col, kw in (("control", -0.17, k.CONTROL, dict(fmt="o", mfc="white", mec=k.CONTROL, mew=1.8)),
                                  ("floor", 0.17, k.REFUSAL, dict(fmt="s", mec="white", mew=1.2))):
            r = D[key][part]
            ax.errorbar(r["d"], i + dy, xerr=[[r["d"] - r["lo"]], [r["hi"] - r["d"]]], color=col, elinewidth=2.2,
                        capsize=0, **kw)
    ax.axvline(0, color=k.st.INK, lw=1.0)
    ax.set_xlim(-0.158, 0.014)
    ax.set_xticks([-0.15, -0.1, -0.05, 0], [f"{k.MINUS}0.15", f"{k.MINUS}0.1", f"{k.MINUS}0.05", "0"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=k.st.GRID, lw=0.8)
    ax.set_xlabel("ΔAUROC")
    k.panel(ax, letter, name)
axes[0].set_yticks(range(len(LEVELS)), LEVELS)
axes[0].set_ylim(2.6, -0.6)
k.legend(fig, [k.Line2D([], [], color=k.REFUSAL, marker="s", ls="none", mec="white", mew=1.2, label="Floor"),
               k.Line2D([], [], color=k.CONTROL, marker="o", ls="none", mfc="white", mec=k.CONTROL, mew=1.8,
                        label="Random control")])
k.layout(fig, w_pad=2.0)
k.save(fig, "fig3-control")
