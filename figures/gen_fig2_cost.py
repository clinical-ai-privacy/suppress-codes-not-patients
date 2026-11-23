"""Figure 2. What a strict release costs a prediction model: change in AUROC against raw codes for
everyone, with the Nadeau-Bengio 95 % interval (K = 5, m = 3)."""
import km2fig as k

plt = k.setup()
D = k.data()["cost"]
ROWS = [("floor_block", "Floor, block", k.REFUSAL, "s"), ("floor_chapter", "Floor, chapter", k.REFUSAL, "s"),
        ("sloc_code", "S-loc, code", k.SUPPRESSION, "o"), ("sloc_block", "S-loc, block", k.SUPPRESSION, "o"),
        ("b3gs_chapter", "B3gs, chapter", k.HYBRID, "^")]
PANELS = [("eicu_death", "eICU-CRD\ndeath"), ("mimic_readmit30", "MIMIC-IV\nreadmission"),
          ("mimic_death", "MIMIC-IV\ndeath")]
fig, axes = plt.subplots(1, 3, figsize=(k.W, 3.9), sharey=True)
for ax, (key, name), letter in zip(axes, PANELS, "abc"):
    for i, (rk, _, col, mk) in enumerate(ROWS):
        r = D[key][rk]
        if abs(r["auroc"] - 0.5) < 1e-9:        # one class left in training: not estimable
            continue
        ax.errorbar(r["d"], i, xerr=[[r["d"] - r["lo"]], [r["hi"] - r["d"]]], fmt=mk, color=col, mec="white",
                    mew=1.2, elinewidth=2.2, capsize=0)
    ax.axvline(0, color=k.st.INK, lw=1.0)
    ax.set_xlim(-0.158, 0.014)
    ax.set_xticks([-0.1, 0], [f"{k.MINUS}0.1", "0"])
    ax.set_xticks([-0.15, -0.05], minor=True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="both", color=k.st.GRID, lw=0.8)
    ax.set_xlabel("ΔAUROC")
    k.panel(ax, letter, name)
axes[0].set_yticks(range(len(ROWS)), [r[1] for r in ROWS])
axes[0].set_ylim(len(ROWS) - 0.5, -0.5)
k.legend(fig, [k.Line2D([], [], color=c, marker=m, ls="none", mec="white", mew=1.2, label=n)
               for c, m, n in ((k.REFUSAL, "s", "Refusal"), (k.SUPPRESSION, "o", "Suppression"),
                               (k.HYBRID, "^", "Hybrid"))])
k.layout(fig, w_pad=1.2)
k.save(fig, "fig2-cost")
