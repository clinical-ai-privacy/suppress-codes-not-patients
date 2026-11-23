"""Figure 1. The refusal floor against the top level of the hierarchy (official ICD tree, K = 5)."""
import km2fig as k

plt = k.setup()
D = k.data()["floor"]
x = range(len(D["levels"]))
fig, axes = plt.subplots(1, 2, figsize=(k.W, 3.5), sharey=True)
for ax, db, name, letter in zip(axes, ("mimic", "eicu"), ("MIMIC-IV", "eICU-CRD"), "ab"):
    ax.plot(x, D[db]["m3"], color=k.REFUSAL, ls="-", marker="o", mec="white", mew=1.2, clip_on=False, zorder=3)
    ax.plot(x, D[db]["m2"], color=k.REFUSAL, ls="--", marker="s", mfc="white", mec=k.REFUSAL, mew=1.8, clip_on=False,
            zorder=3)
    ax.set_xticks(list(x), D["levels"])
    ax.set_xlim(-0.3, len(D["levels"]) - 0.7)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Top level of the hierarchy")
    k.panel(ax, letter, name)
axes[0].set_ylabel("Patients refused, %")
k.legend(fig, [k.Line2D([], [], color=k.REFUSAL, ls="-", marker="o", mec="white", mew=1.2, label="$m$ = 3"),
               k.Line2D([], [], color=k.REFUSAL, ls="--", marker="s", mfc="white", mec=k.REFUSAL, mew=1.8,
                        label="$m$ = 2")])
k.layout(fig, w_pad=2.0)
k.save(fig, "fig1-floor")
