"""Figure 4. Whom the release hurts: eICU-CRD, block level (K = 5, m = 3). The floor refuses patients;
S-loc removes single codes and releases every patient."""
import km2fig as k

plt = k.setup()
D = k.data()["whom"]
fig, axes = plt.subplots(1, 2, figsize=(k.W, 3.5), sharey=True)
F = dict(color=k.REFUSAL, ls="-", marker="o", mec="white", mew=1.2)
S = dict(color=k.SUPPRESSION, ls="--", marker="s", mec="white", mew=1.2)
for ax, suffix, ticks, xlabel, name, letter in (
        (axes[0], "len", ["1–2", "3–5", "6–10", "11–20", "≥21"], "Codes per patient",
         "By record length", "a"),
        (axes[1], "band", [str(10 * i) for i in range(10)], "Age band, years", "By age", "b")):
    x = range(len(ticks))
    ax.plot(x, D[f"floor_refused_{suffix}"], **F)
    ax.plot(x, D[f"sloc_removed_{suffix}"], **S)
    ax.set_xticks(list(x), ticks)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    k.panel(ax, letter, name)
axes[0].set_ylabel("Share removed, %")
k.legend(fig, [k.Line2D([], [], label="Floor: patients refused", **F),
               k.Line2D([], [], label="S-loc: codes removed", **S)])
k.layout(fig, w_pad=2.0)
k.save(fig, "fig4-whom")
