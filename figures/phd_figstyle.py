"""One figure style for the plots (matplotlib).

The same font (Roboto, bundled in fonts/), the same colours (navy, clay and grey inks), names in medium
weight, and a legend strip under a hairline at the bottom of the figure. A figure is read without its
text, so the text inside it only names things (axis, series); explanations go to the caption. This
module holds the palette, the font, the sizes and the line weights, so that every figure script uses
the same ones.

    import phd_figstyle as st
    st.apply()
    fig, ax = plt.subplots(figsize=st.size(0.42))
    ax.plot(x, y, **st.series(0))
    st.save(fig, out_dir, "fig-name")
"""
from __future__ import annotations

from pathlib import Path

# Categorical palette, in fixed order. Checked with a palette validator:
# all pairs pass colour-blind separation (worst 15.5), normal-vision separation (worst 18.5) and contrast
# on white. Two advisory checks do not pass and are accepted for the sake of one style: navy is darker than
# the recommended band and clay is muted. Every series therefore also carries its own marker and dash.
NAVY, CLAY, BLUE = "#2E3192", "#8E5A43", "#0076A8"
PALETTE = (NAVY, CLAY, BLUE)
MARKERS = ("o", "s", "^")
DASHES = ("-", "--", (0, (1, 1.6)))

INK = "#333333"        # names, axis labels (node names of the diagrams)
MUTED = "#606A70"      # ticks, secondary lines, tags
GRID = "#E6E6E6"       # hairlines, the legend line
SOFT = "#F4F5F7"       # shaded bands, quiet fills (the system zone of the diagrams)
TINT = "#F1F1FA"       # fill of what the work contributes, with a navy outline
DOT = "#CCCCCC"        # background marks
SURFACE = "#FFFFFF"

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT = ["Roboto", "Arial", "DejaVu Sans"]
FONT_CSS = "Roboto, Arial, sans-serif"
MONO_CSS = "'Roboto Mono', Menlo, monospace"
WIDTH_IN = 6.5         # full text width of an article page
TEXT_PT = 8.0          # axis labels, legend, panel names at printed size; node names of a diagram match it
LINE_PT = 1.8
MARKER_PT = 5.5


def _register_fonts() -> None:
    """Make the bundled Roboto known to matplotlib, so the figure does not depend on installed fonts."""
    from matplotlib import font_manager

    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))


def apply() -> None:
    """Set matplotlib defaults. Call once, before creating a figure."""
    import matplotlib as mpl

    _register_fonts()
    mpl.rcParams.update({
        "mathtext.fontset": "custom", "mathtext.rm": "Roboto", "mathtext.it": "Roboto:italic",
        "mathtext.bf": "Roboto", "mathtext.sf": "Roboto", "mathtext.cal": "Roboto:italic",
        "mathtext.tt": "Roboto", "mathtext.bfit": "Roboto:italic", "mathtext.default": "it",
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": TEXT_PT,
        "axes.titlesize": TEXT_PT, "axes.labelsize": TEXT_PT, "xtick.labelsize": TEXT_PT - 0.5,
        "ytick.labelsize": TEXT_PT - 0.5, "legend.fontsize": TEXT_PT,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "lines.linewidth": LINE_PT, "lines.markersize": MARKER_PT,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def size(height_ratio: float = 0.42, width: float = 1.0) -> tuple[float, float]:
    """Figure size in inches: `width` as a share of the text width, height as a share of that width."""
    w = WIDTH_IN * width
    return (w, w * height_ratio)


def series(i: int, line: bool = True, marker: bool = True) -> dict:
    """Plot keywords of series i: colour, plus marker and dash so that the series survives grey print."""
    kw = {"color": PALETTE[i], "ls": DASHES[i] if line else "none"}
    if marker:
        kw.update(marker=MARKERS[i], mec=SURFACE, mew=1.0)
    return kw


def panel(ax, letter: str, name: str = "") -> None:
    """Panel title: a letter and at most three words, in the weight of a node name."""
    ax.set_title(f"({letter}) {name}".strip(), loc="left", pad=6, fontweight="medium")


def legend_below(fig, handles, labels=None, ncol: int | None = None, x: float = 0.5) -> None:
    """One legend for the whole figure: a strip under a hairline, as in the diagrams.

    Leave room for it: `fig.tight_layout(rect=(0, 0.12, 1, 1))`. `x` is kept for old calls and ignored.
    """
    from matplotlib.lines import Line2D

    labels = labels or [h.get_label() for h in handles]
    w, h = fig.get_size_inches()
    y_line = 0.34 / h                                       # the strip is 0.34 in high at any figure size
    fig.add_artist(Line2D([0.0, 1.0], [y_line, y_line], transform=fig.transFigure, color=GRID, lw=0.8))
    fig.legend(handles, labels, loc="lower left", ncol=ncol or len(handles), handlelength=2.6,
               columnspacing=2.2, borderaxespad=0.0, borderpad=0.0, bbox_to_anchor=(0.08 / w, 0.07 / h))


def save(fig, out_dir, stem: str) -> None:
    """Vector PDF for LaTeX and a 300 dpi PNG for Word and slides, from the same figure."""
    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
