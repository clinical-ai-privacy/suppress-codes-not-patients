"""Shared settings for the figures of "Suppress Codes, Not Patients".

The style is set in phd_figstyle (Roboto, navy / clay / blue, legend strip under a hairline). Two
things differ, because the publisher asks for them: the figure is drawn at the
two-column width with 12 pt text (OUP, Making Figures Accessible: 12-14 pt), and it is also saved as
a 600 dpi TIFF. One colour means one thing in every figure: clay = refusal (the floor), navy =
suppression (S-loc), blue = hybrid, grey = random control. Marker, dash or hatch repeat the colour.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

import phd_figstyle as st                # noqa: E402

W = 7.0            # inches
PT = 12.0          # every text in the figure
STRIP = 0.50       # inches, height of the legend strip
REFUSAL, SUPPRESSION, HYBRID, CONTROL = st.CLAY, st.NAVY, st.BLUE, st.MUTED
MINUS = "−"


def setup():
    st.FONT_DIR = HERE / "fonts"
    st.apply()
    matplotlib.rcParams.update({
        "font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT, "xtick.labelsize": PT,
        "ytick.labelsize": PT, "legend.fontsize": PT, "lines.linewidth": 2.2, "lines.markersize": 8.5,
        "xtick.color": st.INK, "ytick.color": st.INK, "axes.unicode_minus": True})
    return plt


def data():
    return json.loads((HERE / "data" / "km2_figures_data.json").read_text())


def panel(ax, letter, name):
    ax.set_title(f"({letter}) {name}", loc="left", pad=8, fontweight="medium")


def legend(fig, handles, ncol=None):
    """One legend for the figure: a strip under a hairline at the bottom."""
    w, h = fig.get_size_inches()
    y = STRIP / h
    fig.add_artist(Line2D([0.0, 1.0], [y, y], transform=fig.transFigure, color=st.GRID, lw=0.8))
    fig.legend(handles=handles, loc="lower left", ncol=ncol or len(handles), handlelength=2.4,
               columnspacing=2.0, borderaxespad=0.0, borderpad=0.0, bbox_to_anchor=(0.10 / w, 0.13 / h))


def layout(fig, **kw):
    w, h = fig.get_size_inches()
    fig.tight_layout(rect=(0, (STRIP + 0.06) / h, 1, 1), **kw)


def save(fig, stem):
    """Vector PDF, 300 dpi PNG preview and a 600 dpi RGB TIFF (LZW), from the same figure."""
    from PIL import Image
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    tmp = HERE / f"{stem}_600.png"
    fig.savefig(tmp, dpi=600)
    Image.open(tmp).convert("RGB").save(HERE / f"{stem}.tif", dpi=(600, 600), compression="tiff_lzw")
    tmp.unlink()
    plt.close(fig)
    print("wrote", stem)
