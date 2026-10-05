"""Big-data pipeline figure (two-column): raw S6 ageing files -> Spark map -> per-cycle table -> Spark SQL reduce ->
block inputs -> forecasting. Same style as make_framework_figure.py; Spark stages in the accent colour.

Writes paper/figures/spark_pipeline.png (300 dpi). Run from the repository root::

    python scripts/make_spark_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from make_framework_figure import ACCENT, ACCENT_FILL, FILL, INK, INK2, RULE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, GAP = 0.172, 0.03
BOXES = [
    ("Raw S6 ageing data", ["827 files: one per cell", "and 100-cycle block", "132 GB raw (16 GB xz)",
                              "~1.4 billion rows", "SHA-256 manifest"], False),
    ("Spark map", ["one task per block", "parse raw rows into steps", "cycle validity, 7 responses",
                   "local[N] worker cores"], True),
    ("Per-cycle table", ["~83,000 cycles", "one row per cycle", "cached in Spark memory",
                         "no protocol-scaled values"], True),
    ("Spark SQL reduce", ["groupBy (cell, block)", "mean, last, slope over", "valid cycles; >= 50 valid",
                          "-> 21 block inputs"], True),
    ("Forecasting", ["block ending at check c", "joins origin c (causal)", "TFT + EWC, residual",
                     "on linear extrapolation"], False),
]
SYMBOLS = {">=": "≥", "->": "→", " x 7": " × 7"}


def typeset(text: str) -> str:
    for a, b in SYMBOLS.items():
        text = text.replace(a, b)
    return text


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 1.32))      # drawn at print size: include at the full text width
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=0.9)
    for i, (title, lines, accent) in enumerate(BOXES):
        x = 0.005 + i * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, 0.17), W, 0.81, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=ACCENT_FILL if accent else FILL, edgecolor=ACCENT if accent else RULE,
                                    linewidth=0.9))
        ax.text(x + 0.008, 0.93, title, fontsize=7.0, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.008, 0.76, "\n".join(typeset(t) for t in lines), fontsize=6.0, color=INK2, va="top",
                linespacing=1.3)
        if i < len(BOXES) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.002, 0.57), (x + W + GAP - 0.002, 0.57), **arrow))
    x0, x1 = 0.005 + (W + GAP), 0.005 + 3 * (W + GAP) + W
    ax.text((x0 + x1) / 2, 0.065, "Apache Spark 4.2 (PySpark), local mode on one machine; output validated against a "
            "pandas reference implementation", fontsize=6.0, color=ACCENT, ha="center", va="center")
    fig.tight_layout(pad=0.2)
    out = ROOT / "paper" / "figures" / "spark_pipeline.png"
    fig.savefig(out, dpi=300)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
