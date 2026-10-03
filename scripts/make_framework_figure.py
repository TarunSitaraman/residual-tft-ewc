"""Framework diagram for the paper (two-column figure): data -> verification -> labels/samples -> protocols -> models -> evaluation.

Writes paper/figures/framework.png (300 dpi). Run from the repository root::

    python scripts/make_framework_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, RULE, FILL, ACCENT, ACCENT_FILL = "#14212e", "#4a5560", "#9aa5ad", "#f3f5f4", "#2a78d6", "#e8f1fc"

BOXES = [
    # (x, y, w, h, title, lines, accent)
    (0.01, 0.55, 0.22, 0.40, "Public data",
     ["S1  6 NMC cells, second life", "S6  20 LFP cells, 7 usage cases", "S7-S9  369 first-life cells",
      "(NMC, NCM, NCA, blend)"], False),
    (0.26, 0.55, 0.22, 0.40, "Verification and audit",
     ["checksums vs repository records", "labels reproduced from raw files", "no protocol currents as inputs",
      "88 automated data checks"], False),
    (0.51, 0.55, 0.22, 0.40, "Labels and samples",
     ["SOH = C_k / C_1 x 100 %", "origin check k, horizon h = 1-3", "target: SOH change k to k+h",
      "inputs known at k only"], False),
    (0.76, 0.55, 0.23, 0.40, "Purged splits",
     ["per cell, in time order:", "train <= 60 %, validation 60-80 %,", "test >= 80 %; no label after",
      "a later split's origin"], False),
    (0.01, 0.04, 0.22, 0.40, "Protocols",
     ["P1  continual stream of tasks", "P2  unseen cell or usage case", "P3  pretrain on first-life pool,",
      "      then the S6 stream"], True),
    (0.26, 0.04, 0.22, 0.40, "Models",
     ["backbones: MLP, TFT", "fine-tuning, replay, EWC, joint", "baselines: persistence, linear",
      "extrapolation, ridge, boosting"], True),
    (0.51, 0.04, 0.22, 0.40, "Evaluation",
     ["error matrix R (RMSE, pp)", "final error, forgetting, BWT", "Holm-corrected paired tests",
      "replication on a 2nd seed set"], True),
    (0.76, 0.04, 0.23, 0.40, "Reuse assessment",
     ["forecast SOH to class", "C0 > 95 %    C1 90-95 %", "C2 80-90 %    C3 <= 80 %",
      "agreement with measured class"], True),
]

# Typographic replacements applied when drawing (keeps the source ASCII-only).
SYMBOLS = {" x 100": " × 100", "<=": "≤", ">=": "≥", "1-3": "1–3", "60-80": "60–80",
           "90-95": "90–95", "80-90": "80–90", "S7-S9": "S7–S9", "SOH to class": "SOH → class",
           "k to k+h": "k → k+h"}


def typeset(text: str) -> str:
    for a, b in SYMBOLS.items():
        text = text.replace(a, b)
    return text


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, y, w, h, title, lines, accent in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=ACCENT_FILL if accent else FILL, edgecolor=ACCENT if accent else RULE,
                                    linewidth=0.9))
        ax.text(x + 0.012, y + h - 0.05, title, fontsize=7.2, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.012, y + h - 0.15, "\n".join(typeset(t) for t in lines), fontsize=5.9, color=INK2, va="top",
                linespacing=1.35)
    arrow = dict(arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=0.9)
    for x0, x1 in ((0.23, 0.26), (0.48, 0.51), (0.73, 0.76)):
        ax.add_patch(FancyArrowPatch((x0 + 0.002, 0.75), (x1 - 0.002, 0.75), **arrow))
        ax.add_patch(FancyArrowPatch((x0 + 0.002, 0.24), (x1 - 0.002, 0.24), **arrow))
    # Row change, routed through the gap between the rows.
    ax.plot([0.875, 0.875, 0.12], [0.55, 0.495, 0.495], color=INK2, linewidth=0.9, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((0.12, 0.495), (0.12, 0.442), **arrow))
    fig.tight_layout(pad=0.2)
    out = ROOT / "paper" / "figures" / "framework.png"
    fig.savefig(out, dpi=300)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
