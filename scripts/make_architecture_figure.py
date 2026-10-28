"""Architecture of the proposed model (one-column figure): the TFT correction network (a) and the
EWC training loss and shrunk residual forecast with its interval (b).

Sizes follow src/models/tft_continual.py with the settings used from D21 on (19 shared inputs,
window W=4, d_model=8, two heads, 12,708 parameters). Writes paper/figures/architecture.png (300 dpi).
Run from the repository root::

    python scripts/make_architecture_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, RULE, FILL, ACCENT, ACCENT_FILL = "#14212e", "#4a5560", "#9aa5ad", "#f3f5f4", "#2a78d6", "#e8f1fc"
WARM, WARM_FILL = "#b4632a", "#fcefe4"

# (title, detail) from input to output, drawn top to bottom.
TFT = [
    ("Input window", "W = 4 latest origins of the cell,\n19 inputs + missing flags"),
    ("Per-variable embedding", "each input to d = 8"),
    ("Variable selection", "gated residual networks,\nsoftmax weights per input"),
    ("LSTM encoder", "gate, add & norm skip"),
    ("Masked attention", "2 heads, past origins only;\nGRN, gate, add & norm"),
    ("Linear head", r"correction $g_\theta(\mathbf{x})$ at origin $k$"),
]

LOSS = ("Training on task $t$", "MSE on the correction plus the\nEWC penalty over finished tasks $j$:",
        r"$\sum_{j<t}\frac{\lambda}{2}\sum_p F_{j,p}\,(\theta_p-\theta^*_{j,p})^2$")
EXTRAP = ("Linear extrapolation", r"$\hat\Delta^{\mathrm{lin}} = r_k\,(t_{k+h}-t_k)$")
SHRUNK = ("Shrunk correction", r"$\hat\Delta = \hat\Delta^{\mathrm{lin}} + \alpha_h\, g_\theta(\mathbf{x})$")
INTERVAL = ("Conformal interval and reuse class",
            r"$\hat\Delta \pm q_h$ per horizon; forecast SOH mapped to classes C0–C3")


def box(ax, x: float, y: float, w: float, h: float, title: str, detail: str, edge: str, face: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(x + 0.018, y + h - 0.015, title, fontsize=6.4, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.018, y + h - 0.045, detail, fontsize=5.6, color=INK2, va="top", linespacing=1.25)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", mutation_scale=7, color=INK2, linewidth=0.8)

    # (a) TFT correction network, top to bottom.
    ax.text(0.0, 0.998, r"(a) correction network $g_\theta$ (TFT)", fontsize=6.6, color=INK, va="top")
    x, w, h, gap, top = 0.005, 0.46, 0.098, 0.03, 0.955
    ys = [top - h - i * (h + gap) for i in range(len(TFT))]
    for (title, detail), y in zip(TFT, ys):
        box(ax, x, y, w, h, title, detail, ACCENT, ACCENT_FILL)
    for y0, y1 in zip(ys, ys[1:]):
        ax.add_patch(FancyArrowPatch((x + w / 2, y0), (x + w / 2, y1 + h), **arrow))

    # (b) training loss beside the upper TFT layers; forecast chain beside the head.
    ax.text(0.53, 0.998, "(b) training and forecast", fontsize=6.6, color=INK, va="top")
    x2, w2 = 0.53, 0.455
    loss_y, loss_h = ys[1], ys[0] + h - ys[1]
    box(ax, x2, loss_y, w2, loss_h, LOSS[0], LOSS[1], WARM, WARM_FILL)
    ax.text(x2 + 0.018, loss_y + 0.012, LOSS[2], fontsize=5.8, color=INK2, va="bottom")
    ax.add_patch(FancyArrowPatch((x2, ys[1] + h / 2), (x + w, ys[1] + h / 2), arrowstyle="-|>", mutation_scale=7,
                                 color=WARM, linewidth=0.8, linestyle=(0, (3, 2))))

    box(ax, x2, ys[4], w2, h, *EXTRAP, RULE, FILL)
    box(ax, x2, ys[5], w2, h, *SHRUNK, RULE, FILL)
    ax.add_patch(FancyArrowPatch((x2 + w2 / 2, ys[4]), (x2 + w2 / 2, ys[5] + h), **arrow))
    ax.add_patch(FancyArrowPatch((x + w, ys[5] + h / 2), (x2, ys[5] + h / 2), **arrow))

    # Interval under both columns.
    ih = 0.075
    iy = ys[5] - gap - ih
    box(ax, x, iy, x2 + w2 - x, ih, *INTERVAL, RULE, FILL)
    ax.add_patch(FancyArrowPatch((x2 + w2 / 2, ys[5]), (x2 + w2 / 2, iy + ih), **arrow))

    ax.set_ylim(iy - 0.01, 1)
    fig.tight_layout(pad=0.15)
    out = ROOT / "paper" / "figures" / "architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
