"""Compose the dashboard figures of the paper from screenshots of the results dashboard.

The cards are captured by scripts/capture_dashboard_cards.js from dashboard/index.html, whose charts read only the
committed exports in results/dashboard/ (scripts/export_dashboard_data.py). This script lays the cards out in rows,
adds panel letters and writes paper/figures/dash_*.png.

Usage (from the repository root):
    python -m http.server 8765 --bind 127.0.0.1 &
    node scripts/capture_dashboard_cards.js /tmp/dashboard_cards
    python scripts/make_dashboard_figures.py /tmp/dashboard_cards
"""
from __future__ import annotations

import string
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

# Figure name -> rows of card screenshots (pane_index), as laid out on the page.
FIGURES = {
    # Full-page figure: the S6 stream and per-condition detail.
    "dash_s6": [["continual_0", "continual_1"], ["continual_4"], ["accuracy_0"]],
    # Full-page figure: intervals on unseen S6 conditions and the 124 unseen S7/S10 conditions.
    "dash_unseen": [["reuse_2"], ["many_2", "many_3"], ["many_4"]],
    # Skill against persistence and error by horizon (S6, P2).
    "dash_horizon": [["accuracy_1", "accuracy_2"]],
    # The registered questions of the replay ensemble V2 (D38).
    "dash_d38": [["d38_0"]],
}

WIDTH = 3600  # output width in pixels (about 510 dpi at the 7.16 in text width)
GAP = 30
LABEL_H = 70


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("texgyreheros-bold.otf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    for p in Path("/usr/share").rglob("texgyreheros-bold.otf"):
        return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def compose_row(cards: list[Image.Image], width: int) -> Image.Image:
    """Scale the cards of one row to a common height so that together they fill the width."""
    h = max(c.height for c in cards)
    scaled = [c.resize((round(c.width * h / c.height), h), Image.LANCZOS) for c in cards]
    total = sum(c.width for c in scaled) + GAP * (len(scaled) - 1)
    k = (width - GAP * (len(scaled) - 1)) / (total - GAP * (len(scaled) - 1))
    scaled = [c.resize((round(c.width * k), round(c.height * k)), Image.LANCZOS) for c in scaled]
    row = Image.new("RGB", (width, max(c.height for c in scaled)), "white")
    x = 0
    for c in scaled:
        row.paste(c, (x, 0))
        x += c.width + GAP
    return row, [c.width for c in scaled]


def main(src: Path) -> None:
    f = font(52)
    for name, rows in FIGURES.items():
        parts, letter = [], iter(string.ascii_lowercase)
        labelled = sum(len(r) for r in rows) > 1  # panel letters only for multi-panel figures
        for row in rows:
            cards = [Image.open(src / f"{c}.png").convert("RGB") for c in row]
            img, widths = compose_row(cards, WIDTH)
            top = LABEL_H if labelled else 0
            band = Image.new("RGB", (WIDTH, top + img.height), "white")
            band.paste(img, (0, top))
            d = ImageDraw.Draw(band)
            x = 0
            for w in widths if labelled else []:
                d.text((x + 4, 4), f"({next(letter)})", fill="black", font=f)
                x += w + GAP
            parts.append(band)
        fig = Image.new("RGB", (WIDTH, sum(p.height for p in parts) + GAP * (len(parts) - 1)), "white")
        y = 0
        for p in parts:
            fig.paste(p, (0, y))
            y += p.height + GAP
        out = OUT / f"{name}.png"
        fig.save(out, optimize=True)
        print(out.relative_to(ROOT), fig.size)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "dashboard_cards"))
