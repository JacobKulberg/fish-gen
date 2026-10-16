"""Key colours derived from an animal's colour genes: the renderer's base, belly, fin and pattern colours,
the designer's colour swatches, and the palette summary written into the JSON."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from .genome import ColorGenes

RGB = tuple[int, int, int]

# Slot indices used by the renderer. Index 0 is always transparent.
TRANSPARENT = 0
OUTLINE = 1
SHADOW = 2
BASE = 3
HIGHLIGHT = 4
BELLY = 5
FIN = 6
FIN_DARK = 7
ACCENT = 8
EYE = 9
ACCENT2 = 10

SLOT_NAMES = (
    "transparent", "outline", "shadow", "base", "highlight",
    "belly", "fin", "fin_dark", "accent", "eye", "accent2",
)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _shift_hue(h: float, target: float, amount: float) -> float:
    """Move hue h towards target hue by `amount` (in hue units), the short way round."""
    d = ((target - h + 0.5) % 1.0) - 0.5
    step = max(-amount, min(amount, d))
    return (h + step) % 1.0


def _rgb(h: float, s: float, v: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp(s), _clamp(v))
    return (round(r * 255), round(g * 255), round(b * 255))


def hex_rgb(color: str) -> RGB:
    """'#rrggbb' -> (r, g, b)."""
    c = color.strip().lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected a #rrggbb colour, got {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


@dataclass
class Palette:
    colors: list[RGB]  # indexed by the slot constants above

    def rgba(self, slot: int) -> tuple[int, int, int, int]:
        if slot == TRANSPARENT:
            return (0, 0, 0, 0)
        return (*self.colors[slot], 255)

    def to_dict(self) -> dict[str, str]:
        return {
            name: "#%02x%02x%02x" % c
            for name, c in zip(SLOT_NAMES[1:], self.colors[1:])
        }


def build_palette(g: ColorGenes) -> Palette:
    h, s, v = g.hue, g.saturation, g.value
    # Classic pixel-art hue shifting: shadows drift towards blue/purple,
    # highlights drift towards yellow.
    shadow = _rgb(_shift_hue(h, 0.7, 0.06), s + 0.1, v * 0.68)
    base = _rgb(h, s, v)
    highlight = _rgb(_shift_hue(h, 0.15, 0.05), s * 0.7, min(1.0, v * 1.18 + 0.08))
    outline = _rgb(_shift_hue(h, 0.7, 0.08), min(1.0, s + 0.2), 0.16 + v * 0.08)

    bl = g.belly_lightness
    belly = _rgb(_shift_hue(h, 0.15, 0.04), s * (1 - 0.75 * bl), min(1.0, v + (1 - v) * bl * 0.9 + 0.05))

    fh = h + g.fin_hue_shift
    if g.fin_color:
        fin = hex_rgb(g.fin_color)
        fh, fs, fv = colorsys.rgb_to_hsv(*(c / 255 for c in fin))
        fin_dark = _rgb(_shift_hue(fh, 0.7, 0.05), fs + 0.1, fv * 0.62)
    elif abs(g.fin_hue_shift) < 0.15:
        # Same hue family as the body: fins read as thinner/translucent, so go lighter.
        fin = _rgb(_shift_hue(fh, 0.15, 0.03), s * 0.75, min(1.0, v * 1.05 + 0.12))
    else:
        fin = _rgb(fh, s * 0.9, min(1.0, v * 0.95 + 0.05))
    if not g.fin_color:
        fin_dark = _rgb(_shift_hue(fh, 0.7, 0.05), s + 0.1, v * 0.62)

    ah = h + g.accent_hue_shift
    if g.accent_color:
        accent = hex_rgb(g.accent_color)
    elif abs(g.accent_hue_shift) < 0.2:
        # Same-ish hue: make the pattern contrast via value instead.
        accent = _rgb(ah, min(1.0, s + 0.15), v * 0.5)
    else:
        accent = _rgb(ah, min(1.0, s + 0.1), min(1.0, v + 0.05))

    if g.accent2_color:
        accent2 = hex_rgb(g.accent2_color)
    else:
        accent2 = _rgb(h + 0.33, min(1.0, s + 0.2), min(1.0, v + 0.15))

    eye = (245, 245, 235)
    return Palette([(0, 0, 0), outline, shadow, base, highlight, belly, fin, fin_dark, accent, eye, accent2])
