"""Named fish presets: a body archetype plus genome overrides on top of a seeded random fish.

The seed still varies everything a preset leaves alone, so "Goldfish, seed 3" and
"Goldfish, seed 4" are recognisably goldfish but not identical.
"""

from __future__ import annotations

from .genome import ARCHETYPES, FishGenome

RANDOM = "Random"

# name -> (archetype, overrides). Overrides use genome field names; nested genes
# ("colors", "dorsal", "anal", "pelvic") take a dict of their own fields.
SPECIES: dict[str, tuple[str | None, dict]] = {
    RANDOM: (None, {}),
    "Goldfish": ("round", dict(
        tail_shape="fan", tail_length=0.36, tail_spread=1.25, peduncle=0.28,
        dorsal=dict(present=True, shape="rounded", height=0.14), anal=dict(shape="rounded"),
        colors=dict(hue=0.07, saturation=0.85, value=0.95, fin_hue_shift=0.0, belly_lightness=0.45,
                    pattern="none", fin_color=None, accent_color=None),
    )),
    "Clownfish": ("classic", dict(
        height=0.42, tail_shape="rounded", eye_size=0.3,
        dorsal=dict(present=True, shape="rounded", start=0.25, end=0.75, height=0.12),
        anal=dict(present=True, shape="rounded"),
        colors=dict(hue=0.065, saturation=0.9, value=0.95, belly_lightness=0.15, pattern="stripes",
                    pattern_scale=0.0, pattern_offset=0.25, accent_color="#f4f4ee", fin_color="#f08a24"),
    )),
    "Angelfish": ("deep", dict(
        height=0.7, nose=0.45, tail_shape="lunate", tail_length=0.3,
        dorsal=dict(present=True, shape="sail", start=0.3, end=0.7, height=0.42, sweep=0.85),
        anal=dict(present=True, shape="sail", start=0.45, end=0.8, height=0.4, sweep=0.85),
        pelvic=dict(present=True, height=0.2),
        colors=dict(hue=0.12, saturation=0.15, value=0.88, belly_lightness=0.4, pattern="stripes",
                    pattern_scale=0.0, pattern_offset=0.1, accent_color="#2b2a35", fin_color=None, fin_hue_shift=0.0),
    )),
    "Tuna": ("torpedo", dict(
        height=0.3, nose=0.5, peduncle=0.12, tail_shape="lunate", tail_length=0.28, tail_spread=1.7,
        dorsal=dict(present=True, shape="triangle", height=0.1), anal=dict(present=True, shape="triangle", height=0.09),
        colors=dict(hue=0.62, saturation=0.6, value=0.6, belly_lightness=1.0, pattern="none",
                    fin_color="#e8c840", accent_color=None),
    )),
    "Eel": ("slender", dict(
        height=0.16, nose=0.6, taper=0.6, peduncle=0.3, tail_shape="pointed", tail_length=0.3,
        dorsal=dict(present=True, shape="long", start=0.3, end=0.98, height=0.06, sweep=0.2),
        anal=dict(present=True, shape="long", start=0.55, end=0.98, height=0.05, sweep=0.2),
        pelvic=dict(present=False), pectoral_size=0.08, wave_count=1.4, swim_amplitude=0.75,
        colors=dict(hue=0.22, saturation=0.5, value=0.55, belly_lightness=0.7, pattern="none",
                    fin_hue_shift=0.0, fin_color=None, accent_color=None),
    )),
    "Pufferfish": ("round", dict(
        height=0.75, thickness=0.75, nose=1.0, tail_shape="rounded", tail_length=0.18, tail_spread=0.8,
        eye_size=0.3, eye_height=0.45,
        dorsal=dict(present=True, shape="rounded", start=0.68, end=0.82, height=0.08),
        anal=dict(present=True, shape="rounded", start=0.7, end=0.84, height=0.07),
        pelvic=dict(present=False),
        colors=dict(hue=0.13, saturation=0.6, value=0.85, belly_lightness=1.0, pattern="spots",
                    pattern_scale=0.6, accent_color="#4a3a24", fin_color=None, fin_hue_shift=0.0),
    )),
    "Betta": ("classic", dict(
        height=0.32, tail_shape="fan", tail_length=0.5, tail_spread=1.6,
        dorsal=dict(present=True, shape="sail", start=0.45, end=0.85, height=0.24, sweep=0.9),
        anal=dict(present=True, shape="long", start=0.35, end=0.95, height=0.24, sweep=0.8),
        pelvic=dict(present=True, height=0.18),
        colors=dict(hue=0.97, saturation=0.85, value=0.75, fin_hue_shift=0.0, belly_lightness=0.0,
                    pattern="none", fin_color=None, accent_color=None),
    )),
    "Neon tetra": ("torpedo", dict(
        height=0.3, tail_shape="forked", eye_size=0.32,
        colors=dict(hue=0.6, saturation=0.5, value=0.45, belly_lightness=0.9, pattern="lateral",
                    pattern_scale=0.3, accent_color="#3fe0ff", fin_color="#b9c8d6"),
    )),
    "Koi": ("torpedo", dict(
        height=0.32, tail_shape="rounded", pectoral_size=0.18,
        colors=dict(hue=0.1, saturation=0.06, value=0.95, belly_lightness=0.3, pattern="spots",
                    pattern_scale=0.2, accent_color="#e24a1e", fin_color="#f2ece4"),
    )),
    "Discus": ("round", dict(
        height=0.85, nose=0.9, peak=0.5, tail_shape="truncate", tail_length=0.16,
        dorsal=dict(present=True, shape="long", start=0.25, end=0.92, height=0.1),
        anal=dict(present=True, shape="long", start=0.45, end=0.92, height=0.1),
        colors=dict(hue=0.04, saturation=0.75, value=0.85, belly_lightness=0.0, pattern="stripes",
                    pattern_scale=1.0, accent_color="#3b8fd1", fin_color=None, fin_hue_shift=0.0),
    )),
    "Mackerel": ("torpedo", dict(
        height=0.26, peduncle=0.12, tail_shape="forked", tail_spread=1.5,
        colors=dict(hue=0.45, saturation=0.55, value=0.6, belly_lightness=1.0, pattern="saddle",
                    pattern_scale=0.4, accent_color="#1c3440", fin_color=None, fin_hue_shift=0.0),
    )),
    "Piranha": ("deep", dict(
        height=0.55, nose=0.7, tail_shape="truncate", eye_size=0.22,
        colors=dict(hue=0.58, saturation=0.15, value=0.6, belly_lightness=0.9, pattern="gradient",
                    accent_color=None, fin_color="#3a3f4a"),
    )),
}

SPECIES_NAMES = tuple(SPECIES)


def make_genome(species: str, seed: int, archetype: str | None = None) -> FishGenome:
    """A random fish for ``seed`` shaped into ``species``.

    ``archetype`` only applies to the "Random" species (presets fix their own body type).
    """
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}; choose from {', '.join(SPECIES)}")
    preset_arch, overrides = SPECIES[species]
    if archetype is not None and archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}")
    g = FishGenome.random(seed, archetype=preset_arch or archetype)
    for key, value in overrides.items():
        if isinstance(value, dict):
            sub = getattr(g, key)
            for k, v in value.items():
                setattr(sub, k, v)
        else:
            setattr(g, key, value)
    return g
