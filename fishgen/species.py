"""Named presets, each belonging to one animal kind (see kinds.py).

A preset asks its kind's generator for a random animal (fish of a given body type,
or a shark / ray / ... of a given variant) and then pins the genes that make the
species itself. The seed still varies everything a preset leaves alone, so
"Goldfish, seed 3" and "Goldfish, seed 4" are recognisably goldfish but not identical.
"""

from __future__ import annotations

from .genome import FishGenome
from .kinds import individualise, random_genome

RANDOM = "Random"

# Fish: name -> (archetype, overrides). Overrides use genome field names; nested genes
# ("colors", "dorsal", "anal", "pelvic", "dorsal2", "wings", "arms", "spines")
# take a dict of their own fields.
FISH: dict[str, tuple[str, dict]] = {
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


def C(hue, saturation, value, belly=1.0, pattern="none", scale=0.5, offset=0.0, accent=None, fin=None,
      fin_tips=0.0) -> dict:
    """A complete colour override, so a preset's colours never depend on the seed."""
    return dict(hue=hue, saturation=saturation, value=value, belly_lightness=belly, pattern=pattern,
                pattern_scale=scale, pattern_offset=offset, accent_color=accent, fin_color=fin,
                fin_tips=fin_tips, fin_hue_shift=0.0, accent_hue_shift=0.0)


# Other kinds: name -> (kind, variant, overrides). The kind's generator (kinds.py) builds a
# random animal of that kind and variant; the preset only pins what makes the species itself.
OTHERS: dict[str, tuple[str, str, dict]] = {
    # ------------------------------------------------------------------ sharks
    "Tiger shark": ("shark", "requiem", dict(
        nose=0.95, height=0.28, tail_length=0.34,
        colors=C(0.1, 0.25, 0.58, pattern="tiger", scale=0.3, accent="#3b352c"),
    )),
    "Great white shark": ("shark", "requiem", dict(
        height=0.33, nose=0.5, peduncle=0.11, tail_spread=1.8,
        colors=C(0.58, 0.12, 0.55),
    )),
    "Hammerhead shark": ("shark", "hammerhead", dict(
        hammer=0.13, height=0.25, eye_size=0.2,
        colors=C(0.12, 0.15, 0.6),
    )),
    "Whale shark": ("shark", "requiem", dict(
        height=0.3, thickness=0.95, nose=1.0, peak=0.33, eye_size=0.1, eye_height=0.1,
        dorsal=dict(start=0.45, end=0.6, height=0.12),
        colors=C(0.6, 0.35, 0.45, belly=0.85, pattern="spots", scale=0.9, offset=0.3, accent="#e6ebef"),
    )),
    "Mako shark": ("shark", "requiem", dict(
        height=0.24, nose=0.38, peduncle=0.1, tail_spread=1.8, eye_size=0.2,
        colors=C(0.62, 0.65, 0.6),
    )),
    "Blacktip reef shark": ("shark", "requiem", dict(
        height=0.25, nose=0.6,
        colors=C(0.1, 0.15, 0.72, fin_tips=0.3, accent="#16161a"),
    )),
    # ------------------------------------------------------------------ whales & dolphins
    "Orca": ("whale", "whale", dict(
        height=0.3, nose=0.8, eye_pos=0.1, eye_height=0.2, pectoral_size=0.16,
        dorsal=dict(present=True, shape="triangle", start=0.38, end=0.5, height=0.3, sweep=0.15),
        colors=C(0.62, 0.1, 0.13, pattern="orca", accent="#f2f2ee", fin="#1a1b20"),
    )),
    "Bottlenose dolphin": ("whale", "dolphin", dict(
        beak=0.07, height=0.24, nose=0.6,
        dorsal=dict(present=True, shape="triangle", start=0.42, end=0.56, height=0.13, sweep=0.9),
        colors=C(0.58, 0.12, 0.62, belly=0.8),
    )),
    "Common dolphin": ("whale", "dolphin", dict(
        beak=0.08, height=0.23, nose=0.55,
        dorsal=dict(present=True, shape="triangle", start=0.42, end=0.56, height=0.13, sweep=0.9),
        colors=C(0.62, 0.3, 0.35, pattern="lateral", scale=0.6, accent="#d8c58e", fin="#2b3240"),
    )),
    "Beluga": ("whale", "whale", dict(
        height=0.3, nose=1.0, peak=0.42, dorsal=dict(present=False), tail_length=0.18, pectoral_size=0.14,
        colors=C(0.55, 0.05, 0.93, belly=0.4),
    )),
    "Humpback whale": ("whale", "whale", dict(
        height=0.27, nose=0.7, pectoral_size=0.36, eye_pos=0.16, eye_height=-0.1, tail_spread=1.6,
        dorsal=dict(present=True, shape="rounded", start=0.6, end=0.66, height=0.04, sweep=0.5),
        colors=C(0.62, 0.15, 0.25, belly=0.9, fin="#c9ccd2"),
    )),
    # ------------------------------------------------------------------ rays
    "Cownose ray": ("ray", "eagle", dict(
        nose=1.0, wings=dict(start=0.07, end=0.42, span=0.42, sweep=0.75, flap=0.85, wave=0.2),
        dorsal=dict(present=True, shape="triangle", start=0.45, end=0.5, height=0.04, sweep=0.6),
        colors=C(0.08, 0.5, 0.62),
    )),
    "Manta ray": ("ray", "eagle", dict(
        nose=1.0, taper=2.0, peduncle=0.3,
        wings=dict(start=0.04, end=0.4, span=0.62, sweep=0.65, flap=0.8, wave=0.15),
        colors=C(0.6, 0.2, 0.2),
    )),
    "Spotted eagle ray": ("ray", "eagle", dict(
        nose=0.5, wings=dict(start=0.09, end=0.42, span=0.45, sweep=0.75, flap=0.85, wave=0.2),
        colors=C(0.62, 0.3, 0.26, pattern="spots", scale=0.5, offset=0.1, accent="#eef0f0"),
    )),
    "Stingray": ("ray", "stingray", dict(
        nose=0.8, wings=dict(start=0.02, end=0.52, span=0.22, sweep=0.45, flap=0.3, wave=1.3),
        colors=C(0.09, 0.3, 0.5),
    )),
    "Bluespotted stingray": ("ray", "stingray", dict(
        nose=0.8, wings=dict(start=0.02, end=0.5, span=0.2, sweep=0.45, flap=0.3, wave=1.3),
        colors=C(0.11, 0.5, 0.68, pattern="spots", scale=0.4, offset=0.6, accent="#2f8ae0"),
    )),
    # ------------------------------------------------------------------ squid & cuttlefish
    "Squid": ("squid", "squid", dict(
        height=0.22, arms=dict(length=0.45, tentacle_length=0.75),
        colors=C(0.97, 0.35, 0.85, belly=0.5, pattern="spots", scale=1.0, accent="#b0384a"),
    )),
    "Giant squid": ("squid", "squid", dict(
        height=0.18, wings=dict(end=0.2, span=0.08),
        arms=dict(length=0.55, tentacle_length=1.0, thickness=0.028),
        colors=C(0.0, 0.7, 0.6, belly=0.3),
    )),
    "Cuttlefish": ("squid", "cuttlefish", dict(
        colors=C(0.08, 0.3, 0.62, belly=0.5, pattern="tiger", scale=0.2, accent="#4a3c30"),
    )),
    # ------------------------------------------------------------------ octopus
    "Octopus": ("octopus", "octopus", dict(
        height=0.7, arms=dict(length=1.3, thickness=0.05, spread=0.75, curl=0.6),
        colors=C(0.03, 0.6, 0.6, belly=0.45, pattern="spots", scale=0.7, accent="#7a2a1c"),
    )),
    "Blue-ringed octopus": ("octopus", "octopus", dict(
        height=0.65, arms=dict(length=1.1, thickness=0.045, spread=0.7, curl=0.55),
        colors=C(0.12, 0.6, 0.82, belly=0.3, pattern="rings", scale=0.4, accent="#2e6bff"),
    )),
    "Dumbo octopus": ("octopus", "dumbo", dict(
        colors=C(0.97, 0.35, 0.88, belly=0.4),
    )),
    # ------------------------------------------------------------------ urchins
    "Purple sea urchin": ("urchin", "short", dict(
        spines=dict(count=70, length=0.35, sway=0.12), colors=C(0.8, 0.55, 0.45, belly=0.0, fin="#8a4bb0"),
    )),
    "Long-spined urchin": ("urchin", "long", dict(
        spines=dict(count=50, length=0.9, sway=0.12), colors=C(0.7, 0.2, 0.15, belly=0.0, fin="#25222e"),
    )),
    "Red pencil urchin": ("urchin", "short", dict(
        spines=dict(count=24, length=0.45, sway=0.06), colors=C(0.02, 0.55, 0.45, belly=0.0, fin="#d0533a"),
    )),
    "Green sea urchin": ("urchin", "short", dict(
        spines=dict(count=110, length=0.2, sway=0.12), colors=C(0.25, 0.5, 0.45, belly=0.0, fin="#8db04a"),
    )),
}

# name -> (kind, archetype for fish / variant for other kinds, overrides)
SPECIES: dict[str, tuple[str, str | None, dict]] = {
    RANDOM: ("fish", None, {}),
    **{name: ("fish", arch, ov) for name, (arch, ov) in FISH.items()},
    **OTHERS,
}
SPECIES_NAMES = tuple(SPECIES)


def kind_of(species: str) -> str:
    return SPECIES[species][0]


def species_for(kind: str) -> tuple[str, ...]:
    """Presets of one kind, after "Random" (a random animal of that kind)."""
    return (RANDOM, *[n for n, (k, _, _) in SPECIES.items() if k == kind and n != RANDOM])


def genome_for(seed: int, species: str | None = None, kind: str | None = None) -> FishGenome:
    """The animal for command-line style options: a species, a kind, both, or neither (a random fish)."""
    if species not in (None, RANDOM):
        if kind is not None and kind != kind_of(species):
            raise ValueError(f"{species} is a {kind_of(species)}, not a {kind}")
        return make_genome(species, seed)
    return random_genome(kind or "fish", seed)


def make_genome(species: str, seed: int, archetype: str | None = None, kind: str = "fish") -> FishGenome:
    """An animal for ``seed`` shaped into ``species``.

    "Random" gives a random animal of ``kind``; ``archetype`` (fish body type) only
    applies to a random fish. Presets fix their own kind and body type.
    """
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}; choose from {', '.join(SPECIES)}")
    preset_kind, preset_shape, overrides = SPECIES[species]
    if species == RANDOM:
        return random_genome(kind, seed, archetype=archetype)
    if preset_kind == "fish":
        g = random_genome("fish", seed, archetype=preset_shape)
    else:
        g = random_genome(preset_kind, seed, variant=preset_shape)
    for key, value in overrides.items():
        if isinstance(value, dict):
            sub = getattr(g, key)
            for k, v in value.items():
                setattr(sub, k, v)
        else:
            setattr(g, key, value)
    individualise(g, species)
    return g
