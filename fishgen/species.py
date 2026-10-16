"""Named presets, each belonging to one animal kind (see kinds.py).

A preset asks its kind's generator for a random animal (fish of a given body type,
or a shark / ray / ... of a given variant) and then pins the genes that make the
species itself. The seed still varies everything a preset leaves alone, so
"Goldfish, seed 3" and "Goldfish, seed 4" are recognisably goldfish but not identical.
"""

from __future__ import annotations

import random

from .genome import FinGenes, FishGenome
from .kinds import individualise, random_genome

RANDOM = "Random"

# Fish: name -> (archetype, overrides). Overrides use genome field names; nested genes
# ("colors", "dorsal", "anal", "pelvic", "dorsal2", "wings", "arms", "spines")
# take a dict of their own fields.
FISH: dict[str, tuple[str, dict]] = {
    "Goldfish": ("round", dict(
        size=0.65, tail_shape="fan", tail_length=0.36, tail_spread=1.25, peduncle=0.28,
        dorsal=dict(present=True, shape="rounded", height=0.14), anal=dict(shape="rounded"),
        colors=dict(hue=0.07, saturation=0.85, value=0.95, fin_hue_shift=0.0, belly_lightness=0.45,
                    pattern="none", fin_color=None, accent_color=None),
    )),
    "Clownfish": ("classic", dict(
        size=0.6, height=0.42, tail_shape="rounded", eye_size=0.3,
        dorsal=dict(present=True, shape="rounded", start=0.25, end=0.75, height=0.12),
        anal=dict(present=True, shape="rounded"),
        colors=dict(hue=0.065, saturation=0.9, value=0.95, belly_lightness=0.15, pattern="stripes",
                    pattern_scale=0.0, pattern_offset=0.25, accent_color="#f4f4ee", fin_color="#f08a24"),
    )),
    "Angelfish": ("deep", dict(
        # A tall flat disc with a pointed snout; dorsal and anal fins sweep up and back into
        # long streamers, thread-like pelvic fins, and black bars (one through the eye) that
        # run on across the fins.
        size=0.85, height=0.62, thickness=0.35, peak=0.38, nose=0.85, taper=0.9, peduncle=0.22,
        eye_pos=0.13, eye_size=0.22, eye_height=0.3, tail_shape="lunate", tail_length=0.3, tail_spread=1.25,
        dorsal=dict(present=True, shape="pennant", start=0.3, end=0.66, height=0.48, sweep=0.95),
        anal=dict(present=True, shape="pennant", start=0.42, end=0.74, height=0.46, sweep=0.95),
        pelvic=dict(present=True, start=0.3, height=0.32),
        colors=dict(hue=0.12, saturation=0.12, value=0.9, belly_lightness=0.4, pattern="stripes",
                    pattern_scale=0.0, pattern_offset=0.065, accent_color="#24222b", fin_color="#e9e3cf",
                    fin_hue_shift=0.0, pattern_on_fins=True),
    )),
    "Tuna": ("torpedo", dict(
        # Deep-chested torpedo: dark blue back, silver belly, a yellow flank stripe, yellow
        # second dorsal and anal fins, and a dark (not yellow) crescent tail.
        size=1.4, height=0.34, thickness=0.75, nose=0.5, peduncle=0.1, eye_size=0.14, eye_height=0.3,
        tail_shape="lunate", tail_length=0.28, tail_spread=1.7,
        dorsal=dict(present=True, shape="triangle", start=0.3, end=0.45, height=0.1),
        dorsal2=dict(present=True, shape="falcate", start=0.55, end=0.64, height=0.12, sweep=0.7),
        anal=dict(present=True, shape="falcate", start=0.6, end=0.68, height=0.11, sweep=0.7),
        colors=dict(hue=0.63, saturation=0.65, value=0.5, belly_lightness=1.0, pattern="lateral",
                    pattern_scale=0.0, fin_color="#e8c840", accent_color="#d9c25a", tail_body_color=True),
    )),
    "Eel": ("slender", dict(
        # One long, unbroken body: the dorsal and anal fins run on round the tip of the tail.
        size=1.3, height=0.1, peak=0.3, nose=0.6, taper=1.7, peduncle=0.05, tail_shape="none",
        dorsal=dict(present=True, shape="long", start=0.32, end=1.0, height=0.04, sweep=0.1),
        anal=dict(present=True, shape="long", start=0.55, end=1.0, height=0.035, sweep=0.1),
        pelvic=dict(present=False), pectoral_size=0.07, wave_count=1.4, swim_amplitude=0.75,
        colors=dict(hue=0.22, saturation=0.5, value=0.55, belly_lightness=0.7, pattern="none",
                    fin_hue_shift=0.0, fin_color="#7c8a44", accent_color=None),
    )),
    "Pufferfish": ("classic", dict(
        # Small and slim when calm; see the puff_right / puff_left animations for the spiky ball.
        size=0.75, height=0.38, thickness=0.75, nose=0.7, peak=0.4, tail_shape="rounded", tail_length=0.18,
        tail_spread=0.8, eye_size=0.32, eye_height=0.45, eye_pos=0.14,
        dorsal=dict(present=True, shape="rounded", start=0.68, end=0.8, height=0.08),
        anal=dict(present=True, shape="rounded", start=0.7, end=0.82, height=0.07),
        pelvic=dict(present=False),
        colors=dict(hue=0.13, saturation=0.6, value=0.85, belly_lightness=1.0, pattern="spots",
                    pattern_scale=0.6, accent_color="#4a3a24", fin_color="#e8d27a", fin_hue_shift=0.0),
    )),
    "Betta": ("classic", dict(
        size=0.6, height=0.32, tail_shape="fan", tail_length=0.5, tail_spread=1.6,
        dorsal=dict(present=True, shape="sail", start=0.45, end=0.85, height=0.24, sweep=0.9),
        anal=dict(present=True, shape="long", start=0.35, end=0.95, height=0.24, sweep=0.8),
        pelvic=dict(present=True, height=0.18),
        colors=dict(hue=0.97, saturation=0.85, value=0.75, fin_hue_shift=0.0, belly_lightness=0.0,
                    pattern="none", fin_color=None, accent_color=None),
    )),
    "Neon tetra": ("torpedo", dict(
        # Electric blue stripe from the eye to the tail over a red lower rear half.
        size=0.45, height=0.28, tail_shape="forked", eye_size=0.32,
        dorsal=dict(present=True, shape="triangle", start=0.42, end=0.54, height=0.08, sweep=0.5),
        anal=dict(present=True, shape="long", start=0.6, end=0.85, height=0.06),
        colors=dict(hue=0.58, saturation=0.45, value=0.38, belly_lightness=0.9, belly_level=-0.15, pattern="neon",
                    accent_color="#3fe8ff", accent2_color="#e8283c", fin_color="#c7d2dc"),
    )),
    "Koi": ("torpedo", dict(
        size=1.1, height=0.32, tail_shape="rounded", pectoral_size=0.18,
        colors=dict(hue=0.1, saturation=0.06, value=0.95, belly_lightness=0.3, pattern="spots",
                    pattern_scale=0.2, accent_color="#e24a1e", fin_color="#f2ece4"),
    )),
    "Discus": ("round", dict(
        size=0.85, height=0.85, nose=0.9, peak=0.5, tail_shape="truncate", tail_length=0.16,
        dorsal=dict(present=True, shape="long", start=0.25, end=0.92, height=0.1),
        anal=dict(present=True, shape="long", start=0.45, end=0.92, height=0.1),
        colors=dict(hue=0.04, saturation=0.75, value=0.85, belly_lightness=0.0, pattern="stripes",
                    pattern_scale=1.0, accent_color="#3b8fd1", fin_color=None, fin_hue_shift=0.0),
    )),
    "Mackerel": ("slender", dict(
        # Long and skinny: a small spiny first dorsal, a smaller second dorsal near the tail, a
        # neat forked tail, and wavy dark bars over a steel-blue back.
        size=1.0, height=0.19, thickness=0.7, nose=0.75, peak=0.38, taper=1.1, peduncle=0.1,
        tail_shape="forked", tail_length=0.18, tail_spread=1.25, eye_size=0.24, eye_height=0.3,
        dorsal=dict(present=True, shape="triangle", start=0.28, end=0.4, height=0.07, sweep=0.5),
        dorsal2=dict(present=True, shape="triangle", start=0.6, end=0.67, height=0.05, sweep=0.5),
        anal=dict(present=True, shape="triangle", start=0.62, end=0.69, height=0.045, sweep=0.5),
        pelvic=dict(present=True, start=0.36, height=0.05),
        colors=dict(hue=0.56, saturation=0.45, value=0.62, belly_lightness=1.0, belly_level=-0.1,
                    pattern="mackerel", pattern_scale=0.4, accent_color="#152433", accent2_color="#3f6e86",
                    fin_color="#9fb2bf", fin_hue_shift=0.0),
    )),
    "Piranha": ("deep", dict(
        size=0.8, height=0.55, nose=0.7, tail_shape="truncate", eye_size=0.22,
        colors=dict(hue=0.58, saturation=0.15, value=0.6, belly_lightness=0.9, pattern="gradient",
                    accent_color=None, fin_color="#3a3f4a"),
    )),
}


def C(hue, saturation, value, belly=1.0, pattern="none", scale=0.5, offset=0.0, accent=None, fin=None,
      fin_tips=0.0, accent2=None, belly_level=None, on_fins=False, tail_body=False) -> dict:
    """A complete colour override, so a preset's colours never depend on the seed."""
    return dict(hue=hue, saturation=saturation, value=value, belly_lightness=belly, pattern=pattern,
                pattern_scale=scale, pattern_offset=offset, accent_color=accent, fin_color=fin,
                fin_tips=fin_tips, fin_hue_shift=0.0, accent_hue_shift=0.0, accent2_color=accent2,
                belly_level=belly_level, pattern_on_fins=on_fins, tail_body_color=tail_body)


# Other kinds: name -> (kind, variant, overrides). The kind's generator (kinds.py) builds a
# random animal of that kind and variant; the preset only pins what makes the species itself.
OTHERS: dict[str, tuple[str, str, dict]] = {
    # ------------------------------------------------------------------ sharks
    "Tiger shark": ("shark", "requiem", dict(
        nose=0.95, height=0.28, tail_length=0.34, teeth=True, size=1.9,
        colors=C(0.1, 0.25, 0.58, pattern="tiger", scale=0.3, accent="#3b352c", fin="#857a68", tail_body=True),
    )),
    "Great white shark": ("shark", "requiem", dict(
        # Stout torpedo with a conical snout, a big triangular dorsal set back over the
        # pectorals' trailing edge, huge pectorals, and a mouthful of teeth.
        height=0.31, nose=1.35, flat_head=0.2, peak=0.4, peduncle=0.11, tail_spread=1.8, tail_length=0.34,
        dorsal=dict(start=0.39, end=0.55, height=0.24, shape="triangle", sweep=0.55),
        pectoral_size=0.36, pectoral_droop=1.0, mouth=0.13, mouth_height=-0.62, teeth=True, size=2.2,
        colors=C(0.58, 0.12, 0.55, belly_level=-0.12, fin="#7f8790", tail_body=True),
    )),
    "Hammerhead shark": ("shark", "hammerhead", dict(
        hammer=0.14, height=0.25, eye_size=0.2, size=2.0,
        dorsal=dict(height=0.3, shape="falcate", sweep=0.6),
        colors=C(0.12, 0.15, 0.6, fin="#8f8a7c", tail_body=True),
    )),
    "Whale shark": ("shark", "requiem", dict(
        # Huge, broad and flat-headed with a wide mouth right at the front, ridges along
        # the flanks, a chequerboard of pale spots and lines, and a big crescent tail.
        height=0.26, thickness=1.2, nose=0.4, peak=0.32, flat_head=0.95, back_arch=0.6, taper=1.1,
        eye_pos=0.07, eye_size=0.09, eye_height=-0.15, mouth=0.06, mouth_height=-0.15, teeth=False,
        tail_length=0.38, tail_spread=1.9, pectoral_size=0.32,
        dorsal=dict(start=0.5, end=0.66, height=0.15, shape="triangle", sweep=0.6),
        dorsal2=dict(start=0.8, height=0.05), size=3.0,
        colors=C(0.6, 0.4, 0.4, belly=0.9, belly_level=-0.3, pattern="whale_shark", scale=0.4, offset=0.3,
                 accent="#eef3f6", accent2="#a7bccc", fin="#4a5a70", tail_body=True),
    )),
    "Mako shark": ("shark", "requiem", dict(
        # Slim, with a long needle-pointed snout.
        height=0.22, nose=1.75, flat_head=0.1, peduncle=0.1, tail_spread=1.8, eye_size=0.2, teeth=True,
        size=1.7, colors=C(0.62, 0.65, 0.6, fin="#3a4f9a", tail_body=True),
    )),
    "Blacktip reef shark": ("shark", "requiem", dict(
        height=0.25, nose=1.3, size=1.5,
        dorsal=dict(height=0.19, shape="triangle"),
        colors=C(0.1, 0.15, 0.72, fin_tips=0.45, accent="#0b0b0e", fin="#a39d8f", tail_body=True),
    )),
    # ------------------------------------------------------------------ whales & dolphins
    "Orca": ("whale", "whale", dict(
        # The dorsal fin is set per individual: a tall straight male blade or a curved female sickle.
        height=0.3, nose=0.8, flat_head=0.0, eye_pos=0.1, eye_height=0.2, pectoral_size=0.2, pectoral_droop=0.6,
        size=2.4, colors=C(0.62, 0.1, 0.13, belly=0.0, pattern="orca", accent="#f2f2ee", fin="#1a1b20",
                           accent2="#7d828c"),
    )),
    "Bottlenose dolphin": ("whale", "dolphin", dict(
        beak=0.12, height=0.21, nose=0.6, melon=0.012, size=1.8,
        dorsal=dict(present=True, shape="falcate", start=0.42, end=0.56, height=0.13, sweep=0.75),
        colors=C(0.58, 0.12, 0.62, belly=0.8),
    )),
    "Common dolphin": ("whale", "dolphin", dict(
        beak=0.09, height=0.2, nose=0.55, size=1.6,
        dorsal=dict(present=True, shape="falcate", start=0.42, end=0.55, height=0.14, sweep=0.8),
        colors=C(0.62, 0.3, 0.35, pattern="lateral", scale=0.6, accent="#d8c58e", fin="#2b3240"),
    )),
    "Beluga": ("whale", "whale", dict(
        # Big round echolocation melon over a short beak, a flexible neck, no dorsal fin.
        height=0.3, nose=0.5, peak=0.5, beak=0.03, melon=0.085, neck=0.3, flat_head=0.0, back_arch=0.45,
        eye_pos=0.15, eye_height=0.0, dorsal=dict(present=False), tail_length=0.18,
        pectoral_size=0.15, pectoral_shape="lobe", mouth=0.05, mouth_height=-0.3, size=2.1,
        colors=C(0.55, 0.05, 0.93, belly=0.4),
    )),
    "Humpback whale": ("whale", "whale", dict(
        # Long, thin flat head, enormous white flippers, throat grooves, dark above with a white belly.
        height=0.26, nose=1.25, peak=0.42, flat_head=0.7, back_arch=0.55, pectoral_size=0.55,
        pectoral_droop=0.42, eye_pos=0.2, eye_height=-0.35, tail_spread=1.7, pleats=True, size=2.9,
        dorsal=dict(present=True, shape="rounded", start=0.6, end=0.66, height=0.04, sweep=0.5),
        colors=C(0.62, 0.15, 0.22, belly=0.9, belly_level=-0.5, fin="#dfe2e6", tail_body=True),
    )),
    # ------------------------------------------------------------------ rays
    "Cownose ray": ("ray", "eagle", dict(
        # Blunt, squared-off head with the two rounded lobes of the "cow nose" under it.
        nose=0.3, thickness=1.8, lobes="round", lobe_length=0.045, size=1.4,
        wings=dict(start=0.1, end=0.4, span=0.44, sweep=1.25, flap=0.85, wave=0.2),
        dorsal=dict(present=True, shape="triangle", start=0.45, end=0.5, height=0.04, sweep=0.6),
        colors=C(0.08, 0.5, 0.62),
    )),
    "Manta ray": ("ray", "eagle", dict(
        # Huge swept-back wings, a broad squared front with the two cephalic "horns", short tail.
        nose=0.3, thickness=2.4, lobes="horns", lobe_length=0.09, size=2.6,
        wings=dict(start=0.05, end=0.38, span=0.66, sweep=1.3, flap=0.8, wave=0.15),
        colors=C(0.6, 0.2, 0.2, belly=1.0),
    )),
    "Spotted eagle ray": ("ray", "eagle", dict(
        # Narrow head jutting out in front of the wings with a rounded duck-bill snout.
        nose=0.55, thickness=1.5, lobes="none", lobe_length=0.0, size=1.6,
        wings=dict(start=0.13, end=0.42, span=0.47, sweep=1.25, flap=0.85, wave=0.2),
        colors=C(0.62, 0.3, 0.26, pattern="spots", scale=0.5, offset=0.1, accent="#eef0f0"),
    )),
    "Stingray": ("ray", "stingray", dict(
        # Wide diamond disc that takes in the head, with a long whip tail.
        nose=0.85, size=1.2,
        wings=dict(start=0.0, end=0.5, span=0.26, sweep=0.45, shape="diamond", flap=0.3, wave=1.3),
        colors=C(0.09, 0.3, 0.5),
    )),
    "Bluespotted stingray": ("ray", "stingray", dict(
        # Small oval disc, rounded snout.
        nose=0.6, size=0.9,
        wings=dict(start=0.0, end=0.48, span=0.2, sweep=0.45, shape="round", flap=0.3, wave=1.3),
        colors=C(0.11, 0.5, 0.68, pattern="spots", scale=0.4, offset=0.6, accent="#2f8ae0"),
    )),
    # ------------------------------------------------------------------ squid & cuttlefish
    "Squid": ("squid", "squid", dict(
        size=1.0, height=0.22, arms=dict(length=0.45, tentacle_length=0.75),
        colors=C(0.97, 0.35, 0.85, belly=0.5, pattern="spots", scale=1.0, accent="#b0384a"),
    )),
    "Giant squid": ("squid", "squid", dict(
        size=2.6, height=0.18, wings=dict(end=0.22, span=0.12, dihedral=0.35),
        arms=dict(length=0.55, tentacle_length=1.0, thickness=0.028),
        colors=C(0.0, 0.7, 0.6, belly=0.3),
    )),
    "Cuttlefish": ("squid", "cuttlefish", dict(
        # Iridescent: rose-violet skin with teal and gold colour waves rolling down it.
        size=0.9, colors=C(0.86, 0.45, 0.82, belly=0.5, pattern="shimmer", scale=0.4,
                           accent="#4ce6d2", accent2="#ffc35a"),
    )),
    # ------------------------------------------------------------------ octopus
    "Octopus": ("octopus", "octopus", dict(
        size=1.0, height=0.7, arms=dict(length=1.3, thickness=0.05, spread=0.75, curl=0.6),
        colors=C(0.03, 0.6, 0.6, belly=0.45, pattern="spots", scale=0.7, accent="#7a2a1c"),
    )),
    "Blue-ringed octopus": ("octopus", "octopus", dict(
        size=0.55, height=0.65, arms=dict(length=1.1, thickness=0.045, spread=0.7, curl=0.55),
        colors=C(0.12, 0.6, 0.82, belly=0.3, pattern="rings", scale=0.4, accent="#2e6bff"),
    )),
    "Dumbo octopus": ("octopus", "dumbo", dict(
        size=0.7, colors=C(0.97, 0.35, 0.88, belly=0.4),
    )),
    # ------------------------------------------------------------------ urchins
    "Purple sea urchin": ("urchin", "short", dict(
        size=0.65, spines=dict(count=320, length=0.33, sway=0.12),
        colors=C(0.8, 0.55, 0.45, belly=0.0, fin="#8a4bb0"),
    )),
    "Long-spined urchin": ("urchin", "long", dict(
        size=0.75, spines=dict(count=200, length=0.9, sway=0.12),
        colors=C(0.7, 0.2, 0.15, belly=0.0, fin="#25222e"),
    )),
    "Red pencil urchin": ("urchin", "short", dict(
        size=0.65, spines=dict(count=110, length=0.42, sway=0.06),
        colors=C(0.02, 0.55, 0.45, belly=0.0, fin="#d0533a"),
    )),
    "Green sea urchin": ("urchin", "short", dict(
        size=0.6, spines=dict(count=450, length=0.22, sway=0.12),
        colors=C(0.25, 0.5, 0.45, belly=0.0, fin="#8db04a"),
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


def _orca(g: FishGenome, r: random.Random):
    """Orcas are male or female at random, and the dorsal fins could hardly be more different."""
    if r.random() < 0.5:  # male: a tall, straight, upright triangle
        g.dorsal = FinGenes(True, 0.4, 0.5, r.uniform(0.3, 0.36), "triangle", r.uniform(0.0, 0.1))
    else:  # female (and young): a shorter sickle curving back
        g.dorsal = FinGenes(True, 0.4, 0.52, r.uniform(0.14, 0.17), "falcate", r.uniform(0.75, 0.9))


def _pufferfish(g: FishGenome, r: random.Random):
    """Small and slim until something upsets it."""
    g.puff = r.uniform(0.78, 0.88)
    g.puff_spines = r.randint(70, 95)


# Per-species finishing touches that need more than fixed overrides.
SPECIAL = {"Orca": _orca, "Pufferfish": _pufferfish}


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
    if species in SPECIAL:
        SPECIAL[species](g, random.Random(f"special:{species}:{seed}"))
    return g
