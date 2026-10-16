"""Named presets, each belonging to one animal kind (see kinds.py).

A preset asks its kind's generator for a random animal (fish of a given body type,
or a shark / ray / ... of a given variant) and then pins the genes that make the
species itself. The seed still varies everything a preset leaves alone, so
"Goldfish, seed 3" and "Goldfish, seed 4" are recognisably goldfish but not identical.

Every preset's real adult length (``length_cm``; disc width for rays, shell diameter for
urchins) comes from the size database in sizes.py: each individual is drawn from its species'
measured adult range, so animals drawn to scale come out the right size next to each other. Some species come in colour varieties (``morphs``): the seed picks one, e.g.
a koi is a Kohaku, Sanke, Showa, Ogon or Chagoi.
"""

from __future__ import annotations

import random

from .genome import FinGenes, FishGenome
from .kinds import individualise, random_genome
from .sizes import SIZES

RANDOM = "Random"


def C(hue, saturation, value, belly=1.0, pattern="none", scale=0.5, offset=0.0, accent=None, fin=None,
      fin_tips=0.0, accent2=None, belly_level=None, on_fins=False, tail_body=False, tail=None, fin_edge=None,
      accent_edge=None, belly_pattern=False) -> dict:
    """A complete colour override, so a preset's colours never depend on the seed."""
    return dict(hue=hue, saturation=saturation, value=value, belly_lightness=belly, pattern=pattern,
                pattern_scale=scale, pattern_offset=offset, accent_color=accent, fin_color=fin,
                fin_tips=fin_tips, fin_hue_shift=0.0, accent_hue_shift=0.0, accent2_color=accent2,
                belly_level=belly_level, pattern_on_fins=on_fins, tail_body_color=tail_body, tail_color=tail,
                fin_edge_color=fin_edge, accent_edge_color=accent_edge, belly_pattern=belly_pattern)


# Fish: name -> (archetype, overrides). Overrides use genome field names; nested genes
# ("colors", "dorsal", "anal", "pelvic", "dorsal2", "wings", "arms", "spines")
# take a dict of their own fields. "morphs" is a list of (variety, weight, overrides).
FISH: dict[str, tuple[str, dict]] = {
    "Goldfish": ("round", dict(
        tail_shape="fan", tail_length=0.36, tail_spread=1.25, peduncle=0.28,
        scales=0.035, silver=0.35, gloss=0.5, fin_alpha=0.6, iris_color="#d8a850",
        dorsal=dict(present=True, shape="rounded", height=0.14), anal=dict(shape="rounded"),
        colors=C(0.07, 0.85, 0.95, belly=0.45),
        morphs=[
            ("Orange", 4, {}),
            ("Red and white", 2, dict(colors=dict(hue=0.1, saturation=0.06, value=0.95, pattern="koi",
                                                  pattern_scale=0.45, accent_color="#e0401c"))),
            ("Calico", 1, dict(silver=0.1, colors=dict(hue=0.6, saturation=0.15, value=0.85, pattern="calico",
                                                       pattern_scale=0.5, accent_color="#e8701c",
                                                       accent2_color="#1c1c22"))),
            ("Lemon", 1, dict(colors=dict(hue=0.13, saturation=0.75, value=0.98))),
        ],
    )),
    "Clownfish": ("classic", dict(
        # Three white bars edged in black, black-rimmed fins, a rounded tail.
        height=0.44, peak=0.36, nose=0.5, taper=1.0, peduncle=0.32, tail_shape="rounded",
        tail_length=0.24, eye_size=0.27, scales=0.025, fin_alpha=0.92, pectoral_size=0.18,
        iris_color="#e8a020", gloss=0.45,
        dorsal=dict(present=True, shape="rounded", start=0.25, end=0.75, height=0.12),
        anal=dict(present=True, shape="rounded"),
        colors=C(0.065, 0.92, 0.95, belly=0.15, pattern="stripes", scale=0.0, offset=0.25, accent="#f6f6f0",
                 fin="#f08a24", fin_edge="#141012", accent_edge="#141012"),
        morphs=[
            ("Ocellaris", 5, {}),
            ("Black", 1, dict(colors=dict(hue=0.05, saturation=0.4, value=0.1, fin_color="#1c1716"))),
        ],
    )),
    "Angelfish": ("deep", dict(
        # A tall flat disc with a pointed snout; dorsal and anal fins sweep up and back into
        # long streamers, thread-like pelvic fins, and black bars (one through the eye) that
        # run on across the fins.
        height=0.62, thickness=0.35, peak=0.38, nose=0.85, taper=0.9, peduncle=0.22,
        eye_pos=0.13, eye_size=0.22, eye_height=0.3, tail_shape="lunate", tail_length=0.3, tail_spread=1.25,
        scales=0.02, silver=0.45, fin_alpha=0.55, iris_color="#c83a2a",
        dorsal=dict(present=True, shape="pennant", start=0.3, end=0.66, height=0.48, sweep=0.95),
        anal=dict(present=True, shape="pennant", start=0.42, end=0.74, height=0.46, sweep=0.95),
        pelvic=dict(present=True, start=0.3, height=0.32),
        colors=C(0.12, 0.12, 0.9, belly=0.4, pattern="stripes", scale=0.0, offset=0.065, accent="#24222b",
                 fin="#e9e3cf", on_fins=True),
        morphs=[
            ("Silver", 4, {}),
            ("Marble", 1, dict(colors=dict(pattern="blotches", accent_color="#1c1a20", accent2_color="#f0ece0",
                                           pattern_on_fins=False))),
            ("Black", 1, dict(silver=0.15, colors=dict(value=0.12, saturation=0.2, pattern="none",
                                                       fin_color="#1e1d24"))),
        ],
    )),
    "Tuna": ("torpedo", dict(
        # Yellowfin: deep-chested torpedo, metallic blue-black back, silver flanks, a yellow
        # flank stripe, long yellow second dorsal and anal sickles, yellow finlets.
        height=0.3, thickness=0.75, nose=0.5, peduncle=0.07, eye_size=0.14, eye_height=0.3,
        tail_shape="lunate", tail_length=0.28, tail_spread=1.7, finlets=8, silver=0.75, scales=0.012,
        fin_alpha=0.95, gloss=0.6, iris_color="#c0c4c8",
        dorsal=dict(present=True, shape="triangle", start=0.3, end=0.45, height=0.1),
        dorsal2=dict(present=True, shape="falcate", start=0.55, end=0.64, height=0.2, sweep=0.8),
        anal=dict(present=True, shape="falcate", start=0.6, end=0.68, height=0.19, sweep=0.8),
        colors=C(0.6, 0.7, 0.3, belly_level=0.2, pattern="lateral", scale=0.0, fin="#e8c840",
                 accent="#d9c25a", tail_body=True),
    )),
    "Eel": ("slender", dict(
        # European eel: one long, unbroken, slimy body; the dorsal and anal fins run on round the
        # tip of the tail.
        height=0.1, peak=0.3, nose=0.6, taper=1.7, peduncle=0.05, tail_shape="none",
        scales=0.0, gloss=0.7, fin_alpha=0.75, iris_color="#c8b070",
        dorsal=dict(present=True, shape="long", start=0.32, end=1.0, height=0.04, sweep=0.1),
        anal=dict(present=True, shape="long", start=0.55, end=1.0, height=0.035, sweep=0.1),
        pelvic=dict(present=False), pectoral_size=0.07, wave_count=1.4, swim_amplitude=0.75,
        colors=C(0.2, 0.45, 0.42, belly=0.75, fin="#7c8a44"),
    )),
    "Pufferfish": ("classic", dict(
        # Small and slim when calm; see the puff_right / puff_left animations for the spiky ball.
        height=0.38, thickness=0.75, nose=0.7, peak=0.4, tail_shape="rounded", tail_length=0.18,
        tail_spread=0.8, eye_size=0.32, eye_height=0.45, eye_pos=0.14, scales=0.0, gloss=0.35, fin_alpha=0.6,
        iris_color="#58b0a0",
        dorsal=dict(present=True, shape="rounded", start=0.68, end=0.8, height=0.08),
        anal=dict(present=True, shape="rounded", start=0.7, end=0.82, height=0.07),
        pelvic=dict(present=False),
        colors=C(0.13, 0.6, 0.85, pattern="spots", scale=0.6, accent="#4a3a24", fin="#e8d27a"),
    )),
    "Betta": ("classic", dict(
        # Siamese fighting fish: huge, flowing fins that fold and ripple like cloth (billow), a
        # halfmoon tail as tall as the fish is long, and a deep anal fin draping back under it.
        height=0.3, tail_shape="fan", tail_length=0.55, tail_spread=2.6, scales=0.03,
        silver=0.25, fin_alpha=0.85, tail_alpha=0.88, iris_color="#3a2020", billow=1.0,
        swim_amplitude=0.25, mouth=0.03, mouth_height=0.1,
        dorsal=dict(present=True, shape="sail", start=0.55, end=0.93, height=0.26, sweep=1.0),
        anal=dict(present=True, shape="sail", start=0.38, end=0.99, height=0.36, sweep=1.0),
        pelvic=dict(present=True, start=0.34, height=0.2),
        colors=C(0.97, 0.85, 0.72, belly=0.0, fin="#c41a2a"),
        morphs=[
            ("Red", 3, {}),
            ("Royal blue", 3, dict(colors=dict(hue=0.62, saturation=0.85, value=0.8, fin_color="#2c5ad8"))),
            ("Black orchid", 1, dict(colors=dict(hue=0.72, saturation=0.5, value=0.18, fin_color="#3a2a58",
                                                 fin_edge_color="#b8c0e8"))),
            ("Koi", 1, dict(colors=dict(hue=0.1, saturation=0.06, value=0.92, pattern="calico",
                                        accent_color="#e83a20", accent2_color="#1c2a50"))),
            ("Pastel", 1, dict(colors=dict(hue=0.9, saturation=0.15, value=0.95, fin_color="#e8d0e0"))),
        ],
    )),
    "Neon tetra": ("torpedo", dict(
        # Electric blue stripe from the eye to the tail over a red lower rear half; see-through fins.
        # The real tail is nearly clear, which vanishes in pixel art, so it is drawn a bit bigger and
        # more solid than life (its own alpha) to keep the forked tail readable at 16-32 px.
        height=0.28, tail_shape="forked", tail_length=0.3, tail_spread=1.6, eye_size=0.34,
        scales=0.03, fin_alpha=0.3, tail_alpha=0.8, iris_color="#b8c8d8",
        dorsal=dict(present=True, shape="triangle", start=0.42, end=0.54, height=0.08, sweep=0.5),
        anal=dict(present=True, shape="long", start=0.6, end=0.85, height=0.06),
        colors=C(0.58, 0.45, 0.38, belly=0.9, belly_level=-0.15, pattern="neon", accent="#3fe8ff",
                 accent2="#e8283c", fin="#c7d2dc", tail="#9aa8b8"),
    )),
    "Koi": ("torpedo", dict(
        height=0.3, tail_shape="rounded", pectoral_size=0.18, scales=0.03, silver=0.15,
        fin_alpha=0.7, iris_color="#d0c8b8", mouth=0.04, mouth_height=-0.2,
        colors=C(0.1, 0.06, 0.95, belly=0.3, pattern="koi", scale=0.45, accent="#e24a1e", fin="#f2ece4"),
        morphs=[
            ("Kohaku", 4, {}),
            ("Taisho Sanke", 3, dict(colors=dict(accent2_color="#141418"))),
            ("Showa", 2, dict(colors=dict(hue=0.6, saturation=0.15, value=0.12, accent_color="#e24a1e",
                                          accent2_color="#f2f0ea", pattern_scale=0.5))),
            ("Ogon", 1, dict(silver=0.7, colors=dict(hue=0.12, saturation=0.65, value=0.95, pattern="none",
                                                     fin_color="#f0c858"))),
            ("Chagoi", 1, dict(silver=0.2, colors=dict(hue=0.08, saturation=0.45, value=0.5, pattern="none",
                                                       fin_color="#8a6a48"))),
        ],
    )),
    "Discus": ("round", dict(
        height=0.85, nose=0.9, peak=0.5, tail_shape="truncate", tail_length=0.16, scales=0.015,
        fin_alpha=0.75, iris_color="#d02820",
        dorsal=dict(present=True, shape="long", start=0.25, end=0.92, height=0.1),
        anal=dict(present=True, shape="long", start=0.45, end=0.92, height=0.1),
        colors=C(0.04, 0.75, 0.75, belly=0.0, pattern="discus", scale=0.6, accent="#3fc0d1"),
        morphs=[
            ("Red turquoise", 3, {}),
            ("Blue diamond", 2, dict(colors=dict(hue=0.58, saturation=0.65, value=0.8, pattern="none"))),
            ("Pigeon blood", 1, dict(colors=dict(hue=0.08, saturation=0.55, value=0.95, pattern="discus",
                                                 accent_color="#d8402a", pattern_scale=0.3))),
        ],
    )),
    "Mackerel": ("slender", dict(
        # Long and skinny: a small spiny first dorsal, a smaller second dorsal near the tail, a
        # neat forked tail, wavy dark bars over a steel-blue back, and mirror-silver flanks.
        height=0.19, thickness=0.7, nose=0.75, peak=0.38, taper=1.1, peduncle=0.08,
        tail_shape="forked", tail_length=0.18, tail_spread=1.25, eye_size=0.24, eye_height=0.3, finlets=5,
        silver=0.65, scales=0.008, fin_alpha=0.7, gloss=0.6, iris_color="#c8ccd0",
        dorsal=dict(present=True, shape="triangle", start=0.28, end=0.4, height=0.07, sweep=0.5),
        dorsal2=dict(present=True, shape="triangle", start=0.6, end=0.67, height=0.05, sweep=0.5),
        anal=dict(present=True, shape="triangle", start=0.62, end=0.69, height=0.045, sweep=0.5),
        pelvic=dict(present=True, start=0.36, height=0.05),
        colors=C(0.47, 0.5, 0.4, belly_level=-0.05, pattern="mackerel", scale=0.4, accent="#0e1a22",
                 accent2="#2f5e76", fin="#8fa2af"),
    )),
    "Piranha": ("deep", dict(
        # Red-bellied piranha: sparkly silver-grey flanks, red throat and belly, a jaw of teeth.
        height=0.55, nose=0.7, tail_shape="truncate", eye_size=0.22, scales=0.012,
        silver=0.5, fin_alpha=0.8, iris_color="#c83020", mouth=0.06, mouth_height=-0.35, teeth=True,
        colors=C(0.58, 0.12, 0.55, belly=0.0, pattern="red_belly", accent="#c83a28", fin="#3a3f4a"),
    )),
    "Yellow tang": ("deep", dict(
        # A bright yellow disc with a long pointed snout.
        height=0.62, thickness=0.35, peak=0.45, nose=1.35, taper=1.0, peduncle=0.16,
        eye_pos=0.18, eye_height=0.35, eye_size=0.2, tail_shape="truncate", tail_length=0.2, tail_spread=1.0,
        scales=0.008, fin_alpha=0.85, iris_color="#2a2a2a", mouth=0.025, mouth_height=0.0,
        dorsal=dict(present=True, shape="long", start=0.25, end=0.92, height=0.16),
        anal=dict(present=True, shape="long", start=0.45, end=0.92, height=0.15),
        pelvic=dict(present=True, start=0.35, height=0.08),
        colors=C(0.135, 0.95, 0.97, belly=0.1, fin="#f4d018"),
    )),
    "Blue tang": ("deep", dict(
        # Regal blue tang: royal blue with the black "palette" marking and a yellow tail.
        height=0.52, thickness=0.35, peak=0.42, nose=0.8, peduncle=0.14, eye_pos=0.14,
        eye_size=0.22, tail_shape="truncate", tail_length=0.22, tail_spread=1.2, scales=0.008, fin_alpha=0.9,
        iris_color="#202020", mouth=0.025, mouth_height=0.0,
        dorsal=dict(present=True, shape="long", start=0.22, end=0.92, height=0.12),
        anal=dict(present=True, shape="long", start=0.45, end=0.92, height=0.11),
        colors=C(0.6, 0.85, 0.85, belly=0.1, pattern="tang", accent="#0c0c14", fin="#1a3a9a",
                 fin_edge="#0c0c14", tail="#f2c818"),
    )),
    "Rainbow trout": ("torpedo", dict(
        # Olive back, a pink band along the flank, black speckles over the back and fins.
        height=0.25, thickness=0.65, nose=0.7, peduncle=0.12, eye_size=0.18, scales=0.008,
        silver=0.4, fin_alpha=0.85, iris_color="#c8b880", mouth=0.07, mouth_height=-0.15,
        tail_shape="truncate", tail_length=0.2, tail_spread=1.2,
        dorsal=dict(present=True, shape="rounded", start=0.38, end=0.5, height=0.09),
        anal=dict(present=True, shape="triangle", start=0.68, end=0.78, height=0.07),
        pelvic=dict(present=True, start=0.5, height=0.06),
        colors=C(0.22, 0.35, 0.5, belly=0.95, belly_level=-0.3, pattern="trout", scale=0.8, accent="#1c1c18",
                 accent2="#d86a80", fin="#8a8a66", on_fins=True),
    )),
    "Atlantic salmon": ("torpedo", dict(
        # Bright sea-run salmon: mirror-silver with a blue-green back and small black X spots.
        height=0.24, thickness=0.65, nose=0.75, peduncle=0.09, eye_size=0.16, scales=0.008,
        silver=0.75, fin_alpha=0.8, gloss=0.55, iris_color="#c8ccd0", mouth=0.07, mouth_height=-0.15,
        tail_shape="truncate", tail_length=0.2, tail_spread=1.3,
        dorsal=dict(present=True, shape="triangle", start=0.38, end=0.5, height=0.09),
        anal=dict(present=True, shape="triangle", start=0.7, end=0.78, height=0.07),
        pelvic=dict(present=True, start=0.5, height=0.06),
        colors=C(0.5, 0.35, 0.45, belly_level=0.05, pattern="salmon", scale=0.7, accent="#141418",
                 fin="#6a7a80"),
    )),
    "Great barracuda": ("slender", dict(
        # Long silver torpedo with an underslung jaw full of fangs, dark bars and black blotches.
        height=0.15, thickness=0.75, nose=1.4, peak=0.45, taper=1.1, peduncle=0.09,
        flat_head=0.4, eye_size=0.26, eye_pos=0.12, eye_height=0.35, scales=0.008, silver=0.65,
        fin_alpha=0.85, gloss=0.6, iris_color="#d8d0b0", mouth=0.14, mouth_height=-0.3, teeth=True, scariness=0.75,
        tail_shape="forked", tail_length=0.2, tail_spread=1.6,
        dorsal=dict(present=True, shape="triangle", start=0.38, end=0.46, height=0.08),
        dorsal2=dict(present=True, shape="triangle", start=0.72, end=0.78, height=0.06),
        anal=dict(present=True, shape="triangle", start=0.72, end=0.79, height=0.06),
        pelvic=dict(present=True, start=0.45, height=0.05),
        colors=C(0.52, 0.2, 0.55, belly_level=-0.05, pattern="barracuda", scale=0.4, accent="#2a3238",
                 accent2="#0e1014", fin="#5a6870"),
    )),
    "Ocean sunfish": ("round", dict(
        # Mola: a huge upright disc that ends in a scalloped "clavus" instead of a tail, with
        # tall dorsal and anal fins it sculls from side to side, and thick mottled skin.
        height=0.9, thickness=0.32, peak=0.4, nose=0.5, taper=0.8, peduncle=0.55,
        eye_pos=0.16, eye_size=0.1, eye_height=0.25, tail_shape="rounded", tail_length=0.1, tail_spread=0.62,
        scales=0.0, fin_alpha=1.0, gloss=0.2, iris_color="#303030", mouth=0.02, mouth_height=0.0,
        swim_amplitude=0.12, pectoral_size=0.1,
        dorsal=dict(present=True, shape="falcate", start=0.56, end=0.86, height=0.5, sweep=0.25),
        anal=dict(present=True, shape="falcate", start=0.56, end=0.86, height=0.47, sweep=0.25),
        pelvic=dict(present=False),
        colors=C(0.58, 0.08, 0.58, belly=0.6, pattern="blotches", scale=0.15, accent="#7e888f", accent2="#6a737c",
                 fin="#5f6970", tail_body=True),
    )),
}


# Other kinds: name -> (kind, variant, overrides). The kind's generator (kinds.py) builds a
# random animal of that kind and variant; the preset only pins what makes the species itself.
OTHERS: dict[str, tuple[str, str, dict]] = {
    # ------------------------------------------------------------------ sharks
    "Tiger shark": ("shark", "requiem", dict(
        nose=0.95, height=0.28, tail_length=0.34, teeth=True, scariness=0.7, scars=0.4,
        colors=C(0.1, 0.25, 0.58, pattern="tiger", scale=0.3, accent="#3b352c", fin="#857a68", tail_body=True),
    )),
    "Great white shark": ("shark", "requiem", dict(
        # Stout torpedo with a conical snout, a big triangular dorsal set back over the
        # pectorals' trailing edge, huge pectorals, a sharp grey-over-white line and a mouthful of teeth.
        height=0.31, nose=0.95, flat_head=0.25, peak=0.4, peduncle=0.09, tail_spread=1.8, tail_length=0.3,
        tail_shape="lunate",
        dorsal=dict(start=0.39, end=0.55, height=0.24, shape="triangle", sweep=0.55),
        pectoral_size=0.34, pectoral_droop=0.75, mouth=0.13, mouth_height=-0.62, teeth=True, scariness=0.9,
        scars=0.7,
        colors=C(0.57, 0.14, 0.4, belly_level=-0.12, fin="#5f6770", tail_body=True),
    )),
    "Hammerhead shark": ("shark", "hammerhead", dict(
        hammer=0.14, height=0.25, eye_size=0.2, teeth=True, scariness=0.35,
        dorsal=dict(height=0.3, shape="falcate", sweep=0.6),
        colors=C(0.12, 0.15, 0.6, fin="#8f8a7c", tail_body=True),
    )),
    "Whale shark": ("shark", "requiem", dict(
        # Huge, broad and flat-headed with a wide mouth right at the front, ridges along
        # the flanks, a chequerboard of pale spots and lines, and a big crescent tail.
        height=0.26, thickness=1.2, nose=0.4, peak=0.32, flat_head=0.95, back_arch=0.6, taper=1.1,
        eye_pos=0.07, eye_size=0.09, eye_height=-0.15, mouth=0.06, mouth_height=-0.15, teeth=False, scariness=0.0,
        tail_length=0.38, tail_spread=1.9, pectoral_size=0.32, swim_amplitude=0.3, tail_shape="lunate",
        dorsal=dict(start=0.5, end=0.66, height=0.15, shape="triangle", sweep=0.6),
        dorsal2=dict(start=0.8, height=0.05), scars=0.5,
        colors=C(0.6, 0.4, 0.4, belly=0.9, belly_level=-0.3, pattern="whale_shark", scale=0.4, offset=0.3,
                 accent="#eef3f6", accent2="#a7bccc", fin="#4a5a70", tail_body=True),
    )),
    "Mako shark": ("shark", "requiem", dict(
        # Slim, with a long needle-pointed snout and a metallic indigo back.
        height=0.22, nose=1.75, flat_head=0.1, peduncle=0.08, tail_spread=1.8, eye_size=0.2, teeth=True,
        scariness=0.8,
        tail_shape="lunate",
        gloss=0.45, colors=C(0.62, 0.65, 0.5, fin="#3a4f9a", tail_body=True),
    )),
    "Blacktip reef shark": ("shark", "requiem", dict(
        height=0.25, nose=1.3, teeth=True, scariness=0.4,
        dorsal=dict(height=0.19, shape="triangle"),
        colors=C(0.1, 0.15, 0.72, fin_tips=0.45, accent="#0b0b0e", fin="#a39d8f", tail_body=True),
    )),
    "Blue shark": ("shark", "requiem", dict(
        # Slender and graceful, with very long scythe pectorals, a long snout and a vivid blue back.
        height=0.2, nose=1.6, flat_head=0.1, peduncle=0.09, tail_spread=1.6, tail_length=0.38, eye_size=0.24,
        pectoral_size=0.29, pectoral_droop=0.85, gloss=0.45, teeth=False, scariness=0.0,
        dorsal=dict(start=0.42, end=0.54, height=0.15, shape="triangle", sweep=0.6),
        colors=C(0.62, 0.72, 0.62, belly_level=-0.1, fin="#3a5ab8", tail_body=True),
    )),
    "Nurse shark": ("shark", "requiem", dict(
        # Sluggish bottom-dweller: broad flat head, tiny eyes, two rounded dorsals set far back
        # and a long tail with hardly any lower lobe.
        height=0.22, thickness=1.05, nose=0.45, flat_head=0.8, peak=0.3, back_arch=0.55, eye_size=0.1,
        tail_length=0.42, tail_spread=1.1, pectoral_size=0.24, pectoral_droop=0.7, gloss=0.2,
        teeth=False, scariness=0.0, mouth=0.04, mouth_height=-0.5,
        dorsal=dict(start=0.52, end=0.64, height=0.12, shape="rounded", sweep=0.3),
        dorsal2=dict(start=0.7, height=0.09),
        colors=C(0.08, 0.4, 0.48, belly=0.5, belly_level=-0.4, fin="#7a6448", tail_body=True),
    )),
    # ------------------------------------------------------------------ whales & dolphins
    "Orca": ("whale", "whale", dict(
        # The dorsal fin is set per individual: a tall straight male blade or a curved female sickle.
        height=0.3, nose=0.8, flat_head=0.0, eye_pos=0.1, eye_height=0.2, pectoral_size=0.2, pectoral_droop=0.6,
        gloss=0.75, scars=0.25,
        colors=C(0.62, 0.1, 0.06, belly=0.0, pattern="orca", accent="#f2f2ee", fin="#121318",
                 accent2="#7d828c"),
    )),
    "Bottlenose dolphin": ("whale", "dolphin", dict(
        beak=0.12, eye_pos=0.19, height=0.21, nose=0.6, melon=0.012, gloss=0.7, scars=0.25,
        dorsal=dict(present=True, shape="falcate", start=0.42, end=0.56, height=0.13, sweep=0.75),
        colors=C(0.58, 0.12, 0.55, belly=0.8),
    )),
    "Common dolphin": ("whale", "dolphin", dict(
        # The tan-and-grey hourglass on each flank under a dark cape.
        beak=0.09, eye_pos=0.16, height=0.2, nose=0.55, gloss=0.7,
        dorsal=dict(present=True, shape="falcate", start=0.42, end=0.55, height=0.14, sweep=0.8),
        colors=C(0.62, 0.3, 0.2, belly_level=-0.4, pattern="dolphin_cape", scale=0.6, accent="#c8b07a",
                 accent2="#8a9098", fin="#20262f"),
    )),
    "Beluga": ("whale", "whale", dict(
        # Big round echolocation melon over a short beak, a flexible neck, no dorsal fin.
        height=0.3, nose=0.5, peak=0.5, beak=0.03, melon=0.085, neck=0.3, flat_head=0.0, back_arch=0.45,
        eye_pos=0.15, eye_height=0.0, dorsal=dict(present=False), tail_length=0.18,
        pectoral_size=0.15, pectoral_shape="lobe", mouth=0.05, mouth_height=-0.3, gloss=0.5,
        scars=0.35,
        colors=C(0.12, 0.06, 0.93, belly=0.4),
    )),
    "Humpback whale": ("whale", "whale", dict(
        # Long, thin flat head studded with knobs, enormous white flippers, throat grooves,
        # barnacles, dark above with a mottled white belly.
        height=0.26, nose=0.75, peak=0.42, flat_head=0.75, back_arch=0.55, pectoral_size=0.55,
        pectoral_droop=0.42, eye_pos=0.2, eye_height=-0.35, tail_spread=1.7, pleats=True,
        knobs=1.0, scars=0.4, gloss=0.45,
        dorsal=dict(present=True, shape="rounded", start=0.6, end=0.66, height=0.04, sweep=0.5),
        colors=C(0.6, 0.12, 0.25, belly=0.9, belly_level=-0.5, pattern="humpback", accent="#d8dce0",
                 fin="#dfe2e6"),
    )),
    "Sperm whale": ("whale", "whale", dict(
        # A huge square head a third of the body long, a narrow underslung jaw, wrinkled
        # dark-grey skin scarred by squid, a low hump instead of a dorsal fin, small paddle flippers.
        height=0.25, nose=0.12, peak=0.3, flat_head=1.0, back_arch=0.55, taper=1.4, eye_pos=0.3,
        eye_height=-0.4, pectoral_size=0.1, mouth=0.22, mouth_height=-0.85, gloss=0.35,
        scars=1.0, tail_spread=1.5,
        dorsal=dict(present=True, shape="rounded", start=0.62, end=0.72, height=0.035, sweep=0.4),
        colors=C(0.08, 0.12, 0.25, belly=0.15),
    )),
    # ------------------------------------------------------------------ rays
    "Cownose ray": ("ray", "eagle", dict(
        # Blunt, squared-off head with the two rounded lobes of the "cow nose" under it.
        nose=0.3, thickness=1.8, lobes="round", lobe_length=0.045,
        wings=dict(start=0.1, end=0.36, span=0.4, sweep=1.25, flap=0.85, wave=0.2),
        dorsal=dict(present=True, shape="triangle", start=0.45, end=0.5, height=0.04, sweep=0.6),
        colors=C(0.08, 0.5, 0.5),
    )),
    "Manta ray": ("ray", "eagle", dict(
        # Huge swept-back wings, a broad squared front with the two cephalic fins either side
        # of a wide mouth, a short tail.
        nose=0.3, thickness=2.4, height=0.2, peak=0.3, taper=3.0, peduncle=0.1, lobes="horns",
        lobe_length=0.11, eye_pos=0.07, eye_height=0.1, eye_size=0.16, mouth=0.04,
        mouth_height=-0.15, scars=0.3,
        wings=dict(start=0.05, end=0.6, span=0.85, sweep=1.15, flap=0.75, wave=0.15),
        colors=C(0.6, 0.15, 0.1, belly=1.0, pattern="chevron", accent="#e8ecec", belly_pattern=True),
        morphs=[
            ("Chevron", 6, {}),
            ("Black", 1, dict(colors=dict(belly_lightness=0.0, pattern="none", value=0.07))),
        ],
    )),
    "Spotted eagle ray": ("ray", "eagle", dict(
        # A rounded duck-bill head jutting out in front of the wings, white spots on an inky
        # back, and a whip tail longer than the body.
        nose=0.55, thickness=1.5, lobes="none", lobe_length=0.0,
        wings=dict(start=0.06, end=0.27, span=0.34, sweep=1.25, flap=0.85, wave=0.2),
        colors=C(0.62, 0.3, 0.16, pattern="spots", scale=1.0, offset=0.1, accent="#eef0f0"),
    )),
    "Stingray": ("ray", "stingray", dict(
        # Southern stingray: a wide diamond disc that takes in the head, rippling as it swims,
        # with eyes on top and a long whip tail.
        nose=0.85,
        wings=dict(start=0.0, end=0.5, span=0.28, sweep=0.45, shape="diamond", flap=0.16, wave=1.3),
        colors=C(0.09, 0.3, 0.42),
    )),
    "Bluespotted stingray": ("ray", "stingray", dict(
        # Bluespotted ribbontail: a small oval disc, rounded snout, electric blue spots.
        nose=0.6,
        wings=dict(start=0.0, end=0.48, span=0.2, sweep=0.45, shape="round", flap=0.16, wave=1.3),
        colors=C(0.11, 0.5, 0.68, pattern="spots", scale=0.4, offset=0.6, accent="#2f8ae0"),
    )),
    # ------------------------------------------------------------------ squid & cuttlefish
    "Squid": ("squid", "squid", dict(
        height=0.22, arms=dict(length=0.45, tentacle_length=0.75),
        colors=C(0.97, 0.35, 0.85, belly=0.5, pattern="spots", scale=1.0, accent="#b0384a"),
    )),
    "Giant squid": ("squid", "squid", dict(
        height=0.18, wings=dict(end=0.22, span=0.12, dihedral=0.35),
        arms=dict(length=0.55, tentacle_length=1.0, thickness=0.028),
        colors=C(0.0, 0.7, 0.55, belly=0.3),
    )),
    "Cuttlefish": ("squid", "cuttlefish", dict(
        # Iridescent: rose-violet skin with teal and gold colour waves rolling down it.
        colors=C(0.86, 0.45, 0.82, belly=0.5, pattern="shimmer", scale=0.4,
                 accent="#4ce6d2", accent2="#ffc35a"),
        morphs=[
            ("Rose", 2, {}),
            ("Zebra", 1, dict(colors=dict(hue=0.08, saturation=0.3, value=0.75, pattern="tiger",
                                          accent_color="#2a201c"))),
        ],
    )),
    # ------------------------------------------------------------------ octopus
    "Octopus": ("octopus", "octopus", dict(
        height=0.7, arms=dict(length=1.3, thickness=0.05, spread=0.75, curl=0.6),
        colors=C(0.03, 0.6, 0.6, belly=0.45, pattern="blotches", scale=0.7, accent="#7a2a1c",
                 accent2="#e8b090"),
    )),
    "Blue-ringed octopus": ("octopus", "octopus", dict(
        height=0.65, arms=dict(length=1.1, thickness=0.045, spread=0.7, curl=0.55),
        colors=C(0.12, 0.6, 0.82, belly=0.3, pattern="rings", scale=0.4, accent="#2e6bff",
                 accent_edge="#141010"),
    )),
    "Dumbo octopus": ("octopus", "dumbo", dict(
        colors=C(0.97, 0.35, 0.88, belly=0.4),
    )),
    # ------------------------------------------------------------------ urchins
    "Purple sea urchin": ("urchin", "short", dict(
        # Strongylocentrotus purpuratus: a dense coat of stout purple spines.
        spines=dict(count=100, length=0.38, sway=0.1, taper=0.8),
        colors=C(0.8, 0.55, 0.38, belly=0.0, fin="#8a4bb0"),
    )),
    "Long-spined urchin": ("urchin", "long", dict(
        # Diadema: needle-thin black spines longer than the shell is wide, iridescent blue spots
        # on the shell and an orange-rimmed anal cone on top.
        spines=dict(count=70, length=0.95, sway=0.12, taper=0.95),
        colors=C(0.7, 0.2, 0.12, belly=0.0, fin="#25222e", pattern="spots", scale=0.35, accent="#3d8be0",
                 accent2="#e07a2a"),
    )),
    "Red pencil urchin": ("urchin", "short", dict(
        # Heterocidaris / Heterocentrotus: a few thick, blunt, red pencil spines.
        spines=dict(count=34, length=0.5, sway=0.05, taper=0.2),
        colors=C(0.01, 0.6, 0.4, belly=0.0, fin="#d0533a"),
    )),
    "Green sea urchin": ("urchin", "short", dict(
        # Strongylocentrotus droebachiensis: short, fine, crowded green spines.
        spines=dict(count=130, length=0.26, sway=0.1, taper=0.75),
        colors=C(0.25, 0.5, 0.4, belly=0.0, fin="#8db04a"),
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


def morphs_of(species: str) -> tuple[str, ...]:
    """Names of a species' colour varieties (empty if it only comes in one)."""
    return tuple(m[0] for m in SPECIES[species][2].get("morphs", ()))


def genome_for(seed: int, species: str | None = None, kind: str | None = None) -> FishGenome:
    """The animal for command-line style options: a species, a kind, both, or neither (a random fish)."""
    if species not in (None, RANDOM):
        if kind is not None and kind != kind_of(species):
            raise ValueError(f"{species} is a {kind_of(species)}, not a {kind}")
        return make_genome(species, seed)
    return random_genome(kind or "fish", seed)


def _orca(g: FishGenome, r: random.Random):
    """Orcas are male or female at random, and the dorsal fins could hardly be more different."""
    if r.random() < 0.5:  # male: a tall, straight, upright triangle, and a bigger animal
        g.dorsal = FinGenes(True, 0.4, 0.5, r.uniform(0.3, 0.36), "triangle", r.uniform(0.0, 0.1))
        g.length_cm = SIZES["Orca male"].draw(r)
    else:  # female (and young): a shorter sickle curving back
        g.dorsal = FinGenes(True, 0.4, 0.52, r.uniform(0.14, 0.17), "falcate", r.uniform(0.75, 0.9))


def _pufferfish(g: FishGenome, r: random.Random):
    """Small and slim until something upsets it."""
    g.puff = r.uniform(0.78, 0.88)
    g.puff_spines = r.randint(70, 95)


def _sperm_whale(g: FishGenome, r: random.Random):
    """Bulls are half as long again as cows."""
    if r.random() < 0.5:
        g.length_cm = SIZES["Sperm whale male"].draw(r)


# Per-species finishing touches that need more than fixed overrides.
SPECIAL = {"Orca": _orca, "Pufferfish": _pufferfish, "Sperm whale": _sperm_whale}


def _apply(g: FishGenome, overrides: dict):
    for key, value in overrides.items():
        if key == "morphs":
            continue
        if isinstance(value, dict):
            sub = getattr(g, key)
            for k, v in value.items():
                setattr(sub, k, v)
        else:
            setattr(g, key, value)


def make_genome(species: str, seed: int, archetype: str | None = None, kind: str = "fish",
                morph: str | None = None) -> FishGenome:
    """An animal for ``seed`` shaped into ``species``.

    "Random" gives a random animal of ``kind``; ``archetype`` (fish body type) only
    applies to a random fish. Presets fix their own kind and body type. A species with
    colour varieties picks one from the seed unless ``morph`` names it.
    """
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}; choose from {', '.join(SPECIES)}")
    preset_kind, preset_shape, overrides = SPECIES[species]
    if species == RANDOM:
        return random_genome(kind, seed, archetype=archetype)
    if preset_kind == "fish":
        g = random_genome("fish", seed, archetype=preset_shape)
        g.silver = 0.0  # matte unless the species is silvery
    else:
        g = random_genome(preset_kind, seed, variant=preset_shape)
    _apply(g, overrides)
    morphs = overrides.get("morphs")
    if morphs:
        names = [m[0] for m in morphs]
        if morph is None:
            r = random.Random(f"morph:{species}:{seed}")
            morph = r.choices(names, weights=[m[1] for m in morphs])[0]
        elif morph not in names:
            raise ValueError(f"{species} comes as {', '.join(names)}, not {morph!r}")
        _apply(g, morphs[names.index(morph)][2])
        g.morph = morph
    individualise(g, species)
    g.length_cm = SIZES[species].draw(random.Random(f"size:{species}:{seed}"))
    if species in SPECIAL:
        SPECIAL[species](g, random.Random(f"special:{species}:{seed}"))
    return g
