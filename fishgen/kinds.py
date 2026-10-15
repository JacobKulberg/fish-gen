"""Animal kinds: decide what an animal is first, then draw every gene to suit it.

``random_genome("shark", seed)`` is always shark-shaped (shark tail, gill slits,
stiff fins...) and only its proportions, colours and markings vary with the seed.
Fish use ``FishGenome.random``; every other kind has its own generator here.

Some kinds have variants (a shark is a requiem shark or a hammerhead, a ray is
an eagle-type flapper or a round stingray, ...). A random animal picks one;
species presets ask for theirs.
"""

from __future__ import annotations

import colorsys
import random

from .genome import KINDS, ArmGenes, ColorGenes, FinGenes, FishGenome, SpineGenes, WingGenes

LABELS = {
    "fish": "Fish",
    "shark": "Shark",
    "whale": "Whale & dolphin",
    "ray": "Ray",
    "squid": "Squid & cuttlefish",
    "octopus": "Octopus",
    "urchin": "Sea urchin",
}
VARIANTS = {
    "shark": ("requiem", "hammerhead"),
    "whale": ("dolphin", "whale"),
    "ray": ("eagle", "stingray"),
    "squid": ("squid", "cuttlefish"),
    "octopus": ("octopus", "dumbo"),
    "urchin": ("short", "long"),
}


def kind_label(kind: str) -> str:
    return LABELS[kind]


def kind_from_label(label: str) -> str:
    return next(k for k, v in LABELS.items() if v == label)


def _hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _colors(hue, saturation, value, belly, pattern="none", scale=0.5, offset=0.0,
            accent=None, fin=None, fin_tips=0.0) -> ColorGenes:
    return ColorGenes(hue=hue % 1.0, saturation=saturation, value=value, fin_hue_shift=0.0, accent_hue_shift=0.0,
                      belly_lightness=belly, pattern=pattern, pattern_scale=scale, pattern_offset=offset,
                      fin_color=fin, accent_color=accent, fin_tips=fin_tips)


def _off() -> FinGenes:
    return FinGenes(False, 0.5, 0.6, 0.05, "triangle", 0.5)


def _base(seed: int, kind: str, **genes) -> FishGenome:
    """A genome with every gene a non-fish kind doesn't set switched off or neutral."""
    defaults = dict(
        seed=seed, kind=kind, height=0.3, thickness=0.8, peak=0.4, nose=0.6, taper=1.0, peduncle=0.15,
        back_arch=0.5, eye_pos=0.1, eye_height=0.3, eye_size=0.2, tail_shape="none", tail_length=0.2,
        tail_spread=1.0, dorsal=_off(), anal=_off(), pelvic=_off(), pectoral_size=0.0, swim_amplitude=0.5,
        wave_count=0.8, gill_slits=0, dorsal2=_off(), pectoral_flap=1.0,
    )
    defaults.update(genes)
    return FishGenome(**defaults)


def _variant(r: random.Random, kind: str, variant: str | None, weights) -> str:
    drawn = r.choices(VARIANTS[kind], weights=weights)[0]  # always drawn: keeps the RNG stream stable
    if variant is None:
        return drawn
    if variant not in VARIANTS[kind]:
        raise ValueError(f"unknown {kind} variant {variant!r}; choose from {', '.join(VARIANTS[kind])}")
    return variant


# ---------------------------------------------------------------------------- sharks
def _shark(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "shark", variant, [5, 1])
    U = r.uniform
    hammer = U(0.11, 0.15) if v == "hammerhead" else 0.0
    d_start = U(0.3, 0.36)
    tall = U(0.14, 0.2) if v == "requiem" else U(0.2, 0.26)
    blue = r.random() < 0.25
    hue = U(0.58, 0.64) if blue else r.choice([U(0.55, 0.62), U(0.08, 0.13)])
    sat = U(0.4, 0.65) if blue else U(0.08, 0.3)
    val = U(0.45, 0.72)
    pattern = r.choices(["none", "tiger", "spots"], weights=[6, 1.5, 1])[0]
    accent = {"tiger": _hex(hue, sat + 0.1, val * 0.45), "spots": "#e6ebef", "none": None}[pattern]
    tips = r.random() < 0.2
    if tips:
        accent = accent or "#16161a"
    return _base(
        seed, "shark", height=U(0.23, 0.32), thickness=U(0.68, 0.85), peak=U(0.34, 0.42),
        nose=1.0 if hammer else U(0.4, 1.0), taper=U(1.0, 1.25), peduncle=U(0.1, 0.16), back_arch=U(0.52, 0.58),
        eye_pos=U(0.08, 0.12), eye_height=U(0.25, 0.4), eye_size=U(0.12, 0.2), eye_style="dark",
        gill_slits=5, hammer=hammer,
        tail_shape="heterocercal", tail_length=U(0.26, 0.36), tail_spread=U(1.4, 1.9),
        dorsal=FinGenes(True, d_start, d_start + U(0.13, 0.17), tall, "triangle", U(0.5, 0.8)),
        dorsal2=FinGenes(True, U(0.72, 0.76), 0.81, U(0.035, 0.06), "triangle", 0.5),
        anal=FinGenes(True, U(0.75, 0.78), 0.84, U(0.03, 0.05), "triangle", 0.5),
        pelvic=FinGenes(True, U(0.56, 0.6), 0.65, U(0.05, 0.07), "triangle", 0.5),
        pectoral_size=U(0.18, 0.26), pectoral_flap=U(0.1, 0.2),
        swim_amplitude=U(0.4, 0.5), wave_count=U(0.65, 0.8),
        colors=_colors(hue, sat, val, 1.0, pattern, U(0.2, 0.9), U(0.0, 1.0), accent, None, 0.3 if tips else 0.0),
    )


# ---------------------------------------------------------------------------- whales & dolphins
def _whale(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "whale", variant, [1, 1])
    U = r.uniform
    dolphin = v == "dolphin"
    if r.random() < 0.9:
        d_start = U(0.38, 0.6) if not dolphin else U(0.4, 0.5)
        dorsal = FinGenes(True, d_start, d_start + U(0.06, 0.14), U(0.08, 0.16) if dolphin else U(0.04, 0.3),
                          r.choice(["triangle", "triangle", "rounded", "sail"]), U(0.15, 0.9))
    else:
        dorsal = _off()
    long_flippers = not dolphin and r.random() < 0.25
    white = r.random() < 0.12
    hue = U(0.55, 0.65)
    val = U(0.9, 0.95) if white else U(0.12, 0.65)
    pattern = "none" if white else r.choices(["none", "orca", "lateral", "saddle", "spots", "gradient"],
                                               weights=[3, 1 if val < 0.3 else 0, 1.5, 1.5, 1, 1])[0]
    accent = {"orca": "#f2f2ee", "lateral": _hex(U(0.1, 0.15), 0.35, 0.85), "saddle": _hex(hue, 0.2, val * 0.5),
              "spots": _hex(hue, 0.1, min(1.0, val + 0.35)), "gradient": None, "none": None}[pattern]
    # Dark animals keep dark flippers, though some (humpback-like) have pale ones.
    fin = (_hex(hue, 0.05, 0.85) if r.random() < 0.25 else _hex(hue, 0.1, val * 0.85)) if val < 0.3 else None
    return _base(
        seed, "whale", height=U(0.22, 0.26) if dolphin else U(0.26, 0.32), thickness=U(0.75, 0.85),
        peak=U(0.36, 0.42), nose=U(0.5, 0.65) if dolphin else U(0.7, 1.0), beak=U(0.05, 0.09) if dolphin else 0.0,
        taper=U(1.2, 1.4), peduncle=U(0.1, 0.14), back_arch=0.5,
        eye_pos=U(0.1, 0.14), eye_height=U(-0.1, 0.2), eye_size=U(0.12, 0.16), eye_style="dark",
        swim_vertical=True, tail_shape="flukes", tail_length=U(0.18, 0.24), tail_spread=U(1.1, 1.6),
        dorsal=dorsal, pectoral_size=U(0.3, 0.38) if long_flippers else U(0.12, 0.2), pectoral_flap=0.15,
        swim_amplitude=U(0.4, 0.5), wave_count=U(0.5, 0.6),
        colors=_colors(hue, U(0.05, 0.3), val, U(0.4, 0.6) if white else U(0.6, 1.0), pattern, U(0.4, 0.8), 0.0,
                       accent, fin),
    )


# ---------------------------------------------------------------------------- rays
def _ray(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "ray", variant, [3, 2])
    U = r.uniform
    if v == "eagle":
        wings = WingGenes(True, U(0.03, 0.09), U(0.38, 0.44), U(0.38, 0.62), U(0.6, 0.8), "pointed",
                          U(0.75, 0.9), U(0.1, 0.25))
        dorsal = FinGenes(True, 0.45, 0.5, 0.04, "triangle", 0.6) if r.random() < 0.5 else _off()
    else:
        wings = WingGenes(True, U(0.01, 0.04), U(0.46, 0.54), U(0.18, 0.24), U(0.4, 0.5), "round",
                          U(0.25, 0.35), U(1.1, 1.5))
        dorsal = _off()
    hue = r.choice([U(0.06, 0.12), U(0.58, 0.64)])
    sat, val = U(0.15, 0.55), U(0.2, 0.68)
    pattern = r.choices(["none", "spots"], weights=[3, 1.5])[0]
    accent = ("#eef0f0" if val < 0.4 else r.choice(["#2f8ae0", _hex(hue, sat + 0.2, val * 0.4)])) \
        if pattern == "spots" else None
    return _base(
        seed, "ray", height=U(0.08, 0.1), thickness=U(2.2, 2.6), peak=U(0.2, 0.24), nose=U(0.5, 1.0),
        taper=U(2.0, 3.0), peduncle=U(0.3, 0.38), back_arch=0.6,
        eye_pos=U(0.08, 0.12), eye_height=0.75, eye_size=0.25, eye_style="dark",
        wings=wings, dorsal=dorsal, swim_amplitude=U(0.06, 0.1), wave_count=U(1.2, 1.5),  # stiff body: rays swim with their wings
        camera_tilt=U(0.85, 0.95), turn_lag=0.5,  # near top-down, so both wings flap alike
        colors=_colors(hue, sat, val, 1.0, pattern, U(0.3, 0.7), U(0.0, 1.0), accent),
    )


# ---------------------------------------------------------------------------- squid & cuttlefish
def _squid(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "squid", variant, [7, 3])
    U = r.uniform
    hue = r.choice([U(-0.05, 0.02), U(0.05, 0.1)])
    sat, val = U(0.3, 0.7), U(0.55, 0.88)
    if v == "squid":
        shape = dict(height=U(0.18, 0.24), thickness=0.9, nose=U(0.35, 0.45), peak=0.4, eye_pos=0.76)
        wings = WingGenes(True, 0.0, U(0.2, 0.3), U(0.08, 0.12), 0.5, "pointed", 0.4, 0.6)
        arms = ArmGenes(8, U(0.4, 0.6), U(0.028, 0.034), 2, U(0.7, 1.1), U(0.4, 0.55), U(0.25, 0.35))
        pattern = r.choices(["none", "spots"], weights=[1, 2])[0]
    else:
        shape = dict(height=U(0.27, 0.32), thickness=U(1.1, 1.2), nose=U(0.8, 0.9), peak=0.4, eye_pos=0.74)
        wings = WingGenes(True, 0.03, U(0.65, 0.72), U(0.04, 0.05), 0.5, "round", 0.7, 2.0)
        arms = ArmGenes(8, U(0.2, 0.26), 0.03, 0, 1.0, 0.15, 0.3)
        pattern = r.choices(["none", "tiger", "spots"], weights=[1, 2, 1])[0]
    accent = _hex(hue, sat + 0.2, val * 0.55) if pattern != "none" else None
    return _base(
        seed, "squid", taper=1.0, peduncle=0.55, back_arch=0.5, eye_height=0.25, eye_size=0.3,
        swim_amplitude=0.2, wave_count=0.8, pulse=U(0.1, 0.14), wings=wings, arms=arms,
        colors=_colors(hue, sat, val, U(0.3, 0.5), pattern, U(0.2, 1.0), U(0.0, 1.0), accent), **shape,
    )


# ---------------------------------------------------------------------------- octopus
def _octopus(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "octopus", variant, [5, 1])
    U = r.uniform
    hue = r.choice([U(0.0, 0.1), U(0.94, 0.99), U(0.1, 0.14)])
    sat, val = U(0.35, 0.65), U(0.55, 0.88)
    if v == "octopus":
        shape = dict(height=U(0.6, 0.75), thickness=0.85, peak=0.32, taper=1.6, peduncle=0.32, eye_pos=0.68)
        arms = ArmGenes(8, U(0.9, 1.4), U(0.04, 0.055), 0, 1.0, U(0.6, 0.8), U(0.45, 0.65))
        wings, pulse = WingGenes(), 0.08
        pattern = r.choices(["none", "spots", "rings"], weights=[2, 2, 1])[0]
    else:
        shape = dict(height=U(0.75, 0.85), thickness=0.95, peak=0.38, taper=1.4, peduncle=0.4, eye_pos=0.62)
        arms = ArmGenes(8, U(0.5, 0.7), 0.06, 0, 1.0, U(0.9, 1.1), 0.3)
        wings, pulse = WingGenes(True, 0.2, 0.36, U(0.17, 0.23), 0.5, "round", 0.7, 0.0), 0.04
        pattern = "none"
    accent = {"rings": "#2e6bff", "spots": _hex(hue, sat + 0.2, val * 0.55), "none": None}[pattern]
    return _base(
        seed, "octopus", nose=1.0, back_arch=0.5, eye_height=0.25, eye_size=0.22,
        swim_amplitude=0.2, wave_count=0.8, pulse=pulse, wings=wings, arms=arms,
        colors=_colors(hue, sat, val, U(0.3, 0.45), pattern, U(0.4, 0.8), U(0.0, 1.0), accent), **shape,
    )


# ---------------------------------------------------------------------------- urchins
def _urchin(seed: int, r: random.Random, variant: str | None) -> FishGenome:
    v = _variant(r, "urchin", variant, [4, 1])
    U = r.uniform
    hue, sat, val = r.choice([(U(0.77, 0.85), U(0.4, 0.6), U(0.35, 0.5)), (U(0.0, 0.04), U(0.5, 0.65), U(0.4, 0.55)),
                              (U(0.22, 0.3), U(0.4, 0.55), U(0.4, 0.5)), (U(0.65, 0.75), U(0.1, 0.25), U(0.12, 0.18))])
    if v == "long":
        spines = SpineGenes(r.randint(40, 60), U(0.75, 1.0), U(0.08, 0.14))
    else:
        spines = SpineGenes(r.randint(25, 120), 0.0, U(0.06, 0.15))
        spines.length = 0.2 + 0.3 * (1 - (spines.count - 25) / 95) + U(-0.03, 0.03)  # fewer spines, longer
    spine = _hex(hue, sat + 0.05, min(1.0, val * 1.6 + 0.05))
    return _base(
        seed, "urchin", profile="sphere", height=U(0.7, 0.8), thickness=1.0 / 0.75, back_arch=0.55,
        eye_style="none", swim_amplitude=0.0, wave_count=0.0, turn_lag=0.0, camera_tilt=0.25, spines=spines,
        colors=_colors(hue, sat, val, 0.0, fin=spine),
    )


_GENERATORS = {"shark": _shark, "whale": _whale, "ray": _ray, "squid": _squid, "octopus": _octopus,
               "urchin": _urchin}


def _jit_hex(color: str | None, r: random.Random) -> str | None:
    """Nudge a #rrggbb colour's hue, saturation and brightness a little."""
    if not color:
        return color
    h, s_, v = colorsys.rgb_to_hsv(*(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    return _hex(h + r.uniform(-0.03, 0.03), s_ * r.uniform(0.8, 1.2), v * r.uniform(0.88, 1.12))


def individualise(g: FishGenome, salt: str) -> FishGenome:
    """Seeded individual variation for a species preset: proportions, fin/wing/arm sizes and
    colours shift within a natural range, so two animals of a species never look the same
    while keeping everything that makes them that species (their kind's parts, markings).
    ``g.wings``/``g.arms``/``g.spines`` only vary when the animal has them."""
    r = random.Random(f"individual:{salt}:{g.seed}")
    U = r.uniform

    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    g.height *= U(0.86, 1.14)
    g.thickness *= U(0.9, 1.1)
    g.nose = clip(g.nose + U(-0.15, 0.15), 0.3, 1.0)
    g.peduncle *= U(0.88, 1.12)
    g.eye_size *= U(0.85, 1.15)
    if g.kind in ("fish", "shark", "whale"):
        g.tail_length *= U(0.86, 1.14)
        g.tail_spread *= U(0.86, 1.14)
        g.pectoral_size *= U(0.82, 1.2)
        g.beak *= U(0.75, 1.25)
        g.hammer *= U(0.85, 1.15)
        if g.dorsal.present:
            g.dorsal.height *= U(0.78, 1.22)
            shift = U(-0.03, 0.03)
            g.dorsal.start += shift
            g.dorsal.end += shift
            g.dorsal.sweep = clip(g.dorsal.sweep + U(-0.15, 0.15), 0.0, 1.0)
    if g.wings.present:
        g.wings.span *= U(0.85, 1.15)
        g.wings.sweep = clip(g.wings.sweep + U(-0.07, 0.07), 0.0, 1.0)
        g.wings.flap *= U(0.9, 1.1)
    if g.arms.count:
        g.arms.length *= U(0.8, 1.2)
        g.arms.spread *= U(0.8, 1.2)
        g.arms.curl = clip(g.arms.curl + U(-0.15, 0.15), 0.05, 1.0)
        g.arms.thickness *= U(0.9, 1.12)
        g.arms.tentacle_length *= U(0.85, 1.2)
        g.pulse *= U(0.8, 1.2)
    if g.spines.count:
        g.spines.count = max(8, round(g.spines.count * U(0.8, 1.25)))
        g.spines.length *= U(0.85, 1.15)
    c = g.colors
    c.hue = (c.hue + U(-0.035, 0.035)) % 1.0
    c.saturation = clip(c.saturation * U(0.75, 1.3), 0.0, 1.0)
    c.value = clip(c.value * U(0.85, 1.15) + U(-0.03, 0.07), 0.08, 1.0)  # additive part shows on dark animals
    if c.belly_lightness > 0.2:
        c.belly_lightness = clip(c.belly_lightness + U(-0.15, 0.15), 0.2, 1.0)
    c.pattern_scale = clip(c.pattern_scale + U(-0.3, 0.3), 0.0, 1.0)
    c.pattern_offset = r.random()
    if g.kind in ("whale", "ray") and c.pattern == "none" and r.random() < 0.4:
        # Unmarked whales and rays pick up pale individual blotches and scars.
        c.pattern, c.pattern_scale = "spots", U(0.0, 0.35)
        c.accent_color = _hex(c.hue, c.saturation * 0.4, min(1.0, c.value + 0.3))
    c.fin_color = _jit_hex(c.fin_color, r)
    c.accent_color = _jit_hex(c.accent_color, r)
    return g


def random_genome(kind: str, seed: int, archetype: str | None = None, variant: str | None = None) -> FishGenome:
    """A random animal of ``kind`` for ``seed``.

    ``archetype`` (fish body type) only applies to fish; ``variant`` picks one of
    ``VARIANTS[kind]`` instead of letting the seed choose.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {', '.join(KINDS)}")
    if kind == "fish":
        return FishGenome.random(seed, archetype=archetype)
    r = random.Random(f"{kind}:{seed}")
    return _GENERATORS[kind](seed, r, variant)
