"""Random fish "DNA": every shape, fin and colour decision, derived from a seed."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

TAIL_SHAPES = ("forked", "rounded", "truncate", "lunate", "pointed", "fan")
FIN_SHAPES = ("triangle", "rounded", "sail", "long")
PATTERNS = ("none", "stripes", "spots", "lateral", "saddle", "gradient")


@dataclass
class FinGenes:
    present: bool
    start: float  # position along the body (0 = nose, 1 = tail base)
    end: float
    height: float  # relative to body length
    shape: str
    sweep: float  # how far the fin tip leans backwards (0..1)


@dataclass
class ColorGenes:
    hue: float  # 0..1
    saturation: float
    value: float
    fin_hue_shift: float  # added to hue for fins
    accent_hue_shift: float  # added to hue for the pattern colour
    belly_lightness: float  # 0 = belly same as body, 1 = very pale belly
    pattern: str
    pattern_scale: float  # stripe count / spot density
    pattern_offset: float


@dataclass
class FishGenome:
    seed: int
    # --- body (all lengths relative to body length = 1) ---
    height: float  # max body height
    thickness: float  # max body width (left-right), relative to height
    peak: float  # where along the body the height peaks (0..1)
    nose: float  # nose bluntness exponent (small = pointy, big = blunt)
    taper: float  # how quickly the rear thins out
    peduncle: float  # tail-stem thickness, relative to height
    back_arch: float  # how much of the height sits above the spine (0.5 = symmetric)
    # --- head ---
    eye_pos: float  # along body
    eye_height: float  # -1..1 within the body cross-section
    eye_size: float  # relative to body height
    # --- fins ---
    tail_shape: str
    tail_length: float
    tail_spread: float  # how tall the tail is, relative to body height
    dorsal: FinGenes = field(default=None)  # type: ignore[assignment]
    anal: FinGenes = field(default=None)  # type: ignore[assignment]
    pelvic: FinGenes = field(default=None)  # type: ignore[assignment]
    pectoral_size: float = 0.15
    # --- motion ---
    swim_amplitude: float = 0.35  # radians of tail swing
    wave_count: float = 0.8  # body waves along the length
    # --- colour ---
    colors: ColorGenes = field(default=None)  # type: ignore[assignment]

    @classmethod
    def random(cls, seed: int) -> "FishGenome":
        r = random.Random(seed)

        # Body archetype controls proportions so we get believable variety
        # (eel-ish, torpedo, classic, deep-bodied disc) instead of noise.
        archetype = r.choices(
            ["slender", "torpedo", "classic", "deep", "round"],
            weights=[1, 3, 4, 2, 1.5],
        )[0]
        height = {
            "slender": r.uniform(0.2, 0.26),
            "torpedo": r.uniform(0.24, 0.34),
            "classic": r.uniform(0.32, 0.45),
            "deep": r.uniform(0.5, 0.72),
            "round": r.uniform(0.55, 0.75),
        }[archetype]
        peak = r.uniform(0.3, 0.48) if archetype != "round" else r.uniform(0.42, 0.52)
        nose = r.uniform(0.35, 0.8) if archetype != "round" else r.uniform(0.6, 1.0)
        taper = r.uniform(0.8, 1.6) if archetype != "round" else r.uniform(0.5, 0.9)

        tail_shape = r.choice(TAIL_SHAPES)
        if archetype == "slender" and r.random() < 0.5:
            tail_shape = "pointed"

        def fin(present_p: float, start: tuple, end: tuple, h: tuple) -> FinGenes:
            s = r.uniform(*start)
            return FinGenes(
                present=r.random() < present_p,
                start=s,
                end=max(s + 0.08, r.uniform(*end)),
                height=r.uniform(*h),
                shape=r.choice(FIN_SHAPES),
                sweep=r.uniform(0.2, 0.9),
            )

        dorsal = fin(0.95, (0.2, 0.45), (0.5, 0.8), (0.07, 0.2))
        anal = fin(0.8, (0.55, 0.7), (0.72, 0.9), (0.06, 0.18))
        pelvic = fin(0.6, (0.32, 0.45), (0.42, 0.55), (0.06, 0.13))

        hue = r.random()
        colors = ColorGenes(
            hue=hue,
            saturation=r.uniform(0.45, 0.9),
            value=r.uniform(0.6, 0.9),
            fin_hue_shift=r.choice([0.0, 0.0, r.uniform(-0.1, 0.1), r.uniform(0.35, 0.65)]),
            accent_hue_shift=r.choice([0.0, r.uniform(-0.15, 0.15), r.uniform(0.4, 0.6)]),
            belly_lightness=r.uniform(0.0, 1.0),
            pattern=r.choices(PATTERNS, weights=[3, 3, 2, 2, 1.5, 1.5])[0],
            pattern_scale=r.uniform(0.0, 1.0),
            pattern_offset=r.random(),
        )

        return cls(
            seed=seed,
            height=height,
            thickness=r.uniform(0.35, 0.6) if archetype in ("deep", "round") else r.uniform(0.5, 0.8),
            peak=peak,
            nose=nose,
            taper=taper,
            peduncle=r.uniform(0.12, 0.3),
            back_arch=r.uniform(0.45, 0.65),
            eye_pos=r.uniform(0.08, 0.16),
            eye_height=r.uniform(0.15, 0.45),
            eye_size=r.uniform(0.18, 0.32),
            tail_shape=tail_shape,
            tail_length=r.uniform(0.18, 0.32),
            tail_spread=r.uniform(1.0, 1.8) if archetype in ("slender", "torpedo") else r.uniform(0.7, 1.2),
            dorsal=dorsal,
            anal=anal,
            pelvic=pelvic,
            pectoral_size=r.uniform(0.1, 0.2),
            swim_amplitude=r.uniform(0.55, 0.75),
            wave_count=r.uniform(0.6, 1.0),
            colors=colors,
        )

    def to_dict(self) -> dict:
        return asdict(self)
