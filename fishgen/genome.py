"""Random fish "DNA": every shape, fin and colour decision, derived from a seed."""

from __future__ import annotations

import colorsys
import random
from dataclasses import asdict, dataclass, field

TAIL_SHAPES = ("forked", "rounded", "truncate", "lunate", "pointed", "fan")
# Tail shapes random fish never pick, used by the non-fish presets.
# "flukes" lie flat (whales, dolphins); "heterocercal" is the shark tail.
SPECIAL_TAILS = ("heterocercal", "flukes", "none")
ALL_TAIL_SHAPES = TAIL_SHAPES + SPECIAL_TAILS
FIN_SHAPES = ("triangle", "rounded", "sail", "long")
# Fin shapes random fish never pick: "falcate" is a dolphin/shark sickle (trailing edge hollowed
# towards the tail), "pennant" an angelfish streamer whose tip trails far behind.
SPECIAL_FINS = ("falcate", "pennant")
ALL_FIN_SHAPES = FIN_SHAPES + SPECIAL_FINS
PATTERNS = ("none", "stripes", "spots", "lateral", "saddle", "gradient", "bands", "tiger", "shimmer")
# Patterns random fish never pick.
SPECIAL_PATTERNS = ("orca", "rings", "neon", "whale_shark", "mackerel")
ALL_PATTERNS = PATTERNS + SPECIAL_PATTERNS
ARCHETYPES = ("slender", "torpedo", "classic", "deep", "round")
# What sort of animal a genome is. Decided first; every other gene is drawn to suit it (see kinds.py).
KINDS = ("fish", "shark", "whale", "ray", "squid", "octopus", "urchin")
PROFILES = ("fish", "sphere")  # body outline along its length
EYE_STYLES = ("fish", "dark", "none")  # white eye with pupil / plain dark eye / no eyes
WING_SHAPES = ("pointed", "round", "diamond")  # eagle-ray wing / round disc / stingray diamond disc
PECTORAL_SHAPES = ("lobe", "blade")  # fluttering fish fin / long stiff shark or whale fin
LOBE_SHAPES = ("none", "horns", "round")  # forward head lobes on rays: manta horns, cownose lobes


@dataclass
class FinGenes:
    present: bool
    start: float  # position along the body (0 = nose, 1 = tail base)
    end: float
    height: float  # relative to body length
    shape: str
    sweep: float  # how far the fin tip leans backwards (0..1)


@dataclass
class WingGenes:
    """Flat sheets out of the flanks: ray wings, squid fins, dumbo-octopus ears."""

    present: bool = False
    start: float = 0.05  # root leading edge, along the body
    end: float = 0.45  # root trailing edge
    span: float = 0.4  # how far each wing reaches out from the flank (body length = 1)
    sweep: float = 0.6  # how far back the tip sits (0 = level with the root's front, 1 = its back)
    shape: str = "pointed"  # "pointed" (eagle rays, mantas), "round" (round disc) or "diamond" (stingray disc)
    flap: float = 0.6  # flap amplitude, as a fraction of the span
    wave: float = 0.25  # flap waves along the chord (0 = flap in one piece, >1 = ripple)
    dihedral: float = 0.06  # resting upward tilt of the wings, radians (squid fins stand up in a V)


@dataclass
class ArmGenes:
    """Cephalopod arms trailing behind the head."""

    count: int = 0
    length: float = 0.8  # relative to body length
    thickness: float = 0.08  # root radius, relative to body length
    tentacles: int = 0  # extra feeding tentacles (squid have 2)
    tentacle_length: float = 1.3
    spread: float = 0.35  # how far the arms fan out
    curl: float = 0.3  # how much the arm tips writhe


@dataclass
class SpineGenes:
    """Urchin spines radiating from the body."""

    count: int = 0
    length: float = 0.35  # relative to body length
    sway: float = 0.15  # radians of spine wobble


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
    # Optional exact colours ("#rrggbb") that replace the hue-derived fin / pattern colours.
    fin_color: str | None = None
    accent_color: str | None = None
    fin_tips: float = 0.0  # 0 = off; else the outer fraction of each fin painted in the pattern colour
    accent2_color: str | None = None  # second pattern colour (neon tetra red, shimmer, bands); derived if None
    belly_level: float | None = None  # where the pale belly starts (-1 belly .. 1 back); None = from belly_lightness
    pattern_on_fins: bool = False  # stripes carry on across the dorsal, anal and tail fins (angelfish)
    tail_body_color: bool = False  # tail fin painted in the body colours instead of the fin colour


@dataclass
class FishGenome:
    seed: int
    # --- body (all lengths relative to body length = 1) ---
    height: float  # max body height
    thickness: float  # max body width (left-right), relative to height
    peak: float  # where along the body the height peaks (0..1)
    nose: float  # nose shape exponent (small = blunt and rounded, 1 = conical, above 1 = needle-pointed)
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
    # --- other animals (defaults leave an ordinary fish untouched) ---
    kind: str = "fish"  # one of KINDS
    profile: str = "fish"
    beak: float = 0.0  # dolphin rostrum length (0 = none)
    eye_style: str = "fish"
    gill_slits: int = 1  # 1 = a single fish gill line, 5 = shark gill slits, 0 = none
    dorsal2: FinGenes = field(default_factory=lambda: FinGenes(False, 0.75, 0.82, 0.05, "triangle", 0.5))
    pectoral_flap: float = 1.0  # 1 = fluttering fish fins, 0 = stiff shark/dolphin fins
    hammer: float = 0.0  # hammerhead: how far the head wing reaches each side
    swim_vertical: bool = False  # whales and dolphins beat their tail up and down
    wings: WingGenes = field(default_factory=WingGenes)
    arms: ArmGenes = field(default_factory=ArmGenes)
    spines: SpineGenes = field(default_factory=SpineGenes)
    pulse: float = 0.0  # mantle squeeze per swim cycle (squid jet propulsion)
    camera_tilt: float = 0.0  # extra camera tilt, e.g. to see a ray from above
    turn_lag: float = 0.9  # how far the tail trails the head through a turn (0 = turn rigidly)
    size: float = 1.0  # how big the animal is next to others: the sprite canvas scales by this
    pectoral_shape: str = "lobe"  # one of PECTORAL_SHAPES
    pectoral_droop: float = 0.45  # how steeply the pectoral fins angle down (blades)
    melon: float = 0.0  # forehead bulge (dolphins, belugas), relative to body length
    neck: float = 0.0  # neck pinch behind the head (belugas)
    flat_head: float = 0.0  # 0 = head tapers to the middle; 1 = flat top line, the nose up level with the back
    mouth: float = 0.0  # mouth length drawn on the head (0 = none)
    mouth_height: float = -0.35  # where the mouth sits on the head (-1 belly .. 1 back)
    teeth: bool = False
    lobes: str = "none"  # one of LOBE_SHAPES
    lobe_length: float = 0.0
    puff: float = 0.0  # pufferfish: how much it inflates when it puffs up (0 = can't)
    puff_spines: int = 0  # spikes that stand up when puffed
    pleats: bool = False  # throat grooves (rorqual whales)

    @classmethod
    def random(cls, seed: int, archetype: str | None = None) -> "FishGenome":
        """Random fish for ``seed``. ``archetype`` (one of ARCHETYPES) forces the body type."""
        r = random.Random(seed)

        # Body archetype controls proportions so we get believable variety
        # (eel-ish, torpedo, classic, deep-bodied disc) instead of noise.
        # Always draw it so a forced archetype doesn't shift the rest of the RNG stream.
        drawn = r.choices(ARCHETYPES, weights=[1, 3, 4, 2, 1.5])[0]
        if archetype is not None and archetype not in ARCHETYPES:
            raise ValueError(f"unknown archetype {archetype!r}; choose from {', '.join(ARCHETYPES)}")
        archetype = archetype or drawn
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
            saturation=r.uniform(0.55, 0.95),
            value=r.uniform(0.62, 0.95),
            fin_hue_shift=r.choice([0.0, r.uniform(-0.1, 0.1), r.uniform(0.15, 0.3), r.uniform(0.35, 0.65)]),
            accent_hue_shift=r.choice([0.0, r.uniform(-0.15, 0.15), r.uniform(0.25, 0.4), r.uniform(0.4, 0.6)]),
            belly_lightness=r.uniform(0.0, 1.0),
            pattern=r.choices(PATTERNS, weights=[2, 3, 2.5, 2, 1.5, 1.5, 2, 1.5, 1.2])[0],
            pattern_scale=r.uniform(0.0, 1.0),
            pattern_offset=r.random(),
        )
        # A second pattern colour, often a bold contrast (used by bands, shimmer and lateral lines).
        a2 = (hue + r.choice([r.uniform(0.25, 0.4), r.uniform(0.6, 0.75), 0.5])) % 1.0
        colors.accent2_color = "#%02x%02x%02x" % tuple(round(c * 255) for c in colorsys.hsv_to_rgb(
            a2, r.uniform(0.6, 0.95), r.uniform(0.75, 1.0)))
        colors.pattern_on_fins = colors.pattern == "stripes" and r.random() < 0.3
        size = {"slender": r.uniform(1.0, 1.25), "torpedo": r.uniform(0.85, 1.1), "classic": r.uniform(0.7, 0.95),
                "deep": r.uniform(0.75, 0.95), "round": r.uniform(0.55, 0.75)}[archetype]

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
            size=size,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FishGenome":
        """Inverse of to_dict. Genes missing from older dicts keep their defaults."""
        d = dict(d)
        for name in ("dorsal", "anal", "pelvic", "dorsal2"):
            if name in d:
                d[name] = FinGenes(**d[name])
        for name, kind in (("colors", ColorGenes), ("wings", WingGenes), ("arms", ArmGenes), ("spines", SpineGenes)):
            if name in d:
                d[name] = kind(**d[name])
        return cls(**d)
